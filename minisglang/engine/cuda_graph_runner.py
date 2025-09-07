import bisect
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Callable, TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from minisglang.engine.model_runner import ModelRunner

import tqdm

import torch
import torch.distributed

logger = logging.getLogger(__name__)
from minisglang.engine.batch import Batch, Mode

from minisglang.utils import get_available_gpu_memory, get_nvgpu_memory_capacity

# Detect whether the current forward pass is in capture mode
is_capture_mode = False


def get_is_capture_mode():
    return is_capture_mode


@contextmanager
def model_capture_mode():
    global is_capture_mode
    is_capture_mode = True

    yield

    is_capture_mode = False
    
@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream
    

@contextmanager
def graph_capture():
    pass

# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val
    
class CudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: "ModelRunner"):
        # Parse args
        self.model_runner = model_runner
        self.graphs = {}
        self.output_buffers = {}
        # TODO parameters in ServerArgs
        self.enable_torch_compile = True
        self.disable_padding = False

        self.tp_rank = model_runner.tp_rank
        self.tp_size = model_runner.tp_size

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        logger.info(f"Capture cuda graph bs {self.capture_bs}")
        self.capture_forward_mode = Mode.DECODE
        self.num_tokens_per_bs = 1

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.model_runner.attn_backend.init_cuda_graph_state(self.max_bs)
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), 1, dtype=torch.int32, device="cpu"
        )

        # Graph inputs
        with torch.device("cuda"):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.page_table_ids = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.seq_lens = torch.full((self.max_bs,), 1, dtype=torch.int32)
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
        
        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )
    def can_run(self, batch: Batch):
        cuda_graph_bs = batch.batch_size()
        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )
        return is_bs_supported 
    
    def capture(self) -> None:
        capture_range = (
            tqdm.tqdm(list(reversed(self.capture_bs)))
            if self.tp_rank == 0
            else reversed(self.capture_bs)
        )
        for bs in capture_range:
            if self.tp_rank == 0:
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.tp_rank,
                    empty_cache=False
                )
                capture_range.set_description(
                    f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                )
            
            graph, logits_output = self.capture_one_batch_size(bs)
            self.graphs[bs] = graph
            self.output_buffers[bs] = logits_output
            
    
    def capture_one_batch_size(self, bs:int):
        graph = torch.cuda.CUDAGraph()
        num_tokens = bs * self.num_tokens_per_bs
        
        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        page_table_ids = self.page_table_ids[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        
        self.num_token_non_padded[...] = num_tokens
        
        batch = Batch(
            reqs=None,
            page_manager=self.model_runner.page_manager,
            kvcache=self.model_runner.kvcache,
            tree_cache=None
        )
        batch.mode = self.capture_forward_mode
        batch.input_ids = input_ids
        batch.page_table_ids = page_table_ids
        batch.seq_lens = seq_lens
        batch.seq_lens_cpu = seq_lens.cpu()
        batch.out_cache_loc = out_cache_loc
        batch.positions = positions

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            page_table_ids=page_table_ids,
            seq_lens=seq_lens,
            forward_mode=batch.mode
        )
        
        # Run and capture
        def run_once():
            return self.model_runner.forward(batch)
        
        torch.cuda.synchronize()
        torch.distributed.barrier()
        run_once()  # Warm up
        
        global global_graph_memory_pool
        with torch.cuda.graph(graph, pool=global_graph_memory_pool):
            logits_output = run_once()
            
        global_graph_memory_pool = graph.pool()
        return graph, logits_output
        
    def replay_prepare(
        self,
        batch: Batch,
    ):
        raw_bs = batch.batch_size()
        raw_num_token = raw_bs * self.num_tokens_per_bs
        
        # Padding
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()
            
        # Common inputs
        self.input_ids[:raw_num_token].copy_(batch.input_ids)
        self.page_table_ids[:raw_bs].copy_(batch.page_table_ids)
        self.seq_lens[:raw_bs].copy_(batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(batch.positions)
        
        # TODO seq_lens_cpu ?
        if bs != raw_bs:
            self.seq_lens_cpu.fill_(1)
        self.seq_lens_cpu[:raw_bs].copy_(batch.seq_lens[:raw_bs].to("cpu"))
        
        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            page_table_ids=self.page_table_ids[:bs],
            seq_lens=self.seq_lens[:bs],
            seq_lens_sum=batch.seq_lens_sum + (bs - raw_bs) * 1,
            forward_mode=self.capture_forward_mode,
            seq_lens_cpu=self.seq_lens_cpu[:bs],
        )
        
        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs
    
    def replay(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        self.replay_prepare(batch)
        
        # Replay
        self.graphs[self.bs].replay()

        logits_output = self.output_buffers[self.bs]
        return logits_output[: self.raw_num_token]
        
        


def get_batch_sizes_to_capture(model_runner: "ModelRunner"):
    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs

    if capture_bs is None:
        if server_args.disable_cuda_graph_padding:
            capture_bs = list(range(1, 33)) + list(range(48, 161, 16))
        else:
            capture_bs = [1, 2, 4, 8] + list(range(16, 161, 8))

        gpu_mem = get_nvgpu_memory_capacity()
        if gpu_mem is not None:
            if gpu_mem > 90 * 1024:  # H200, H20
                capture_bs += list(range(160, 257, 8))
            if gpu_mem > 160 * 1000:  # B200, MI300
                capture_bs += list(range(256, 513, 16))

    if max(capture_bs) > model_runner.page_manager.max_req_num:
        # In some cases (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs += [model_runner.page_manager.max_req_num]

    if server_args.cuda_graph_max_bs:
        capture_bs = [bs for bs in capture_bs if bs <= server_args.cuda_graph_max_bs]
        if max(capture_bs) < server_args.cuda_graph_max_bs:
            capture_bs += list(
                range(max(capture_bs), server_args.cuda_graph_max_bs + 1, 16)
            )
    capture_bs = [
        bs for bs in capture_bs if bs <= model_runner.page_manager.max_req_num
    ]
    capture_bs = list(sorted(set(capture_bs)))
    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs

CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --cuda-graph-max-bs to a smaller value (e.g., 16)\n"
    "3. disable torch compile by not using --enable-torch-compile\n"
    "4. disable CUDA graph by --disable-cuda-graph. (Not recommended. Huge performance loss)\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
