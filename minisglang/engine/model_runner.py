import time
from typing import Optional
from click import Option
from minisglang.layers.attention_backends.flash_attention_backend import FlashAttentionBackend
from minisglang.memory import page_manager
from minisglang.utils import get_available_gpu_memory
import torch
from torch import nn
import os
from glob import glob
from safetensors import safe_open
import torch.distributed as dist

import logging
logger = logging.getLogger(__name__)

from minisglang.engine.batch import Batch
from minisglang.utils.model_config import ModelConfig
from minisglang.layers.sampler import Sampler
from minisglang.memory.kvcache import KVCache
from minisglang.memory.page_manager import PageManager
from minisglang.layers.attention_backends.torch_native_backend import (
    TorchNativeAttnBackend,
)
from minisglang.models.llama import LlamaForCausalLM
from minisglang.utils.args import ServerArgs

from minisglang.memory.radix_cache import PagedRadixCache
from minisglang.engine.cuda_graph_runner import CudaGraphRunner

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def get_model(
    model_config: ModelConfig,
) -> nn.Module:
    model = LlamaForCausalLM(model_config.hf_config)
    # load weight
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(model_config.model_path, "*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))
    return model


class ModelRunner:
    def __init__(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        tp_rank: int,
    ):
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.model_config = model_config
        self.device = server_args.device
        self.dtype = model_config.dtype
        self.page_size = server_args.page_size
        self.mem_fraction_static = server_args.mem_fraction_static
        
        torch.cuda.set_device(self.tp_rank)
        torch.set_default_device("cuda")
        torch.set_default_dtype(self.model_config.dtype)
        
        # Get memory before model loading
        min_per_gpu_memory = self.init_torch_distributed()
        self.tp_cpu_group = dist.new_group(ranks=[i for i in range(self.tp_size)], backend="gloo")
        
        # Load the model
        self.model = get_model(self.model_config)
        self.sampler = Sampler()

        # init memory
        self.init_memory_pool(min_per_gpu_memory, server_args.max_running_requests, server_args.max_total_tokens)

        # self.attn_backend = TorchNativeAttnBackend(self)
        self.attn_backend = FlashAttentionBackend(self)
        
        self.init_cuda_graphs()
        
        
    def init_torch_distributed(self):
        logger.info("Init torch distributed begin.")
        torch.get_device_module(self.device).set_device(self.tp_rank)

        before_avail_memory = get_available_gpu_memory(self.device, self.tp_rank)
        
        dist.init_process_group(
            "nccl", "tcp://localhost:33000", world_size=self.tp_size, rank=self.tp_rank, device_id=torch.device("cuda", self.tp_rank)
        )
        
        min_per_gpu_memory = get_available_gpu_memory(
            self.device, self.tp_rank, distributed=self.tp_size > 1
        )
        local_gpu_memory = get_available_gpu_memory(self.device, self.tp_rank)
        logger.info(
            f"Init torch distributed ends. mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return min_per_gpu_memory
    
    def init_memory_pool(
        self,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)
        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )
            
        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logging.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{self.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)
            
        
        self.max_total_num_tokens = (
            self.max_total_num_tokens
            // self.server_args.page_size
            * self.server_args.page_size
        )
        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )
            
        self.page_manager = PageManager(
            page_size=self.page_size,
            max_req_num=max_num_reqs,
            max_context_len=self.model_config.context_len + 4,
            device=self.device,
        )
        
        self.kvcache = KVCache(
            size=self.max_total_num_tokens,
            page_size=self.page_size,
            head_num=self.model_config.get_num_kv_heads_per_GPU(self.tp_size),
            head_dim=self.model_config.head_dim,
            dtype=self.dtype,
            layer_num=self.model_config.num_hidden_layers,
            device=self.device,
        )
        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.tp_rank):.2f} GB"
        )
    
    def profile_max_num_token(self, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            self.device, self.tp_rank, distributed=self.tp_size > 1
        )
        cell_size = (
            self.model_config.get_num_kv_heads_per_GPU(self.tp_size)
            * self.model_config.head_dim
            * self.model_config.num_hidden_layers
            * 2
            * torch._utils._element_size(self.dtype)
        )
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        max_num_token = int(rest_memory * (1 << 30) // cell_size)
        return max_num_token
    
    def forward(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        can_run_cuda_graph = bool(
            batch.mode.is_decode()
            and self.cuda_graph_runner
            and self.cuda_graph_runner.can_run(batch)
        )
        if can_run_cuda_graph:
            return self.cuda_graph_runner.replay(batch)

        self.attn_backend.init_forward_metadata(batch)
        batch.attn_backend = self.attn_backend
        
        print(f"[TP {self.tp_rank}] Running batch: {batch.input_ids.shape=} {batch.positions.shape=}")
        logits_output = self.model.forward(batch.input_ids, batch.positions, batch)
        print(f"[TP {self.tp_rank}] Model forward done. {logits_output.shape=}")

        # logits_output: shape = (bs, vocab_size)
        return logits_output
    
    def sample(
        self,
        logits: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        
        next_token_ids = self.sampler(
            logits=logits, sampling_info=batch.sampling_info
        )
        print(f"sample. {next_token_ids.shape=}")
        return next_token_ids
            
    
    
    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_mem_usage = 0
        
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.tp_rank)
        logger.info(
            f"Capture cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.cuda_graph_runner = CudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.tp_rank)
        self.cuda_graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.cuda_graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )