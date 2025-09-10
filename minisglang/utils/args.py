
import argparse
import dataclasses

from dataclasses import dataclass
import random
import tempfile
from typing import List, Optional

from minisglang.utils import get_nvgpu_memory_capacity, is_port_available



@dataclass
class ServerArgs:
    model_path: str = None
    tokenizer_path: str = None
    device: str = "cuda"
    
    # Port for the HTTP server
    host: str = "127.0.0.1"
    port: int = 30000
    
    # Memory and scheduling
    mem_fraction_static: float = 0.8
    max_running_requests: Optional[int] = None
    max_total_tokens: int = 1024 * 1024
    schedule_policy: str = "fcfs"
    page_size: int = 1
    schedule_conservativeness: float = 1.0
    
    # Other runtime options
    tp_size: int = 1
    stream_interval: int = 1
    stream_output: bool = False
    
    # Optimization/debug options
    cuda_graph_max_bs: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    enable_torch_compile: bool = False
    
    # Logging
    log_level: str = "info"
    
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # Model and port args
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=ServerArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--host", type=str, default=ServerArgs.host, help="The host of the server."
        )
        parser.add_argument(
            "--port", type=int, default=ServerArgs.port, help="The port of the server."
        )

        parser.add_argument(
            "--device",
            type=str,
            default=ServerArgs.device,
            help="The device to use ('cuda', 'xpu', 'hpu', 'cpu'). Defaults to auto-detection if not specified.",
        )
        
        # Memory and scheduling
        parser.add_argument(
            "--mem-fraction-static",
            type=float,
            default=ServerArgs.mem_fraction_static,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-running-requests",
            type=int,
            default=ServerArgs.max_running_requests,
            help="The maximum number of running requests.",
        )
        parser.add_argument(
            "--max-total-tokens",
            type=int,
            default=ServerArgs.max_total_tokens,
            help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. "
            "This option is typically used for development and debugging purposes.",
        )
        parser.add_argument(
            "--schedule-policy",
            type=str,
            default=ServerArgs.schedule_policy,
            choices=["lpm", "random", "fcfs", "dfs-weight"],
            help="The scheduling policy of the requests.",
        )
        parser.add_argument(
            "--schedule-conservativeness",
            type=float,
            default=ServerArgs.schedule_conservativeness,
            help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            default=ServerArgs.page_size,
            help="The number of tokens in a page.",
        )
        
        # Other runtime options
        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=ServerArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=ServerArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--stream-output",
            action="store_true",
            help="Whether to output as a sequence of disjoint segments.",
        )
        
        # Optimization/debug options
        parser.add_argument(
            "--cuda-graph-max-bs",
            type=int,
            default=ServerArgs.cuda_graph_max_bs,
            help="Set the maximum batch size for cuda graph. It will extend the cuda graph capture batch size to this value.",
        )
        parser.add_argument(
            "--cuda-graph-bs",
            type=int,
            nargs="+",
            help="Set the list of batch sizes for cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Disable cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph-padding",
            action="store_true",
            help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile. Experimental feature.",
        )
        
        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
    
    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        
        gpu_mem = get_nvgpu_memory_capacity()
        
        # Set cuda graph max batch size
        if self.cuda_graph_max_bs is None:
            # Based on detailed statistics, when serving TP1/TP2 models on lower-end GPUs with HBM<25G, you can either disable cuda graph or set `cuda_graph_max_bs` to a very small value to reduce the memory overhead of creating cuda graphs, with almost no impact on performance. However, when serving models with TP4 or TP8, we need to enable cuda graph to maintain high performance. In this case, we can set `cuda_graph_max_bs` to 80 (half of the default value 160) to reduce the memory overhead of creating cuda graphs. Looking at the logs from TP4 serving of qwen2-72b, a value of 80 is sufficient and can reduce the memory overhead of creating cuda graphs on lower-end GPUs compared to the original 160, avoiding OOM issues.
            if gpu_mem is not None and gpu_mem < 35 * 1024:
                if self.tp_size < 4:
                    self.cuda_graph_max_bs = 8
                else:
                    self.cuda_graph_max_bs = 80
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.tp_size = args.tensor_parallel_size
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def url(self):
        return f"http://{self.host}:{self.port}"

def prepare_server_args(argv: List[str]) -> ServerArgs:
    """
    Prepare the server arguments from the command line arguments.

    Args:
        args: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The server arguments.
    """
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args)
    return server_args




ZMQ_TCP_PORT_DELTA = 233


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for tokenizer to receive inputs from detokenizer (zmq)
    tokenizer_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str
    # The ipc filename for detokenizer to receive inputs from scheduler (zmq)
    detokenizer_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    @staticmethod
    def init_new(server_args) -> "PortArgs":
        port = server_args.port + random.randint(100, 1000)
        while True:
            if is_port_available(port):
                break
            if port < 60000:
                port += 42
            else:
                port -= 43

        # Normal case, use IPC within a single node
        return PortArgs(
            tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            nccl_port=port,
        )
        
        
"""Global configurations"""

import os


class GlobalConfig:
    def __init__(self):
        # Verbosity level
        # 0: do not output anything
        # 2: output final text after every run
        self.verbosity = 0

        # Default backend of the language
        self.default_backend = None

        # Runtime constants: New generation token ratio estimation
        self.default_init_new_token_ratio = float(
            os.environ.get("SGLANG_INIT_NEW_TOKEN_RATIO", 0.7)
        )
        self.default_min_new_token_ratio_factor = float(
            os.environ.get("SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR", 0.14)
        )
        self.default_new_token_ratio_decay_steps = float(
            os.environ.get("SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS", 600)
        )

        # Runtime constants: others
        self.retract_decode_steps = 20
        self.flashinfer_workspace_size = os.environ.get(
            "FLASHINFER_WORKSPACE_SIZE", 384 * 1024 * 1024
        )

        # Output tokenization configs
        self.skip_special_tokens_in_output = True
        self.spaces_between_special_tokens_in_out = True

        # Interpreter optimization configs
        self.enable_precache_with_tracing = True
        self.enable_parallel_encoding = True


global_config = GlobalConfig()
