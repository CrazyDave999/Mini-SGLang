
import argparse
import dataclasses

from dataclasses import dataclass
import random
import tempfile
from typing import List, Optional

from minisglang.utils import is_port_available



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
    max_running_requests: int = 8
    max_total_tokens: int = 1024 * 1024
    schedule_policy: str = "fcfs"
    page_size: int = 1
    
    # Other runtime options
    tp_size: int = 1
    stream_interval: int = 1
    stream_output: bool = False
    
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
        
        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        
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
        