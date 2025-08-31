"""Launch the inference server."""

import os
import sys

from minisglang.entrypoints.http_server import launch_server
from minisglang.utils.args import prepare_server_args
from minisglang.utils import configure_logger, kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    configure_logger(server_args)

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
