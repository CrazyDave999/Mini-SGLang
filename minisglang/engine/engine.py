import asyncio
from typing import List, Optional, Union, Dict
import zmq
import multiprocessing as mp

import logging

logger = logging.getLogger(__name__)

from minisglang.engine.tokenizer import TokenizerManager
from minisglang.utils.args import PortArgs, ServerArgs
from minisglang.utils.io_struct import GenerateReqInput
from minisglang.utils.ipc import get_zmq_socket
from minisglang.engine.scheduler import run_scheduler_process


class Engine:
    def __init__(self, server_args: ServerArgs):
        context = zmq.asyncio.Context(2)
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, "ipc:///tmp/engine2scheduler", bind=True
        )
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, "ipc:///tmp/scheduler2engine", bind=True
        )

        tokenizer_manager = _launch_subprocesses(server_args)
        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager

    def generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        stream: bool = False,
    ) -> Dict:
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=stream,
        )
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = loop.run_until_complete(generator.__anext__())
            return ret

    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        stream: bool = False,
    ):
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=stream,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)
        if stream is True:
            return generator
        else:
            return await generator.__anext__()


def _launch_subprocesses(server_args: ServerArgs) -> TokenizerManager:

    # Allocate ports for inter-process communications
    port_args = PortArgs.init_new(server_args)
    logger.info(f"{port_args=}")

    # run the scheduler subprocesses
    scheduler_procs = []
    scheduler_pipe_readers = []
    for tp_rank in range(server_args.tp_size):
        reader, writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=run_scheduler_process,
            args=(server_args, port_args, server_args.model_path, tp_rank, writer),
        )
        proc.start()
        scheduler_procs.append(proc)
        scheduler_pipe_readers.append(reader)

    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            raise

        if data["status"] != "ready":
            raise RuntimeError("Scheduler initialization failed")

    # Launch tokenizer manager process in the main process
    tokenizer_manager = TokenizerManager(server_args, port_args)

    return tokenizer_manager
