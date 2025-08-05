from typing import List, Optional, Union, Dict
from transformers import AutoTokenizer, PreTrainedTokenizer
import zmq
import multiprocessing as mp

from dataclasses import dataclass

from minisglang.utils.ipc import get_zmq_socket
from minisglang.engine.batch import Req, Batch
from minisglang.engine.scheduler import run_scheduler_process


class Engine:
    def __init__(self, model_path: str, tokenizer_path: str, tp_size: int):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        context = zmq.asyncio.Context(2)
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, "ipc:///tmp/engine2scheduler", bind=True
        )
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, "ipc:///tmp/scheduler2engine", bind=True
        )

        # run the scheduler subprocesses
        scheduler_procs = []
        scheduler_pipe_readers = []
        for tp_rank in range(tp_size):
            reader, writer = mp.Pipe(duplex=False)
            proc = mp.Process(
                target=run_scheduler_process, args=(model_path, tp_rank, writer)
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

    def generate(
        self,
        prompts: List[str],
        stream: bool = False,
    ) -> Dict:
        if stream:
            pass
        else:
            for prompt in prompts:
                input_ids = self.tokenizer.encode(prompt)
                req = Req(input_ids=input_ids)
                self._send_one_req(req)
            while True:
                try:
                    result = self.recv_from_scheduler.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

    def _send_one_req(
        self,
        req: Req,
    ):
        # send one request to the scheduler
        self.send_to_scheduler.send_pyobj(req)
        pass
