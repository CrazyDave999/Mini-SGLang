from typing import List
import zmq
import torch

from minisglang.engine.model_runner import ModelRunner
from minisglang.utils.ipc import get_zmq_socket
from dataclasses import dataclass


@dataclass
class GenerationBatchResult:
    bid: int
    logits_output: torch.Tensor
    next_token_ids: torch.Tensor
    
    
class Scheduler:
    """
    For receiving reqs from tokenizer, managing the waiting queue 
    """
    def __init__(
        self,
        model_path: str,
        tp_rank: int,
    ):

        self.tp_rank = tp_rank
        self.model_runner = ModelRunner(model_path, tp_rank)
        
        context = zmq.Context(2)
        if tp_rank == 0:
            self.recv_from_engine = get_zmq_socket(
                context, zmq.PULL, "ipc:///tmp/scheduler", bind=False
            )
        else:
            self.recv_from_engine = None
            
        self.waiting_queue: List[Req] = []
        
    def loop(self):
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            
            batch = self.get_next_batch_to_run()

            if batch is not None:
                result = self.run_batch(batch)
                self.process_batch_result(result)

    def recv_requests(self) -> List[Req]:
        """Receive requests from the engine."""
        if self.tp_rank == 0:
            recv_reqs = []
            while True:
                try:
                    recv_req = self.recv_from_engine.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            # TODO recv_reqs = ??
            recv_reqs = []
        return recv_reqs

    def process_input_requests(self, recv_reqs: List[Req]):
        self.waiting_queue.extend(recv_reqs)

    def get_next_batch_to_run(self) -> Batch:
        """Get the next batch to run."""
    
    def run_batch(self, batch: Batch) -> GenerationBatchResult:
        logits_output, next_token_ids = self.model_runner.forward(batch)
        return GenerationBatchResult(
            bid=batch.bid,
            logits_output=logits_output,
            next_token_ids=next_token_ids,
        )
    def process_batch_result(self, result: GenerationBatchResult):
        pass
    
    
def run_scheduler_process(
    model_path: str,
    tp_rank: int,
    pipe_writer,
):
    scheduler = Scheduler(model_path, tp_rank)
    pipe_writer.send(
        {
            "status": "ready",
        }
    )
    scheduler.loop()
