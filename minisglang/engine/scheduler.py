from typing import List
import zmq
import torch

from minisglang.engine.model_runner import ModelRunner
from minisglang.utils.ipc import get_zmq_socket
from minisglang.engine.batch import Batch, Req
from dataclasses import dataclass


@dataclass
class GenerationBatchResult:
    bid: int
    logits_output: torch.Tensor
    next_token_ids: torch.Tensor


@dataclass
class BatchTokenIDOut:
    rids: List[int]
    finished_reasons: List[str]


class Scheduler:
    """
    For receiving reqs from tokenizer, managing the waiting queue
    """

    def __init__(
        self,
        model_path: str,
        tp_rank: int,
        max_running_requests: int,
    ):

        self.tp_rank = tp_rank
        self.max_running_requests = max_running_requests
        self.model_runner = ModelRunner(model_path, tp_rank)

        context = zmq.Context(2)
        if tp_rank == 0:
            self.recv_from_engine = get_zmq_socket(
                context, zmq.PULL, "ipc:///tmp/engine2scheduler", bind=False
            )
            self.send_to_engine = get_zmq_socket(
                context, zmq.PUSH, "ipc:///tmp/scheduler2engine", bind=False
            )
        else:
            self.recv_from_engine = None

        self.waiting_queue: List[Req] = []
        # self.running_batch: Batch = Batch(reqs=[])

    def loop(self):
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch is not None:
                result = self.run_batch(batch)
                self.process_batch_result(result)

            self.last_batch = batch

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
        if self.last_batch is not None and self.last_batch.mode.is_prefill():
            self.last_batch.filter_batch()
            if not self.last_batch.is_empty():
                self.last_batch.prepare_for_decode()
                return self.last_batch

        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            return new_batch

        return None

    def get_new_batch_prefill(self) -> Batch:
        if len(self.waiting_queue) == 0:
            return None

        can_run_list: List[Req] = self.waiting_queue[
            : min(len(self.waiting_queue), self.max_running_requests)
        ]

        self.waiting_queue = [
            req for req in self.waiting_queue if req not in set(can_run_list)
        ]

        new_batch = Batch(
            reqs=can_run_list, page_manager=self.page_manager, kvcache=self.kvcache
        )
        new_batch.prepare_for_prefill()

        return new_batch

    # def update_running_batch(self, batch: Batch) -> Batch:
    #     """Update the current running decoding batch."""

    def run_batch(self, batch: Batch) -> GenerationBatchResult:
        logits_output, next_token_ids = self.model_runner.forward(batch)
        return GenerationBatchResult(
            bid=batch.bid,
            logits_output=logits_output,
            next_token_ids=next_token_ids,
        )

    def process_batch_result(self, batch: Batch, result: GenerationBatchResult):
        next_token_ids, bid = result.next_token_ids, result.bid
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req.output_ids.append(next_token_id)
            req.check_finished()

        self.stream_output(batch.reqs)

    def stream_output(self, reqs: List[Req]):
        rids = []
        finished_reasons = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                finished_reasons.append(req.finished_reason)

        if rids:
            self.send_to_engine.send_pyobj(
                BatchTokenIDOut(rids=rids, finished_reasons=finished_reasons)
            )


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
