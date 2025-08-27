import random
from typing import List
import zmq
import torch
from enum import Enum
import torch.distributed as dist

from minisglang.engine.model_runner import ModelRunner
from minisglang.utils.ipc import get_zmq_socket
from minisglang.engine.batch import Batch, Req
from dataclasses import dataclass

from minisglang.utils.args import ServerArgs
from minisglang.memory.radix_cache import PagedRadixCache
from minisglang.utils.ipc import broadcast_pyobj


@dataclass
class GenerationBatchResult:
    bid: int
    logits_output: torch.Tensor
    next_token_ids: torch.Tensor


@dataclass
class BatchTokenIDOut:
    rids: List[int]
    finished_reasons: List[str]

    # for incremental decoding
    decode_ids_list: List[List[int]]
    read_offsets: List[int]


class SchedulePolicyMode(Enum):
    LPM = "lpm"
    FCFS = "fcfs"
    RANDOM = "random"


class SchedulePolicy:
    def __init__(
        self,
        policy: str,
        tree_cache: PagedRadixCache,
    ):
        self.tree_cache = tree_cache
        self.policy_mode = SchedulePolicyMode(policy.upper())

    def calc_priority(self, waiting_queue: List[Req]):
        policy_mode = self.policy_mode
        # Turn off the expensive prefix matching and sorting when the #queue is large.
        if policy_mode == SchedulePolicyMode.LPM and len(waiting_queue) > 128:
            policy_mode = SchedulePolicyMode.FCFS

        if policy_mode == SchedulePolicyMode.LPM:
            self._compute_prefix_matches(waiting_queue)
            SchedulePolicy._sort_by_longest_prefix(waiting_queue)
        elif policy_mode == SchedulePolicyMode.FCFS:
            pass
        elif policy_mode == SchedulePolicyMode.RANDOM:
            SchedulePolicy._sort_randomly(waiting_queue)
        else:
            raise ValueError(f"Unknown schedule policy mode: {policy_mode}")

    @staticmethod
    def _sort_by_longest_prefix(waiting_queue: List[Req]):
        waiting_queue.sort(key=lambda req: len(req.prefix), reverse=True)

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        random.shuffle(waiting_queue)

    def _compute_prefix_matches(self, waiting_queue: List[Req]):
        """
        Computes and caches the matching prefixes for requests in the waiting queue
        """
        for r in waiting_queue:
            r.prefix_ppns, r.last_node = self.tree_cache.match_prefix(
                key=r.fill_ids,
            )


class Scheduler:
    """
    For receiving reqs from tokenizer, managing the waiting queue
    """

    def __init__(
        self,
        server_args: ServerArgs,
        model_path: str,
        tp_rank: int,
    ):

        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.max_running_requests = server_args.max_running_requests
        self.stream_interval = server_args.stream_interval

        self.model_runner = ModelRunner(
            server_args=server_args,
            model_path=model_path,
            tp_rank=tp_rank,
            device="cuda",
        )
        self.page_manager = self.model_runner.page_manager
        self.kvcache = self.model_runner.kvcache

        self.tree_cache = PagedRadixCache(
            page_manager=self.page_manager, kvcache=self.kvcache
        )

        self.policy = SchedulePolicy(
            policy=server_args.schedule_policy,
            tree_cache=self.tree_cache,
        )

        context = zmq.Context(2)
        if tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, "ipc:///tmp/tokenizer2scheduler", bind=False
            )
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, "ipc:///tmp/scheduler2tokenizer", bind=False
            )
        else:
            self.recv_from_tokenizer = None

        self.waiting_queue: List[Req] = []
        self.running_batch: Batch = Batch(reqs=[])

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
        """Receive requests from the engine. Only tp rank 0 will receive reqs from Engine by zmq. Other ranks should get reqs by broadcast"""
        if self.tp_rank == 0:
            recv_reqs = []
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = None

        if self.tp_size != 1:
            recv_reqs = broadcast_pyobj(recv_reqs, rank=self.tp_rank)

        return recv_reqs

    def process_input_requests(self, recv_reqs: List[Req]):
        self.waiting_queue.extend(recv_reqs)

    def get_next_batch_to_run(self) -> Batch:
        """Get the next batch to run."""
        if self.last_batch is not None and self.last_batch.mode.is_prefill():
            # last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch()

            if not self.last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)

        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            # Run prefill first if possible
            return new_batch
        else:
            # Run decode batch
            if not self.running_batch.is_empty():
                self.running_batch.filter_batch()
                # TODO check if decode out of memory
                self.running_batch.prepare_for_decode()
                return self.running_batch

        return None

    def get_new_batch_prefill(self) -> Batch:
        running_bs = self.running_batch.batch_size()
        if running_bs >= self.max_running_requests:
            return None

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue)

        can_run_list = []

        for req in self.waiting_queue:
            if running_bs + len(can_run_list) >= self.max_running_requests:
                break
            # TODO do some max token check
            req.init_next_round_input()
            can_run_list.append(req)

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
        if batch.mode.is_prefill():
            self.process_output_prefill(batch, result)
        elif batch.mode.is_decode():
            self.process_output_decode(batch, result)

    def process_output_prefill(self, batch: Batch, result: GenerationBatchResult):
        next_token_ids, bid = result.next_token_ids.to_list(), result.bid
        for req, next_token_id in zip(batch.reqs, next_token_ids):
            req.output_ids.append(next_token_id)
            req.check_finished()
            if req.finished():
                self.tree_cache.cache_finished_req(req)
            else:
                self.tree_cache.cache_unfinished_req(req)

        self.stream_output(batch.reqs)

    def process_output_decode(self, batch: Batch, result: GenerationBatchResult):
        next_token_ids, bid = result.next_token_ids.to_list(), result.bid
        for req, next_token_id in zip(batch.reqs, next_token_ids):
            req.output_ids.append(next_token_id)
            req.check_finished()
            if req.finished():
                self.tree_cache.cache_finished_req(req)

        self.stream_output(batch.reqs)

    def stream_output(self, reqs: List[Req]):
        rids = []
        finished_reasons = []

        decode_ids_list = []
        read_offsets = []

        for req in reqs:
            if req.finished() or len(req.output_ids % self.stream_interval == 0):
                rids.append(req.rid)
                finished_reasons.append(req.finished_reason)
                decode_ids, read_offset = req.init_incremental_detokenize()
                decode_ids_list.append(decode_ids)
                read_offsets.append(read_offset)

        if rids:
            self.send_to_tokenizer.send_pyobj(
                BatchTokenIDOut(
                    rids=rids,
                    finished_reasons=finished_reasons,
                    decode_ids_list=decode_ids_list,
                    read_offsets=read_offsets,
                )
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
