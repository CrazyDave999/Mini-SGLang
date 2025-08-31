import random
from types import SimpleNamespace
from typing import List, Optional
from minisglang.utils import TypeBasedDispatcher
from minisglang.utils.io_struct import (
    FlushCacheReqInput,
    FlushCacheReqOutput,
    TokenizedGenerateReqInput,
)
from minisglang.utils.model_config import ModelConfig
import zmq
import torch
from enum import Enum

import logging

logger = logging.getLogger(__name__)

from minisglang.engine.model_runner import ModelRunner
from minisglang.utils.ipc import get_zmq_socket
from minisglang.engine.batch import BaseFinishReason, Batch, Req
from dataclasses import dataclass

from minisglang.utils.args import PortArgs, ServerArgs
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
    finished_reasons: List[BaseFinishReason]

    # for incremental decoding
    decoded_texts: List[str]
    decode_ids: List[List[int]]
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
        self.policy_mode = SchedulePolicyMode(policy.lower())

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
        port_args: PortArgs,
        model_path: str,
        tp_rank: int,
    ):
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.max_running_requests = server_args.max_running_requests
        self.stream_interval = server_args.stream_interval

        self.model_config = ModelConfig(model_path)

        self.model_runner = ModelRunner(
            server_args=server_args,
            model_config=self.model_config,
            tp_rank=tp_rank,
        )
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
            ),
            self.model_runner.page_manager.max_req_num,
        )
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
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
                context, zmq.PULL, port_args.scheduler_input_ipc_name, bind=False
            )
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, bind=False
            )
        else:
            self.recv_from_tokenizer = None
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        self.waiting_queue: List[Req] = []
        self.running_batch: Batch = Batch(reqs=[])

        self.cur_batch: Optional[Batch] = None
        # The last forward batch
        self.last_batch: Optional[Batch] = None

        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (FlushCacheReqInput, self.flush_cache),
            ]
        )

        # Print debug info
        logger.info(
            f"max_total_num_tokens={self.model_runner.max_total_num_tokens}, "
            f"max_running_requests={self.server_args.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

    def loop(self):
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch is not None:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)

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
            recv_reqs = broadcast_pyobj(
                recv_reqs, rank=self.tp_rank, dist_group=self.model_runner.tp_cpu_group
            )

        # if len(recv_reqs) > 0:
        #     logger.info(f"[TP {self.tp_rank}] Recv reqs: {recv_reqs}")
        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            output = self._request_dispatcher(recv_req)
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Create a new request
        req = Req(
            rid=recv_req.rid,
            origin_input_ids=recv_req.input_ids,
            sampling_params=recv_req.sampling_params,
        )
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )
        self.waiting_queue.append(req)

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
            req.init_next_round_input(
                self.tree_cache,
            )
            can_run_list.append(req)

        self.waiting_queue = [
            req for req in self.waiting_queue if req not in set(can_run_list)
        ]
        
        if len(can_run_list) > 0:
            self.log_prefill_stats(can_run_list, running_bs)

        new_batch = (
            Batch(
                reqs=can_run_list, page_manager=self.page_manager, kvcache=self.kvcache
            )
            if can_run_list
            else None
        )
        if new_batch is not None:
            new_batch.prepare_for_prefill()

        return new_batch

    def log_prefill_stats(
        self,
        can_run_list: List[Req],
        running_bs: int,
    ):
        num_new_seq = len(can_run_list)
        f = (
            f"Prefill batch. "
            f"#new-seq: {num_new_seq}, "
        )
        f += f"#running-req: {running_bs}, "
        f += f"#queue-req: {len(self.waiting_queue)}, "
        logger.info(f"[TP {self.tp_rank}] {f}")
        
        
    def log_decode_stats(
        self, batch: Batch
    ):
        num_running_reqs = len(batch.reqs)
        msg = f"Decode batch. #running-req: {num_running_reqs}"
        msg += (
            f"#queue-req: {len(self.waiting_queue)}, "
        )
        logger.info(f"[TP {self.tp_rank}] {msg}")
    # def update_running_batch(self, batch: Batch) -> Batch:
    #     """Update the current running decoding batch."""

    def run_batch(self, batch: Batch) -> GenerationBatchResult:
        logits_output, next_token_ids = self.model_runner.forward(batch)
        batch.output_ids = next_token_ids
        return GenerationBatchResult(
            bid=batch.bid,
            logits_output=logits_output,
            next_token_ids=next_token_ids,
        )

    def flush_cache(self, _: FlushCacheReqInput) -> FlushCacheReqOutput:
        if len(self.waiting_queue) == 0 and self.running_batch.is_empty():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            self.page_manager.clear()
            self.kvcache.clear()

            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            is_succuss = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            is_succuss = False
        return FlushCacheReqOutput(success=is_succuss)

    def process_batch_result(self, batch: Batch, result: GenerationBatchResult):
        if batch.mode.is_prefill():
            self.process_output_prefill(batch, result)
        elif batch.mode.is_decode():
            self.process_output_decode(batch, result)

    def process_output_prefill(self, batch: Batch, result: GenerationBatchResult):
        next_token_ids, bid = result.next_token_ids.tolist(), result.bid
        for req, next_token_id in zip(batch.reqs, next_token_ids):
            req.output_ids.append(next_token_id)
            req.check_finished()
            if req.finished():
                self.tree_cache.cache_finished_req(req)
            else:
                self.tree_cache.cache_unfinished_req(req)

        self.stream_output(batch.reqs)

    def process_output_decode(self, batch: Batch, result: GenerationBatchResult):
        self.log_decode_stats(batch)
        next_token_ids, bid = result.next_token_ids.tolist(), result.bid
        for req, next_token_id in zip(batch.reqs, next_token_ids):
            req.output_ids.append(next_token_id)
            req.check_finished()
            if req.finished():
                self.tree_cache.cache_finished_req(req)

        self.stream_output(batch.reqs)

    def stream_output(self, reqs: List[Req]):
        rids = []
        finished_reasons = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []

        for req in reqs:
            if req.finished() or len(req.output_ids) % self.stream_interval == 0:
                rids.append(req.rid)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decode_ids, read_offset = req.init_incremental_detokenize()
                decoded_texts.append(req.decoded_text)
                decode_ids_list.append(decode_ids)
                read_offsets.append(read_offset)

        if rids:
            self.send_to_tokenizer.send_pyobj(
                BatchTokenIDOut(
                    rids=rids,
                    finished_reasons=finished_reasons,
                    decoded_texts=decoded_texts,
                    decode_ids=decode_ids_list,
                    read_offsets=read_offsets,
                )
            )


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    model_path: str,
    tp_rank: int,
    pipe_writer,
):
    scheduler = Scheduler(server_args, port_args, model_path, tp_rank)
    pipe_writer.send(
        {
            "status": "ready",
        }
    )
    scheduler.loop()
