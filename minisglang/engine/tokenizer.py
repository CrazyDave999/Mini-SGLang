import asyncio
import dataclasses
import enum
from http import HTTPStatus
import os
import signal
import threading
import time
from typing import Any, Dict, List, Optional, Union

from lark import logger
from minisglang.layers.sampler import SamplingParams
from minisglang.utils.model_config import ModelConfig
from sympy import li
import zmq
import zmq.asyncio
from transformers import AutoTokenizer
import fastapi
import uvloop

from minisglang.engine.batch import Req
from minisglang.engine.scheduler import BatchTokenIDOut
from minisglang.utils.args import PortArgs, ServerArgs
from minisglang.utils.io_struct import (
    BatchStrOut,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from minisglang.utils.ipc import get_zmq_socket
from minisglang.utils import _Communicator, LimitedCapacityDict, TypeBasedDispatcher, find_printable_text

import logging
logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))



@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: List
    finished: bool
    event: asyncio.Event
    obj: Any

    # For metrics
    created_time: float
    finished_time: float = 0.0
    first_token_time: float = 0.0
    last_time: float = 0.0
    last_completion_tokens: int = 1

    # For streaming output
    last_output_offset: int = 0


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    decoded_text: str
    decode_ids: List[int]
    surr_offset: (
        int  # 一个输出字符可能涉及多个tokens，目前能安全解码的最后一个token的位置
    )
    read_offset: int  # 目前读取的最后位置


class TokenizerManager:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs
    ):
        self.server_args = server_args
        self.port_args = port_args
        self.model_path = server_args.model_path
        self.tokenizer_path = server_args.tokenizer_path
        
        self.model_config = ModelConfig(
            model_path=server_args.model_path,
        )
        self.context_len = self.model_config.context_len

        context = zmq.asyncio.Context(2)
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, self.port_args.scheduler_input_ipc_name, bind=True
        )
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, self.port_args.tokenizer_ipc_name, bind=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(server_args.tokenizer_path)

        self.created_loop: bool = False
        self.rid_to_state: Dict[str, ReqState] = {}
        
        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)

        # Communicators
        self.flush_cache_communicator = _Communicator(self.send_to_scheduler, 1)

        self._result_dispatcher = TypeBasedDispatcher(
            [
                (BatchTokenIDOut, self._handle_batch_token_id_out),
                (FlushCacheReqOutput, self.flush_cache_communicator.handle_recv),
            ]
        )

    async def generate_request(
        self,
        obj: GenerateReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        """generator for results coming from batch request"""
        created_time = time.time()
        self.auto_create_handle_loop()
        
        obj.normalize_batch_and_args()

        is_single = obj.is_single
        if is_single:
            tokenized_obj = await self._tokenize_one_request(obj)
            # logger.info(f"Sending tokenized request to scheduler: {tokenized_obj=}")
            self._send_one_request(obj, tokenized_obj, created_time)
            async for response in self._wait_one_response(obj, request):
                yield response
        else:
            async for response in self._handle_batch_request(
                obj, request, created_time
            ):
                yield response

    def get_internal_state(self):
        # TODO
        return None

    async def flush_cache(self) -> FlushCacheReqOutput:
        return (await self.flush_cache_communicator(FlushCacheReqInput()))[0]

    def auto_create_handle_loop(self):
        if self.created_loop:
            return

        self.created_loop = True
        loop = asyncio.get_event_loop()

        loop.create_task(self.loop())

    async def loop(self):
        while True:
            recv_obj = await self.recv_from_scheduler.recv_pyobj()
            self._result_dispatcher(recv_obj)

    def _send_one_request(
        self,
        obj: GenerateReqInput,
        tokenized_obj: TokenizedGenerateReqInput,
        created_time: Optional[float] = None,
    ):
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[obj.rid] = state
        self.send_to_scheduler.send_pyobj(tokenized_obj)

    async def _wait_one_response(
        self,
        obj: GenerateReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        """Wait for the response of one request."""
        state = self.rid_to_state[obj.rid]

        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(obj.rid)
                    raise ValueError(
                        "Request is disconnected from the client side. "
                        f"Abort request {obj.rid}"
                    )
                continue

            out = state.out_list[-1]
            if state.finished:
                del self.rid_to_state[obj.rid]
                # Check if this was an abort/error created by scheduler
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if (
                        finish_reason.get("type") == "abort"
                        and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                    ):
                        raise ValueError(finish_reason["message"])

                yield out
                break
            state.event.clear()
            yield out
    async def _tokenize_one_request(
        self,
        obj: GenerateReqInput,
    ):
        input_text = obj.text
        if obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            input_ids = self.tokenizer.encode(input_text)

        sampling_params = SamplingParams(**obj.sampling_params)

        self._validate_token_len(obj, input_ids)

        return TokenizedGenerateReqInput(
            rid=obj.rid,
            input_text=input_text,
            input_ids=input_ids,
            sampling_params=sampling_params,
        )
        
    def _validate_token_len(
        self, obj: GenerateReqInput, input_ids: List[int]
    ) -> None:
        """Validates that the input token count and the requested token count doesn't exceed the model's context length."""

        input_token_num = len(input_ids) if input_ids is not None else 0
        # Check if input alone exceeds context length
        if input_token_num >= self.context_len:
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        # Check total tokens (input + max_new_tokens)
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if (
            max_new_tokens is not None
            and (max_new_tokens + input_token_num) >= self.context_len
        ):
            total_tokens = max_new_tokens + input_token_num
            error_msg = (
                f"Requested token count exceeds the model's maximum context length "
                f"of {self.context_len} tokens. You requested a total of {total_tokens} "
                f"tokens: {input_token_num} tokens from the input messages and "
                f"{max_new_tokens} tokens for the completion. Please reduce the number "
                f"of tokens in the input messages or the completion to fit within the limit."
            )
            raise ValueError(error_msg)

    async def _handle_batch_request(
        self,
        obj: GenerateReqInput,
        request: Optional[fastapi.Request] = None,
        created_time: Optional[float] = None,
    ):
        bs = obj.batch_size
        generators = []
        rids = []

        # batch tokenization
        tokenized_objs = await self._batch_tokenize_and_process(bs, obj)
        for i, tokenized_obj in enumerate(tokenized_objs):
            tmp_obj = obj[i]
            # send the tokenized req input and collect the generators(from scheduler output)
            self._send_one_request(tmp_obj, tokenized_obj, created_time)
            generators.append(self._wait_one_response(tmp_obj, request))
            rids.append(tmp_obj.rid)

        rid_to_index = {rid: i for i, rid in enumerate(rids)}
        task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}

        while task_map:
            done, _ = await asyncio.wait(
                task_map.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                gen = task_map.pop(task)
                try:
                    result = task.result()
                    result["index"] = rid_to_index[result["meta_info"]["id"]]
                    yield result
                    new_task = asyncio.create_task(gen.__anext__())
                    task_map[new_task] = gen
                except StopAsyncIteration:
                    pass

    async def _batch_tokenize_and_process(
        self, batch_size: int, obj: GenerateReqInput
    ) -> List[TokenizedGenerateReqInput]:
        """Handle batch tokenization for text inputs only."""

        # Collect requests and texts
        requests = [obj[i] for i in range(batch_size)]
        texts = [req.text for req in requests]

        # Batch tokenize all texts
        encoded = self.tokenizer(texts)
        input_ids_list = encoded["input_ids"]

        # Process all requests
        tokenized_objs = []
        for i, req in enumerate(requests):
            self._validate_token_len(obj[i], input_ids_list[i])
            tokenized_objs.append(
                TokenizedGenerateReqInput(
                    rid=req.rid, input_text=req.text, input_ids=input_ids_list[i]
                )
            )
        return tokenized_objs

    def _handle_batch_token_id_out(self, recv_obj: BatchTokenIDOut):
        """receive from scheduler stream output. detokenize the tokens and return the texts"""
        bs = len(recv_obj.rids)

        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if rid not in self.decode_status:
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=recv_obj.decode_ids[i],
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                self.decode_status[rid] = s
            else:
                s = self.decode_status[rid]
                s.decode_ids = recv_obj.decode_ids[i]

            read_ids.append(
                self._trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

        surr_texts = self.tokenizer.batch_decode(
            surr_ids,
        )
        read_texts = self.tokenizer.batch_decode(
            read_ids,
        )

        # Incremental decoding
        output_strs = []
        for i in range(bs):
            s = self.decode_status[recv_obj.rids[i]]
            new_text = read_texts[i][len(surr_texts[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming chunk: update the decode status
                if len(new_text) > 0 and not new_text.endswith("�"):
                    s.decoded_text = s.decoded_text + new_text
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    new_text = ""
                else:
                    new_text = find_printable_text(new_text)

            output_strs.append(
                self._trim_matched_stop(
                    s.decoded_text + new_text,
                    recv_obj.finished_reasons[i],
                )
            )

        # return BatchStrOut(
        #     rids=recv_obj.rids,
        #     finished_reasons=recv_obj.finished_reasons,
        #     output_strs=output_strs,
        #     output_ids=None
        # )
        # TODO simplify code
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                continue

            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
            }
            out_dict = {
                "text": output_strs[i],
                "meta_info": meta_info,
            }
            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                state.finished_time = time.time()
                meta_info["e2e_latency"] = state.finished_time - state.created_time

            state.out_list.append(out_dict)
            state.event.set()

    def _trim_matched_stop(self, output: Union[str, List[int]], finished_reason: Dict):
        if not finished_reason:
            return output
        
        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            return output[:pos] if pos != -1 else output
        
        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            assert len(output) > 0
            return output[:-1]
        return output


class SignalHandler:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    def signal_handler(self, signum=None, frame=None):
        logger.warning(
            f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down..."
        )
        self.tokenizer_manager.gracefully_exit = True
