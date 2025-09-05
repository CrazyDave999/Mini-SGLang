from typing import List, Optional, Union, TYPE_CHECKING
import dataclasses
from minisglang.utils.args import ServerArgs
import torch
import logging

logger = logging.getLogger(__name__)

from minisglang.memory.kvcache import KVCache
from enum import Enum
from minisglang.layers.sampler import SamplingParams

from minisglang.utils.args import global_config
if TYPE_CHECKING:
    from minisglang.memory.page_manager import PageManager
    from minisglang.memory.radix_cache import PagedRadixCache
INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5


class Mode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"

    def is_decode(self):
        return self == Mode.DECODE

    def is_prefill(self):
        return self == Mode.PREFILL


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message="Unknown error", status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


class Req:
    def __init__(
        self,
        rid: int,
        origin_input_ids: List[int],
        sampling_params: SamplingParams,
    ):

        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.output_ids = []
        self.fill_ids = None
        self.prefix_ppns = []
        self.page_table_id = None

        self.sampling_params: SamplingParams = sampling_params

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""

        self.last_node = None

        self.finished_reason = None

    def finished(self):
        return self.finished_reason is not None

    def init_next_round_input(self, tree_cache: "PagedRadixCache"):
        self.fill_ids = self.origin_input_ids + self.output_ids
        self.prefix_ppns, self.last_node = tree_cache.match_prefix(
            key=self.adjust_max_prefix_ids()
        )
        # logger.info(
        #     f"init_next_round_input Req {self.rid=} {self.fill_ids=} {self.prefix_ppns=}"
        # )

    def adjust_max_prefix_ids(self):
        self.fill_ids = self.origin_input_ids + self.output_ids
        input_len = len(self.fill_ids)

        # FIXME: To work around some bugs in logprob computation, we need to ensure each
        # request has at least one token. Later, we can relax this requirement and use `input_len`.
        max_prefix_len = input_len - 1

        if self.sampling_params.max_new_tokens > 0:
            # Need at least one token to compute logits
            max_prefix_len = min(max_prefix_len, input_len - 1)

        max_prefix_len = max(max_prefix_len, 0)
        return self.fill_ids[:max_prefix_len]

    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None
        if first_iter:
            self.read_offset = len(self.origin_input_ids)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )

        all_ids = self.origin_input_ids + self.output_ids
        return all_ids[self.surr_offset :], self.read_offset - self.surr_offset

    def check_finished(self):
        if self.finished():
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            return

        last_token_id = self.output_ids[-1]

        matched_eos = False
        if self.sampling_params.stop_token_ids:
            matched_eos = last_token_id in self.sampling_params.stop_token_ids

        if matched_eos:
            self.finished_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
            return
        
    def reset_for_retract(self):
        self.prefix_ppns = []
        self.last_node = None
        self.page_table_id = None


bid = 0


@dataclasses.dataclass
class Batch:
    bid: int
    reqs: List[Req] = None
    page_manager: "PageManager" = None
    page_size: int = 1
    kvcache: KVCache = None
    tree_cache: "PagedRadixCache" = None

    page_table_ids: torch.Tensor = None

    seq_lens: torch.Tensor = None
    prefix_lens: torch.Tensor = None

    mode: Mode = None
    input_ids: torch.Tensor = None

    output_ids: torch.Tensor = None
    out_cache_loc: torch.Tensor = None
    device: str = "cuda"
    
    # The sum of all sequence lengths
    seq_lens_sum: int = None

    positions: torch.Tensor = None

    attn_backend = None

    def __init__(
        self,
        reqs: List[Req],
        page_manager=None,
        kvcache: KVCache = None,
        tree_cache: "PagedRadixCache" = None,
    ):
        self.reqs = reqs
        self.page_manager = page_manager
        self.page_size = self.page_manager.page_size if self.page_manager else None
        self.kvcache = kvcache
        self.tree_cache = tree_cache
        global bid
        bid += 1
        self.bid = bid

    def is_empty(self):
        return len(self.reqs) == 0

    def batch_size(self):
        return len(self.reqs) if self.reqs is not None else self.seq_lens.shape[0]

    def alloc_req_slots(self, num_reqs: int):
        page_table_ids = self.page_manager.allocate(num_reqs)
        assert page_table_ids is not None, "No free slots in page table"
        return page_table_ids

    def prepare_for_prefill(self):
        self.mode = Mode.PREFILL

        batch_size = len(self.reqs)
        page_table_ids = self.alloc_req_slots(batch_size)

        reqs = self.reqs
        input_ids = [
            r.fill_ids[len(r.prefix_ppns) * self.page_size :] for r in reqs
        ]

        seq_lens = [len(r.fill_ids) for r in reqs]
        prefix_lens = [len(r.prefix_ppns) * self.page_size for r in reqs]

        page_table_ids_tensor = torch.tensor(
            page_table_ids,
            dtype=torch.int64,
        ).to(self.device, non_blocking=True)
        input_ids_tensor = torch.tensor(sum(input_ids, []), dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        seq_lens_tensor = torch.tensor(
            seq_lens,
            dtype=torch.int64,
        ).to(self.device, non_blocking=True)

        prefix_lens_tensor = torch.tensor(
            prefix_lens,
            dtype=torch.int64,
        ).to(self.device, non_blocking=True)

        # allocate pages for newly added tokens
        new_pages_list: List[List[int]] = self.kvcache.allocate_pages_prefill(
            seq_lens, prefix_lens
        )

        for i, (req, new_pages) in enumerate(zip(reqs, new_pages_list)):
            # set page_table_id
            req.page_table_id = page_table_ids[i]

            prefix_ppns_tensor = torch.tensor(
                req.prefix_ppns, dtype=torch.int32
            ).to(self.device, non_blocking=True)
            new_pages_tensor = torch.tensor(
                new_pages, dtype=torch.int32
            ).to(self.device, non_blocking=True)
            # write ppns to page table (prefix & newly allocated pages)
            ppns = torch.cat((prefix_ppns_tensor, new_pages_tensor))
            self.page_manager.write_ppns_prefill(req.page_table_id, ppns)

        # set fields
        self.input_ids = input_ids_tensor
        self.page_table_ids = page_table_ids_tensor
        self.seq_lens = seq_lens_tensor
        self.prefix_lens = prefix_lens_tensor
        self.positions = torch.cat(
            [
                torch.arange(pl, sl, dtype=torch.int64)
                for pl, sl in zip(prefix_lens, seq_lens)
            ]
        )

        page_table_id_mask = torch.cat(
            [
                torch.tensor([req_id] * (sl - pl), dtype=torch.int64)
                for req_id, pl, sl in zip(page_table_ids, prefix_lens, seq_lens)
            ]
        )

        # out_cache_loc is obtained by indexing the page table with positions

        self.out_cache_loc = self.page_manager.page_table[
            page_table_id_mask, self.positions // self.page_size
        ] * self.page_size + (self.positions % self.page_size)
        self.seq_lens_sum = sum(seq_lens)

    def prepare_for_decode(self):
        self.mode = Mode.DECODE
        bs = len(self.reqs)

        # update tensors
        self.input_ids = self.output_ids
        self.output_ids = None
        self.seq_lens = self.seq_lens + 1
        self.positions = self.seq_lens - 1

        # allocate pages and write ppns to page_table
        ppns_list: List[List[int]] = self.kvcache.allocate_pages_decode(self.seq_lens)
        last_vpns = self.positions // self.page_size

        for i, (req, ppns) in enumerate(zip(self.reqs, ppns_list)):
            if len(ppns) > 0:
                ppns = torch.tensor(ppns, device=self.device, dtype=torch.int32)
                vpns = torch.tensor(
                    [last_vpns[i]], device=self.device, dtype=torch.int32
                )
                self.page_manager.write_ppns_decode(req.page_table_id, vpns, ppns)

        page_table_id_mask = self.page_table_ids

        self.out_cache_loc = self.page_manager.page_table[
            page_table_id_mask, self.positions // self.page_size
        ] * self.page_size + (self.positions % self.page_size)
        self.seq_lens_sum += bs

    def filter_batch(
        self,
        keep_indices: Optional[List[int]] = None,
    ):
        """filter out finished requests"""
        if keep_indices is None:
            keep_indices = [i for i in range(len(self.reqs)) if not self.reqs[i].finished()]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests
            self.reqs = []
            return
        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return
        
        keep_indices_device = torch.tensor(keep_indices, dtype=torch.int64).to(
            self.device, non_blocking=True
        )

        self.reqs = [self.reqs[i] for i in keep_indices]
        self.page_table_ids = self.page_table_ids[keep_indices_device]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.prefix_lens = self.prefix_lens[keep_indices_device]
        self.input_ids = self.input_ids[keep_indices_device]
        self.output_ids = self.output_ids[keep_indices_device]
        self.out_cache_loc = None
        self.seq_lens_sum = self.seq_lens.sum().item()

    def merge_batch(self, other):
        """merge the last prefill batch into the running decode batch"""
        self.page_table_ids = torch.cat([self.page_table_ids, other.page_table_ids])
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.out_cache_loc = None
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = torch.cat([self.output_ids, other.output_ids])
        self.reqs.extend(other.reqs)

    def check_decode_mem(self, buf_multiplier: float = 1.0):
        num_new_pages = sum(
            1 for seqlen in self.seq_lens if seqlen % self.page_size == 0
        )
        num_pages_required = num_new_pages * buf_multiplier

        if self.kvcache.available_page_num() >= num_pages_required:
            return True

        self.tree_cache.evict(num_pages_required)
        return self.kvcache.available_page_num() >= num_pages_required

    def retract_decode(self):
        """Retract the decoding requests when there is not enough memory."""
        sorted_indices = [i for i in range(len(self.reqs))]
        sorted_indices.sort(
            key=lambda i: (
                len(self.reqs[i].output_ids),
                -len(self.reqs[i].origin_input_ids),
            ),
            reverse=True,
        )

        retracted_reqs = []
        seq_lens_cpu = self.seq_lens.cpu().numpy()
        first_iter = True
        
        def get_required_num_pages(num_reqs: int):
            return num_reqs * (global_config.retract_decode_steps + self.page_size - 1) // self.page_size

        while (
            self.kvcache.available_page_num() < get_required_num_pages(len(sorted_indices))
            or first_iter
        ):
            if len(sorted_indices) == 1:
                # Corner case: only one request left
                assert (
                    self.kvcache.available_page_num() > 0
                ), "No space left for only one request"
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)

            last_uncached_pos = len(req.prefix_ppns)
            ppns = self.page_manager.page_table[
                req.page_table_id,
                last_uncached_pos : seq_lens_cpu[idx] // self.page_size,
            ]
            self.kvcache._free_pages(ppns.tolist())
            self.page_manager.free([req.page_table_id])

            # release the last node
            self.tree_cache.dec_lock_ref(req.last_node)

            # evit the rectrated reqs and reuse their memory immediately
            residual_num_pages = (
                get_required_num_pages(len(sorted_indices))
                - self.kvcache.available_page_num()
            )
            residual_size = max(0, residual_size)
            self.tree_cache.evict(residual_num_pages)
            
            req.reset_for_retract()
            
            if len(retracted_reqs) == 0:
                # Corner case: only one request left
                raise ValueError(
                    "Failed to retract any request. No space left for only one request."
                )
                
        self.filter_batch()
        # Reqs in batch are filtered
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens + global_config.retract_decode_steps * len(self.reqs)
        ) / total_max_new_tokens
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio
