from typing import List
import dataclasses
import torch

from minisglang.memory.kvcache import KVCache
from enum import Enum
from minisglang.layers.sampler import SamplingParams


class Mode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"

    def is_decode(self):
        return self == Mode.DECODE

    def is_prefill(self):
        return self == Mode.PREFILL


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

        self.finished_reason = None

    def finished(self):
        return self.finished_reason is not None

    def check_finished(self):
        if self.finished():
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = "max_new_tokens"
            return

        last_token_id = self.output_ids[-1]

        matched_eos = False
        if self.sampling_params.stop_token_ids:
            matched_eos = last_token_id in self.sampling_params.stop_token_ids

        if matched_eos:
            self.finished_reason = "stop_token_ids"
            return


@dataclasses.dataclass
class Batch:
    bid: int
    reqs: List[Req] = None
    page_manager = None
    kvcache: KVCache = None

    page_table_ids: torch.Tensor = None

    seq_lens: torch.Tensor = None
    prefix_lens: torch.Tensor = None

    mode: Mode = None
    input_ids: torch.Tensor = None

    output_ids: torch.Tensor = None
    device: str = "cuda"

    out_cache_loc: torch.Tensor = None
    positions: torch.Tensor = None

    attn_backend = None

    def __init__(
        self,
        reqs: List[Req],
        page_manager=None,
        kvcache: KVCache = None,
    ):
        self.reqs = reqs
        self.page_manager = page_manager
        self.kvcache = kvcache
        self.device = self.kvcache.device

    def is_empty(self):
        return len(self.reqs) == 0

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
            r.fill_ids[len(r.prefix_ppns) * self.page_manager.page_size :] for r in reqs
        ]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        prefix_lens = [len(r.prefix_ppns) * self.page_manager.page_size for r in reqs]

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

            new_pages = torch.tensor(
                new_pages,
                device=self.device,
            )
            # write ppns to page table (prefix & newly allocated pages)
            prefix_ppns = torch.tensor(
                req.prefix_ppns,
                device=self.device,
            )
            ppns = torch.cat((prefix_ppns, new_pages))
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
            page_table_id_mask, self.positions // self.page_manager.page_size
        ] + (self.positions % self.page_manager.page_size)

    def prepare_for_decode(self):
        self.mode = Mode.DECODE
        batch_size = len(self.reqs)

        self.input_ids = self.output_ids
        self.output_ids = None
        self.seq_lens = self.seq_lens + 1

        ppns_list: List[List[int]] = self.kvcache.allocate_pages_decode(self.seq_lens)
        last_vpns = self.seq_lens // self.page_manager.page_size

        for i, (req, ppns) in enumerate(zip(self.reqs, ppns_list)):
            if len(ppns) > 0:
                ppns = torch.tensor(ppns, device=self.device, dtype=torch.int32)
                vpns = torch.tensor([last_vpns[i]], device=self.device, dtype=torch.int32)
                self.page_manager.write_ppns_decode(req.page_table_id, vpns, ppns)

        self.positions = self.seq_lens - 1
        page_table_id_mask = self.page_table_ids

        self.out_cache_loc = self.page_manager.page_table[
            page_table_id_mask, self.positions // self.page_manager.page_size
        ] + (self.positions % self.page_manager.page_size)

    def filter_batch(self):
        """filter out finished requests"""
        keep_indices = [i for i in range(len(self.reqs)) if not self.reqs[i].finished()]

        if len(keep_indices) == 0:
            self.reqs = []
            return

        self.reqs = [self.reqs[i] for i in keep_indices]
        self.page_table_ids = [self.page_table_ids[i] for i in keep_indices]
        self.seq_lens = [self.seq_lens[i] for i in keep_indices]
        self.prefix_lens = [self.prefix_lens[i] for i in keep_indices]
        self.input_ids = self.input_ids[keep_indices]
        self.output_ids = self.output_ids[keep_indices]
