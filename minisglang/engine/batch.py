import dataclasses
from minisglang.memory.page_manager import PageManager
from minisglang.memory.kvcache import KVCache
from enum import Enum


class Mode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"

    def is_decode(self):
        return self == Mode.DECODE

    def is_prefill(self):
        return self == Mode.PREFILL


@dataclasses.dataclass
class Req:
    def __init__(self, mode: Mode):
        self.mode = mode
        self.fill_ids = None
        self.output_ids = []
        self.prefix_indices = []


@dataclasses.dataclass
class Batch:
    bid: int
    reqs: List[Req]
    page_manager: PageManager
    kvcache: KVCache = None

    page_table_ids: List[int] = None

    seq_lens: List[int]
    prefix_lens: List[int]

    mode: Mode
    input_ids: torch.Tensor = None

    out_cache_loc: torch.Tensor = None
    output_ids: torch.Tensor = None

    def __init__(
        self,
        reqs: List[Req],
        page_table: PageManager = None,
        kvcache: KVCache = None,
    ):
        self.reqs = reqs
        self.page_table = page_table
        self.kvcache = kvcache

    def alloc_req_slots(self, num_reqs: int):
        page_table_ids = self.page_manager.allocate(num_reqs)
        assert page_table_ids is not None, "No free slots in page table"
        return page_table_ids

    def alloc_token_slots_prefill(
        self,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        last_page_ids: torch.Tensor,
        extend_num_tokens: int,
    ) -> torch.Tensor:
        # TODO evict when no available pages

        out_cache_loc = self.kvcache.allocate_prefill(
            seq_lens, prefix_lens, last_page_ids, extend_num_tokens
        )
        assert out_cache_loc is not None, "allocate token slots failed"
        return out_cache_loc

    def alloc_token_slots_decode(
        self,
        seq_lens: torch.Tensor,
        last_page_ids: torch.Tensor,
    ) -> torch.Tensor:
        out_cache_loc = self.kvcache.allocate_decode(seq_lens, last_page_ids)
        assert out_cache_loc is not None, "allocate token slots failed"
        return out_cache_loc

    def prepare_for_prefill(self):
        self.mode = Mode.PREFILL

        batch_size = len(self.reqs)
        page_table_ids = self.alloc_req_slots(batch_size)

        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]

        page_table_ids_tensor = torch.tensor(
            page_table_ids,
            dtype=torch.int64,
        ).to(self.device, non_blocking=True)
        input_ids_tensor = torch.tensor(
            input_ids,
            dtype=torch.int64,
        ).to(self.device, non_blocking=True)
        seq_lens_tensor = torch.tensor(
            seq_lens,
            dtype=torch.int64,
        ).to(self.device, non_blocking=True)
        prefix_lens_tensor = torch.tensor(
            prefix_lens,
            dtype=torch.int64,
        ).to(self.device, non_blocking=True)
