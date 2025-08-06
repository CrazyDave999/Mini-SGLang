import dataclasses
from minisglang.memory.page_manager import PageManager
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
    def __init__(self, mode: Mode):
        self.rid = None
        self.mode = mode
        self.fill_ids = None
        self.output_ids = []
        self.prefix_ppns = []
        self.page_table_id = None
        self.finished_reason = None
        self.sampling_params: SamplingParams = None
        
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
    reqs: List[Req]
    page_manager: PageManager
    kvcache: KVCache = None

    page_table_ids: List[int] = None

    seq_lens: List[int]
    prefix_lens: List[int]

    mode: Mode
    input_ids: torch.Tensor = None

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
        
    def is_empty(self):
        return len(self.reqs) == 0

    def alloc_req_slots(self, num_reqs: int):
        page_table_ids = self.page_manager.allocate(num_reqs)
        assert page_table_ids is not None, "No free slots in page table"
        return page_table_ids

    # def alloc_token_slots_prefill(
    #     self,
    #     seq_lens: torch.Tensor,
    #     prefix_lens: torch.Tensor,
    #     last_page_ids: torch.Tensor,
    #     extend_num_tokens: int,
    # ) -> torch.Tensor:
    #     # TODO evict when no available pages

    #     out_cache_loc = self.kvcache.allocate_prefill(
    #         seq_lens, prefix_lens, last_page_ids, extend_num_tokens
    #     )
    #     assert out_cache_loc is not None, "allocate token slots failed"
    #     return out_cache_loc

    def alloc_pages_prefill(
        self, seq_lens: List[int], prefix_lens: List[int]
    ) -> List[List[int]]:
        # TODO evict when no available pages

        new_pages = self.kvcache.allocate_pages_prefill(seqlens, prefix_lens)
        assert new_pages is not None, "allocate token slots failed"
        return new_pages

    # def alloc_token_slots_decode(
    #     self,
    #     seq_lens: torch.Tensor,
    #     last_page_ids: torch.Tensor,
    # ) -> torch.Tensor:
    #     out_cache_loc = self.kvcache.allocate_decode(seq_lens, last_page_ids)
    #     assert out_cache_loc is not None, "allocate token slots failed"
    #     return out_cache_loc

    def prepare_for_prefill(self):
        self.mode = Mode.PREFILL

        batch_size = len(self.reqs)
        page_table_ids = self.alloc_req_slots(batch_size)

        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]

        # page_table_ids_tensor = torch.tensor(
        #     page_table_ids,
        #     dtype=torch.int64,
        # ).to(self.device, non_blocking=True)
        # input_ids_tensor = torch.tensor(
        #     input_ids,
        #     dtype=torch.int64,
        # ).to(self.device, non_blocking=True)
        # seq_lens_tensor = torch.tensor(
        #     seq_lens,
        #     dtype=torch.int64,
        # ).to(self.device, non_blocking=True)
        # prefix_lens_tensor = torch.tensor(
        #     prefix_lens,
        #     dtype=torch.int64,
        # ).to(self.device, non_blocking=True)

        # allocate pages for newly added tokens
        new_pages_list = self.alloc_pages_prefill(seq_lens, prefix_lens)

        for i, (req, new_pages) in enumerate(zip(reqs, new_pages_list)):
            # set page_table_id
            req.page_table_id = page_table_ids[i]
            # write ppns to page table (prefix & newly allocated pages)
            ppns = req.prefix_ppns + new_pages
            self.page_manager.write_ppns_prefill(req.page_table_id, ppns)

    def prepare_for_decode(self):
        self.mode = Mode.DECODE
        batch_size = len(self.reqs)

        self.input_ids = self.output_ids
        self.output_ids = None
        self.seq_lens = self.seq_lens + 1

        ppns_list: List[List[int]] = self.kvcache.allocate_pages_decode(self.seq_lens)
        last_vpn_list: List[int] = self.seq_lens // self.page_manager.page_size
        vpns_list: List[List[int]] = [
            [last_vpn_list[i]] if ppns_list[i] else [] for i in range(len(ppns_list))
        ]
        self.page_manager.write_ppns_decode(self.page_table_ids, vpns_list, ppns_list)

    def filter_batch(self):
        keep_indices = [
            i for i in range(len(self.reqs)) if not self.reqs[i].finished()
        ]
        
        if len(keep_indices) == 0:
            self.reqs = []
            return
        
        self.reqs = [self.reqs[i] for i in keep_indices]
        self.page_table_ids = [self.page_table_ids[i] for i in keep_indices]
        self.seq_lens = [self.seq_lens[i] for i in keep_indices]
        self.prefix_lens = [self.prefix_lens[i] for i in keep_indices]
        self.input_ids = self.input_ids[keep_indices] 
        self.output_ids = self.output_ids[keep_indices]
        