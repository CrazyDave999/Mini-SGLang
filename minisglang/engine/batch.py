import dataclasses
import minisglang.engine.scheduler.Mode as Mode
from minisglang.memory.page_table import PageTable
from minisglang.memory.kvcache import KVCache

class Req:
    mode: Mode
    def __init__(self, mode: Mode):
        self.mode = mode
        pass

@dataclasses.dataclass
class Batch:
    reqs: List[Req]
    page_table: PageTable
    kvcache: KVCache = None
    
    seq_lens: List[int]
    
    mode: Mode
    input_ids: torch.Tensor = None
    input_embeds: torch.Tensor = None
    out_cache_loc: torch.Tensor = None
    output_ids: torch.Tensor = None
    
    def __init__(
        self,
        reqs: List[Req],
        page_table: PageTable = None,
        kvcache: KVCache = None,
    ):
        self.reqs = reqs
        self.page_table = page_table
        self.kvcache = kvcache
