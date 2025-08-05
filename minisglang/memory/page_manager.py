from minisglang.engine.batch import Req, Batch


class PageManager:
    """
    a mapping for paged kvcache: (req_id, vpn) -> ppn, shape: (batch_size, max_page_num)
    """

    def __init__(
        self,
        page_size: int,
        max_req_num: int,
        max_page_num: int,
        device: str,
    ):
        self.page_size = page_size
        self.max_req_num = max_req_num
        self.max_page_num = max_page_num
        self.page_table = torch.zeros(
            (max_req_num, max_page_num), dtype=torch.int32, device=device
        )
        self.free_slots = [i for i in range(max_req_num)]


    def translate_token(self, req_id: int, token_id: int):
        vpn, offset = token_id // self.page_size, token_id % self.page_size
        return self.page_table[req_id, vpn] + offset
    
    def allocate(self, num_reqs: int) -> List[int]:
        """ allocate req ids in the page table"""
        assert num_reqs <= len(self.free_slots)
        allocated = self.free_slots[:num_reqs]
        self.free_slots = self.free_slots[num_reqs:]
        return allocated
    
    def free(self, page_table_ids: List[int]):
        """ free req ids in the page table"""
        self.free_slots.extend(page_table_ids)
        