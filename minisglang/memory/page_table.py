class PageTable:
    """
    a mapping for paged kvcache: (req_id, vpn) -> ppn, shape: (batch_size, max_page_num)
    """

    def __init__(
        self,
        page_size: int,
        max_batch_size: int,
        max_page_num: int,
        device: str,
    ):
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.max_page_num = max_page_num
        self.page_table = torch.zeros(
            (max_batch_size, max_page_num), dtype=torch.int32, device=device
        )

    def translate_token(req_id: int, token_id: int):
        vpn, offset = token_id // self.page_size, token_id % self.page_size
        return self.page_table[req_id, vpn] + offset
