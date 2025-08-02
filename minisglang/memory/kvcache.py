import torch


class KVCache:
    def __init__(
        self,
        page_num: int,
        page_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
    ):
        self.page_num = page_num
        self.page_size = page_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.layer_num = layer_num
        self.device = device

        self.k_cache = [
            torch.zeros(
                (page_num, page_size, head_num, head_dim), dtype=dtype, device=device
            )
            for _ in range(layer_num)
        ]
        self.v_cache = [
            torch.zeros(
                (page_num, page_size, head_num, head_dim), dtype=dtype, device=device
            )
            for _ in range(layer_num)
        ]
