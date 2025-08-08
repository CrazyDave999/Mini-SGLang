from typing import List, Tuple
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
                (page_num * page_size, head_num, head_dim), dtype=dtype, device=device
            )
            for _ in range(layer_num)
        ]
        self.v_cache = [
            torch.zeros(
                (page_num * page_size, head_num, head_dim), dtype=dtype, device=device
            )
            for _ in range(layer_num)
        ]
        self.free_pages = [i for i in range(page_num)]

    def allocate_pages_prefill(
        self,
        seq_lens: List[int],
        prefix_lens: List[int],
    ) -> List[List[int]]:
        ret = []
        for seq_len, prefix_len in zip(seq_lens, prefix_lens):
            page_num = (seq_len + self.page_size - 1) // self.page_size - (prefix_len + self.page_size - 1) // self.page_size
            ret.append(self._allocate_pages(page_num))
        return ret
    
    def allocate_pages_decode(
        self,
        seq_lens: List[int],
    ) -> List[List[int]]:
        return self.allocate_pages_prefill(
            seq_lens=seq_lens,
            prefix_lens=seq_lens - 1,
        )
            

    def _allocate_pages(self, num_pages: int) -> List[int]:
        if num_pages > len(self.free_pages):
            return None
        allocated = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        return allocated

    def _free_pages(self, pages: List[int]):
        self.free_pages.extend(pages)

    def write_kv(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # print(f"k shape: {cache_k.shape}, v shape: {cache_v.shape}")
        self.k_cache[layer_id][loc] = cache_k
        self.v_cache[layer_id][loc] = cache_v

    def get_kv_cache(
        self,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer_id], self.v_cache[layer_id]
