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
        self.free_pages = [i for i in range(max_page_num)]

    def allocate_prefill(
        self,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        last_page_ids: torch.Tensor,
        extend_num_tokens: int,
    ) -> torch.Tensor:
        """return the indices in the kv cache of the newly allocated tokens"""
        batch_size = seq_lens.size(0)
        device = self.device
        page_size = self.page_size

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        cur_start = 0
        for i in range(batch_size):
            cur_extend_num = seq_lens[i] - prefix_lens[i]
            cur_last_offset = prefix_lens[i] % page_size
            cur_remain_in_page = page_size - cur_last_offset
            if cur_extend_num <= cur_remain_in_page:
                out_indices[cur_start:cur_start + cur_extend_num] = torch.arange(
                    last_page_ids[i] * page_size + cur_last_offset,
                    last_page_ids[i] * page_size + cur_last_offset + cur_extend_num,
                    device=device,
                )
            else:
                if cur_remain_in_page > 0:
                    out_indices[cur_start : cur_start + cur_remain_in_page] = torch.arange(
                        last_page_ids[i] * page_size + cur_last_offset,
                        last_page_ids[i] * page_size + page_size,
                        device=device,
                    )
                    cur_extend_num -= cur_remain_in_page
                    cur_start += cur_remain_in_page

                cur_num_full_pages = cur_extend_num // page_size
                cur_num_remain = cur_extend_num % page_size

                ppns = self._allocate_pages(cur_num_full_pages + 1 if cur_num_remain > 0 else cur_num_full_pages)
                if cur_num_full_pages > 0:
                    paddrs = torch.Tensor(ppns[:cur_num_full_pages], device=device).unsqueeze(
                        1
                    ) * page_size + torch.arange(
                        self.page_size, device=device
                    )
                    out_indices[cur_start : cur_start + cur_num_full_pages * page_size] = paddrs.view(-1)
                    cur_start += cur_num_full_pages * page_size
                if cur_num_remain > 0:
                    out_indices[cur_start : cur_start + cur_num_remain] = (
                        ppns[cur_num_full_pages] * page_size
                        + torch.arange(cur_num_remain, device=device)
                    )
                    cur_start += cur_num_remain
                    
        return out_indices
    
    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        last_page_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.allocate_prefill(
            seq_lens=seq_lens,
            prefix_lens=seq_lens - 1,
            last_page_ids=last_page_ids,
            extend_num_tokens=len(seq_lens),
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
        page_ids = loc // self.page_size
        offsets = loc % self.page_size

        self.k_cache[layer_id][page_ids, offsets] = cache_k
        self.v_cache[layer_id][page_ids, offsets] = cache_v

    def get_kv_cache(
        self,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer_id], self.v_cache[layer_id]
