from pyexpat import model
import torch
import torch.nn as nn

from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# from flash_attn import flash_attn_with_kvcache

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisglang.memory.page_manager import PageManager
    from minisglang.engine.model_runner import ModelRunner


from minisglang.engine.batch import Batch
from minisglang.layers.attention import Attention



@dataclass
class FlashAttentionMetaData:
    """
    Metadata is only initialized once in the model forward pass
    """

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seqlen_q: int = 1
    # Maximum sequence length for key
    max_seqlen_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None


class FlashAttentionBackend:
    def __init__(self, model_runner: "ModelRunner"):
        self.forward_metadata: FlashAttentionMetaData = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.page_manager = model_runner.page_manager
        self.page_size = model_runner.page_size
        self.page_num = model_runner.kvcache.page_num

    def init_forward_metadata(self, batch: Batch):
        metadata = FlashAttentionMetaData()
        seqlens_in_batch = batch.seq_lens
        batch_size = seqlens_in_batch.shape[0]
        device = batch.device
        if batch.mode.is_decode():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seqlen_k = seqlens_in_batch.max().item()
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.page_table = batch.page_manager.page_table[
                batch.page_table_ids, : metadata.max_seqlen_k // self.page_size + 1
            ]
        else:
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seqlen_k = seqlens_in_batch.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.page_table = batch.page_manager.page_table[
                batch.page_table_ids, : metadata.max_seqlen_k // self.page_size + 1
            ]
            metadata.cu_seqlens_q = metadata.cu_seqlens_k
            metadata.max_seqlen_q = metadata.max_seqlen_k

        self.forward_metadata = metadata

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
        batch: Batch,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            # write the newly calculated kv to the kv cache
            batch.kvcache.write_kv(
                layer.layer_id,
                batch.out_cache_loc,
                k.view(-1, layer.num_kv_heads, layer.head_dim),
                v.view(-1, layer.num_kv_heads, layer.head_dim)
            )
            
        window_size = (-1, -1)
        
        metadata = self.forward_metadata
        page_table = metadata.page_table
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seqlen_q
        max_seqlen_k = metadata.max_seqlen_k
        k_cache, v_cache = batch.kvcache.get_kv_cache(layer.layer_id)
        # print(f"{q.shape=} {k_cache.shape=} {v_cache.shape=}")
        
        # FA3
        q = q.contiguous().view(-1, layer.num_heads, layer.head_dim)
        k_cache=k_cache.view(self.page_num, self.page_size, layer.num_kv_heads, layer.head_dim)
        v_cache=v_cache.view(self.page_num, self.page_size, layer.num_kv_heads, layer.head_dim)
        # print(f"{q.shape=} {k_cache.shape=} {v_cache.shape=}")
        o = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=None,
            causal=True,
            window_size=window_size,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
        )
        o = o.view(-1, layer.num_heads * layer.head_dim)
        
        # FA2
        # o = flash_attn_with_kvcache(
        #     q=q.contiguous().view(-1, layer.num_heads, layer.head_dim),
        #     k_cache=k_cache.view(self.page_num, self.page_size, layer.num_kv_heads, layer.head_dim),
        #     v_cache=v_cache.view(self.page_num, self.page_size, layer.num_kv_heads, layer.head_dim),
        #     cache_seqlens=cache_seqlens,
        #     block_table=page_table,
        #     causal=True,
        #     window_size=window_size,
        #     return_softmax_lse=False,
        # )
        
        return o
    
