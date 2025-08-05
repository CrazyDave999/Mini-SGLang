from abc import ABC, abstractmethod
from minisglang.engine.batch import Batch
import torch
import torch.nn as nn

from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from dataclasses import dataclass

from minisglang.engine.model_runner import ModelRunner


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
    def __init__(self, model_runner: ModelRunner):
        self.forward_metadata: FlashAttentionMetaData = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.page_table = model_runner.page_table
        self.page_size = model_runner.page_size

    def init_forward_metadata(self, batch: Batch):
        metadata = FlashAttentionMetaData()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = len(seqlens_in_batch)
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
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            # write the newly calculated kv to the kv cache
            cache_loc = batch.out_cache_loc
            batch.kvcache.write_kv(layer_id, cache_loc, k, v)
            
        window_size = (-1, -1)
        
        metadata = self.forward_metadata
        page_table = metadata.page_table
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seqlen_q
        max_seqlen_k = metadata.max_seqlen_k
        k_cache, v_cache = batch.kvcache.get_kv_cache(layer_id)
        
        o = flash_attn_with_kvcache(
            q=q.contiguous(),
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
            softcap=None,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
        )
        return o
    

      
class Attention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float = 1.0,
    ):
        super().__init__()
        layer_id = layer_id
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        # call the flash_attention backend
        return batch.attn_backend.forward(
            q=q,
            k=k,
            v=v,
            layer_id=batch.layer_id,
            batch=batch,
        )
