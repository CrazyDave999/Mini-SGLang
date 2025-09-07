from pyexpat import model
import torch
import torch.nn as nn

from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# from flash_attn import flash_attn_with_kvcache

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from minisglang.memory.page_manager import PageManager
    from minisglang.engine.model_runner import ModelRunner


from minisglang.engine.batch import Batch, Mode
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
            metadata.max_seqlen_k = batch.seq_lens_cpu.max().item()
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

    def init_cuda_graph_state(self, max_bs: int):
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "page_table_draft_decode": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        page_table_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: Mode,
    ):
        metadata = FlashAttentionMetaData()
        device = seq_lens.device
        if forward_mode.is_decode():
            # Normal Decode. Get sequence information
            metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
            batch_size = len(seq_lens)
            device = seq_lens.device
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            # Precompute maximum sequence length
            metadata.max_seq_len_k = seq_lens.max().item()
            # Precompute page table
            metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                page_table_ids, :
            ]
            # Precompute cumulative sequence lengths
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            self.decode_cuda_graph_metadata[bs] = metadata
        else:
            raise NotImplementedError()

        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        page_table_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: Mode,
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        page_table_ids = page_table_ids[:bs]
        device = seq_lens.device
        metadata = None

        if forward_mode.is_decode():
            # Normal Decode
            metadata = self.decode_cuda_graph_metadata[bs]
            max_len = seq_lens_cpu.max().item()
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size
            metadata.max_seq_len_k = max_len
            
            normal_decode_set_metadata(
                cache_seqlens_int32=metadata.cache_seqlens_int32,
                cu_seqlens_k=metadata.cu_seqlens_k,
                page_table=metadata.page_table,
                origin_page_table=self.page_manager.page_table,
                page_table_ids=page_table_ids,
                max_seq_pages=max_seq_pages,
                seq_lens=seq_lens,
                seq_len_delta=0,
            )
        else:
            raise NotImplementedError()
        
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
                v.view(-1, layer.num_kv_heads, layer.head_dim),
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
        k_cache = k_cache.view(
            self.page_num, self.page_size, layer.num_kv_heads, layer.head_dim
        )
        v_cache = v_cache.view(
            self.page_num, self.page_size, layer.num_kv_heads, layer.head_dim
        )
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


def normal_decode_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    origin_page_table: torch.Tensor,
    page_table_ids: torch.Tensor,
    max_seq_pages: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len_delta: int,
):
    cache_seqlens_int32.copy_(seq_lens + seq_len_delta)
    cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32))

    page_table[:, :max_seq_pages].copy_(
        origin_page_table[page_table_ids, :max_seq_pages]
    )
