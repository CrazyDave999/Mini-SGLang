from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn.functional import scaled_dot_product_attention

from minisglang.engine.batch import Batch
from minisglang.layers.attention import Attention

if TYPE_CHECKING:
    from minisglang.engine.model_runner import ModelRunner


class TorchNativeAttnBackend:
    def __init__(self, model_runner: ModelRunner):
        self.forward_metadata = None
        self.device = model_runner.device
        self.page_size = model_runner.page_size

    def init_forward_metadata(self, batch: Batch):
        """Init the metadata for a forward pass."""
        pass

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        page_table_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            page_table: [max_num_reqs, max_page_num]
            page_table_ids: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            page_table_id = page_table_ids[seq_idx]

            per_req_positions = torch.arange(seq_len_kv, device=query.device)
            per_req_tokens = page_table[
                page_table_id, per_req_positions // self.page_size
            ] * self.page_size + (per_req_positions % self.page_size)

            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            # print shapes
            # print(f"per_req_query_redudant: {per_req_query_redudant.shape}, ")
            # print(f"per_req_key: {per_req_key.shape}, ")
            # print(f"per_req_value: {per_req_value.shape}, ")
            # print(f"output: {output.shape}")

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        page_table_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            page_table_id = page_table_ids[seq_idx]

            per_req_positions = torch.arange(seq_len_kv, device=query.device)
            per_req_tokens = page_table[
                page_table_id, per_req_positions // self.page_size
            ] * self.page_size + (per_req_positions % self.page_size)
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: Attention,
        batch: Batch,
    ):
        if k is not None:
            assert v is not None
            # write the newly calculated kv to the kv cache
            batch.kvcache.write_kv(
                layer.layer_id,
                batch.out_cache_loc,
                k.view(-1, layer.num_kv_heads, layer.head_dim),
                v.view(-1, layer.num_kv_heads, layer.head_dim),
            )

        o = torch.empty_like(q)

        use_gqa = layer.num_heads != layer.num_kv_heads

        q_ = q.view(-1, layer.num_heads, layer.head_dim)
        o_ = o.view(-1, layer.num_heads, layer.head_dim)

        k_cache, v_cache = batch.kvcache.get_kv_cache(layer.layer_id)

        self._run_sdpa_forward_extend(
            query=q_,
            output=o_,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=batch.page_manager.page_table,
            page_table_ids=batch.page_table_ids,
            seq_lens=batch.seq_lens,
            extend_prefix_lens=batch.prefix_lens,
            extend_seq_lens=batch.seq_lens - batch.prefix_lens,
            enable_gqa=use_gqa,
            causal=True,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: Attention,
        batch: Batch,
    ):
        if k is not None:
            assert v is not None
            # write the newly calculated kv to the kv cache
            batch.kvcache.write_kv(
                layer.layer_id,
                batch.out_cache_loc,
                k.view(-1, layer.num_kv_heads, layer.head_dim),
                v.view(-1, layer.num_kv_heads, layer.head_dim),
            )
        q = q.reshape(-1, layer.num_heads * layer.head_dim)
        o = torch.empty_like(q)

        use_gqa = layer.num_heads != layer.num_kv_heads

        q_ = q.view(-1, layer.num_heads, layer.head_dim)
        o_ = o.view(-1, layer.num_heads, layer.head_dim)

        k_cache, v_cache = batch.kvcache.get_kv_cache(layer.layer_id)

        self._run_sdpa_forward_decode(
            query=q_,
            output=o_,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=batch.page_manager.page_table,
            page_table_ids=batch.page_table_ids,
            seq_lens=batch.seq_lens,
            enable_gqa=use_gqa,
            causal=False,
        )

        return o
