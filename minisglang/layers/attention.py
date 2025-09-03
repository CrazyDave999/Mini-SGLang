from minisglang.engine.batch import Batch
import torch
import torch.nn as nn

      
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
        self.layer_id = layer_id
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
            layer=self,
            batch=batch,
        )
        
        # # call the torch native
        # if batch.mode.is_prefill():
        #     return batch.attn_backend.forward_extend(
        #         q=q,
        #         k=k,
        #         v=v,
        #         layer=self,
        #         batch=batch
        #     )
        # else:
        #     return batch.attn_backend.forward_decode(
        #         q=q,
        #         k=k,
        #         v=v,
        #         layer=self,
        #         batch=batch
        #     )
