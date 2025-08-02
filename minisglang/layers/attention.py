from abc import ABC, abstractmethod
from minisglang.engine.batch import Batch
import torch
import torch.nn as nn

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float = 1.0,
    ):
        super().__init__()
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
        pass
    
