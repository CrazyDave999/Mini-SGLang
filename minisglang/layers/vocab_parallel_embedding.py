import torch
from torch import nn
from typing import Optional, Tuple, Dict, Any
import torch.distributed as dist
from torch.nn import functional as F
from minisglang.engine.batch import Batch
import logging
logger = logging.getLogger(__name__)

class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.num_embeddings = num_embeddings
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim, dtype=dtype)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        assert x is not None
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(num_embeddings, embedding_dim, dtype=dtype)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.num_embeddings_per_partition, dtype=dtype)
            )
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, batch: Batch):
        """
        Args:
            x (torch.Tensor):  shape = (sum_of_input_ids_len, hidden_size_per_partition)
        """
        # print(f"ParallelLMHead forward. {x.shape=}, {batch.mode=}")
        if batch.mode.is_prefill():
            last_indices = torch.cumsum(batch.seq_lens - batch.prefix_lens, dim=0) - 1
            # print(f"Before prefill handling. {last_indices.shape=} {last_indices=}")
            x = x[last_indices].contiguous()
        # x: shape = (bs, hidden_size)
        # print(f"After prefill handling. {x.shape=}, {self.weight.shape=}")
        logits = F.linear(x, self.weight)
        # x: shape = (bs, vocab_size_per_partition))
        # print(f"After linear. {logits.shape=}")
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        # print(f"After all gather. {logits.shape=}")
        # logits: shape = (bs, vocab_size)
        return logits
