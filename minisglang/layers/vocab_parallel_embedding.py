import torch
from torch import nn
from typing import Optional, Tuple, Dict, Any
import torch.distributed as dist

class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank() 
        self.tp_size = dist.get_world_size()
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        