import torch
import torch.nn as nn
import torch.nn.functional as F
from minisglang.utils import divide
import torch.distributed as dist

class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.params_dtype = params_dtype
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("LinearBase forward not implemented. Subclasses should implement this method.")
    

class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        params_dtype: Optional[torch.dtype] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
        )
        self.tp_rank = tp_rank
        self.tp_size = tp_size

            
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
    
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        bias: bool = True,
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
        )
        
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size:int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = True,
        params_dtype: Optional[torch.dtype] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads if total_num_kv_heads is not None else total_num_heads
        # Divide the weight matrix along the last dimension.
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.num_heads = divide(total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        
        input_size = self.hidden_size
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * self.tp_size * self.head_size
        )
        self.output_sizes = [
            self.num_heads * self.head_size * self.tp_size,
            self.num_kv_heads * self.head_size * self.tp_size,
            self.num_kv_heads * self.head_size * self.tp_size,
        ]
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            params_dtype=params_dtype,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        
class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        params_dtype: Optional[torch.dtype] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
        )
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
