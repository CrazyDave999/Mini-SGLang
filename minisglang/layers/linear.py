import torch
import torch.nn as nn
import torch.nn.functional as F
from minisglang.utils import divide
import torch.distributed as dist
from torch.nn.parameter import Parameter
from typing import Optional, List


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype
        self.output_dim = 0

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "LinearBase forward not implemented. Subclasses should implement this method."
        )


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            dtype=dtype,
        )

        if tp_rank is None:
            tp_rank = dist.get_rank()
        if tp_size is None:
            tp_size = dist.get_world_size()
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        self.output_size_per_partition = divide(output_size, tp_size)

        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, self.input_size, dtype=dtype)
        )
        self.weight.weight_loader = self.weight_loader

        if bias:
            raise NotImplementedError("Bias is not supported in ColumnParallelLinear")
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = self.output_dim

        param_data = param.data

        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            dtype=dtype,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ):
        output_dim = self.output_dim
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, output_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = (
            total_num_kv_heads if total_num_kv_heads is not None else total_num_heads
        )
        # Divide the weight matrix along the last dimension.
        if tp_rank is None:
            tp_rank = dist.get_rank()
        if tp_size is None:
            tp_size = dist.get_world_size()
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
            dtype=dtype,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        output_dim = self.output_dim
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, output_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            dtype=dtype,
        )
        if tp_rank is None:
            tp_rank = dist.get_rank()
        if tp_size is None:
            tp_size = dist.get_world_size()

        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.input_size_per_partition = divide(input_size, tp_size)
        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.input_size_per_partition, dtype=dtype)
        )
        self.weight
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = self.output_dim

        param_data = param.data
        shard_size = param_data.size(output_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
