"""
Torch-native implementation for FusedMoE. This is used for torch.compile.
It is based on https://github.com/pytorch-labs/gpt-fast/blob/32971d3129541c5bfb4f715abc33d1c5f408d204/mixtral-moe/model.py#L204
"""

from typing import Callable, Optional

import torch
from torch.nn import functional as F

from minisglang.layers.activation import SiluAndMul
from minisglang.layers.moe.topk import select_experts


def fused_moe_forward_native(
    layer: torch.nn.Module,
    x: torch.Tensor, # shape: (M, I)
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:

    if apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        torch_native=True,
    )
    # topk_weights: shape: (M, topk)
    # topk_ids: shape: (M, topk)

    w13_weights = layer.w13_weight[topk_ids] # shape: (M, topk, O * 2, I)
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)

    # w1_weights: shape: (M, topk, O, I)
    # w3_weights: shape: (M, topk, O, I)

    w2_weights = layer.w2_weight[topk_ids] # shape: (M, topk, O, I)
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights) # shape: (M, topk, O)
    if activation == "silu":
        x1 = F.silu(x1)
    elif activation == "gelu":
        x1 = F.gelu(x1)
    else:
        raise ValueError(f"Unsupported activation: {activation=}")
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights) # shape: (M, topk, O)
    expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights) # shape: (M, topk, I)
    return torch.einsum("tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype)) # shape: (M, I)

