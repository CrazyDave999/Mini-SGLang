from typing import List, Optional, TYPE_CHECKING

import torch
from torch import nn

from dataclasses import dataclass

if TYPE_CHECKING:
    from minisglang.engine.batch import Batch
@dataclass
class SamplingParams:
    max_new_tokens: int = 128
    stop_token_ids: Optional[List[int]] = None
    temperature: float = 0.0
    ignore_eos: bool = False


@dataclass
class SamplingBatchInfo:
    # Basic batched sampling params
    temperatures: torch.Tensor

    @classmethod
    def from_batch(cls, batch: "Batch"):
        reqs = batch.reqs
        device = batch.device
        temperatures = (
            torch.tensor(
                [r.sampling_params.temperature for r in reqs],
                dtype=torch.float,
            )
            .view(-1, 1)
            .to(device, non_blocking=True)
        )
        return cls(temperatures=temperatures)
    
    def merge_batch(self, other: "SamplingBatchInfo"):
        for item in [
            "temperatures",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            setattr(self, item, torch.cat([self_val, other_val]))
    def filter_batch(self, keep_indices_device: torch.Tensor):
        for item in [
            "temperatures",
        ]:
            value = getattr(self, item, None)
            setattr(self, item, value[keep_indices_device])
class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
    ):
        # logits: shape = (bs, vocab_size)
        print(f"sample. {logits.shape=}, {sampling_info.temperatures.shape=}")
        logits = logits.to(torch.float)
        # greedy_tokens: shape = (bs,)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(sampling_info.temperatures)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10
        # sample_tokens: shape = (bs,)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1) + epsilon
        ).argmax(dim=-1)
        next_token_ids = torch.where(sampling_info.temperatures.view(-1) == 0, greedy_tokens, sample_tokens)
        # print(f"{sampling_info.temperatures.shape=} {greedy_tokens.shape=} {sample_tokens.shape=} {next_token_ids.shape=}")
        return next_token_ids
