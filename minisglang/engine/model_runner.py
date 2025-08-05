import torch
from torch import nn
from minisglang.engine.batch import Batch
from minisglang.utils.model_config import ModelConfig
from minisglang.layers.sampler import Sampler
from minisglang.memory.kvcache import KVCache
from minisglang.memory.page_manager import PageManager
from minisglang.layers.attention import FlashAttentionBackend


def get_model(
    model_config: ModelConfig,
) -> nn.Module:
    raise NotImplementedError()


class ModelRunner:
    def __init__(
        self,
        model_path: str,
        tp_rank: int,
        device: str,
    ):
        self.tp_rank = tp_rank
        self.model_config = ModelConfig(model_path)
        self.model = get_model(self.model_config)
        self.sampler = Sampler()
        self.device = device

        self.attn_backend = FlashAttentionBackend(self)

    def loop(self):
        while True:
            # receive batch from engine
            batch = self.recv_batch()
            if batch is not None:
                logits_output, next_token_ids = self.forward(batch)

                # post processing

    def recv_batch(self) -> Batch:
        """Receive a batch from the engine."""

    def forward(
        self,
        batch: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        self.attn_backend.init_forward_metadata(batch)
        logits_output = self.model.forward(batch.input_ids, batch.positions, batch)

        next_token_ids = self.sampler(
            logits=logits_output, temperatures=batch.temperatures
        )

        return logits_output, next_token_ids
