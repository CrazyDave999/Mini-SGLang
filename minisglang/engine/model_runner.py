import torch
from torch import nn
from minisglang.engine.batch import Batch
from minisglang.utils.model_config import ModelConfig
from minisglang.layers.sampler import Sampler
from minisglang.memory.kvcache import KVCache
from minisglang.memory.page_manager import PageManager
from minisglang.layers.attention import FlashAttentionBackend
from minisglang.models.llama import LlamaForCausalLM
import os
from glob import glob
from safetensors import safe_open

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def get_model(
    model_config: ModelConfig,
) -> nn.Module:
    model = LlamaForCausalLM(model_config.hf_config)
    # load weight
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
    return model


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
