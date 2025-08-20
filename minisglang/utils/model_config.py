from transformers import AutoConfig, PretrainedConfig
import torch

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class ModelConfig:
    def __init__(
        self,
        model_path: str,
    ) -> None:
        self.model_path = model_path
        self.hf_config = AutoConfig.from_pretrained(model_path)
        self.hf_text_config = get_hf_text_config(self.hf_config)

        self.context_len = get_context_length(self.hf_text_config)

        self.head_dim = getattr(
            self.hf_text_config,
            "head_dim",
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads,
        )
        self.num_key_value_heads = getattr(
            self.hf_text_config, "num_key_value_heads", None
        )

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.hf_text_config.num_attention_heads

        self.hidden_size = self.hf_text_config.hidden_size
        self.num_hidden_layers = self.hf_text_config.num_hidden_layers
        self.vocab_size = self.hf_text_config.vocab_size

        config_dtype = getattr(self.hf_text_config, "torch_dtype", None)
        if config_dtype is not None:
            self.dtype = config_dtype
        else:
            raise ValueError(f"Unknown dtype: {config_dtype}")
        
    def get_num_kv_heads_per_GPU(self, tp_size: int) -> int:
        return max(1, self.num_key_value_heads // tp_size)



def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    class_name = config.architectures[0]
    if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
        # We support non-hf version of llava models, so we do not want to
        # read the wrong values from the unused default text_config.
        # NOTE(HandH1998): We set `torch_dtype` of config to `torch.float16` for the weights, as
        # `torch.float16` is default used for image features in `python/sglang/srt/models/llava.py`.
        setattr(config, "torch_dtype", torch.float16)
        return config

    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    if hasattr(config, "language_config"):
        return config.language_config
    else:
        return config


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model configs."""
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048
