"""Model loading strategies: dtype / attention / 4bit quantization."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from onlyone.utils.config import ModelConfig

_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizer:
    tok = AutoTokenizer.from_pretrained(cfg.name_or_path, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Right padding for training; rollout engines set their own.
    tok.padding_side = "right"
    return tok


def load_backbone(cfg: ModelConfig) -> PreTrainedModel:
    kwargs: dict = dict(
        torch_dtype=_DTYPE[cfg.dtype],
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=cfg.trust_remote_code,
    )
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_DTYPE[cfg.dtype],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, **kwargs)
    model.config.use_cache = False  # training default; rollout re-enables it
    return model
