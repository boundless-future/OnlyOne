"""Shared test fixtures: a tiny randomly-initialized Llama on CPU.

No downloads — the whole point is that CI can verify numerical correctness
on any machine.
"""

from __future__ import annotations

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from onlyone.models.unified import UnifiedModel
from onlyone.utils.config import ModelConfig

VOCAB = 64
SEQ = 12


@pytest.fixture(scope="session")
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        name_or_path="tiny",  # unused by from_model
        dtype="fp32",
        attn_implementation="eager",
        use_lora=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "v_proj"],
    )


@pytest.fixture()
def tiny_backbone() -> LlamaForCausalLM:
    # Function-scoped: each test gets a fresh backbone so PEFT adapters never
    # accumulate across tests.
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture()
def um(tiny_backbone, tiny_cfg) -> UnifiedModel:
    torch.manual_seed(1)  # deterministic LoRA init
    return UnifiedModel.from_model(tiny_backbone, tiny_cfg)


def make_batch(batch_size: int = 3, seq: int = SEQ, n_completion: int = 5):
    """Random token batch; first (seq - n_completion) positions masked in labels."""
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, VOCAB, (batch_size, seq), generator=g)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : seq - n_completion] = -100
    return input_ids, attention_mask, labels


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Self-contained tiny model + tokenizer directory on disk (no downloads).

    Used by end-to-end tests that exercise the real loading path
    (AutoModelForCausalLM.from_pretrained / AutoTokenizer.from_pretrained).
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny_model")

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    ))
    model.save_pretrained(d)

    tok = Tokenizer(models.WordLevel(
        vocab={"<pad>": 0, "<eos>": 1} | {f"w{i}": i + 2 for i in range(125)},
        unk_token="<eos>",
    ))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>"
    )
    fast.save_pretrained(d)
    return str(d)
