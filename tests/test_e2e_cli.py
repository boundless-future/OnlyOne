"""True end-to-end: local tiny model dir + jsonl -> CLI config -> train -> ckpt.

Builds a self-contained tiny model + tokenizer directory (no downloads), then
runs the same code path as `onlyone train --config ...`.
"""

from __future__ import annotations

import json

import pytest
import torch
import yaml


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

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


def test_cli_train_end_to_end(tiny_model_dir, tmp_path):
    # --- dataset
    data_path = tmp_path / "train.jsonl"
    rows = [
        {"prompt": "w1 w2", "completion": "w3 w4 w5"},
        {"prompt": "w6 w7", "completion": "w8 w9 w10"},
    ] * 8
    with open(data_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # --- config through the same loader the CLI uses
    cfg_dict = {
        "model": {
            "name_or_path": tiny_model_dir,
            "dtype": "fp32",
            "attn_implementation": "eager",
            "use_lora": True,
            "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "v_proj"],
        },
        "data": {
            "train_path": str(data_path),
            "template": "raw",
            "max_len": 32,
            "packing": False,
        },
        "train": {
            "output_dir": str(tmp_path / "out"),
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "lr": 5e-3,
            "warmup_ratio": 0.0,
            "max_steps": 10,
            "gradient_checkpointing": False,
            "logging_steps": 5,
            "save_steps": 10,
            "device": "cpu",
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f)

    from onlyone.utils.config import load_config
    from onlyone.trainers.sft import build_sft_trainer

    cfg = load_config(cfg_path)
    trainer, dataloader = build_sft_trainer(cfg)

    losses: list[float] = []
    orig_log = trainer.tracker.log
    trainer.tracker.log = lambda m, s: (losses.append(m["loss"]), orig_log(m, s))
    trainer.train(dataloader)

    assert len(losses) >= 2 and losses[-1] < losses[0], f"loss 未下降: {losses}"

    # --- checkpoint artifacts
    ckpt = tmp_path / "out" / "final"
    assert (ckpt / "adapter_model.safetensors").exists()
    assert (ckpt / "trainer_state.pt").exists()
    state = torch.load(ckpt / "trainer_state.pt", weights_only=False)
    assert state["global_step"] == 10
