"""End-to-end smoke for ORPO: preference data -> build_orpo_trainer -> loss drops
and preference accuracy rises on a fixed toy set."""

from __future__ import annotations

import json

import torch
import yaml


def test_orpo_end_to_end(tiny_model_dir, tmp_path):
    data_path = tmp_path / "pref.jsonl"
    # Fixed toy pairs: chosen always "w3 w4", rejected "w5 w6".
    rows = [{"prompt": "w1 w2", "chosen": "w3 w4", "rejected": "w5 w6"}] * 16
    with open(data_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cfg_dict = {
        "model": {
            "name_or_path": tiny_model_dir,
            "dtype": "fp32", "attn_implementation": "eager",
            "use_lora": True, "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "v_proj"],
        },
        "data": {"train_path": str(data_path), "template": "raw",
                 "max_len": 32, "packing": False},
        "train": {
            "output_dir": str(tmp_path / "out"),
            "per_device_batch_size": 4, "gradient_accumulation_steps": 1,
            "lr": 5e-3, "warmup_ratio": 0.0, "max_steps": 20,
            "gradient_checkpointing": False, "logging_steps": 5,
            "save_steps": 10_000, "device": "cpu",
        },
        "algo": {"name": "orpo", "orpo_lambda": 0.5},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f)

    from onlyone.trainers.orpo import build_orpo_trainer
    from onlyone.utils.config import load_config

    cfg = load_config(cfg_path)
    trainer, dataloader = build_orpo_trainer(cfg)

    history: list[dict] = []
    orig_log = trainer.tracker.log
    trainer.tracker.log = lambda m, s: (history.append(m), orig_log(m, s))
    trainer.train(dataloader)

    assert len(history) >= 2
    first, last = history[0], history[-1]
    assert last["loss"] < first["loss"], f"loss 未下降: {[h['loss'] for h in history]}"
    # ORPO's signature behavior: reward margin (chosen - rejected avg logp) grows
    assert last["reward_margin"] > first["reward_margin"], \
        f"margin 未提升: {[h['reward_margin'] for h in history]}"


def test_kto_end_to_end(tiny_model_dir, tmp_path):
    data_path = tmp_path / "bin.jsonl"
    rows = (
        [{"prompt": "w1 w2", "completion": "w3 w4", "label": True}] * 8
        + [{"prompt": "w1 w2", "completion": "w5 w6", "label": False}] * 8
    )
    with open(data_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cfg_dict = {
        "model": {
            "name_or_path": tiny_model_dir,
            "dtype": "fp32", "attn_implementation": "eager",
            "use_lora": True, "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "v_proj"],
        },
        "data": {"train_path": str(data_path), "template": "raw",
                 "max_len": 32, "packing": False},
        "train": {
            "output_dir": str(tmp_path / "out"),
            "per_device_batch_size": 8, "gradient_accumulation_steps": 1,
            "lr": 5e-3, "warmup_ratio": 0.0, "max_steps": 20,
            "gradient_checkpointing": False, "logging_steps": 5,
            "save_steps": 10_000, "device": "cpu",
        },
        "algo": {"name": "kto", "kto_beta": 0.1},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f)

    from onlyone.trainers.kto import build_kto_trainer
    from onlyone.utils.config import load_config

    cfg = load_config(cfg_path)
    trainer, dataloader = build_kto_trainer(cfg)

    history: list[dict] = []
    orig_log = trainer.tracker.log
    trainer.tracker.log = lambda m, s: (history.append(m), orig_log(m, s))
    trainer.train(dataloader)

    assert len(history) >= 2
    # KTO signature: good-sample rewards rise relative to bad-sample rewards
    first, last = history[0], history[-1]
    gap_first = first["reward_good_mean"] - first["reward_bad_mean"]
    gap_last = last["reward_good_mean"] - last["reward_bad_mean"]
    assert gap_last > gap_first, f"好坏样本 reward 差距未拉开: {gap_first} -> {gap_last}"


def test_gsm8k_eval_runs(tiny_model_dir, tmp_path):
    """eval_gsm8k executes end-to-end on a tiny model (accuracy will be ~0 —
    we only verify the plumbing and report shape)."""
    data_path = tmp_path / "gsm8k.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"question": "w1 w2?", "answer": "...\n#### 8"}) + "\n")

    from onlyone.eval.benchmarks import eval_gsm8k
    from onlyone.models.loading import load_tokenizer
    from onlyone.models.unified import UnifiedModel
    from onlyone.utils.config import ModelConfig

    mc = ModelConfig(
        name_or_path=tiny_model_dir, dtype="fp32", attn_implementation="eager",
        use_lora=False,
    )
    um = UnifiedModel(mc)
    tok = load_tokenizer(mc)
    report = eval_gsm8k(um, tok, str(data_path), template="raw",
                        max_new_tokens=8, device="cpu")
    assert report["n"] == 1
    assert 0.0 <= report["accuracy"] <= 1.0
    assert len(report["samples"]) == 1
