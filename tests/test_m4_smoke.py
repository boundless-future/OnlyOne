"""M4 end-to-end smoke: GRPO full loop on a tiny model.

The loop: prompts -> rollout -> rewards -> advantages -> clipped update.
With a random tiny model most math rewards are 0, so we register a custom
reward keyed on a token the sampler will produce with varying frequency —
that guarantees reward variance inside groups, which is what GRPO needs.
"""

from __future__ import annotations

import json

import yaml


def test_grpo_end_to_end(tiny_model_dir, tmp_path):
    prompts_path = tmp_path / "prompts.jsonl"
    rows = [{"prompt": "w1 w2", "meta": {}}] * 8
    with open(prompts_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cfg_dict = {
        "model": {
            "name_or_path": tiny_model_dir,
            "dtype": "fp32", "attn_implementation": "eager",
            "use_lora": True, "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "v_proj"],
        },
        "data": {"train_path": str(prompts_path), "template": "raw",
                 "max_len": 32, "packing": False},
        "train": {
            "output_dir": str(tmp_path / "out"),
            "per_device_batch_size": 1, "gradient_accumulation_steps": 1,
            "lr": 1e-3, "warmup_ratio": 0.0, "max_steps": 6,
            "gradient_checkpointing": False, "logging_steps": 2,
            "save_steps": 10_000, "device": "cpu",
        },
        "algo": {"name": "grpo"},
        "grpo": {
            "prompts_path": str(prompts_path),
            "group_size": 4,
            "prompts_per_step": 4,
            "rewards": ["length_bucket"],
            "clip_eps": 0.2,
            "kl_beta": 0.01,
            "kl_circuit_breaker": 0.0,
            "max_new_tokens": 8,
            "temperature": 1.2,   # 高温度保证小模型候选多样性
            "top_p": 1.0,
            "rollout_batch_size": 16,
            "engine": "hf",
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f)

    # 自定义奖励:按 completion 长度分桶,天然带组内方差
    from onlyone.rewards.registry import register
    try:
        register("length_bucket", lambda p, r, m: float(len(r) % 5) / 5.0)
    except KeyError:
        pass  # 已注册(重复跑测试时)

    from onlyone.trainers.grpo import build_grpo_trainer
    from onlyone.utils.config import load_config

    cfg = load_config(cfg_path)
    trainer = build_grpo_trainer(cfg)

    history: list[dict] = []
    orig_log = trainer.tracker.log
    trainer.tracker.log = lambda m, s: (history.append(m), orig_log(m, s))
    trainer.train()

    assert trainer.global_step == 6
    assert len(history) >= 2
    for h in history:
        assert h["kl"] >= 0, "k3 KL 必须非负"
        assert 0.0 <= h["clip_frac"] <= 1.0
        assert "reward_mean" in h and "completion_chars" in h


def test_grpo_skip_guard_against_degenerate_rewards(tiny_model_dir, tmp_path):
    """If rewards never vary (model all-right/all-wrong), the trainer must
    abort with an informative error instead of looping forever."""
    prompts_path = tmp_path / "prompts.jsonl"
    with open(prompts_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "w1 w2", "meta": {}}) + "\n")

    cfg_dict = {
        "model": {
            "name_or_path": tiny_model_dir, "dtype": "fp32",
            "attn_implementation": "eager", "use_lora": True,
            "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "v_proj"],
        },
        "data": {"train_path": str(prompts_path), "template": "raw",
                 "max_len": 32, "packing": False},
        "train": {
            "output_dir": str(tmp_path / "out"), "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1, "lr": 1e-3, "max_steps": 5,
            "gradient_checkpointing": False, "logging_steps": 1,
            "save_steps": 10_000, "device": "cpu",
        },
        "algo": {"name": "grpo"},
        "grpo": {
            "prompts_path": str(prompts_path), "group_size": 2,
            "prompts_per_step": 1, "rewards": ["always_same"],
            "max_new_tokens": 4, "temperature": 1.0, "top_p": 1.0,
            "rollout_batch_size": 4, "engine": "hf",
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f)

    from onlyone.rewards.registry import register
    try:
        register("always_same", lambda p, r, m: 0.5)  # 恒定:所有组退化
    except KeyError:
        pass

    import pytest

    from onlyone.trainers.grpo import build_grpo_trainer
    from onlyone.utils.config import load_config

    cfg = load_config(cfg_path)
    trainer = build_grpo_trainer(cfg)
    with pytest.raises(RuntimeError, match="连续 .* 次 rollout"):
        trainer.train()
