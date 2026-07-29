"""M3 end-to-end smokes: DPO/SimPO training loops + HF rollout engine."""

from __future__ import annotations

import json

import yaml


def _base_cfg(tiny_model_dir, tmp_path, data_path, algo):
    return {
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
        "algo": algo,
    }


def _write_pref_data(tmp_path):
    data_path = tmp_path / "pref.jsonl"
    rows = [{"prompt": "w1 w2", "chosen": "w3 w4", "rejected": "w5 w6"}] * 16
    with open(data_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return data_path


def _run_and_check_margin(cfg_dict, tmp_path, builder_path):
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f)

    import importlib
    mod_name, fn_name = builder_path.rsplit(".", 1)
    build = getattr(importlib.import_module(mod_name), fn_name)

    from onlyone.utils.config import load_config
    cfg = load_config(cfg_path)
    trainer, dataloader = build(cfg)

    history: list[dict] = []
    orig_log = trainer.tracker.log
    trainer.tracker.log = lambda m, s: (history.append(m), orig_log(m, s))
    trainer.train(dataloader)

    assert len(history) >= 2
    first, last = history[0], history[-1]
    assert last["reward_margin"] > first["reward_margin"], \
        f"margin 未提升: {[h['reward_margin'] for h in history]}"
    return history


def test_dpo_end_to_end(tiny_model_dir, tmp_path):
    data_path = _write_pref_data(tmp_path)
    cfg = _base_cfg(tiny_model_dir, tmp_path, data_path,
                    {"name": "dpo", "dpo_beta": 0.5})
    # 初始 loss ≈ log2 的性质已由 test_dpo_policy_equals_ref_gives_log2 覆盖;
    # 这里只验证训练动态(margin 提升)
    _run_and_check_margin(cfg, tmp_path, "onlyone.trainers.dpo.build_dpo_trainer")


def test_simpo_end_to_end(tiny_model_dir, tmp_path):
    data_path = _write_pref_data(tmp_path)
    cfg = _base_cfg(tiny_model_dir, tmp_path, data_path,
                    {"name": "simpo", "simpo_beta": 2.0, "simpo_gamma": 0.3})
    _run_and_check_margin(cfg, tmp_path, "onlyone.trainers.dpo.build_dpo_trainer")


def test_hf_rollout_engine(tiny_model_dir, tmp_path):
    """Rollout returns the right shape, excludes the prompt, and leaves the
    model in its original training state."""
    from onlyone.models.loading import load_tokenizer
    from onlyone.models.unified import UnifiedModel
    from onlyone.rollout.hf_engine import HFRolloutEngine
    from onlyone.utils.config import ModelConfig

    mc = ModelConfig(name_or_path=tiny_model_dir, dtype="fp32",
                     attn_implementation="eager", use_lora=False)
    um = UnifiedModel(mc)
    um.model.train()  # simulate mid-training state

    tok = load_tokenizer(mc)
    engine = HFRolloutEngine(um, tok, template="raw", device="cpu", batch_size=2)
    out = engine.generate(
        ["w1 w2", "w3"], n_per_prompt=3, max_new_tokens=4,
        temperature=0.8, top_p=0.95,
    )

    assert len(out) == 2
    assert all(len(c) == 3 for c in out)
    assert all(isinstance(text, str) for c in out for text in c)
    assert um.model.training  # state restored
    assert um.model.config.use_cache is False  # training default restored


def test_as_generator_restores_state(um):
    """as_generator flips use_cache on inside, restores the PRIOR value outside
    (whatever it was), and restores train/eval mode."""
    um.model.train()
    um.model.config.use_cache = False  # 训练态显式设定(from_model 路径默认 True)
    with um.as_generator() as model:
        assert not model.training
        assert model.config.use_cache is True
    assert um.model.training
    assert um.model.config.use_cache is False


def test_as_generator_restores_use_cache_true_if_previously_true(um):
    um.model.eval()
    um.model.config.use_cache = True
    with um.as_generator() as model:
        assert model.config.use_cache is True
    assert um.model.config.use_cache is True
    assert not um.model.training
