"""Checkpoint rotation and resume-from-checkpoint (LoRA adapter path).

Resume correctness has one subtle invariant: the ref adapter must stay at
the run's STARTING policy (frozen by snapshot_ref before any resume load),
while the policy adapter, optimizer, scheduler, step counter and prompt RNG
all restore from the checkpoint.
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from onlyone.models.unified import UnifiedModel
from tests.test_grpo import _batch_with_adv, _trainer


def _fresh_um(tiny_cfg):
    """Second independently-built model with identical init (same seeds)."""
    torch.manual_seed(0)
    backbone = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )).eval()
    torch.manual_seed(1)
    return UnifiedModel.from_model(backbone, tiny_cfg)


def _train_one_step(trainer, batch):
    loss, _ = trainer.compute_loss(batch)
    loss.backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.global_step += 1


def _lora_params(model, adapter):
    return {n: p for n, p in model.named_parameters() if f".{adapter}." in n}


def test_resume_restores_full_training_state(um, tiny_cfg, tmp_path):
    um.snapshot_ref()
    um.model.train()
    trainer1 = _trainer(um, tmp_path / "run")
    batch = _batch_with_adv()
    _train_one_step(trainer1, batch)
    _train_one_step(trainer1, batch)
    trainer1.rng.sample([1, 2, 3, 4], 2)  # move the prompt RNG off its start
    trainer1.save_checkpoint()  # -> step2

    um2 = _fresh_um(tiny_cfg)
    um2.snapshot_ref()  # ref anchor = fresh init; must happen BEFORE resume load
    um2.model.train()
    trainer2 = _trainer(um2, tmp_path / "run2")
    trainer2.load_checkpoint(str(tmp_path / "run" / "step2"))

    assert trainer2.global_step == 2

    # policy adapter weights transferred exactly
    p1 = _lora_params(um.model, "default")
    p2 = _lora_params(um2.model, "default")
    assert p1 and all(torch.equal(p1[k], p2[k]) for k in p1)

    # ref adapter is still the fresh init (lora_B starts at zero), i.e. the
    # resume did NOT contaminate the KL anchor with trained weights
    ref_b = [p for n, p in um2.model.named_parameters()
             if "lora_B" in n and ".ref." in n]
    assert ref_b and all(p.abs().sum() == 0 for p in ref_b)

    # optimizer moments actually restored (non-zero after 2 real steps)
    exp_avgs = [s["exp_avg"] for s in trainer2.optimizer.state.values()]
    assert exp_avgs and any(t.abs().sum() > 0 for t in exp_avgs)

    assert (trainer2.scheduler.get_last_lr()[0]
            == trainer1.scheduler.get_last_lr()[0])
    assert trainer2.rng.getstate() == trainer1.rng.getstate()

    # and the resumed trainer can keep training (loss is finite)
    loss, _ = trainer2.compute_loss(batch)
    assert torch.isfinite(loss)


def test_checkpoint_rotation_keeps_last_n_and_final(um, tmp_path):
    um.snapshot_ref()
    trainer = _trainer(um, tmp_path)
    trainer.cfg.keep_last_n_checkpoints = 2
    for step in (10, 20, 30):
        trainer.global_step = step
        trainer.save_checkpoint()

    remaining = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
    assert remaining == ["step20", "step30"]  # step10 rotated out

    trainer.save_checkpoint(final=True)
    remaining = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
    assert remaining == ["final", "step20", "step30"]  # final never rotates


def test_checkpoint_rotation_disabled_by_default(um, tmp_path):
    um.snapshot_ref()
    trainer = _trainer(um, tmp_path)
    for step in (10, 20, 30):
        trainer.global_step = step
        trainer.save_checkpoint()
    remaining = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
    assert remaining == ["step10", "step20", "step30"]
