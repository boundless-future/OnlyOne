"""Numerical tests for DPO and SimPO losses (same property-based approach as
ORPO/KTO: boundary values + hand-computed references)."""

from __future__ import annotations

import math

import torch

from onlyone.trainers.dpo import DPOTrainer, SimPOTrainer
from onlyone.utils.config import TrainConfig
from tests.conftest import make_batch


def _train_cfg(tmp_path):
    return TrainConfig(
        output_dir=str(tmp_path), per_device_batch_size=2,
        gradient_accumulation_steps=1, lr=1e-3, max_steps=1,
        gradient_checkpointing=False, device="cpu",
    )


def _pref_batch():
    ids, mask, labels = make_batch(batch_size=4, seq=12, n_completion=6)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# --------------------------------------------------------------------- DPO

def test_dpo_policy_equals_ref_gives_log2(um, tmp_path):
    """Right after snapshot_ref: policy == ref → all rewards 0 →
    loss = -log σ(0) = log 2."""
    um.snapshot_ref()
    trainer = DPOTrainer(um, _train_cfg(tmp_path), beta=0.1)
    loss, metrics = trainer.compute_loss(_pref_batch())
    assert abs(loss.item() - math.log(2)) < 1e-4
    assert metrics["pref_acc"] == 0.0  # all margins exactly 0


def test_dpo_loss_matches_hand_computed(um, tmp_path):
    um.snapshot_ref()
    trainer = DPOTrainer(um, _train_cfg(tmp_path), beta=0.3)
    batch = _pref_batch()
    loss, _ = trainer.compute_loss(batch)

    with torch.no_grad():
        policy = um.logps(**batch)
        with um.as_reference():
            ref = um.logps(**batch)
        pc, pr = policy.chunk(2)
        rc, rr = ref.chunk(2)
        want = -torch.nn.functional.logsigmoid(0.3 * ((pc - rc) - (pr - rr))).mean()
    assert torch.allclose(loss.detach(), want, atol=1e-6)


def test_dpo_ref_receives_no_gradient(um, tmp_path):
    um.snapshot_ref()
    trainer = DPOTrainer(um, _train_cfg(tmp_path), beta=0.1)
    loss, _ = trainer.compute_loss(_pref_batch())
    loss.backward()
    ref_grads = [p.grad for n, p in um.model.named_parameters()
                 if ".ref." in n and p.grad is not None]
    assert not ref_grads


def test_dpo_margin_grows_when_chosen_favored(um, tmp_path):
    """Hand-fixed rewards: chosen +2, rejected -2 → margin 4β, acc 1."""
    um.snapshot_ref()
    trainer = DPOTrainer(um, _train_cfg(tmp_path), beta=0.5)
    real = um.logps
    calls = {"n": 0}
    def fake(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # policy: chosen high, rejected low
            return torch.tensor([2.0, 2.0, -2.0, -2.0])
        return torch.zeros(4)  # ref
    um.logps = fake
    try:
        _, metrics = trainer.compute_loss(_pref_batch())
    finally:
        um.logps = real
    assert abs(metrics["reward_margin"] - 0.5 * 4.0) < 1e-5
    assert metrics["pref_acc"] == 1.0


# ------------------------------------------------------------------- SimPO

def test_simpo_loss_matches_hand_computed(um, tmp_path):
    trainer = SimPOTrainer(um, _train_cfg(tmp_path), beta=2.0, gamma=0.5)
    batch = _pref_batch()
    loss, _ = trainer.compute_loss(batch)

    with torch.no_grad():
        avg = um.logps(**batch, average_per_token=True)
        ac, ar = avg.chunk(2)
        want = -torch.nn.functional.logsigmoid(2.0 * (ac - ar) - 0.5).mean()
    assert torch.allclose(loss.detach(), want, atol=1e-6)


def test_simpo_equal_logps_loss_is_logsigmoid_of_gamma(um, tmp_path):
    """chosen == rejected → loss = -log σ(-γ) = log(1 + e^γ)."""
    trainer = SimPOTrainer(um, _train_cfg(tmp_path), beta=2.0, gamma=0.7)
    real = um.logps
    um.logps = lambda **kw: torch.full((4,), -3.0)
    try:
        loss, metrics = trainer.compute_loss(_pref_batch())
    finally:
        um.logps = real
    want = math.log(1 + math.exp(0.7))
    assert abs(loss.item() - want) < 1e-4
    assert metrics["pref_acc"] == 0.0  # margin 0 < γ


def test_simpo_uses_no_reference(um, tmp_path):
    """SimPO must not require snapshot_ref (reference-free by design)."""
    trainer = SimPOTrainer(um, _train_cfg(tmp_path))  # no snapshot_ref called
    loss, _ = trainer.compute_loss(_pref_batch())
    assert torch.isfinite(loss)
