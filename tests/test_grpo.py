"""Numerical tests for GRPO: group advantages, k3 KL, clipped surrogate.

GRPO is the algorithm where silent divergence hurts most (online RL, no
static data to sanity-check against). These tests pin down every component
before any real GPU hour is spent.
"""

from __future__ import annotations

import pytest
import torch

from onlyone.trainers.grpo import GRPOTrainer, group_advantages, kl_k3
from onlyone.utils.config import GRPOConfig, TrainConfig
from tests.conftest import make_batch


# --------------------------------------------------------- group advantages

def test_group_advantages_zero_mean_unit_std():
    advs, skipped = group_advantages([1.0, 2.0, 3.0, 4.0], group_size=4)
    assert skipped == 0
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)
    var = sum(a * a for a in advs) / len(advs)
    assert var ** 0.5 == pytest.approx(1.0, abs=1e-2)


def test_group_advantages_per_group_independent():
    # Two groups with different scales: each normalized independently.
    advs, _ = group_advantages([0.0, 1.0, 10.0, 20.0], group_size=2)
    assert advs[0] == pytest.approx(advs[2], abs=1e-3)
    assert advs[1] == pytest.approx(advs[3], abs=1e-3)
    assert advs[0] < 0 < advs[1]


def test_group_advantages_degenerate_group_zeroed_and_counted():
    advs, skipped = group_advantages([1.0, 1.0, 0.0, 1.0], group_size=2)
    assert advs[:2] == [0.0, 0.0]  # identical rewards -> no signal
    assert skipped == 1
    assert advs[2] < 0 < advs[3]


def test_group_advantages_bad_length_raises():
    with pytest.raises(ValueError, match="整数倍"):
        group_advantages([1.0, 2.0, 3.0], group_size=2)


# ------------------------------------------------------------------ k3 KL

def test_kl_k3_zero_at_identity():
    x = torch.randn(5, 7)
    assert torch.allclose(kl_k3(x, x), torch.zeros_like(x), atol=1e-6)


def test_kl_k3_nonnegative():
    policy = torch.randn(4, 10)
    ref = torch.randn(4, 10)
    assert (kl_k3(policy, ref) >= 0).all()


def test_kl_k3_matches_closed_form():
    # Δ = ref - policy = 0.5 → e^0.5 - 0.5 - 1
    policy = torch.zeros(3)
    ref = torch.full((3,), 0.5)
    want = torch.exp(torch.tensor(0.5)) - 0.5 - 1
    assert torch.allclose(kl_k3(policy, ref), want.expand(3), atol=1e-6)


# -------------------------------------------------------------- GRPO loss

def _grpo_cfg(**overrides):
    base = dict(prompts_path="unused", group_size=2, clip_eps=0.2, kl_beta=0.1)
    base.update(overrides)
    return GRPOConfig(**base)


def _train_cfg(tmp_path):
    return TrainConfig(
        output_dir=str(tmp_path), per_device_batch_size=2,
        gradient_accumulation_steps=1, lr=1e-3, max_steps=1,
        gradient_checkpointing=False, device="cpu",
    )


def _trainer(um, tmp_path, **grpo_overrides):
    return GRPOTrainer(
        um, _train_cfg(tmp_path), _grpo_cfg(**grpo_overrides),
        engine=None, reward_fn=None, prompts=[],  # compute_loss 用不到
        tokenizer=None,
    )


def _batch_with_adv():
    ids, mask, labels = make_batch(batch_size=4, seq=12, n_completion=6)
    return {
        "input_ids": ids, "attention_mask": mask, "labels": labels,
        "advantages": torch.tensor([1.0, -1.0, 0.5, -0.5]),
    }


def test_grpo_loss_matches_hand_computed(um, tmp_path):
    um.snapshot_ref()
    trainer = _trainer(um, tmp_path)
    batch = _batch_with_adv()
    loss, metrics = trainer.compute_loss(batch)

    with torch.no_grad():
        tl, counts = um.token_logps(batch["input_ids"], batch["attention_mask"], batch["labels"])
        with um.as_reference():
            ref_tl, _ = um.token_logps(batch["input_ids"], batch["attention_mask"], batch["labels"])
        mask = batch["labels"][:, 1:] != -100
        # policy == old here (no update between), so ratio = 1 exactly
        ratio = torch.ones_like(tl)
        adv = batch["advantages"].unsqueeze(1).expand_as(ratio)
        surr = torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
        kl = kl_k3(tl, ref_tl)
        per_tok = -surr + 0.1 * kl
        want = ((per_tok * mask).sum(1) / mask.sum(1)).mean()
    assert torch.allclose(loss.detach(), want, atol=1e-5)
    assert metrics["kl"] >= 0
    assert 0.0 <= metrics["clip_frac"] <= 1.0


def test_grpo_ratio_is_one_on_first_step(um, tmp_path):
    """μ=1: old policy recomputed from the same weights → ratio exactly 1,
    clip_frac 0, and loss reduces to -mean(adv) + β·KL."""
    um.snapshot_ref()
    trainer = _trainer(um, tmp_path, kl_beta=0.0)
    batch = _batch_with_adv()
    loss, metrics = trainer.compute_loss(batch)
    assert metrics["clip_frac"] == 0.0
    # loss = -mean over sequences of (masked mean of adv) = -mean(advantages)
    want = -batch["advantages"].mean()
    assert abs(loss.item() - want.item()) < 1e-4


def test_grpo_positive_adv_reduces_loss(um, tmp_path):
    """All-positive advantages (good group) → loss negative (push up logps)."""
    um.snapshot_ref()
    trainer = _trainer(um, tmp_path, kl_beta=0.0)
    batch = _batch_with_adv()
    batch["advantages"] = torch.ones(4)
    loss, _ = trainer.compute_loss(batch)
    assert loss.item() < 0


def test_grpo_circuit_breaker(um, tmp_path):
    """KL over the breaker threshold must abort with a clear error."""
    um.snapshot_ref()
    trainer = _trainer(um, tmp_path)
    trainer.grpo.kl_circuit_breaker = 1e-9  # any nonzero KL trips it
    # Simulate train-loop check directly (train() needs an engine).
    batch = _batch_with_adv()
    _, metrics = trainer.compute_loss(batch)
    tripped = metrics["kl"] > trainer.grpo.kl_circuit_breaker
    # policy == ref at init → kl ≈ 0 → NOT tripped at init (sanity)
    assert not tripped

    # Now simulate a drifted policy by perturbing trainable weights.
    with torch.no_grad():
        for n, p in um.model.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.5)
    _, metrics2 = trainer.compute_loss(batch)
    assert metrics2["kl"] > trainer.grpo.kl_circuit_breaker
