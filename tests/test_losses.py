"""Numerical tests for ORPO and KTO losses.

These verify mathematical PROPERTIES of the losses (boundary values, symmetry,
hand-computed references), not just that the code runs. Alignment losses that
silently diverge are the #1 risk of self-written trainers (design doc §9).
"""

from __future__ import annotations

import torch

from onlyone.trainers.kto import KTOTrainer
from onlyone.trainers.orpo import ORPOTrainer
from onlyone.utils.config import TrainConfig
from tests.conftest import make_batch

_VOCAB = 64


def _train_cfg(tmp_path):
    return TrainConfig(
        output_dir=str(tmp_path), per_device_batch_size=2,
        gradient_accumulation_steps=1, lr=1e-3, max_steps=1,
        gradient_checkpointing=False, device="cpu",
    )


def _pref_batch():
    """Batch shaped (2B, T): first half chosen, second half rejected."""
    ids, mask, labels = make_batch(batch_size=4, seq=12, n_completion=6)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def test_orpo_loss_matches_hand_computed(um, tmp_path):
    trainer = ORPOTrainer(um, _train_cfg(tmp_path), lambda_or=0.5)
    batch = _pref_batch()
    loss, metrics = trainer.compute_loss(batch)

    # Independent hand computation from token_logps.
    with torch.no_grad():
        token_logps, counts = um.token_logps(**batch)
        seq_logps = token_logps.sum(-1)
        avg = seq_logps / counts.clamp(min=1)
        c, r = avg.chunk(2)
        cc, _ = counts.chunk(2)
        nll = -token_logps.chunk(2)[0].sum() / cc.sum()
        p_c = c.exp().clamp(max=1 - 1e-6)
        p_r = r.exp().clamp(max=1 - 1e-6)
        lo_c = c - torch.log1p(-p_c)
        lo_r = r - torch.log1p(-p_r)
        or_term = -torch.nn.functional.logsigmoid(lo_c - lo_r).mean()
        want = nll + 0.5 * or_term

    assert torch.allclose(loss.detach(), want, atol=1e-5)
    assert metrics["pref_acc"] in (0.0, 0.5, 1.0)


def test_orpo_or_term_vanishes_when_chosen_dominates(um, tmp_path):
    """If chosen logp >> rejected logp, log_odds diff → +inf, or_loss → 0."""
    trainer = ORPOTrainer(um, _train_cfg(tmp_path), lambda_or=1.0)
    batch = _pref_batch()
    # Artificially construct the dominating case by monkeypatching token_logps.
    real = um.token_logps
    def fake(**kwargs):
        tl, counts = real(**kwargs)
        b = tl.shape[0] // 2
        tl = tl.clone()
        tl[:b] = -0.01   # chosen: near-certain
        tl[b:] = -50.0   # rejected: near-impossible
        return tl, counts
    um.token_logps = fake
    try:
        _, metrics = trainer.compute_loss(batch)
    finally:
        um.token_logps = real
    assert metrics["or_loss"] < 1e-3
    assert metrics["pref_acc"] == 1.0


def test_orpo_equal_logps_gives_log2(um, tmp_path):
    """chosen == rejected → log_odds diff = 0 → or_loss = -log σ(0) = log 2."""
    trainer = ORPOTrainer(um, _train_cfg(tmp_path), lambda_or=1.0)
    batch = _pref_batch()
    real = um.token_logps
    def fake(**kwargs):
        tl, counts = real(**kwargs)
        return torch.full_like(tl, -2.0), counts
    um.token_logps = fake
    try:
        _, metrics = trainer.compute_loss(batch)
    finally:
        um.token_logps = real
    assert abs(metrics["or_loss"] - 0.6931) < 1e-3


def _binary_batch(n_good: int = 2, n_bad: int = 2):
    ids, mask, labels = make_batch(batch_size=n_good + n_bad, seq=12, n_completion=6)
    is_good = torch.tensor([1.0] * n_good + [0.0] * n_bad)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels, "is_good": is_good}


def test_kto_policy_equals_ref_gives_half(um, tmp_path):
    """Right after snapshot_ref, policy == ref → all rewards = 0 → z_ref = 0 →
    every loss term = 1 - σ(0) = 0.5; with unit weights, total = 1.0."""
    um.snapshot_ref()
    trainer = KTOTrainer(um, _train_cfg(tmp_path), beta=0.1)
    loss, metrics = trainer.compute_loss(_binary_batch())
    assert abs(loss.item() - 1.0) < 1e-4, f"loss={loss.item()}"
    assert abs(metrics["kl_estimate"]) < 1e-6


def test_kto_gradient_flows_through_policy_not_ref(um, tmp_path):
    um.snapshot_ref()
    trainer = KTOTrainer(um, _train_cfg(tmp_path), beta=0.1)
    loss, _ = trainer.compute_loss(_binary_batch())
    loss.backward()
    ref_grads = [p.grad for n, p in um.model.named_parameters()
                 if ".ref." in n and p.grad is not None]
    assert not ref_grads, "reference adapter 不应收到梯度"


def test_kto_good_reward_pushes_loss_down(um, tmp_path):
    """If good samples already beat baseline, good_loss < 0.5."""
    um.snapshot_ref()
    trainer = KTOTrainer(um, _train_cfg(tmp_path), beta=0.1)
    real = um.logps
    calls = {"n": 0}
    def fake(**kwargs):
        # First call (policy): good +3, bad -3. Second call (ref): zeros.
        calls["n"] += 1
        if calls["n"] == 1:
            return torch.tensor([3.0, 3.0, -3.0, -3.0])
        return torch.zeros(4)
    um.logps = fake
    try:
        _, metrics = trainer.compute_loss(_binary_batch())
    finally:
        um.logps = real
    assert metrics["kto_good_loss"] < 0.5
    assert metrics["kto_bad_loss"] < 0.5  # bad rewards below baseline too
    assert metrics["kto_acc"] == 1.0
