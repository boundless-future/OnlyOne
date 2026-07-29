"""Correctness of the dual-adapter reference mechanism.

The ref slot must be an exact frozen snapshot, must survive training updates
to the default slot, and must restore state cleanly on exit.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import make_batch


def test_ref_equals_default_right_after_snapshot(um):
    um.snapshot_ref()
    input_ids, attention_mask, labels = make_batch()
    with torch.no_grad():
        default_logps = um.logps(input_ids, attention_mask, labels)
    with um.as_reference() as ref_model:
        ref_logps = um.logps(input_ids, attention_mask, labels)
    assert torch.allclose(default_logps, ref_logps, atol=1e-6)


def test_ref_unchanged_after_default_update(um):
    um.snapshot_ref()
    input_ids, attention_mask, labels = make_batch()
    with um.as_reference():
        before = um.logps(input_ids, attention_mask, labels)

    # Simulate a training step: perturb every trainable (default-slot) weight.
    with torch.no_grad():
        for name, p in um.model.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)

    with um.as_reference():
        after = um.logps(input_ids, attention_mask, labels)
    with torch.no_grad():
        new_default = um.logps(input_ids, attention_mask, labels)

    assert torch.allclose(before, after, atol=1e-6), "ref 快照被 default 更新污染"
    assert not torch.allclose(new_default, after, atol=1e-4), "default 更新未生效"


def test_ref_params_are_frozen(um):
    um.snapshot_ref()
    for name, p in um.model.named_parameters():
        if ".ref." in name:
            assert not p.requires_grad, f"{name} 应被冻结"


def test_snapshot_after_training_raises(um):
    um.mark_trained(1)
    with pytest.raises(RuntimeError, match="训练 step 之前"):
        um.snapshot_ref()


def test_as_reference_requires_snapshot(um):
    with pytest.raises(RuntimeError, match="snapshot_ref"):
        with um.as_reference():
            pass


def test_as_reference_restores_state(um):
    um.snapshot_ref()
    um.model.train()
    with um.as_reference():
        assert not um.model.training
        assert um.model.active_adapter == "ref"
    assert um.model.training
    assert um.model.active_adapter == "default"


def test_as_reference_blocks_gradients(um):
    um.snapshot_ref()
    input_ids, attention_mask, labels = make_batch()
    with um.as_reference():
        out = um.logps(input_ids, attention_mask, labels)
        assert not out.requires_grad
