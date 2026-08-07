"""Numerical correctness of UnifiedModel.logps — THE critical test.

Every alignment loss (SFT/ORPO/DPO/GRPO) is built on logps. If this is wrong,
training silently diverges and nothing errors. So we verify against an
independent hand-computed reference, plus edge cases.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.conftest import make_batch


def hand_computed_logps(model, input_ids, attention_mask, labels):
    """Independent reference implementation, intentionally written differently
    (per-token loop instead of gather) to catch indexing/shift bugs."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = F.log_softmax(logits.float(), dim=-1)
    total = torch.zeros(input_ids.shape[0])
    for b in range(input_ids.shape[0]):
        for t in range(input_ids.shape[1] - 1):
            lab = labels[b, t + 1].item()
            if lab == -100:
                continue
            total[b] += logp[b, t, lab]
    return total


def test_logps_matches_hand_computed(um):
    input_ids, attention_mask, labels = make_batch()
    with torch.no_grad():
        got = um.logps(input_ids, attention_mask, labels)
    want = hand_computed_logps(um.model, input_ids, attention_mask, labels)
    assert torch.allclose(got, want, atol=1e-5), f"max diff {(got - want).abs().max()}"


def test_token_logps_chunking_is_numerically_identical(um):
    """Chunked forward (memory fix for 7B full-vocab logits) must match the
    unchunked result exactly — same kernels, just fewer rows per call."""
    input_ids, attention_mask, labels = make_batch()
    with torch.no_grad():
        full_logps, full_counts = um.token_logps(
            input_ids, attention_mask, labels, logps_chunk_size=10_000
        )
        chunked_logps, chunked_counts = um.token_logps(
            input_ids, attention_mask, labels, logps_chunk_size=1
        )
    assert torch.allclose(chunked_logps, full_logps, atol=1e-6), (
        f"max diff {(chunked_logps - full_logps).abs().max()}"
    )
    assert torch.equal(chunked_counts, full_counts)


def test_logps_all_masked_is_zero(um):
    input_ids, attention_mask, labels = make_batch()
    labels[:] = -100
    with torch.no_grad():
        got = um.logps(input_ids, attention_mask, labels)
    assert torch.allclose(got, torch.zeros_like(got))


def test_logps_consistent_with_hf_builtin_loss(um):
    """Cross-check against HF's own shifted CE loss: for a batch where every
    sequence has the same number of completion tokens,
        -logps.sum() / n_tokens  ==  model(labels=...).loss
    HF's loss is the reference implementation of causal-LM shifting; agreeing
    with it validates both our shift and our masking."""
    input_ids, attention_mask, labels = make_batch(n_completion=5)  # uniform count
    with torch.no_grad():
        logps = um.logps(input_ids, attention_mask, labels)
        hf_loss = um.model(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels).loss
    n_tokens = (labels[:, 1:] != -100).sum()
    our_loss = -logps.sum() / n_tokens
    assert torch.allclose(our_loss, hf_loss, atol=1e-4), \
        f"ours={our_loss.item():.6f} hf={hf_loss.item():.6f}"


def test_prompt_label_positions_are_masked(um):
    """If we mark a completion token as prompt (label = -100), its contribution
    is removed from the sum. We verify by flipping the first completion label
    to -100 and checking the difference equals that token's logp."""
    input_ids, attention_mask, labels = make_batch(n_completion=5)
    with torch.no_grad():
        base = um.logps(input_ids, attention_mask, labels)
    first_completion_label = labels.clone()
    # find first non-masked label after position 0; set it to -100
    pos = (first_completion_label[0, 1:] != -100).nonzero(as_tuple=True)[0][0].item() + 1
    first_completion_label[:, pos] = -100
    with torch.no_grad():
        masked = um.logps(input_ids, attention_mask, first_completion_label)

    logits = um.model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    batch_idx = torch.arange(input_ids.shape[0])
    token_logp = logp[batch_idx, pos - 1, labels[:, pos]]
    assert torch.allclose(base - masked, token_logp, atol=1e-5)


def test_average_per_token(um):
    input_ids, attention_mask, labels = make_batch()
    with torch.no_grad():
        summed = um.logps(input_ids, attention_mask, labels)
        avg = um.logps(input_ids, attention_mask, labels, average_per_token=True)
    n = (labels[:, 1:] != -100).sum(dim=-1)
    assert torch.allclose(avg, summed / n, atol=1e-5)


def test_logps_has_gradient(um):
    input_ids, attention_mask, labels = make_batch()
    um.model.train()
    got = um.logps(input_ids, attention_mask, labels)
    got.sum().backward()
    grads = [p.grad for p in um.model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
