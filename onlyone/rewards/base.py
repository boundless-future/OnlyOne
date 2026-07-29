"""Reward function protocol.

A reward maps (prompt, response, meta) -> float. Meta carries per-row ground
truth from the prompts jsonl (e.g. {"gold": "42"} for math checking), so
reward functions stay pure and testable without dataset coupling.

Rule rewards are the single-card default (design doc §2.3): zero VRAM,
deterministic, no reward model to overfit to.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol


class RewardFn(Protocol):
    def __call__(self, prompt: str, response: str, meta: dict[str, Any]) -> float: ...


def combine(fns: list[RewardFn], weights: list[float] | None = None) -> RewardFn:
    """Weighted sum of component rewards (default: equal weights)."""
    if weights is None:
        weights = [1.0] * len(fns)
    if len(weights) != len(fns):
        raise ValueError("weights 与 fns 长度不一致")

    def combined(prompt: str, response: str, meta: dict[str, Any]) -> float:
        return sum(w * fn(prompt, response, meta) for fn, w in zip(fns, weights))

    return combined
