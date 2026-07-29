"""Reward registry: config names -> reward functions.

Configs list reward names (e.g. raft.rewards: ["math_answer", "length"]);
this module resolves and combines them. Factory-style rewards that need
arguments (regex, length) are registered with sensible defaults here —
parameterized variants can be added without changing callers.
"""

from __future__ import annotations

from onlyone.rewards.base import RewardFn, combine
from onlyone.rewards.rules import (
    json_format_reward, length_penalty_reward, math_answer_reward,
    regex_format_reward,
)

_REGISTRY: dict[str, RewardFn] = {
    "math_answer": math_answer_reward,
    "json_format": json_format_reward,
    "length": length_penalty_reward(),
    "think_tag": regex_format_reward(r"<think>.*</think>"),
}


def build_reward(names: list[str]) -> RewardFn:
    """Combine registered rewards by name into a single summed reward."""
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise KeyError(f"未知奖励: {unknown},可选: {sorted(_REGISTRY)}")
    return combine([_REGISTRY[n] for n in names])


def register(name: str, fn: RewardFn) -> None:
    """Register a custom reward (user extension point)."""
    if name in _REGISTRY:
        raise KeyError(f"奖励 '{name}' 已注册")
    _REGISTRY[name] = fn
