"""Rollout engine abstraction.

A RolloutEngine turns (prompts, n_per_prompt) into generated completions.
RAFT and GRPO both consume this interface; the HF implementation is the
zero-dependency fallback, vLLM arrives in M4 as a pure performance swap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RolloutEngine(ABC):
    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n_per_prompt: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[str]]:
        """Return completions[prompt_idx][candidate_idx] (decoded text,
        prompt excluded). Lengths: len(prompts) x n_per_prompt."""
