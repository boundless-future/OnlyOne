"""Rule-based reward functions.

All are deterministic and side-effect free. Each returns a score in a
documented range so they compose predictably under `combine`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from onlyone.eval.benchmarks import answers_match, extract_prediction


def math_answer_reward(prompt: str, response: str, meta: dict[str, Any]) -> float:
    """+1 if the extracted final number equals meta["gold"], else 0.

    Requires prompts jsonl rows to carry meta.gold. Rows without gold score 0
    rather than crashing — a mislabeled dataset shows up as a suspiciously low
    reward mean in the logs, not a stack trace.
    """
    gold = meta.get("gold")
    if gold is None:
        return 0.0
    pred = extract_prediction(response)
    return 1.0 if answers_match(pred, str(gold)) else 0.0


def json_format_reward(prompt: str, response: str, meta: dict[str, Any]) -> float:
    """+1 if the response parses as JSON, else 0. For structured-output tasks."""
    try:
        json.loads(response.strip())
        return 1.0
    except (json.JSONDecodeError, ValueError):
        return 0.0


def regex_format_reward(pattern: str):
    """Factory: +1 if `pattern` (compiled once) matches the response, else 0."""
    compiled = re.compile(pattern, re.DOTALL)

    def reward(prompt: str, response: str, meta: dict[str, Any]) -> float:
        return 1.0 if compiled.search(response) else 0.0

    return reward


def length_penalty_reward(max_chars: int = 512, penalty: float = -0.5):
    """Factory: 0 within budget, `penalty` per 100% overflow beyond max_chars.

    Anti reward-hacking guard (design doc §5): without a length term, RL
    quickly learns that longer completions game verifiable rewards.
    """

    def reward(prompt: str, response: str, meta: dict[str, Any]) -> float:
        overflow = max(0, len(response) - max_chars) / max_chars
        return penalty * overflow

    return reward
