"""Tests for the stage-0 math data preparation helpers."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from scripts.prepare_math_data import (
    build_mix,
    build_probe,
    convert_gsm8k,
    convert_math,
    extract_math_gold,
    parse_level,
    write_jsonl,
)


def _prompt(source: str, level: int, index: int) -> dict:
    return {
        "prompt": f"{source}-{level}-{index}",
        "meta": {"gold": str(index), "level": level, "source": source},
    }


def test_convert_gsm8k_and_math():
    gsm8k = convert_gsm8k([{
        "question": "What is 20 + 22?",
        "answer": "reasoning\n#### 42",
    }])
    assert gsm8k == [{
        "prompt": "What is 20 + 22?",
        "meta": {"gold": "42", "level": 0, "source": "gsm8k"},
    }]

    math_rows = convert_math([{
        "problem": "Simplify.",
        "solution": r"Therefore \boxed{\frac{1}{2}}.",
        "level": "Level 3",
        "type": "Algebra",
    }])
    assert math_rows[0]["meta"] == {
        "gold": r"\frac{1}{2}",
        "level": 3,
        "subject": "Algebra",
        "source": "math",
    }


def test_extract_math_gold_prefers_answer_and_validates():
    assert parse_level("Level ?") is None
    assert extract_math_gold({"answer": " 7 ", "solution": r"\boxed{8}"}) == "7"
    with pytest.raises(ValueError, match="no boxed answer"):
        extract_math_gold({"problem": "bad", "solution": "no final answer"})
    with pytest.raises(ValueError, match="invalid MATH level"):
        parse_level("hard")


def test_build_mix_has_expected_ratio_and_is_deterministic():
    gsm8k = [_prompt("gsm8k", 0, index) for index in range(8)]
    math_rows = [
        *[_prompt("math", 1, index) for index in range(8)],
        *[_prompt("math", 3, index) for index in range(16)],
    ]
    first = build_mix(gsm8k, math_rows, size=16, seed=7)
    second = build_mix(gsm8k, math_rows, size=16, seed=7)
    assert first == second

    buckets = Counter(
        "gsm8k" if row["meta"]["source"] == "gsm8k"
        else "low" if row["meta"]["level"] <= 2
        else "high"
        for row in first
    )
    assert buckets == {"gsm8k": 4, "low": 4, "high": 8}


def test_build_probe_and_write_jsonl(tmp_path):
    gsm8k = [_prompt("gsm8k", 0, index) for index in range(3)]
    math_rows = [
        *[_prompt("math", 1, index) for index in range(3)],
        *[_prompt("math", 2, index) for index in range(3)],
    ]
    probe = build_probe(gsm8k, math_rows, per_level=2, seed=3)
    assert Counter(row["meta"]["level"] for row in probe) == {0: 2, 1: 2, 2: 2}

    path = tmp_path / "nested" / "rows.jsonl"
    assert write_jsonl(path, probe) == 6
    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert loaded == probe
