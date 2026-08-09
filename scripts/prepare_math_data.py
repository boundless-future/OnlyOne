"""Prepare GSM8K, MATH, and MATH-500 JSONL files for math GRPO."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from onlyone.eval.benchmarks import extract_boxed, extract_gold

MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
EXPECTED_COUNTS = {
    "gsm8k_train": 7473,
    "gsm8k_test": 1319,
    "math_train": 7500,
    "math500_test": 500,
}
_LEVEL_RE = re.compile(r"(\d+)")


def parse_level(value: Any) -> int | None:
    """Normalize MATH levels such as Level 3 to an integer."""
    if str(value).strip() in {"?", "Level ?"}:
        return None
    match = _LEVEL_RE.search(str(value))
    if not match:
        raise ValueError(f"invalid MATH level: {value!r}")
    return int(match.group(1))


def extract_math_gold(row: Mapping[str, Any]) -> str:
    """Read an explicit answer or extract the final boxed value from a solution."""
    answer = row.get("answer")
    if answer is not None and str(answer).strip():
        return str(answer).strip()
    gold = extract_boxed(str(row.get("solution", "")))
    if gold is None:
        raise ValueError(f"MATH row has no boxed answer: {row.get('problem', '')[:80]!r}")
    return gold


def convert_gsm8k(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        gold = extract_gold(str(row["answer"]))
        if gold is None:
            raise ValueError(f"GSM8K row has no #### answer: {row['answer']!r}")
        converted.append({
            "prompt": str(row["question"]),
            "meta": {"gold": gold, "level": 0, "source": "gsm8k"},
        })
    return converted


def convert_math(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str = "math",
) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        subject = row.get("subject", row.get("type", "unknown"))
        converted.append({
            "prompt": str(row["problem"]),
            "meta": {
                "gold": extract_math_gold(row),
                "level": parse_level(row["level"]),
                "subject": str(subject),
                "source": source,
            },
        })
    return converted


def _sample(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"requested {count} rows from a bucket containing {len(rows)}")
    return rng.sample(rows, count)


def build_mix(
    gsm8k: list[dict[str, Any]],
    math_rows: list[dict[str, Any]],
    *,
    size: int = 4000,
    seed: int = 42,
    level_weights: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Sample a mixed prompt set with optional per-level weights.

    Args:
        level_weights: mapping from level (0=gsm8k, 1~5=MATH) to relative
            weight. If None, fall back to the legacy 1:1:2 split
            (GSM8K : MATH-L1~2 : MATH-L3~5).
    """
    if size <= 0:
        raise ValueError("mix size must be positive")

    # Build buckets per integer level; gsm8k is level 0.
    buckets: dict[int, list[dict[str, Any]]] = {0: gsm8k}
    for row in math_rows:
        level = row["meta"].get("level")
        if isinstance(level, int):
            buckets.setdefault(level, []).append(row)

    rng = random.Random(seed)

    if level_weights is None:
        # Legacy 1:1:2 split preserved for backwards compatibility.
        if size % 4:
            raise ValueError("legacy mix size must be a multiple of 4")
        unit = size // 4
        low = []
        high = []
        for level, rows in buckets.items():
            if level == 0:
                continue
            (low if level <= 2 else high).extend(rows)
        mixed = _sample(gsm8k, unit, rng) + _sample(low, unit, rng) + _sample(high, unit * 2, rng)
        rng.shuffle(mixed)
        return mixed

    # Normalize weights and compute counts, rounding to integers that sum to size.
    levels = sorted(level_weights.keys())
    total_weight = sum(level_weights[l] for l in levels)
    if total_weight <= 0:
        raise ValueError("level weights must sum to a positive value")

    raw_counts = {l: size * level_weights[l] / total_weight for l in levels}
    counts = {l: int(raw_counts[l]) for l in levels}
    remainder = size - sum(counts.values())
    # Distribute the rounding remainder to the largest fractional parts.
    if remainder:
        for l in sorted(raw_counts, key=lambda x: raw_counts[x] - counts[x], reverse=True)[:remainder]:
            counts[l] += 1

    mixed: list[dict[str, Any]] = []
    for level in levels:
        count = counts[level]
        if count == 0:
            continue
        if level not in buckets:
            raise ValueError(f"level {level} has no available prompts")
        mixed.extend(_sample(buckets[level], count, rng))
    rng.shuffle(mixed)
    return mixed


def build_probe(
    gsm8k: list[dict[str, Any]],
    math_rows: list[dict[str, Any]],
    *,
    per_level: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample the same number of prompts from GSM8K and each MATH level."""
    if per_level <= 0:
        raise ValueError("probe per-level count must be positive")
    rng = random.Random(seed)
    buckets: dict[int, list[dict[str, Any]]] = {0: gsm8k}
    for row in math_rows:
        if row["meta"]["level"] is not None:
            buckets.setdefault(row["meta"]["level"], []).append(row)

    probe = []
    for level in sorted(buckets):
        probe.extend(_sample(buckets[level], per_level, rng))
    rng.shuffle(probe)
    return probe


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_datasets() -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install project dependencies before preparing data") from exc

    gsm8k_train = list(load_dataset("openai/gsm8k", "main", split="train"))
    gsm8k_test = list(load_dataset("openai/gsm8k", "main", split="test"))

    math_train = []
    for subject in MATH_SUBJECTS:
        rows = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        for row in rows:
            normalized = dict(row)
            normalized.setdefault("subject", subject)
            math_train.append(normalized)

    math500_test = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    return gsm8k_train, gsm8k_test, math_train, math500_test


def prepare(
    output_dir: Path,
    *,
    mix_size: int = 4000,
    mix_level_weights: dict[int, float] | None = None,
    probe_per_level: int = 100,
    seed: int = 42,
    check_counts: bool = True,
) -> dict[str, int]:
    gsm8k_raw, gsm8k_test_raw, math_raw, math500_raw = _load_datasets()
    raw_counts = {
        "gsm8k_train": len(gsm8k_raw),
        "gsm8k_test": len(gsm8k_test_raw),
        "math_train": len(math_raw),
        "math500_test": len(math500_raw),
    }
    if check_counts and raw_counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected dataset counts: {raw_counts}; expected {EXPECTED_COUNTS}")

    gsm8k = convert_gsm8k(gsm8k_raw)
    gsm8k_test = convert_gsm8k(gsm8k_test_raw)
    math_rows = convert_math(math_raw)
    math500 = convert_math(math500_raw, source="math500")
    outputs = {
        "gsm8k_train": write_jsonl(output_dir / "gsm8k_train.jsonl", gsm8k),
        "gsm8k_test": write_jsonl(output_dir / "gsm8k_test.jsonl", gsm8k_test),
        "math_train": write_jsonl(output_dir / "math_train.jsonl", math_rows),
        "math500_test": write_jsonl(output_dir / "math500_test.jsonl", math500),
        "prompts_mix": write_jsonl(
            output_dir / "prompts_mix.jsonl",
            build_mix(gsm8k, math_rows, size=mix_size, seed=seed,
                      level_weights=mix_level_weights),
        ),
        "prompts_mix_probe": write_jsonl(
            output_dir / "prompts_mix_probe.jsonl",
            build_probe(gsm8k, math_rows, per_level=probe_per_level, seed=seed),
        ),
    }
    return outputs


def _parse_level_weights(raw: str | None) -> dict[int, float] | None:
    """Parse a JSON object like '{"0":5,"1":5,...}' into level weights."""
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"level weights must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("level weights must be a JSON object")
    weights: dict[int, float] = {}
    for key, value in parsed.items():
        try:
            level = int(key)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"level key must be int, got {key!r}") from exc
        if not isinstance(value, (int, float)):
            raise argparse.ArgumentTypeError(f"weight for level {level} must be numeric")
        if value < 0:
            raise argparse.ArgumentTypeError(f"weight for level {level} must be non-negative")
        weights[level] = float(value)
    if not weights:
        raise argparse.ArgumentTypeError("level weights must not be empty")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/math"))
    parser.add_argument("--mix-size", type=int, default=4000)
    parser.add_argument("--mix-level-weights", type=_parse_level_weights, default=None,
                        help='per-level mix weights as JSON, e.g. \'{"0":5,"1":5,"2":10,"3":25,"4":27.5,"5":27.5}\'')
    parser.add_argument("--probe-per-level", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-count-check", action="store_true")
    args = parser.parse_args()
    counts = prepare(
        args.output_dir,
        mix_size=args.mix_size,
        mix_level_weights=args.mix_level_weights,
        probe_per_level=args.probe_per_level,
        seed=args.seed,
        check_counts=not args.skip_count_check,
    )
    for name, count in counts.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
