"""Stage-1 difficulty probe analysis: per-level keep_rate from scores.jsonl.

Usage:
    python scripts/analyze_probe.py <scores.jsonl> [--group-size 4]

Reads the per-prompt records written by the RAFT flywheel
(raft.output_dir/round_<i>/scores.jsonl) and buckets them by meta.level,
printing pass@1 (mean score) and keep_rate (pass@G: any candidate kept)
per difficulty bucket, plus a verdict per the stage-1 plan table:

    keep_rate 20%~70%  -> sweet spot, include in GRPO prompt set
    keep_rate >80%     -> too easy, downweight
    keep_rate <5%      -> too hard, drop
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def verdict(keep_rate: float) -> str:
    if keep_rate < 0.05:
        return "太难 → 剔除"
    if keep_rate > 0.80:
        return "太简单 → 降占比"
    if 0.20 <= keep_rate <= 0.70:
        return "✅ 甜区"
    return "边缘区间,人工判断"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scores_path", help="raft 输出的 scores.jsonl 路径")
    ap.add_argument("--group-size", type=int, default=None,
                    help="每组候选数(仅用于表头显示,默认从数据推断)")
    args = ap.parse_args()

    buckets: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "kept": 0, "score_sum": 0.0, "n_scores": 0}
    )
    group_size = args.group_size
    n_lines = 0
    with open(args.scores_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n_lines += 1
            meta = row.get("meta") or {}
            level = meta.get("level")
            key = "gsm8k" if level == 0 else (f"L{level}" if level is not None else "unknown")
            scores = row.get("scores") or []
            if group_size is None and scores:
                group_size = len(scores)
            b = buckets[key]
            b["n"] += 1
            b["kept"] += 1 if row.get("kept") else 0
            b["score_sum"] += sum(scores)
            b["n_scores"] += len(scores)

    if n_lines == 0:
        print(f"空文件或无有效行: {args.scores_path}", file=sys.stderr)
        return 1

    def sort_key(k: str) -> tuple[int, str]:
        if k == "gsm8k":
            return (0, k)
        if k.startswith("L") and k[1:].isdigit():
            return (int(k[1:]), k)
        return (99, k)

    g = group_size or "?"
    print(f"文件: {args.scores_path}  共 {n_lines} 条 prompt, 每组 G={g}")
    print(f"{'难度桶':<10}{'条数':>6}{'pass@1':>10}{f'pass@{g}(keep)':>16}  判定")
    print("-" * 60)
    total_n = total_kept = 0
    total_score = 0.0
    total_scores = 0
    for key in sorted(buckets, key=sort_key):
        b = buckets[key]
        keep_rate = b["kept"] / b["n"]
        pass1 = b["score_sum"] / max(b["n_scores"], 1)
        total_n += b["n"]
        total_kept += b["kept"]
        total_score += b["score_sum"]
        total_scores += b["n_scores"]
        print(f"{key:<10}{b['n']:>6}{pass1:>10.1%}{keep_rate:>16.1%}  {verdict(keep_rate)}")
    print("-" * 60)
    overall_keep = total_kept / max(total_n, 1)
    overall_pass1 = total_score / max(total_scores, 1)
    print(f"{'总计':<10}{total_n:>6}{overall_pass1:>10.1%}{overall_keep:>16.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
