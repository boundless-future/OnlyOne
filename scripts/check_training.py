"""Training trend health check from metrics.jsonl.

The Tracker appends one JSON row per logged step to
`{output_dir}/metrics.jsonl`. This script prints the recent rows plus
verdicts against the stage-3 monitoring thresholds (see
docs/math-grpo-runbook.md and the stage-3 execution record §7).

Usage:
    python scripts/check_training.py runs/grpo_7b_math/metrics.jsonl
    python scripts/check_training.py runs/grpo_7b_math/metrics.jsonl --tail 20
    python scripts/check_training.py runs/grpo_7b_math/metrics.jsonl --spark
    python scripts/check_training.py runs/grpo_7b_math/metrics.jsonl --plot trend.png
"""

from __future__ import annotations

import argparse
import json
import sys

COLUMNS = [
    ("step", "{:>5d}"),
    ("loss", "{:>9.4g}"),
    ("kl", "{:>9.4g}"),
    ("clip_frac", "{:>9.4g}"),
    ("reward_mean", "{:>11.4g}"),
    ("completion_chars", "{:>16.4g}"),
    ("vram_peak_gb", "{:>12.4g}"),
]


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_table(rows: list[dict], tail: int) -> None:
    print("  ".join(name for name, _ in COLUMNS))
    for row in rows[-tail:]:
        cells = []
        for name, spec in COLUMNS:
            v = row.get(name)
            cells.append(spec.format(v) if isinstance(v, (int, float)) else "-".rjust(9))
        print("  ".join(cells))


def check(ok: bool, label: str, detail: str, issues: list[str]) -> None:
    print(f"{'✅' if ok else '⚠️ '} {label}: {detail}")
    if not ok:
        issues.append(label)


SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 60) -> str:
    """ASCII-art trend, stdlib only — usable over a plain SSH session."""
    if not values:
        return ""
    if len(values) > width:  # evenly downsample
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo, hi = min(values), max(values)
    if hi == lo:
        return SPARK_CHARS[3] * len(values)
    return "".join(SPARK_CHARS[min(int((v - lo) / (hi - lo) * 7), 7)] for v in values)


def print_sparks(rows: list[dict]) -> None:
    for key in ("reward_mean", "kl", "completion_chars", "vram_peak_gb"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            print(f"{key:<17}{sparkline(vals)}  min={min(vals):.4g} max={max(vals):.4g} last={vals[-1]:.4g}")


def plot(rows: list[dict], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("画图需要 matplotlib: pip install matplotlib(或改用 --spark 看 ASCII 趋势)")

    steps = [r["step"] for r in rows]
    panels = ["loss", "reward_mean", "kl", "clip_frac", "completion_chars", "vram_peak_gb"]
    # English labels: default matplotlib fonts render CJK as boxes.
    thresholds = {
        "kl": [(0.1, "orange", "warn 0.1"), (0.5, "red", "breaker 0.5")],
        "clip_frac": [(0.3, "orange", "lr too big 0.3")],
        "vram_peak_gb": [(30.0, "red", "budget 30G")],
        "completion_chars": [],
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, key in zip(axes.flat, panels):
        pairs = [(s, r[key]) for s, r in zip(steps, rows) if key in r]
        if not pairs:
            ax.set_visible(False)
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, marker=".", lw=1)
        ax.set_title(key)
        ax.grid(alpha=0.3)
        for y, color, label in thresholds.get(key, []):
            ax.axhline(y, color=color, ls="--", lw=0.8, label=label)
            ax.legend(fontsize=7)
        if key == "completion_chars":
            ax.axhline(ys[0] * 2, color="orange", ls="--", lw=0.8, label="2x initial")
            ax.legend(fontsize=7)
    axes[-1][0].set_xlabel("step")
    fig.suptitle(f"training trend: step {steps[0]} → {steps[-1]} ({len(rows)} pts)")
    fig.tight_layout()
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"图已保存: {path}")


def main() -> None:
    # Windows GBK consoles can't print ✅/⚠️ without this.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_jsonl")
    ap.add_argument("--tail", type=int, default=12, help="表尾显示行数")
    ap.add_argument("--spark", action="store_true", help="打印 ASCII 趋势图(零依赖)")
    ap.add_argument("--plot", metavar="PNG", help="保存 matplotlib 趋势图到 PNG")
    args = ap.parse_args()

    rows = load_rows(args.metrics_jsonl)
    if not rows:
        sys.exit("metrics.jsonl 为空 —— 训练还没产出日志行,或 logging_steps 太大")

    print(f"共 {len(rows)} 行,step {rows[0]['step']} → {rows[-1]['step']}\n")
    print_table(rows, args.tail)
    print()

    issues: list[str] = []

    # kl: <0.1 健康缓涨,0.5 熔断
    kls = [r.get("kl", 0.0) for r in rows]
    check(max(kls) < 0.1, "kl", f"max={max(kls):.4f}(健康 <0.1,熔断 0.5)", issues)

    # clip_frac: 看后半段(前半段 policy≈old,天然≈0);0.05~0.2 健康,>0.3 lr 偏大
    tail_rows = rows[len(rows) // 2:] or rows
    clips = [r.get("clip_frac", 0.0) for r in tail_rows]
    check(max(clips) < 0.3, "clip_frac", f"后半段 max={max(clips):.4f}(健康 0.05~0.2)", issues)

    # completion 长度: <2x 初始,防长度黑客
    chars = [r["completion_chars"] for r in rows if "completion_chars" in r]
    if chars:
        ratio = chars[-1] / max(chars[0], 1e-9)
        check(ratio < 2.0, "completion 长度", f"{chars[0]:.0f} → {chars[-1]:.0f}({ratio:.2f}x,阈值 <2x)", issues)

    # 显存峰值: <30G(32G 卡)
    vrams = [r["vram_peak_gb"] for r in rows if "vram_peak_gb" in r]
    if vrams:
        check(max(vrams) < 30.0, "vram_peak", f"max={max(vrams):.2f}G(阈值 <30G)", issues)

    # reward 趋势:后 1/3 均值应高于前 1/3(步数太少时只报告不判定)
    rewards = [r["reward_mean"] for r in rows if "reward_mean" in r]
    if len(rewards) >= 6:
        third = max(len(rewards) // 3, 1)
        first_m = sum(rewards[:third]) / third
        last_m = sum(rewards[-third:]) / third
        check(last_m > first_m, "reward 趋势",
              f"前 1/3 均值 {first_m:.4f} → 后 1/3 {last_m:.4f}(应上升)", issues)
    elif rewards:
        print(f"ℹ️  reward 趋势: 仅 {len(rewards)} 行,不足以判定(当前均值 {sum(rewards)/len(rewards):.4f})")

    # 退化组: 持续过半说明配比失衡
    degens = [r["n_degenerate_groups"] for r in rows if "n_degenerate_groups" in r]
    if degens:
        mean_d = sum(degens) / len(degens)
        check(mean_d < 4.0, "退化组", f"均值 {mean_d:.1f}/8(持续 ≥4 说明配比失衡)", issues)

    print()
    if args.spark:
        print_sparks(rows)
        print()
    if args.plot:
        plot(rows, args.plot)
    if issues:
        print(f"⚠️  {len(issues)} 项需要关注: {', '.join(issues)} —— 对照 runbook §7 预案处理")
        sys.exit(1)
    print("✅ 全部指标健康")


if __name__ == "__main__":
    main()
