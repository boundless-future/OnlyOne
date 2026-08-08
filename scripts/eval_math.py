"""阶段 4 评估:base vs LoRA checkpoint,固定题集 + 纯 math_verify 正确率。

训练内 reward 混有 length 成分,chars 收缩也能挣 reward —— 本脚本是无混淆
的裁判:同一批题、贪婪解码 pass@1、只看对错。base 与 adapter 在同一个
vLLM 实例里先后生成,解码条件严格一致。

用法(服务器,GPU):

  # 对比评估(base + adapter 各生成一遍)
  python scripts/eval_math.py \
    --model /cloud/models/Qwen2.5-7B-Instruct \
    --adapter /root/work/TestResults/grpo_qwen7b_lr1.6e4/step60 \
    --data data/math/math500_test.jsonl \
    --out /root/work/TestResults/grpo_qwen7b_lr1.6e4/eval_step60_math500.json

  # 只跑基线: 去掉 --adapter
  # 评 GSM8K held-out: --data data/math/gsm8k_test.jsonl
  # 快速自检: --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 与训练 rollout 同一条验证过的引擎路径(V0);必须早于任何 vllm import。
os.environ.setdefault("VLLM_USE_V1", "0")


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_summary(records: list[dict], variants: list[str]) -> dict:
    """聚合 overall + 按 level 分桶的正确率/长度。records 每行含
    level 和 {variant}_correct / {variant}_chars 字段。纯函数,不依赖 GPU。"""
    def agg(rs: list[dict]) -> dict:
        out = {"n": len(rs)}
        for v in variants:
            out[f"{v}_acc"] = round(sum(r[f"{v}_correct"] for r in rs) / len(rs), 4)
            out[f"{v}_chars"] = round(sum(r[f"{v}_chars"] for r in rs) / len(rs), 1)
        return out

    levels = sorted({r["level"] for r in records})
    return {
        "overall": agg(records),
        "per_level": {str(lv): agg([r for r in records if r["level"] == lv]) for lv in levels},
    }


def print_report(summary: dict, variants: list[str]) -> None:
    header = f"{'level':<7}{'n':>5}" + "".join(f"{v+'_acc':>12}{v+'_chars':>12}" for v in variants)
    print(header)
    print("-" * len(header))
    for label, agg in [("overall", summary["overall"]),
                       *((f"L{lv}", a) for lv, a in summary["per_level"].items())]:
        row = f"{label:<7}{agg['n']:>5}"
        for v in variants:
            row += f"{agg[f'{v}_acc']:>12.4f}{agg[f'{v}_chars']:>12.1f}"
        print(row)
    if len(variants) == 2:
        a, b = variants
        delta = summary["overall"][f"{b}_acc"] - summary["overall"][f"{a}_acc"]
        print(f"\nΔ accuracy ({b} − {a}): {delta:+.4f}  ({delta * 100:+.1f} pt)")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="基座模型路径")
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint 目录(含 adapter_config.json);不传则只评 base")
    ap.add_argument("--data", required=True, help="评测集 jsonl(prompt + meta.gold/level)")
    ap.add_argument("--out", default=None, help="结果 json;默认存到 adapter 上级目录")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="与训练生成上限一致")
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 条(自检用)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit(f"数据为空: {args.data}")

    from onlyone.data.templates import get_template
    from onlyone.rewards.registry import build_reward

    template = get_template("chatml")
    rendered = [template.render_prompt(r["prompt"]) for r in rows]
    scoring = build_reward(["math_verify"])  # 纯正确率,不含 length

    from vllm import LLM, SamplingParams

    lora_req = None
    if args.adapter:
        cfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text(encoding="utf-8"))
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("eval", 1, args.adapter)
        llm = LLM(model=args.model, enable_lora=True,
                  max_lora_rank=cfg.get("r", 16), max_loras=1,
                  gpu_memory_utilization=args.gpu_mem_util,
                  max_model_len=2048, enforce_eager=True)
    else:
        llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem_util,
                  max_model_len=2048, enforce_eager=True)

    params = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens)

    variants = ["base"] + (["adapter"] if lora_req else [])
    outputs: dict[str, list[str]] = {}
    for v in variants:
        outs = llm.generate(rendered, params,
                            lora_request=lora_req if v == "adapter" else None)
        outputs[v] = [o.outputs[0].text for o in outs]
        print(f"{v} 生成完成 ({len(rows)} 条)")

    records = []
    for i, row in enumerate(rows):
        rec = {"idx": i, "level": row.get("meta", {}).get("level", 0)}
        for v in variants:
            text = outputs[v][i]
            rec[f"{v}_correct"] = float(scoring(row["prompt"], text, row.get("meta", {})) > 0)
            rec[f"{v}_chars"] = len(text)
            rec[f"{v}_completion"] = text
        records.append(rec)

    summary = build_summary(records, variants)
    print_report(summary, variants)

    out = args.out
    if out is None:
        tag = Path(args.adapter).name if args.adapter else "base_only"
        parent = Path(args.adapter).parent if args.adapter else Path(".")
        out = str(parent / f"eval_{Path(args.data).stem}_{tag}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary, "records": records},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
