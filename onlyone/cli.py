"""onlyone CLI: train / eval. (rollout, flywheel arrive in later milestones)."""

from __future__ import annotations

import json
from typing import Optional

import typer

from onlyone.utils.logging import Tracker, setup_console_logging

app = typer.Typer(help="OnlyOne — 单卡大模型对齐全流程框架")


@app.command()
def train(config: str = typer.Option(..., "--config", "-c", help="YAML 配置路径")):
    """Run a training job; cfg.algo.name selects the trainer."""
    from onlyone.utils.config import load_config

    setup_console_logging()
    cfg = load_config(config)
    tracker = Tracker(
        log_with=cfg.train.log_with,
        run_name=cfg.train.run_name,
        output_dir=cfg.train.output_dir,
        config=cfg.model_dump(),
    )

    if cfg.algo.name == "sft":
        from onlyone.trainers.sft import build_sft_trainer as build
    elif cfg.algo.name == "orpo":
        from onlyone.trainers.orpo import build_orpo_trainer as build
    elif cfg.algo.name == "kto":
        from onlyone.trainers.kto import build_kto_trainer as build
    elif cfg.algo.name in ("dpo", "simpo"):
        from onlyone.trainers.dpo import build_dpo_trainer as build
    else:
        raise ValueError(f"未知算法: {cfg.algo.name}")

    trainer, dataloader = build(cfg, tracker=tracker)
    trainer.train(dataloader)


@app.command()
def flywheel(config: str = typer.Option(..., "--config", "-c", help="YAML 配置路径(需含 raft 段)")):
    """Run the RAFT data flywheel: rollout -> filter -> SFT retrain, N rounds."""
    from onlyone.flywheel.raft import run_flywheel
    from onlyone.utils.config import load_config

    setup_console_logging()
    cfg = load_config(config)
    if cfg.raft is None:
        raise ValueError("配置缺少 raft 段,无法运行 flywheel")
    tracker = Tracker(
        log_with=cfg.train.log_with,
        run_name=cfg.train.run_name,
        output_dir=cfg.raft.output_dir,
        config=cfg.model_dump(),
    )
    results = run_flywheel(cfg, tracker=tracker)
    for r in results:
        typer.echo(
            f"round {r.round_idx}: sft={len(r.sft_rows)} pref={len(r.pref_rows)} "
            f"reward_mean={r.reward_mean:.3f} keep_rate={r.keep_rate:.1%}"
        )


@app.command(name="eval")
def evaluate(
    config: str = typer.Option(..., "--config", "-c", help="YAML 配置路径（复用 model/data 段）"),
    benchmark: str = typer.Option("gsm8k", "--benchmark", "-b"),
    eval_path: Optional[str] = typer.Option(None, "--eval-path", help="覆盖配置中的评测集路径"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="待评测的 LoRA checkpoint 目录"),
    limit: Optional[int] = typer.Option(None, "--limit", help="只评测前 N 条"),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens"),
    device: str = typer.Option("auto", "--device"),
):
    """Score a checkpoint on a held-out benchmark (GSM8K for now)."""
    from onlyone.utils.config import load_config

    setup_console_logging()
    if benchmark != "gsm8k":
        raise ValueError(f"未知 benchmark: {benchmark}（v1 仅支持 gsm8k）")

    cfg = load_config(config)
    path = eval_path or cfg.data.eval_path
    if not path:
        raise ValueError("未指定评测集：请用 --eval-path 或配置 data.eval_path")

    import torch

    from onlyone.eval.benchmarks import eval_gsm8k
    from onlyone.models.loading import load_tokenizer
    from onlyone.models.unified import UnifiedModel

    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else (
        device if device != "auto" else "cpu"
    )
    tokenizer = load_tokenizer(cfg.model)
    um = UnifiedModel(cfg.model)
    if adapter:
        from peft import PeftModel

        um.model = PeftModel.from_pretrained(um.model.base_model.model, adapter)
        um.model = um.model.merge_and_unload()  # eval-only: merge for speed
    um.model.to(dev)

    report = eval_gsm8k(
        um, tokenizer, path,
        template=cfg.data.template,
        max_new_tokens=max_new_tokens,
        limit=limit,
        device=dev,
    )
    typer.echo(json.dumps({k: v for k, v in report.items() if k != "samples"},
                          indent=2, ensure_ascii=False))
    for s in report["samples"]:
        typer.echo(f"  gold={s['gold']} pred={s['pred']} | {s['generation'][:120]!r}")


if __name__ == "__main__":
    app()
