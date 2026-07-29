"""onlyone CLI: train / (rollout, eval, flywheel arrive in later milestones)."""

from __future__ import annotations

import typer

from onlyone.utils.logging import Tracker, setup_console_logging

app = typer.Typer(help="OnlyOne — 单卡大模型对齐全流程框架")


@app.command()
def train(config: str = typer.Option(..., "--config", "-c", help="YAML 配置路径")):
    """Run a training job. The config's top-level shape selects the trainer."""
    from onlyone.utils.config import load_config

    setup_console_logging()
    cfg = load_config(config)  # M1: SFTConfig; M2+ will dispatch on config type
    tracker = Tracker(
        log_with=cfg.train.log_with,
        run_name=cfg.train.run_name,
        output_dir=cfg.train.output_dir,
        config=cfg.model_dump(),
    )

    from onlyone.trainers.sft import build_sft_trainer

    trainer, dataloader = build_sft_trainer(cfg, tracker=tracker)
    trainer.train(dataloader)


if __name__ == "__main__":
    app()
