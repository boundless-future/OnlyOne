"""Logging facade: console + optional wandb / tensorboard.

All trainers report through `Tracker` so backends stay swappable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("onlyone")


def setup_console_logging(level: int = logging.INFO) -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)


class Tracker:
    """Thin wrapper over wandb / tensorboard; 'none' mode only logs to console."""

    def __init__(self, log_with: str = "none", run_name: Optional[str] = None,
                 output_dir: Optional[str] = None, config: Optional[dict[str, Any]] = None):
        self.log_with = log_with
        self._writer = None
        if log_with == "wandb":
            import wandb

            wandb.init(project="onlyone", name=run_name, config=config or {})
            self._writer = wandb
        elif log_with == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=f"{output_dir}/tb")

    def log(self, metrics: dict[str, float], step: int) -> None:
        pretty = "  ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        logger.info("step %d  %s", step, pretty)
        if self.log_with == "wandb":
            self._writer.log(metrics, step=step)
        elif self.log_with == "tensorboard":
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, step)

    def finish(self) -> None:
        if self.log_with == "wandb":
            self._writer.finish()
        elif self.log_with == "tensorboard":
            self._writer.close()
