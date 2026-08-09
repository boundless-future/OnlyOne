"""Logging facade: console + metrics.jsonl + optional wandb / tensorboard.

All trainers report through `Tracker` so backends stay swappable.
`metrics.jsonl` is always written when output_dir is given — it is the
durable, zero-dependency measurement record (one JSON object per logged
step), independent of whichever online backend is active. Analyze trends
with `scripts/check_training.py`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("onlyone")


def setup_console_logging(level: int = logging.INFO) -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)


class Tracker:
    """Thin wrapper over wandb / tensorboard; 'none' mode logs to console.

    Regardless of backend, every logged step is also appended to
    `{output_dir}/metrics.jsonl` (when output_dir is set).
    """

    def __init__(self, log_with: str = "none", run_name: Optional[str] = None,
                 output_dir: Optional[str] = None, config: Optional[dict[str, Any]] = None):
        self.log_with = log_with
        self._writer = None
        self._metrics_path = Path(output_dir) / "metrics.jsonl" if output_dir else None
        if self._metrics_path is not None:
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
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
        if self._metrics_path is not None:
            # Open per write: crash-safe append, and the file stays readable
            # by external tools while training is running.
            with self._metrics_path.open("a", encoding="utf-8") as f:
                row = {"step": step, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **metrics}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
