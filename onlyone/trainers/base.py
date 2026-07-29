"""Shared trainer infrastructure.

BaseTrainer owns the optimization loop plumbing (optimizer, schedule,
accumulation, checkpointing, tracking). Subclasses implement `compute_loss`
and may override `extra_metrics`.
"""

from __future__ import annotations

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from onlyone.models.unified import UnifiedModel
from onlyone.utils.config import TrainConfig
from onlyone.utils.logging import Tracker
from onlyone.utils.memory import peak_vram_gb

logger = logging.getLogger("onlyone")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaseTrainer(ABC):
    def __init__(self, model: UnifiedModel, train_cfg: TrainConfig,
                 tokenizer=None, tracker: Tracker | None = None):
        self.um = model
        self.model = model.model
        self.cfg = train_cfg
        self.tokenizer = tokenizer
        self.tracker = tracker or Tracker("none")
        if train_cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(train_cfg.device)
        self.model.to(self.device)

        set_seed(train_cfg.seed)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        warmup = int(train_cfg.max_steps * train_cfg.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup, train_cfg.max_steps
        )
        self.global_step = 0

        if train_cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Return (loss, metrics). metrics values must be python floats."""

    def extra_metrics(self) -> dict[str, float]:
        return {}

    def _to_device(self, batch: dict) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def train(self, dataloader: DataLoader) -> None:
        cfg = self.cfg
        accum = cfg.gradient_accumulation_steps
        self.model.train()
        running: dict[str, float] = {}
        micro_step = 0

        while self.global_step < cfg.max_steps:
            for batch in dataloader:
                if self.global_step >= cfg.max_steps:
                    break
                batch = self._to_device(batch)
                loss, metrics = self.compute_loss(batch)
                (loss / accum).backward()

                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + v
                micro_step += 1

                if micro_step % accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        cfg.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self.um.mark_trained(1)

                    if self.global_step % cfg.logging_steps == 0:
                        n = accum * cfg.logging_steps
                        logs = {k: v / n for k, v in running.items()}
                        logs["lr"] = self.scheduler.get_last_lr()[0]
                        logs["vram_peak_gb"] = peak_vram_gb()
                        logs.update(self.extra_metrics())
                        self.tracker.log(logs, self.global_step)
                        running = {}

                    if self.global_step % cfg.save_steps == 0:
                        self.save_checkpoint()

        self.save_checkpoint(final=True)
        self.tracker.finish()
        logger.info("训练完成: %d steps -> %s", self.global_step, cfg.output_dir)

    def save_checkpoint(self, final: bool = False) -> None:
        name = "final" if final else f"step{self.global_step}"
        path = os.path.join(self.cfg.output_dir, name)
        self.um.save(path)
        state = {
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, os.path.join(path, "trainer_state.pt"))
        with open(os.path.join(path, "train_cfg.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info("checkpoint saved: %s", path)
