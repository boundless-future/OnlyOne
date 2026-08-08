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
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from onlyone.models.unified import DEFAULT_ADAPTER, UnifiedModel
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
            **self.extra_state(),
        }
        torch.save(state, os.path.join(path, "trainer_state.pt"))
        with open(os.path.join(path, "train_cfg.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info("checkpoint saved: %s", path)
        if not final:
            self._rotate_checkpoints()

    def _rotate_checkpoints(self) -> None:
        """Keep only the newest `keep_last_n_checkpoints` step dirs (final is
        never rotated). Disabled when the config value is unset/0."""
        keep = self.cfg.keep_last_n_checkpoints
        if not keep:
            return
        out = Path(self.cfg.output_dir)
        step_dirs = sorted(
            (d for d in out.iterdir()
             if d.is_dir() and d.name.startswith("step") and d.name[4:].isdigit()),
            key=lambda d: int(d.name[4:]),
        )
        for d in step_dirs[: max(len(step_dirs) - keep, 0)]:
            shutil.rmtree(d)
            logger.info("checkpoint rotated out: %s", d)

    def load_checkpoint(self, path: str) -> None:
        """Resume from a save_checkpoint directory (weights + trainer state).

        MUST be called after the builder froze the reference adapter
        (snapshot_ref): the ref anchor stays at the run's starting policy,
        and only the policy adapter is overwritten with trained weights.
        """
        from safetensors.torch import load_file

        adapter_file = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            # peft save strips the adapter name from keys; the official loader
            # maps them back onto adapter_name.
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(
                self.model, load_file(adapter_file), adapter_name=DEFAULT_ADAPTER
            )
        else:
            shards = sorted(Path(path).glob("model*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"checkpoint 里找不到权重文件: {path}")
            sd: dict = {}
            for shard in shards:
                sd.update(load_file(str(shard)))
            # strict=False: tied weights (e.g. Qwen lm_head) are deduplicated
            # on save and re-tied by the model itself.
            self.model.load_state_dict(sd, strict=False)

        state = torch.load(
            os.path.join(path, "trainer_state.pt"),
            map_location=self.device,
            weights_only=False,  # own checkpoint, trusted; holds RNG tuples
        )
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.load_extra_state(state)
        logger.info("resumed from %s (global_step=%d)", path, self.global_step)

    def extra_state(self) -> dict:
        """Subclass-specific state persisted into trainer_state.pt."""
        return {}

    def load_extra_state(self, state: dict) -> None:
        """Restore what extra_state() saved."""
