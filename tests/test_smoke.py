"""End-to-end smoke: SFTTrainer actually reduces loss on a tiny model."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from onlyone.data.datasets import SFTCollator
from onlyone.trainers.base import BaseTrainer
from onlyone.trainers.sft import SFTTrainer
from onlyone.utils.config import TrainConfig
from tests.conftest import make_batch


class ToyDataset(torch.utils.data.Dataset):
    """Fixed random batch repeated — enough to check the optimizer moves loss."""

    def __init__(self, n: int = 16):
        self.items = [make_batch(batch_size=1, seq=12, n_completion=6) for _ in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, mask, labels = self.items[i]
        return {
            "input_ids": ids[0].tolist(),
            "attention_mask": mask[0].tolist(),
            "labels": labels[0].tolist(),
        }


def _collate(features):
    import torch as t
    return {k: t.tensor([f[k] for f in features]) for k in features[0]}


def test_sft_loss_decreases(um, tmp_path):
    cfg = TrainConfig(
        output_dir=str(tmp_path),
        per_device_batch_size=4,
        gradient_accumulation_steps=1,
        lr=1e-2,
        max_steps=20,
        warmup_ratio=0.0,
        gradient_checkpointing=False,
        logging_steps=5,
        save_steps=10_000,  # don't pollute the test with checkpoints
        device="cpu",       # CI / dev machines may have an unusable GPU
    )
    trainer: BaseTrainer = SFTTrainer(um, cfg)
    loader = DataLoader(ToyDataset(), batch_size=4, collate_fn=_collate, shuffle=False)

    losses: list[float] = []
    original_log = trainer.tracker.log
    trainer.tracker.log = lambda m, s: (losses.append(m["loss"]), original_log(m, s))

    trainer.train(loader)

    assert len(losses) >= 2
    assert losses[-1] < losses[0], f"loss 未下降: {losses}"
    assert um._trained_steps == 20
