"""SFT trainer: completion-only cross-entropy.

We deliberately compute the loss from `UnifiedModel.logps` rather than the
model's built-in loss so that SFT shares exactly one logps code path with
DPO/ORPO/GRPO — one place to test, one place to be wrong.
"""

from __future__ import annotations

import torch

from onlyone.trainers.base import BaseTrainer


class SFTTrainer(BaseTrainer):
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        mask = batch["labels"][:, 1:] != -100
        n_tokens = mask.sum().clamp(min=1)

        # token-mean CE: -sum(logp) / n_completion_tokens
        seq_logps = self.um.logps(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = -seq_logps.sum() / n_tokens
        return loss, {"loss": loss.item(), "n_completion_tokens": float(n_tokens.item())}


def build_sft_trainer(cfg, tracker=None) -> tuple[SFTTrainer, torch.utils.data.DataLoader]:
    """Wire config -> model + dataset + dataloader + trainer."""
    from torch.utils.data import DataLoader

    from onlyone.data.datasets import SFTCollator, SFTDataset
    from onlyone.data.packing import PackedSFTDataset
    from onlyone.models.loading import load_tokenizer
    from onlyone.models.unified import UnifiedModel

    tokenizer = load_tokenizer(cfg.model)
    um = UnifiedModel(cfg.model)

    dataset = SFTDataset(cfg.data.train_path, tokenizer, cfg.data.template, cfg.data.max_len)
    if cfg.data.packing:
        dataset = PackedSFTDataset(dataset, cfg.data.max_len)

    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # dataset encodes lazily; workers complicate tokenizer pickling
    )
    trainer = SFTTrainer(um, cfg.train, tokenizer=tokenizer, tracker=tracker)
    return trainer, dataloader
