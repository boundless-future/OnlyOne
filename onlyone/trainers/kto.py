"""KTO: Kahneman-Tversky Optimization (Ethayarajh et al., 2024).

Binary (good/bad) labels instead of preference pairs — the lowest data
barrier of all alignment algorithms. Unlike ORPO it DOES need a reference
policy, which is exactly what UnifiedModel's dual-adapter slot provides at
zero extra VRAM.

    r(x,y) = log π_θ(y|x) - log π_ref(y|x)          (summed over tokens)
    z_ref  = mean over batch of r                    (KL estimate, detached)
    good y:  L = 1 - σ(β·(r - z_ref))     weight λ_D
    bad  y:  L = 1 - σ(β·(z_ref - r))     weight λ_U

z_ref note: TRL estimates KL from deliberately mismatched prompt/completion
pairs; we use the batch mean of r (the common simplification) — same role,
one less forward pass per step.
"""

from __future__ import annotations

import torch

from onlyone.trainers.base import BaseTrainer


class KTOTrainer(BaseTrainer):
    def __init__(self, model, train_cfg, beta: float = 0.1,
                 desirable_weight: float = 1.0, undesirable_weight: float = 1.0,
                 **kwargs):
        super().__init__(model, train_cfg, **kwargs)
        self.beta = beta
        self.w_des = desirable_weight
        self.w_und = undesirable_weight

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        policy_logps = self.um.logps(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        with self.um.as_reference():
            ref_logps = self.um.logps(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

        rewards = policy_logps - ref_logps
        z_ref = rewards.mean().detach()  # KL estimate; no gradient through the baseline

        is_good = batch["is_good"] > 0.5
        good_r, bad_r = rewards[is_good], rewards[~is_good]

        losses, metrics_parts = [], {}
        if good_r.numel() > 0:
            good_loss = (1 - torch.sigmoid(self.beta * (good_r - z_ref))).mean()
            losses.append(self.w_des * good_loss)
            metrics_parts["kto_good_loss"] = good_loss.item()
        if bad_r.numel() > 0:
            bad_loss = (1 - torch.sigmoid(self.beta * (z_ref - bad_r))).mean()
            losses.append(self.w_und * bad_loss)
            metrics_parts["kto_bad_loss"] = bad_loss.item()
        if not losses:
            raise ValueError("KTO batch 中没有有效样本")

        loss = torch.stack(losses).sum()
        with torch.no_grad():
            metrics = {
                "loss": loss.item(),
                "kl_estimate": z_ref.item(),
                "reward_good_mean": good_r.mean().item() if good_r.numel() else 0.0,
                "reward_bad_mean": bad_r.mean().item() if bad_r.numel() else 0.0,
                # Fraction of good samples whose reward already exceeds baseline.
                "kto_acc": (good_r > z_ref).float().mean().item() if good_r.numel() else 0.0,
                **metrics_parts,
            }
        return loss, metrics


def build_kto_trainer(cfg, tracker=None):
    from torch.utils.data import DataLoader

    from onlyone.data.datasets import BinaryCollator, BinaryDataset
    from onlyone.models.loading import load_tokenizer
    from onlyone.models.unified import UnifiedModel

    tokenizer = load_tokenizer(cfg.model)
    um = UnifiedModel(cfg.model)
    um.snapshot_ref()  # freeze the SFT policy as reference BEFORE any update

    dataset = BinaryDataset(cfg.data.train_path, tokenizer, cfg.data.template, cfg.data.max_len)
    collator = BinaryCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    trainer = KTOTrainer(
        um, cfg.train,
        beta=cfg.algo.kto_beta,
        desirable_weight=cfg.algo.kto_desirable_weight,
        undesirable_weight=cfg.algo.kto_undesirable_weight,
        tokenizer=tokenizer, tracker=tracker,
    )
    return trainer, dataloader
