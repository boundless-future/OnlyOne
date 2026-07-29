"""DPO (Rafailov et al., 2023) and SimPO (Meng et al., 2024) trainers.

Both consume the PreferenceCollator's (2B, T) batch (chosen first, then
rejected) and share reward/accuracy metrics; they differ in the implicit
reward definition:

DPO   (needs reference, uses dual-adapter slot):
    r = β · (logp_policy - logp_ref)          (sequence-summed)
    L = -log σ(r_chosen - r_rejected)

SimPO (reference-free, length-normalized):
    r = β · mean_per_token(logp_policy)
    L = -log σ(r_chosen - r_rejected - γ)     (γ: target margin)

SimPO is the most VRAM-friendly pair-based algorithm (no ref forward at all);
DPO is the classic baseline that exercises UnifiedModel.as_reference().
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from onlyone.trainers.base import BaseTrainer


class DPOTrainer(BaseTrainer):
    def __init__(self, model, train_cfg, beta: float = 0.1, **kwargs):
        super().__init__(model, train_cfg, **kwargs)
        self.beta = beta

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

        policy_c, policy_r = policy_logps.chunk(2, dim=0)
        ref_c, ref_r = ref_logps.chunk(2, dim=0)

        reward_c = self.beta * (policy_c - ref_c)
        reward_r = self.beta * (policy_r - ref_r)
        loss = -F.logsigmoid(reward_c - reward_r).mean()

        with torch.no_grad():
            margin = (reward_c - reward_r).mean()
            acc = (reward_c > reward_r).float().mean()
        return loss, {
            "loss": loss.item(),
            "reward_chosen": reward_c.mean().item(),
            "reward_rejected": reward_r.mean().item(),
            "reward_margin": margin.item(),
            "pref_acc": acc.item(),
        }


class SimPOTrainer(BaseTrainer):
    def __init__(self, model, train_cfg, beta: float = 2.0, gamma: float = 0.5, **kwargs):
        super().__init__(model, train_cfg, **kwargs)
        self.beta = beta
        self.gamma = gamma

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        avg_logps = self.um.logps(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            average_per_token=True,  # SimPO's length normalization — no ref needed
        )
        avg_c, avg_r = avg_logps.chunk(2, dim=0)

        reward_c = self.beta * avg_c
        reward_r = self.beta * avg_r
        loss = -F.logsigmoid(reward_c - reward_r - self.gamma).mean()

        with torch.no_grad():
            margin = (reward_c - reward_r).mean()
            acc = (reward_c - reward_r > self.gamma).float().mean()
        return loss, {
            "loss": loss.item(),
            "reward_chosen": reward_c.mean().item(),
            "reward_rejected": reward_r.mean().item(),
            "reward_margin": margin.item(),
            "pref_acc": acc.item(),  # fraction already exceeding target margin γ
        }


def build_dpo_trainer(cfg, tracker=None):
    """Build DPO or SimPO trainer depending on cfg.algo.name."""
    from torch.utils.data import DataLoader

    from onlyone.data.datasets import PreferenceCollator, PreferenceDataset
    from onlyone.models.loading import load_tokenizer
    from onlyone.models.unified import UnifiedModel

    tokenizer = load_tokenizer(cfg.model)
    um = UnifiedModel(cfg.model)

    dataset = PreferenceDataset(cfg.data.train_path, tokenizer, cfg.data.template, cfg.data.max_len)
    collator = PreferenceCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    if cfg.algo.name == "dpo":
        um.snapshot_ref()  # freeze SFT policy as reference BEFORE any update
        trainer = DPOTrainer(um, cfg.train, beta=cfg.algo.dpo_beta,
                             tokenizer=tokenizer, tracker=tracker)
    elif cfg.algo.name == "simpo":
        trainer = SimPOTrainer(um, cfg.train, beta=cfg.algo.simpo_beta,
                               gamma=cfg.algo.simpo_gamma,
                               tokenizer=tokenizer, tracker=tracker)
    else:
        raise ValueError(f"build_dpo_trainer 不支持算法: {cfg.algo.name}")
    return trainer, dataloader
