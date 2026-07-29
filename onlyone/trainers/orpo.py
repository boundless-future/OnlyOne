"""ORPO: Odds-Ratio Preference Optimization (Hong et al., 2024).

Single-stage, reference-free alignment: SFT NLL on chosen + a weighted
odds-ratio penalty that pushes chosen above rejected:

    L = L_SFT(chosen) + λ · L_OR
    L_OR = -log σ( log odds(chosen) - log odds(rejected) )
    log odds(y) = log p(y|x) - log(1 - p(y|x)),  p normalized per token

Why ORPO is our first alignment algorithm (design doc §2.4): no reference
model, no adapter snapshot, memory ≈ plain SFT — the cheapest correct
preference optimization on a single card.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from onlyone.trainers.base import BaseTrainer


class ORPOTrainer(BaseTrainer):
    def __init__(self, model, train_cfg, lambda_or: float = 0.1, **kwargs):
        super().__init__(model, train_cfg, **kwargs)
        self.lambda_or = lambda_or

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        token_logps, counts = self.um.token_logps(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        seq_logps = token_logps.sum(dim=-1)
        # Per-token average log p — ORPO's probability normalization.
        avg_logps = seq_logps / counts.clamp(min=1)

        chosen_logps, rejected_logps = avg_logps.chunk(2, dim=0)
        chosen_counts, _ = counts.chunk(2, dim=0)

        # SFT term: token-mean NLL over chosen sequences only.
        chosen_token_logps, _ = token_logps.chunk(2, dim=0)
        nll = -chosen_token_logps.sum() / chosen_counts.sum().clamp(min=1)

        # Odds-ratio term. odds = p / (1-p); in log space with p = exp(avg_logp):
        # log odds = avg_logp - log1p(-p). exp(avg) can round to exactly 1.0
        # for very confident sequences; the clamp keeps log1p finite.
        p_c = chosen_logps.exp().clamp(max=1 - 1e-6)
        p_r = rejected_logps.exp().clamp(max=1 - 1e-6)
        log_odds_c = chosen_logps - torch.log1p(-p_c)
        log_odds_r = rejected_logps - torch.log1p(-p_r)
        or_loss = -F.logsigmoid(log_odds_c - log_odds_r).mean()

        loss = nll + self.lambda_or * or_loss
        with torch.no_grad():
            margin = (chosen_logps - rejected_logps).mean()
            acc = (chosen_logps > rejected_logps).float().mean()
        return loss, {
            "loss": loss.item(),
            "nll": nll.item(),
            "or_loss": or_loss.item(),
            "reward_margin": margin.item(),
            "pref_acc": acc.item(),
        }


def build_orpo_trainer(cfg, tracker=None):
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
    trainer = ORPOTrainer(
        um, cfg.train, lambda_or=cfg.algo.orpo_lambda, tokenizer=tokenizer, tracker=tracker
    )
    return trainer, dataloader
