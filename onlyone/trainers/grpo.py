"""GRPO: Group Relative Policy Optimization (DeepSeekMath, 2024).

The single-card RL breakthrough (design doc §2.2): no Critic — the value
baseline is replaced by the group mean reward of G candidates sampled from
the SAME prompt:

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)     per candidate i
    ratio_t = exp(logp_policy_t - logp_old_t)              per token
    L = -E[ min(ratio_t·A, clip(ratio_t, 1±ε)·A) ] + β·KL_k3
    KL_k3 = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1  (≥0)

One gradient step per rollout (μ=1), so the "old" policy is simply the
policy right after generation — old token logps are recomputed under
no_grad before the update.

Anti-collapse guards (design doc §5):
- kl_circuit_breaker: abort training if mean KL spikes (policy fleeing ref)
- completion-length metric exposed every step (length-hacking early warning)
- skip groups whose rewards are all identical (advantage undefined → noise)
"""

from __future__ import annotations

import json
import logging
import random

import torch

from onlyone.data.datasets import _encode_pair
from onlyone.rewards.base import RewardFn
from onlyone.rollout.base import RolloutEngine
from onlyone.trainers.base import BaseTrainer

logger = logging.getLogger("onlyone")


def group_advantages(rewards: list[float], group_size: int,
                     eps: float = 1e-4) -> tuple[list[float], int]:
    """Group-relative advantage normalization. Returns (advantages, n_skipped).

    Groups whose rewards are all identical (std < eps) get advantage 0 and are
    counted as skipped — they carry no learning signal and would only add KL
    penalty noise. Callers may drop them entirely.
    """
    if len(rewards) % group_size != 0:
        raise ValueError(f"rewards 数量 {len(rewards)} 不是 group_size {group_size} 的整数倍")
    advantages: list[float] = []
    n_skipped = 0
    for i in range(0, len(rewards), group_size):
        group = rewards[i : i + group_size]
        mean = sum(group) / len(group)
        var = sum((r - mean) ** 2 for r in group) / len(group)
        std = var ** 0.5
        if std < eps:
            advantages.extend([0.0] * len(group))
            n_skipped += 1
        else:
            advantages.extend([(r - mean) / (std + eps) for r in group])
    return advantages, n_skipped


def kl_k3(policy_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """k3 KL estimator (Schulman): KL ≈ e^(Δ) - Δ - 1 with Δ = ref - policy.
    Non-negative, unbiased at 0, smooth. Operates element-wise on token logps."""
    delta = ref_logps - policy_logps
    return torch.exp(delta) - delta - 1


class GRPOTrainer(BaseTrainer):
    """Online RL trainer that generates its own training data each iteration.

    Unlike the dataset-driven trainers, GRPO overrides `train()` entirely:
    there is no static dataloader — each iteration rolls out fresh candidates
    from the current policy. Optimizer/checkpoint/tracker plumbing still
    comes from BaseTrainer.
    """

    def __init__(self, model, train_cfg, grpo_cfg, engine: RolloutEngine,
                 reward_fn: RewardFn, prompts: list[dict],
                 tokenizer=None, template: str = "chatml", max_len: int = 2048,
                 **kwargs):
        super().__init__(model, train_cfg, tokenizer=tokenizer, **kwargs)
        from onlyone.data.templates import get_template
        self.grpo = grpo_cfg
        self.engine = engine
        self.reward_fn = reward_fn
        self.prompts = prompts
        self.template_obj = get_template(template)
        self.max_len = max_len
        self.rng = random.Random(train_cfg.seed)
        self._last_stats: dict[str, float] = {}  # filled by _rollout_and_build_batch

    # ------------------------------------------------------------ data build

    def _rollout_and_build_batch(self) -> dict[str, torch.Tensor] | None:
        """One full iteration of data production: prompts -> candidates ->
        rewards -> advantages -> tokenized batch with per-token fields."""
        g = self.grpo
        sampled = self.rng.sample(self.prompts, min(g.prompts_per_step, len(self.prompts)))
        prompt_texts = [r["prompt"] for r in sampled]
        metas = [r.get("meta", {}) for r in sampled]

        generations = self.engine.generate(
            prompt_texts, n_per_prompt=g.group_size,
            max_new_tokens=g.max_new_tokens,
            temperature=g.temperature, top_p=g.top_p,
        )

        flat_rewards: list[float] = []
        flat_rows: list[dict] = []
        for prompt, meta, candidates in zip(prompt_texts, metas, generations):
            for cand in candidates:
                flat_rewards.append(self.reward_fn(prompt, cand, meta))
                flat_rows.append({"prompt": prompt, "completion": cand})

        advantages, n_skipped = group_advantages(flat_rewards, g.group_size)
        if n_skipped == len(sampled):
            logger.warning("所有组 reward 相同(模型全对或全错),本步跳过")
            return None

        # Tokenize prompt+completion; labels mask the prompt part.
        encoded = [
            _encode_pair(self.tokenizer, self.template_obj,
                         r["prompt"], r["completion"], self.max_len)
            for r in flat_rows
        ]
        max_len = max(len(e["input_ids"]) for e in encoded)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attn, labels = [], [], []
        for e in encoded:
            pad = max_len - len(e["input_ids"])
            input_ids.append(e["input_ids"] + [pad_id] * pad)
            attn.append(e["attention_mask"] + [0] * pad)
            labels.append(e["labels"] + [-100] * pad)

        self._last_stats = {
            "reward_mean": sum(flat_rewards) / len(flat_rewards),
            "reward_min": min(flat_rewards),
            "reward_max": max(flat_rewards),
            "n_degenerate_groups": float(n_skipped),
            "completion_chars": sum(len(r["completion"]) for r in flat_rows) / len(flat_rows),
        }
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "advantages": torch.tensor(advantages, dtype=torch.float),
        }

    # ------------------------------------------------------------------ loss

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        token_logps, counts = self.um.token_logps(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        # Old policy = policy right after generation (μ=1). Recompute detached
        # so the ratio's gradient flows only through the current policy.
        with torch.no_grad():
            old_token_logps, _ = self.um.token_logps(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        with self.um.as_reference():
            ref_token_logps, _ = self.um.token_logps(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

        mask = batch["labels"][:, 1:] != -100  # (B, T-1) completion-token mask
        ratio = torch.exp(token_logps - old_token_logps)

        # Broadcast per-sequence advantage over its completion tokens.
        adv = batch["advantages"].unsqueeze(1).expand_as(ratio)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.grpo.clip_eps, 1 + self.grpo.clip_eps) * adv
        pg_loss = -torch.min(surr1, surr2)

        kl = kl_k3(token_logps, ref_token_logps)
        per_token_loss = pg_loss + self.grpo.kl_beta * kl

        # Per-sequence masked mean, then batch mean (GRPO paper convention).
        seq_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = seq_loss.mean()

        with torch.no_grad():
            kl_mean = (kl * mask).sum() / mask.sum().clamp(min=1)
            clip_frac = ((ratio - 1.0).abs() > self.grpo.clip_eps).float()
            clip_frac = (clip_frac * mask).sum() / mask.sum().clamp(min=1)
        metrics = {
            "loss": loss.item(),
            "kl": kl_mean.item(),
            "clip_frac": clip_frac.item(),
            **self._last_stats,
        }
        return loss, metrics

    # ------------------------------------------------------------------ loop

    def train(self, dataloader=None) -> None:  # noqa: ARG002 - GRPO 自产数据
        cfg = self.cfg
        self.model.train()
        logger.info("GRPO 开始: %d prompts, G=%d, %d steps",
                    len(self.prompts), self.grpo.group_size, cfg.max_steps)

        consecutive_skips = 0
        max_consecutive_skips = 20  # 防死循环:模型全对/全错时不可能学到东西
        while self.global_step < cfg.max_steps:
            batch = self._rollout_and_build_batch()
            if batch is None:
                consecutive_skips += 1
                if consecutive_skips >= max_consecutive_skips:
                    raise RuntimeError(
                        f"连续 {max_consecutive_skips} 次 rollout 所有组 reward 相同。"
                        "模型对当前 prompts 全对或全错 —— 请换难度匹配的数据,"
                        "或先用 SFT/RAFT 把模型带到合适区间。"
                    )
                continue
            consecutive_skips = 0
            batch = self._to_device(batch)

            loss, metrics = self.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            self.um.mark_trained(1)

            if self.grpo.kl_circuit_breaker > 0 and metrics["kl"] > self.grpo.kl_circuit_breaker:
                self.save_checkpoint(final=False)
                raise RuntimeError(
                    f"KL 熔断: step {self.global_step} kl={metrics['kl']:.4f} > "
                    f"{self.grpo.kl_circuit_breaker}。policy 正在逃离 reference,"
                    f"checkpoint 已保存,请检查 kl_beta / lr / reward 设计。"
                )

            if self.global_step % cfg.logging_steps == 0:
                from onlyone.utils.memory import peak_vram_gb
                metrics["lr"] = self.scheduler.get_last_lr()[0]
                metrics["vram_peak_gb"] = peak_vram_gb()
                self.tracker.log(metrics, self.global_step)

            if self.global_step % cfg.save_steps == 0:
                self.save_checkpoint()

        self.save_checkpoint(final=True)
        self.tracker.finish()
        logger.info("GRPO 训练完成: %d steps", self.global_step)


def build_grpo_trainer(cfg, tracker=None):
    from onlyone.models.loading import load_tokenizer
    from onlyone.models.unified import UnifiedModel
    from onlyone.rewards.registry import build_reward

    if cfg.grpo is None:
        raise ValueError("配置缺少 grpo 段")

    tokenizer = load_tokenizer(cfg.model)
    um = UnifiedModel(cfg.model)
    um.snapshot_ref()  # GRPO 的 KL 锚点:SFT 后的策略,训练前冻结

    device = "cuda" if torch.cuda.is_available() and cfg.train.device == "auto" else (
        cfg.train.device if cfg.train.device != "auto" else "cpu"
    )
    um.model.to(device)

    if cfg.grpo.engine == "vllm":
        from onlyone.rollout.vllm_engine import VLLMRolloutEngine
        engine = VLLMRolloutEngine(
            um, tokenizer, template=cfg.data.template, device=device,
            gpu_mem_util=cfg.grpo.vllm_gpu_mem_util,
        )
    else:
        from onlyone.rollout.hf_engine import HFRolloutEngine
        engine = HFRolloutEngine(um, tokenizer, template=cfg.data.template,
                                 device=device, batch_size=cfg.grpo.rollout_batch_size)

    reward_fn = build_reward(cfg.grpo.rewards)
    with open(cfg.grpo.prompts_path, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"prompts 为空: {cfg.grpo.prompts_path}")

    trainer = GRPOTrainer(
        um, cfg.train, cfg.grpo, engine, reward_fn, prompts,
        tokenizer=tokenizer, template=cfg.data.template, max_len=cfg.data.max_len,
        tracker=tracker,
    )
    return trainer
