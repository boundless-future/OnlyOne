"""RAFT / rejection-sampling data flywheel.

One round of the loop (design doc §2.4):

    prompts ──rollout(G candidates)──► reward scoring ──►
    best≥threshold ──► SFT row            (rejection sampling)
    best vs worst  ──► DPO preference pair (free byproduct)
    retrain SFT on kept rows ──► better policy ──► next round

This is the automation heart of OnlyOne: the model generates its own
training data, rule rewards filter it, and DPO pairs fall out for free —
no human preference annotation.

Selection logic (`select_samples`) is a pure function so the flywheel's
data decisions are unit-testable without a model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from onlyone.rewards.base import RewardFn
from onlyone.rollout.base import RolloutEngine

logger = logging.getLogger("onlyone")


@dataclass
class RoundResult:
    round_idx: int
    n_prompts: int
    sft_rows: list[dict[str, Any]] = field(default_factory=list)
    pref_rows: list[dict[str, Any]] = field(default_factory=list)
    reward_mean: float = 0.0
    keep_rate: float = 0.0  # fraction of prompts that yielded an SFT row


def select_samples(
    prompt: str,
    meta: dict[str, Any],
    candidates: list[str],
    reward_fn: RewardFn,
    threshold: float,
    make_preference: bool = True,
) -> tuple[Optional[dict], Optional[dict], list[float]]:
    """Score candidates; return (sft_row, pref_row, scores).

    - sft_row: best candidate if its reward >= threshold, else None
    - pref_row: (best, worst) pair if best >= threshold AND best > worst
      (identical rewards make a meaningless DPO pair), else None
    """
    scores = [reward_fn(prompt, c, meta) for c in candidates]
    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    worst_idx = min(range(len(candidates)), key=lambda i: scores[i])
    best_score, worst_score = scores[best_idx], scores[worst_idx]

    sft_row = None
    if best_score >= threshold:
        sft_row = {"prompt": prompt, "completion": candidates[best_idx]}

    pref_row = None
    if make_preference and sft_row is not None and best_score > worst_score:
        pref_row = {
            "prompt": prompt,
            "chosen": candidates[best_idx],
            "rejected": candidates[worst_idx],
        }
    return sft_row, pref_row, scores


def run_round(
    round_idx: int,
    prompts: list[dict[str, Any]],
    engine: RolloutEngine,
    reward_fn: RewardFn,
    group_size: int,
    threshold: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    make_preference: bool = True,
) -> RoundResult:
    """One rollout+filter round over all prompts."""
    result = RoundResult(round_idx=round_idx, n_prompts=len(prompts))
    all_scores: list[float] = []

    prompt_texts = [r["prompt"] for r in prompts]
    metas = [r.get("meta", {}) for r in prompts]
    generations = engine.generate(
        prompt_texts, n_per_prompt=group_size,
        max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
    )

    for prompt, meta, candidates in zip(prompt_texts, metas, generations):
        sft_row, pref_row, scores = select_samples(
            prompt, meta, candidates, reward_fn, threshold, make_preference
        )
        all_scores.extend(scores)
        if sft_row:
            result.sft_rows.append(sft_row)
        if pref_row:
            result.pref_rows.append(pref_row)

    result.reward_mean = sum(all_scores) / max(len(all_scores), 1)
    result.keep_rate = len(result.sft_rows) / max(len(prompts), 1)
    logger.info(
        "round %d: prompts=%d sft=%d pref=%d reward_mean=%.3f keep_rate=%.1f%%",
        round_idx, result.n_prompts, len(result.sft_rows), len(result.pref_rows),
        result.reward_mean, 100 * result.keep_rate,
    )
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_flywheel(cfg, tracker=None) -> list[RoundResult]:
    """Full RAFT loop: N rounds of rollout -> filter -> SFT retrain.

    Each round writes its artifacts under raft.output_dir/round_<i>/:
    sft.jsonl, preference.jsonl, result.json. Retraining uses the standard
    SFT trainer on the round's kept samples.
    """
    import torch

    from onlyone.models.loading import load_tokenizer
    from onlyone.models.unified import UnifiedModel
    from onlyone.rewards.registry import build_reward
    from onlyone.rollout.hf_engine import HFRolloutEngine

    raft = cfg.raft
    if raft is None:
        raise ValueError("配置缺少 raft 段")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(cfg.model)
    um = UnifiedModel(cfg.model)
    um.model.to(device)
    engine = HFRolloutEngine(um, tokenizer, template=cfg.data.template,
                             device=device, batch_size=raft.rollout_batch_size)
    reward_fn = build_reward(raft.rewards)

    with open(raft.prompts_path, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    results: list[RoundResult] = []
    for round_idx in range(raft.rounds):
        result = run_round(
            round_idx, prompts, engine, reward_fn,
            group_size=raft.group_size, threshold=raft.threshold,
            max_new_tokens=raft.max_new_tokens,
            temperature=raft.temperature, top_p=raft.top_p,
            make_preference=raft.make_preference,
        )
        results.append(result)

        out_dir = Path(raft.output_dir) / f"round_{round_idx}"
        _write_jsonl(out_dir / "sft.jsonl", result.sft_rows)
        _write_jsonl(out_dir / "preference.jsonl", result.pref_rows)
        _write_jsonl(out_dir / "result.json", [{
            "round": round_idx, "reward_mean": result.reward_mean,
            "keep_rate": result.keep_rate,
            "n_sft": len(result.sft_rows), "n_pref": len(result.pref_rows),
        }])
        if tracker:
            tracker.log({
                "raft/reward_mean": result.reward_mean,
                "raft/keep_rate": result.keep_rate,
                "raft/n_sft": float(len(result.sft_rows)),
            }, step=round_idx)

        if raft.train_sft and result.sft_rows:
            _retrain_sft(cfg, um, tokenizer, out_dir / "sft.jsonl", round_idx)

    return results


def _retrain_sft(cfg, um, tokenizer, sft_path: Path, round_idx: int) -> None:
    """Continue training the SAME UnifiedModel on this round's SFT data.

    A fresh optimizer per round is deliberate: RAFT rounds are conceptually
    separate training phases, and reusing moments across distribution shifts
    (new self-generated data) is not obviously correct.
    """
    from torch.utils.data import DataLoader

    from onlyone.data.datasets import SFTCollator, SFTDataset
    from onlyone.trainers.sft import SFTTrainer

    dataset = SFTDataset(str(sft_path), tokenizer, cfg.data.template, cfg.data.max_len)
    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=cfg.train.per_device_batch_size,
                            shuffle=True, collate_fn=collator, num_workers=0)
    trainer = SFTTrainer(um, cfg.train, tokenizer=tokenizer)
    trainer.train(dataloader)
    um.save(str(Path(cfg.raft.output_dir) / f"round_{round_idx}" / "adapter"))
    logger.info("round %d: SFT retrain done on %d rows", round_idx, len(dataset))
