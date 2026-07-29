"""GSM8K generative evaluation.

GSM8K answers end with "#### <number>". We prompt with the question, let the
model generate free-form, and extract the LAST number in the output (matching
how the model is expected to answer after SFT on the GSM8K format).

Dataset jsonl format: {"question": ..., "answer": "... #### 42"}
This is deliberately decoupled from training: any checkpoint can be scored
without touching trainer state.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import torch

from onlyone.data.templates import get_template

logger = logging.getLogger("onlyone")

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")


def extract_gold(answer_field: str) -> Optional[str]:
    """Pull the gold number out of a GSM8K answer field."""
    m = _ANSWER_RE.search(answer_field)
    if not m:
        return None
    return m.group(1).replace(",", "")


def extract_prediction(text: str) -> Optional[str]:
    """Extract the predicted answer: last number after '####' if present,
    else the last number in the text."""
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def answers_match(pred: Optional[str], gold: Optional[str], tol: float = 1e-4) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < tol
    except ValueError:
        return False


@torch.no_grad()
def eval_gsm8k(
    um,
    tokenizer,
    path: str,
    template: str = "chatml",
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
    device: str = "cpu",
) -> dict:
    """Generate answers and score exact-match accuracy.

    Returns {"accuracy": float, "n": int, "n_parse_fail": int, "samples": [...]}.
    `samples` holds the first few generations for eyeballing in the report.
    """
    tmpl = get_template(template)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]

    was_training = um.model.training
    um.model.eval()
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # generation convention

    n_correct, n_parse_fail = 0, 0
    samples = []
    for i, row in enumerate(rows):
        gold = extract_gold(row["answer"])
        prompt_text = tmpl.render_prompt(row["question"])
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
        out = um.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: eval must be deterministic
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_prediction(text)
        if pred is None:
            n_parse_fail += 1
        if answers_match(pred, gold):
            n_correct += 1
        if i < 5:
            samples.append({"gold": gold, "pred": pred, "generation": text[:400]})
        if (i + 1) % 50 == 0:
            logger.info("gsm8k progress: %d/%d, acc=%.3f", i + 1, len(rows), n_correct / (i + 1))

    tokenizer.padding_side = old_padding_side
    if was_training:
        um.model.train()

    n = len(rows)
    return {
        "accuracy": n_correct / max(n, 1),
        "n": n,
        "n_parse_fail": n_parse_fail,
        "samples": samples,
    }
