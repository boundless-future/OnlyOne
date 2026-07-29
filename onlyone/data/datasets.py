"""SFT dataset: jsonl -> tokenized examples with completion-only labels.

Input format (one JSON object per line):
    {"prompt": "...", "completion": "..."}

Output example fields:
    input_ids      (<= max_len)
    attention_mask
    labels         input_ids with prompt tokens and padding masked to -100
"""

from __future__ import annotations

import json
from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from onlyone.data.templates import get_template


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer: PreTrainedTokenizer,
                 template: str = "chatml", max_len: int = 2048):
        self.tokenizer = tokenizer
        self.template = get_template(template)
        self.max_len = max_len
        self.rows: list[dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"数据集为空: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def encode(self, prompt: str, completion: str) -> dict[str, Any]:
        """Tokenize one (prompt, completion) pair; public for reuse/tests."""
        prompt_text = self.template.render_prompt(prompt)
        full_text = self.template.render_full(prompt, completion)

        # add_special_tokens=False: the template owns all special tokens,
        # otherwise e.g. a BOS would be inserted twice across prompt/full.
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        full_ids = full_ids[: self.max_len]
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return self.encode(row["prompt"], row["completion"])


class SFTCollator:
    """Right-pad a batch of variable-length examples."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
