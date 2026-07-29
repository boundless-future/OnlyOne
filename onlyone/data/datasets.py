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


def _encode_pair(tokenizer, template, prompt: str, completion: str,
                 max_len: int) -> dict[str, Any]:
    """Shared tokenization for preference/binary rows."""
    prompt_text = template.render_prompt(prompt)
    full_text = template.render_full(prompt, completion)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_len]
    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(full_ids))):
        labels[i] = -100
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class PreferenceDataset(Dataset):
    """Preference pairs for ORPO/DPO/SimPO.

    Input jsonl: {"prompt": ..., "chosen": ..., "rejected": ...}
    Each row yields {"chosen": enc, "rejected": enc} so the collator can
    interleave them into a single padded batch of 2B sequences.
    Packing is meaningless here (pairs must stay aligned) and is rejected
    by config validation.
    """

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
                    row = json.loads(line)
                    for key in ("prompt", "chosen", "rejected"):
                        if key not in row:
                            raise ValueError(f"Preference 数据缺少字段 '{key}': {path}")
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"数据集为空: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "chosen": _encode_pair(self.tokenizer, self.template,
                                   row["prompt"], row["chosen"], self.max_len),
            "rejected": _encode_pair(self.tokenizer, self.template,
                                     row["prompt"], row["rejected"], self.max_len),
        }


class PreferenceCollator:
    """Pad chosen and rejected into ONE batch shaped (2B, T): chosen first,
    then rejected, so trainers can split with a single chunk(2)."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        seqs = [f["chosen"] for f in features] + [f["rejected"] for f in features]
        max_len = max(len(s["input_ids"]) for s in seqs)
        input_ids, attention_mask, labels = [], [], []
        for s in seqs:
            pad = max_len - len(s["input_ids"])
            input_ids.append(s["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(s["attention_mask"] + [0] * pad)
            labels.append(s["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class BinaryDataset(Dataset):
    """Binary good/bad labels for KTO.

    Input jsonl: {"prompt": ..., "completion": ..., "label": true|false}
    """

    def __init__(self, path: str, tokenizer: PreTrainedTokenizer,
                 template: str = "chatml", max_len: int = 2048):
        self.tokenizer = tokenizer
        self.template = get_template(template)
        self.max_len = max_len
        self.rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if "label" not in row:
                        raise ValueError(f"KTO 数据缺少字段 'label': {path}")
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"数据集为空: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        enc = _encode_pair(self.tokenizer, self.template,
                           row["prompt"], row["completion"], self.max_len)
        enc["is_good"] = bool(row["label"])
        return enc


class BinaryCollator:
    """Pad binary rows; carries the boolean label as a float tensor."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels, is_good = [], [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
            is_good.append(float(f["is_good"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_good": torch.tensor(is_good, dtype=torch.float),
        }
