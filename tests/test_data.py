"""Data layer tests: template masking, collation, packing invariants."""

from __future__ import annotations

import json

from onlyone.data.datasets import SFTCollator, SFTDataset
from onlyone.data.packing import PackedSFTDataset, pack_examples


class FakeTokenizer:
    """Whitespace tokenizer with a fixed small vocab; no downloads."""

    pad_token_id = 0

    def __init__(self):
        self.vocab: dict[str, int] = {}
        for i, w in enumerate(range(200)):
            self.vocab[f"w{i}"] = i + 1

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = []
        for tok in text.replace("<|im_start|>", " imstart ").replace("<|im_end|>", " imend ").split():
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab) + 1
            ids.append(self.vocab[tok])
        return {"input_ids": ids}


def _write_jsonl(tmp_path, rows):
    path = tmp_path / "data.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def test_prompt_tokens_masked(tmp_path):
    path = _write_jsonl(tmp_path, [{"prompt": "w1 w2", "completion": "w3 w4"}])
    ds = SFTDataset(path, FakeTokenizer(), template="raw", max_len=64)
    ex = ds[0]
    n_prompt = len(FakeTokenizer()("w1 w2")["input_ids"])
    assert all(l == -100 for l in ex["labels"][:n_prompt])
    assert ex["labels"][n_prompt:] == ex["input_ids"][n_prompt:]


def test_truncation_keeps_labels_aligned(tmp_path):
    path = _write_jsonl(tmp_path, [{"prompt": "w1 w2", "completion": " ".join(f"w{i}" for i in range(50))}])
    ds = SFTDataset(path, FakeTokenizer(), template="raw", max_len=10)
    ex = ds[0]
    assert len(ex["input_ids"]) == len(ex["labels"]) == 10


def test_collator_padding(tmp_path):
    path = _write_jsonl(tmp_path, [
        {"prompt": "w1", "completion": "w2 w3"},
        {"prompt": "w1", "completion": "w2"},
    ])
    ds = SFTDataset(path, FakeTokenizer(), template="raw", max_len=64)
    batch = SFTCollator(pad_token_id=0)([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"][1].sum() < batch["attention_mask"][0].sum()
    # padded label positions are -100
    assert (batch["labels"][1][batch["attention_mask"][1] == 0] == -100).all()


def test_packing_respects_max_len_and_labels(tmp_path):
    rows = [{"prompt": "w1 w2", "completion": "w3 w4"}] * 5
    path = _write_jsonl(tmp_path, rows)
    ds = SFTDataset(path, FakeTokenizer(), template="raw", max_len=64)
    single_len = len(ds[0]["input_ids"])
    packed = PackedSFTDataset(ds, max_len=single_len * 2 + 1)
    for block in packed:
        assert len(block["input_ids"]) <= single_len * 2 + 1
        assert len(block["input_ids"]) == len(block["labels"])
        # each packed sample keeps its prompt masked: -100 count preserved
    total_masked = sum(sum(1 for l in b["labels"] if l == -100) for b in packed)
    assert total_masked == 5 * 2  # 2 prompt tokens per sample
