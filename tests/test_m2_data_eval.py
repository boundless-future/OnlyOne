"""Preference/Binary dataset tests + answer-extraction tests for GSM8K eval."""

from __future__ import annotations

import json

import pytest

from onlyone.data.datasets import (
    BinaryCollator, BinaryDataset, PreferenceCollator, PreferenceDataset,
)
from onlyone.eval.benchmarks import (
    answers_match, extract_boxed, extract_gold, extract_prediction,
)
from tests.test_data import FakeTokenizer, _write_jsonl


def test_preference_dataset_and_collator(tmp_path):
    path = _write_jsonl(tmp_path, [
        {"prompt": "w1 w2", "chosen": "w3 w4", "rejected": "w5"},
    ])
    ds = PreferenceDataset(path, FakeTokenizer(), template="raw", max_len=64)
    row = ds[0]
    assert set(row) == {"chosen", "rejected"}

    batch = PreferenceCollator(pad_token_id=0)([ds[0], ds[0]])
    # 2 rows -> 4 sequences (chosen first, then rejected)
    assert batch["input_ids"].shape[0] == 4
    chosen, rejected = batch["labels"].chunk(2, dim=0)
    # chosen prompts masked
    assert (chosen[:, :2] == -100).all()
    # rejected row differs in content length but is padded consistently
    assert rejected.shape == chosen.shape


def test_preference_missing_field_raises(tmp_path):
    path = _write_jsonl(tmp_path, [{"prompt": "w1", "chosen": "w2"}])
    with pytest.raises(ValueError, match="rejected"):
        PreferenceDataset(path, FakeTokenizer())


def test_binary_dataset_and_collator(tmp_path):
    path = _write_jsonl(tmp_path, [
        {"prompt": "w1", "completion": "w2 w3", "label": True},
        {"prompt": "w1", "completion": "w4", "label": False},
    ])
    ds = BinaryDataset(path, FakeTokenizer(), template="raw", max_len=64)
    batch = BinaryCollator(pad_token_id=0)([ds[0], ds[1]])
    assert batch["is_good"].tolist() == [1.0, 0.0]
    assert batch["input_ids"].shape[0] == 2


def test_binary_missing_label_raises(tmp_path):
    path = _write_jsonl(tmp_path, [{"prompt": "w1", "completion": "w2"}])
    with pytest.raises(ValueError, match="label"):
        BinaryDataset(path, FakeTokenizer())


# ------------------------------------------------------------- gsm8k parsing

def test_extract_gold():
    assert extract_gold("推理过程……\n#### 42") == "42"
    assert extract_gold("#### 1,000") == "1000"
    assert extract_gold("没有标记") is None


def test_extract_boxed():
    assert extract_boxed(r"answer: \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{1} then \boxed{x^{2} + 1}") == "x^{2} + 1"
    assert extract_boxed(r"answer: \boxed {\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"incomplete \boxed{42") is None
    assert extract_boxed(r"answer: \boxed 2.") == "2"


def test_extract_prediction_priority():
    assert extract_prediction(r"\boxed{7}, but #### 8") == "8"
    assert extract_prediction(r"first 3, final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_prediction():
    assert extract_prediction("所以答案是 #### 16") == "16"
    assert extract_prediction("一共 3+5=8 个") == "8"          # fallback: last number
    assert extract_prediction("20×0.8=16 元") == "16"
    assert extract_prediction("没有数字") is None
    assert extract_prediction("1,000 人参加了活动，最后 500 人离开") == "500"


def test_answers_match():
    assert answers_match("16", "16")
    assert answers_match("16.0", "16")
    assert not answers_match("15", "16")
    assert not answers_match(None, "16")
