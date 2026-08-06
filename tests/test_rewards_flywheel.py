"""Tests for rule rewards, the registry, and RAFT selection logic."""

from __future__ import annotations

import sys

import pytest

from onlyone.flywheel.raft import select_samples
from onlyone.rewards.base import combine
from onlyone.rewards.registry import build_reward, register
from onlyone.rewards.rules import (
    json_format_reward, length_penalty_reward, math_answer_reward,
    math_verify_reward,
    regex_format_reward,
)


# ------------------------------------------------------------------ rewards

def test_math_answer_reward():
    meta = {"gold": "8"}
    assert math_answer_reward("p", "3+5=8", meta) == 1.0
    assert math_answer_reward("p", "答案是 #### 8", meta) == 1.0
    assert math_answer_reward("p", "3+5=9", meta) == 0.0
    assert math_answer_reward("p", "没有数字", meta) == 0.0
    assert math_answer_reward("p", "3+5=8", {}) == 0.0  # no gold -> 0, no crash


def test_json_format_reward():
    assert json_format_reward("p", '{"a": 1}', {}) == 1.0
    assert json_format_reward("p", "[1, 2]", {}) == 1.0
    assert json_format_reward("p", "not json", {}) == 0.0


def test_regex_format_reward():
    r = regex_format_reward(r"<think>.*</think>")
    assert r("p", "<think>reasoning</think> answer", {}) == 1.0
    assert r("p", "plain answer", {}) == 0.0


def test_length_penalty_reward():
    r = length_penalty_reward(max_chars=10, penalty=-0.5)
    assert r("p", "short", {}) == 0.0
    assert r("p", "x" * 10, {}) == 0.0
    assert r("p", "x" * 15, {}) == pytest.approx(-0.25)  # 50% overflow


def test_combine_weighted():
    f = combine(
        [math_answer_reward, length_penalty_reward(max_chars=5, penalty=-1.0)],
        weights=[1.0, 2.0],
    )
    # correct answer (+1) but 6 chars > 5 -> length term -0.2, weighted -0.4
    assert f("p", "ans=8x", {"gold": "8"}) == pytest.approx(1.0 - 0.4)


def test_registry_build_and_unknown():
    fn = build_reward(["math_answer", "length"])
    assert callable(fn)
    with pytest.raises(KeyError, match="未知奖励"):
        build_reward(["nonexistent"])


def test_registry_register_custom():
    register("always_one", lambda p, r, m: 1.0)
    fn = build_reward(["always_one"])
    assert fn("p", "x", {}) == 1.0
    with pytest.raises(KeyError, match="已注册"):
        register("always_one", lambda p, r, m: 1.0)


# ---------------------------------------------------------- RAFT selection

def _scoring_reward(good_word: str):
    def reward(prompt, response, meta):
        return 1.0 if good_word in response else 0.0
    return reward


def test_select_samples_keeps_best_above_threshold():
    candidates = ["bad 1", "good answer", "bad 2", "good other"]
    sft, pref, scores = select_samples(
        "p", {}, candidates, _scoring_reward("good"), threshold=0.9
    )
    assert scores == [0.0, 1.0, 0.0, 1.0]
    assert sft == {"prompt": "p", "completion": "good answer"}  # first best
    assert pref == {"prompt": "p", "chosen": "good answer", "rejected": "bad 1"}


def test_select_samples_rejects_all_below_threshold():
    sft, pref, _ = select_samples(
        "p", {}, ["bad 1", "bad 2"], _scoring_reward("good"), threshold=0.9
    )
    assert sft is None and pref is None


def test_select_samples_no_pref_when_all_equal():
    """All candidates identical score -> no meaningful DPO pair."""
    sft, pref, _ = select_samples(
        "p", {}, ["good a", "good b"], _scoring_reward("good"), threshold=0.9
    )
    assert sft is not None
    assert pref is None  # best == worst, pair would be noise


def test_select_samples_pref_disabled():
    sft, pref, _ = select_samples(
        "p", {}, ["good a", "bad b"], _scoring_reward("good"),
        threshold=0.9, make_preference=False,

    )
    assert sft is not None and pref is None

def test_math_verify_reward(monkeypatch):
    class FakeMathVerify:
        @staticmethod
        def parse(value, **kwargs):
            return value.removeprefix("The answer is $").removesuffix("$")

        @staticmethod
        def verify(gold, prediction, **kwargs):
            equivalents = {("0.5", r"\frac{1}{2}"), ("2", "2.0")}
            return gold == prediction or (gold, prediction) in equivalents

    monkeypatch.setitem(sys.modules, "math_verify", FakeMathVerify)
    assert math_verify_reward("p", r"\boxed{\frac{1}{2}}", {"gold": "0.5"}) == 1.0
    assert math_verify_reward("p", r"\boxed{2.0}", {"gold": "2"}) == 1.0
    assert math_verify_reward("p", r"\boxed{3}", {"gold": "2"}) == 0.0
    assert math_verify_reward("p", "no answer", {"gold": "2"}) == 0.0
    assert math_verify_reward("p", r"\boxed{2}", {}) == 0.0


def test_math_verify_is_registered():
    assert callable(build_reward(["math_verify"]))

def test_math_verify_reward_with_real_dependency():
    pytest.importorskip("math_verify")
    assert math_verify_reward(
        "p", r"\boxed{\frac{1}{2}}", {"gold": "0.5"}
    ) == 1.0
    assert math_verify_reward("p", r"\boxed{3}", {"gold": "2"}) == 0.0
