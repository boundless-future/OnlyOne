"""eval_math.build_summary 的聚合正确性:overall / per-level / 多变体。"""

from __future__ import annotations

import pytest

from scripts.eval_math import build_summary


def _rec(level, base_correct, adapter_correct, base_chars=100, adapter_chars=50):
    return {
        "level": level,
        "base_correct": float(base_correct),
        "adapter_correct": float(adapter_correct),
        "base_chars": base_chars,
        "adapter_chars": adapter_chars,
    }


def test_build_summary_overall_and_per_level():
    records = [
        _rec(0, 1, 1),
        _rec(0, 1, 0),
        _rec(5, 0, 1),
        _rec(5, 0, 0),
    ]
    s = build_summary(records, ["base", "adapter"])
    assert s["overall"]["n"] == 4
    assert s["overall"]["base_acc"] == pytest.approx(0.5)
    assert s["overall"]["adapter_acc"] == pytest.approx(0.5)
    assert s["per_level"]["0"]["base_acc"] == pytest.approx(1.0)
    assert s["per_level"]["0"]["adapter_acc"] == pytest.approx(0.5)
    assert s["per_level"]["5"]["base_acc"] == pytest.approx(0.0)
    assert s["per_level"]["5"]["adapter_acc"] == pytest.approx(0.5)
    # 长度均值:直接检验"变短"假设的字段必须存在
    assert s["overall"]["adapter_chars"] == pytest.approx(50.0)


def test_build_summary_base_only():
    records = [_rec(3, 1, 0), _rec(3, 0, 0)]
    s = build_summary(records, ["base"])
    assert "adapter_acc" not in s["overall"]
    assert s["overall"]["base_acc"] == pytest.approx(0.5)
