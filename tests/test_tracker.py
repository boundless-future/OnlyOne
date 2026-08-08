"""Tracker metrics.jsonl durability: every logged step must land on disk,
independent of the online backend (wandb/tb/none)."""

from __future__ import annotations

import json

from onlyone.utils.logging import Tracker


def test_metrics_jsonl_written_and_appended(tmp_path):
    tracker = Tracker(log_with="none", output_dir=str(tmp_path))
    tracker.log({"loss": 1.5, "kl": 0.01}, step=1)
    tracker.log({"loss": 1.2, "kl": 0.02}, step=2)
    tracker.finish()

    path = tmp_path / "metrics.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["step"] == 1 and rows[0]["loss"] == 1.5
    assert rows[1]["step"] == 2 and rows[1]["kl"] == 0.02
    assert "time" in rows[0]


def test_no_output_dir_means_no_file(tmp_path):
    tracker = Tracker(log_with="none")
    tracker.log({"loss": 1.0}, step=1)  # must not raise
    tracker.finish()
