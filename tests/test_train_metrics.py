"""Tests for training metrics output."""

import json
from pathlib import Path

from adaptive_roi_codec.train import build_train_metrics_report, write_metrics_file


def test_build_train_metrics_report_includes_all_epochs() -> None:
    history = [
        {"epoch": 1.0, "loss": -10.0, "batches": 100.0, "elapsed_s": 60.0},
        {"epoch": 2.0, "loss": -12.0, "batches": 100.0, "elapsed_s": 58.0},
    ]
    report = build_train_metrics_report(experiment_id="exp-a", epoch_history=history)
    assert report["experiment_id"] == "exp-a"
    assert len(report["epochs"]) == 2
    assert report["final"]["epoch"] == 2.0
    assert report["final"]["loss"] == -12.0


def test_write_metrics_file_persists_epoch_history(tmp_path: Path) -> None:
    metrics_path = tmp_path / "train_metrics.json"
    report = build_train_metrics_report(
        experiment_id="exp-b",
        epoch_history=[{"epoch": 1.0, "loss": -1.0, "batches": 3.0, "elapsed_s": 1.5}],
    )
    write_metrics_file(metrics_path, report)
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert loaded["epochs"][0]["batches"] == 3.0
    assert loaded["final"]["loss"] == -1.0
