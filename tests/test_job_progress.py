"""Tests for DataSphere job progress reporting."""

import json
import os
from pathlib import Path

from adaptive_roi_codec.utils.job_progress import JobProgressTracker, report_job_progress


def test_report_job_progress_overwrites_latest_snapshot(tmp_path: Path, monkeypatch) -> None:
    progress_file = tmp_path / "job_progress.json"
    monkeypatch.setenv("JOB_PROGRESS_FILENAME", str(progress_file))

    report_job_progress(25, "epoch 1 batch 100/400")
    report_job_progress(100, "done")

    payload = json.loads(progress_file.read_text(encoding="utf-8").strip())
    assert payload["progress"] == 100
    assert payload["message"] == "done"
    assert payload["create_time"].endswith("+00:00")
    assert "." not in payload["create_time"].split("+")[0]


def test_report_job_progress_noop_without_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("JOB_PROGRESS_FILENAME", raising=False)
    report_job_progress(50, "ignored")
    assert list(tmp_path.iterdir()) == []


def test_job_progress_tracker_maps_phases(tmp_path: Path, monkeypatch) -> None:
    progress_file = tmp_path / "job_progress.json"
    monkeypatch.setenv("JOB_PROGRESS_FILENAME", str(progress_file))
    tracker = JobProgressTracker(total_batches=200, epochs=1)

    tracker.setup("setup")
    tracker.staging(1200, 2400)
    tracker.training(epoch=1, batch=200)
    tracker.complete("done")

    payload = json.loads(progress_file.read_text(encoding="utf-8").strip())
    assert payload["progress"] == 100
    assert payload["message"] == "done"

    tracker.staging(1, 2400)
    mid = json.loads(progress_file.read_text(encoding="utf-8").strip())
    assert 8 <= mid["progress"] <= 12
