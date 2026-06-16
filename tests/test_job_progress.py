"""Tests for DataSphere job progress reporting."""

import json
import os
from pathlib import Path

from adaptive_roi_codec.utils.job_progress import report_job_progress


def test_report_job_progress_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    progress_file = tmp_path / "job_progress.jsonl"
    monkeypatch.setenv("JOB_PROGRESS_FILENAME", str(progress_file))

    report_job_progress(25, "epoch 1 batch 100/400")
    report_job_progress(100, "done")

    lines = progress_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["progress"] == 25
    assert first["message"] == "epoch 1 batch 100/400"
    assert first["create_time"].endswith("+00:00")

    second = json.loads(lines[1])
    assert second["progress"] == 100


def test_report_job_progress_noop_without_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("JOB_PROGRESS_FILENAME", raising=False)
    report_job_progress(50, "ignored")
    assert list(tmp_path.iterdir()) == []
