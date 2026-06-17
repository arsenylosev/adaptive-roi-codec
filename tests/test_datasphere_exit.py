"""Tests for DataSphere job exit helpers."""

import json
import multiprocessing as mp
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from adaptive_roi_codec.utils.datasphere_exit import (
    finalize_datasphere_job,
    terminate_multiprocessing_children,
    write_job_status,
)


def test_terminate_multiprocessing_children_joins_active_children() -> None:
    children_before = len(mp.active_children())
    terminate_multiprocessing_children()
    assert len(mp.active_children()) == children_before


def test_write_job_status_persists_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    status_path = tmp_path / "job_status.json"
    monkeypatch.setenv("TRAIN_STATUS_PATH", str(status_path))

    write_job_status("success", exit_code=0, message="done")

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["exit_code"] == 0
    assert payload["message"] == "done"
    assert payload["create_time"].endswith("+00:00")


def test_finalize_datasphere_job_returns_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TRAIN_STATUS_PATH", str(tmp_path / "job_status.json"))
    monkeypatch.setenv("JOB_PROGRESS_FILENAME", str(tmp_path / "job_progress.json"))
    with patch("adaptive_roi_codec.utils.datasphere_exit.terminate_multiprocessing_children") as terminate:
        with patch("adaptive_roi_codec.utils.datasphere_exit.release_torch_cuda") as release_cuda:
            code = finalize_datasphere_job(0, message="ok")
    terminate.assert_called_once()
    release_cuda.assert_called_once()
    assert code == 0
    progress = json.loads((tmp_path / "job_progress.json").read_text(encoding="utf-8").strip())
    assert progress["progress"] == 100


def test_finalize_datasphere_job_syncs_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TRAIN_STATUS_PATH", str(tmp_path / "job_status.json"))
    monkeypatch.setenv("JOB_PROGRESS_FILENAME", str(tmp_path / "job_progress.json"))
    sync_calls: list[None] = []

    def fake_sync() -> None:
        sync_calls.append(None)

    monkeypatch.setattr(os, "sync", fake_sync, raising=False)
    assert finalize_datasphere_job(0) == 0
    assert sync_calls
