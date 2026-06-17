"""DataSphere Jobs progress reporting."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


def progress_timestamp() -> str:
    """Return a DataSphere-compatible UTC timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def report_job_progress(progress: int, message: str) -> None:
    """Write the latest progress snapshot to JOB_PROGRESS_FILENAME.

    DataSphere sets this path to a ``*.json`` file under the job log directory.
    Overwriting with one JSON object per update avoids parser failures that can
    happen when many JSONL lines are appended to a ``.json`` file.
    """
    progress_file = os.getenv("JOB_PROGRESS_FILENAME")
    if not progress_file:
        return

    entry = {
        "progress": max(0, min(100, int(progress))),
        "message": message,
        "create_time": progress_timestamp(),
    }
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class JobProgressTracker:
    """Map pipeline phases to DataSphere's 0–100 progress bar."""

    SETUP_END = 8
    STAGING_END = 42
    TRAINING_END = 99

    def __init__(self, *, total_batches: int, epochs: int = 1) -> None:
        self.total_batches = max(total_batches, 1)
        self.epochs = max(epochs, 1)
        self._last_progress = -1
        self._training_report_stride = max(1, self.total_batches // 20)

    def _emit(self, progress: int, message: str, *, force: bool = False) -> None:
        clamped = max(0, min(100, int(progress)))
        if not force and clamped <= self._last_progress and clamped < 100:
            return
        self._last_progress = clamped
        report_job_progress(clamped, message)

    def setup(self, message: str) -> None:
        self._emit(1, message, force=True)

    def staging(self, completed: int, total: int) -> None:
        if total <= 0:
            return
        if completed not in {1, total} and completed % max(1, total // 10) != 0:
            return
        span = self.STAGING_END - self.SETUP_END
        progress = self.SETUP_END + int(span * completed / total)
        self._emit(
            progress,
            f"Staging frames to local SSD: {completed}/{total}",
            force=completed in {1, total},
        )

    def training(self, *, epoch: int, batch: int) -> None:
        if batch not in {1, self.total_batches} and batch % self._training_report_stride != 0:
            return
        epoch_fraction = (epoch - 1 + batch / self.total_batches) / self.epochs
        span = self.TRAINING_END - self.STAGING_END
        progress = self.STAGING_END + int(span * epoch_fraction)
        self._emit(
            progress,
            f"Training epoch {epoch}/{self.epochs} batch {batch}/{self.total_batches}",
            force=batch in {1, self.total_batches},
        )

    def finalize(self, message: str) -> None:
        self._emit(self.TRAINING_END, message, force=True)

    def complete(self, message: str) -> None:
        self._emit(100, message, force=True)
