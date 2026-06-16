"""DataSphere Jobs progress reporting."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


def report_job_progress(progress: int, message: str) -> None:
    """Append a progress record to JOB_PROGRESS_FILENAME when running in DataSphere."""
    progress_file = os.getenv("JOB_PROGRESS_FILENAME")
    if not progress_file:
        return

    entry = {
        "progress": max(0, min(100, int(progress))),
        "message": message,
        "create_time": datetime.now(UTC).isoformat(),
    }
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
