"""Clean shutdown helpers for DataSphere Jobs."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

from adaptive_roi_codec.utils.job_progress import progress_timestamp, report_job_progress

logger = logging.getLogger(__name__)


def terminate_multiprocessing_children() -> None:
    """Stop spawned/forked child processes before the job container exits."""
    for child in mp.active_children():
        if child.is_alive():
            logger.info("Terminating child process pid=%s name=%s", child.pid, child.name)
            child.terminate()

    for child in mp.active_children():
        child.join(timeout=5.0)

    for child in mp.active_children():
        if child.is_alive():
            logger.warning("Killing child process pid=%s name=%s", child.pid, child.name)
            child.kill()
            child.join(timeout=2.0)


def release_torch_cuda() -> None:
    """Drop CUDA allocations before interpreter shutdown."""
    try:
        import gc

        import torch
    except ImportError:
        return

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            logger.warning("torch.cuda.synchronize() failed during cleanup")
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.warning("torch.cuda.empty_cache() failed during cleanup")
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except Exception:
                logger.warning("torch.cuda.ipc_collect() failed during cleanup")


def write_job_status(
    status: str,
    *,
    exit_code: int,
    metrics_path: Path | None = None,
    message: str = "",
) -> None:
    """Persist a small status artifact DataSphere can collect as an output."""
    status_path = Path(os.getenv("TRAIN_STATUS_PATH", "job_status.json"))
    payload = {
        "status": status,
        "exit_code": int(exit_code),
        "message": message,
        "create_time": progress_timestamp(),
    }
    if metrics_path is not None:
        payload["metrics_path"] = str(metrics_path)

    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    logger.info("Wrote job status to %s (%s)", status_path, status)


def finalize_datasphere_job(
    exit_code: int = 0,
    *,
    metrics_path: Path | None = None,
    message: str = "",
) -> int:
    """Flush artifacts and return the process exit code for a normal interpreter shutdown."""
    terminate_multiprocessing_children()
    status = "success" if exit_code == 0 else "failed"
    write_job_status(status, exit_code=exit_code, metrics_path=metrics_path, message=message)
    report_job_progress(
        100 if exit_code == 0 else 99,
        message or ("Job finished successfully" if exit_code == 0 else "Job failed"),
    )
    release_torch_cuda()
    sys.stdout.flush()
    sys.stderr.flush()
    if hasattr(os, "sync"):
        os.sync()
    return exit_code
