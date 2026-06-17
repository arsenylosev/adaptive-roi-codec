"""DataSphere Jobs entrypoint with failsafe process exit."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from adaptive_roi_codec.train import train
from adaptive_roi_codec.utils.datasphere_exit import finalize_datasphere_job
from adaptive_roi_codec.utils.env import load_project_env, optional_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive ROI codec (DataSphere entrypoint)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--params", required=False, help="JSON overrides")
    parser.add_argument("--dry-run", action="store_true", help="Run one batch on synthetic data")
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Metrics JSON output path (DataSphere ${METRICS})",
    )
    parser.add_argument(
        "--status-out",
        default=None,
        help="Job status JSON output path (DataSphere ${JOB_STATUS})",
    )
    return parser.parse_args()


def configure_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    metrics_path = Path(
        args.metrics_out or optional_env("TRAIN_METRICS_PATH", "train_metrics.json")
    )
    status_path = Path(
        args.status_out or optional_env("TRAIN_STATUS_PATH", "job_status.json")
    )
    os.environ["TRAIN_METRICS_PATH"] = str(metrics_path)
    os.environ["TRAIN_STATUS_PATH"] = str(status_path)
    return metrics_path, status_path


def main() -> int:
    load_project_env()
    args = parse_args()
    metrics_path, _status_path = configure_output_paths(args)
    exit_code = 0
    message = "Training finished successfully"
    try:
        train(args.config, args.params, dry_run=args.dry_run)
    except Exception as exc:
        logger.exception("Training failed")
        exit_code = 1
        message = str(exc)

    if os.getenv("JOB_PROGRESS_FILENAME"):
        return finalize_datasphere_job(
            exit_code,
            metrics_path=metrics_path if metrics_path.exists() else None,
            message=message,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
