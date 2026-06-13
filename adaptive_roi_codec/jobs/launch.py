"""Launch training jobs on Yandex Datasphere."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from string import Template

import yaml

from adaptive_roi_codec.utils.env import ENV_FILE, REPO_ROOT, load_project_env, require_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE = REPO_ROOT / "jobs" / "configs" / "job_train.yaml.template"
GENERATED_DIR = REPO_ROOT / "jobs" / "configs" / ".generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Datasphere Jobs config and launch GPU training"
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Job config template path",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=REPO_ROOT / "jobs" / "inputs" / "train_input.json",
        help="JSON file passed to the training script as job input",
    )
    parser.add_argument(
        "--config-name",
        default="job_train.yaml",
        help="Output filename inside jobs/configs/.generated/",
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Datasphere project ID (defaults to DATASPHERE_PROJECT_ID from .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render config only; do not call datasphere CLI",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call `datasphere project job execute` after rendering",
    )
    return parser.parse_args()


def render_template(template_path: Path, context: dict[str, str]) -> str:
    raw = template_path.read_text(encoding="utf-8")
    return Template(raw).safe_substitute(context)


def validate_inputs(params_path: Path) -> None:
    if not params_path.exists():
        raise FileNotFoundError(f"Job params file not found: {params_path}")
    with params_path.open(encoding="utf-8") as handle:
        json.load(handle)


def build_context(args: argparse.Namespace) -> dict[str, str]:
    load_project_env()
    project_id = args.project_id or require_env("DATASPHERE_PROJECT_ID")
    s3_connector_id = require_env("S3_CONNECTOR_ID")

    return {
        "JOB_NAME": os.getenv("DATASPHERE_JOB_NAME", "vae-capsule-train"),
        "JOB_DESC": os.getenv(
            "DATASPHERE_JOB_DESC",
            "VAE codec training on Kvasir-Capsule with adaptive ROI quantization",
        ),
        "DATASPHERE_PROJECT_ID": project_id,
        "S3_CONNECTOR_ID": s3_connector_id,
        "S3_DATA_PREFIX": os.getenv("S3_DATA_PREFIX", "kvasir-capsule"),
        "S3_CHECKPOINT_SUBDIR": os.getenv("S3_CHECKPOINT_SUBDIR", "checkpoints"),
        "WORKING_STORAGE_GB": os.getenv("DATASPHERE_WORKING_STORAGE_GB", "150"),
    }


def write_generated_config(content: str, config_name: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_DIR / config_name
    output_path.write_text(content, encoding="utf-8")
    yaml.safe_load(content)
    logger.info("Rendered job config: %s", output_path)
    return output_path


def execute_job(project_id: str, config_path: Path) -> None:
    cmd = [
        "datasphere",
        "project",
        "job",
        "execute",
        "-p",
        project_id,
        "-c",
        str(config_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()

    if not args.template.exists():
        raise FileNotFoundError(f"Template not found: {args.template}")
    if not ENV_FILE.exists():
        logger.warning(
            "No .env file at %s — copy .env.example and fill in Yandex Cloud values",
            ENV_FILE,
        )

    validate_inputs(args.params)
    context = build_context(args)
    rendered = render_template(args.template, context)
    config_path = write_generated_config(rendered, args.config_name)

    if args.dry_run and not args.execute:
        logger.info("Dry run complete. Review %s and run with --execute", config_path)
        return

    if args.execute:
        execute_job(context["DATASPHERE_PROJECT_ID"], config_path)
        return

    logger.info(
        "Config rendered. Launch with:\n"
        "  uv run launch-train --execute\n"
        "or:\n"
        "  datasphere project job execute -p %s -c %s",
        context["DATASPHERE_PROJECT_ID"],
        config_path,
    )


if __name__ == "__main__":
    main()
