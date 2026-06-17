"""Launch training jobs on Yandex Datasphere."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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
EXTRACT_TEMPLATE = REPO_ROOT / "jobs" / "configs" / "job_extract_frames.yaml.template"
GENERATED_DIR = REPO_ROOT / "jobs" / "configs" / ".generated"
GENERATED_TRAIN_PARAMS = GENERATED_DIR / "train_params.json"
DEFAULT_TRAIN_JOB_NAME = "vae-capsule-train"
DEFAULT_SMOKE_TRAIN_JOB_NAME = "vae-capsule-smoke-train"
DEFAULT_SMOKE_TRAIN_JOB_DESC = (
    "GPU smoke training: preprocessed Kvasir frames, capped batches for pipeline validation"
)
DEFAULT_FULL_TRAIN_JOB_DESC = (
    "VAE codec training on Kvasir-Capsule with adaptive ROI quantization"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Datasphere Jobs config and launch GPU training"
    )
    parser.add_argument(
        "--job",
        choices=("train", "extract"),
        default="train",
        help="Job template: GPU training (train) or CPU frame extraction (extract)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Job config template path (overrides --job default)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=REPO_ROOT / "jobs" / "inputs" / "train_input.json",
        help="JSON file passed to the training script as job input",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Override training.batch_size in job params (written to jobs/configs/.generated/train_params.json)",
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
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Block until the job finishes (default: submit with --async and return)",
    )
    parser.add_argument(
        "--async",
        dest="async_submit",
        action="store_true",
        help="Force datasphere --async even for smoke params (overrides smoke sync default)",
    )
    parser.add_argument(
        "--async-output",
        type=Path,
        default=GENERATED_DIR / "last_job_execute.json",
        help="Path for datasphere --async -o metadata (default: jobs/configs/.generated/last_job_execute.json)",
    )
    return parser.parse_args()


def render_template(template_path: Path, context: dict[str, str]) -> str:
    raw = template_path.read_text(encoding="utf-8")
    return Template(raw).safe_substitute(context)


def yaml_quote(value: str) -> str:
    """Return a YAML double-quoted scalar (safe for colons and spaces)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def resolve_datasphere_cli() -> str:
    """Return path to the datasphere CLI from the active Python environment."""
    venv_candidate = Path(sys.executable).resolve().parent / "datasphere"
    if venv_candidate.is_file():
        return str(venv_candidate)

    on_path = shutil.which("datasphere")
    if on_path:
        return on_path

    raise FileNotFoundError(
        "datasphere CLI not found. Install operator dependencies with:\n"
        "  uv sync --extra cloud --extra dev"
    )


def validate_inputs(params_path: Path) -> None:
    if not params_path.exists():
        raise FileNotFoundError(f"Job params file not found: {params_path}")
    with params_path.open(encoding="utf-8") as handle:
        json.load(handle)


def load_params(params_path: Path) -> dict:
    with params_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_batch_size(params_path: Path, cli_batch_size: int | None) -> int | None:
    if cli_batch_size is not None:
        return cli_batch_size
    batch_size = load_params(params_path).get("training", {}).get("batch_size")
    return int(batch_size) if batch_size is not None else None


def materialize_train_params(params_path: Path, batch_size: int | None) -> Path:
    if batch_size is None:
        return params_path
    params = load_params(params_path)
    params.setdefault("training", {})["batch_size"] = batch_size
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_TRAIN_PARAMS.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    logger.info("Rendered train params with batch_size=%s: %s", batch_size, GENERATED_TRAIN_PARAMS)
    return GENERATED_TRAIN_PARAMS


def is_smoke_train_params(params: dict) -> bool:
    if params.get("job_profile") == "smoke":
        return True
    training = params.get("training", {})
    data = params.get("data", {})
    return training.get("max_train_batches") is not None or data.get("max_frames") is not None


def resolve_train_job_base_name(params_path: Path) -> str:
    params = load_params(params_path)
    if is_smoke_train_params(params):
        return DEFAULT_SMOKE_TRAIN_JOB_NAME
    env_name = os.getenv("DATASPHERE_JOB_NAME")
    if env_name:
        return env_name
    return DEFAULT_TRAIN_JOB_NAME


def resolve_train_job_base_desc(params_path: Path) -> str:
    params = load_params(params_path)
    if is_smoke_train_params(params):
        return DEFAULT_SMOKE_TRAIN_JOB_DESC
    env_desc = os.getenv("DATASPHERE_JOB_DESC")
    if env_desc:
        return env_desc
    return DEFAULT_FULL_TRAIN_JOB_DESC


def format_train_job_name(base_name: str, batch_size: int | None) -> str:
    if batch_size is None:
        return base_name
    return f"{base_name}-bs{batch_size}"


def format_train_job_desc(base_desc: str, batch_size: int | None) -> str:
    if batch_size is None:
        return base_desc
    return f"{base_desc} (batch_size={batch_size})"


def params_input_path(params_path: Path) -> str:
    """Return repo-relative path for DataSphere job inputs."""
    resolved = params_path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Params file must live inside the repository to upload as a job input: {params_path}"
        ) from exc


def build_context(args: argparse.Namespace) -> dict[str, str]:
    load_project_env()
    project_id = args.project_id or require_env("DATASPHERE_PROJECT_ID")
    s3_connector_id = require_env("S3_CONNECTOR_ID")
    s3_prefix = os.getenv("S3_DATA_PREFIX", "kvasir-capsule")
    frames_mount = f"/job/s3/{s3_connector_id}/{s3_prefix}/processed/frames"

    batch_size = None
    if args.job == "extract":
        job_name = os.getenv("DATASPHERE_EXTRACT_JOB_NAME", "kvasir-extract-frames")
        job_desc = os.getenv(
            "DATASPHERE_EXTRACT_JOB_DESC",
            "CPU stage-1: decode Kvasir MP4s to .pt tensors on S3",
        )
        params_path = args.params
    else:
        batch_size = resolve_batch_size(args.params, args.batch_size)
        params_path = materialize_train_params(args.params, args.batch_size)
        base_name = resolve_train_job_base_name(params_path)
        base_desc = resolve_train_job_base_desc(params_path)
        job_name = format_train_job_name(base_name, batch_size)
        job_desc = format_train_job_desc(base_desc, batch_size)

    return {
        "JOB_NAME": yaml_quote(job_name),
        "JOB_DESC": yaml_quote(job_desc),
        "PARAMS_INPUT": params_input_path(params_path),
        "TRAIN_BATCH_SIZE": str(batch_size) if batch_size is not None else "",
        "DATASPHERE_PROJECT_ID": project_id,
        "S3_CONNECTOR_ID": s3_connector_id,
        "S3_DATA_PREFIX": s3_prefix,
        "S3_CHECKPOINT_SUBDIR": os.getenv("S3_CHECKPOINT_SUBDIR", "checkpoints"),
        "FRAMES_S3_DIR": frames_mount,
        "WORKING_STORAGE_GB": os.getenv("DATASPHERE_WORKING_STORAGE_GB", "150"),
    }


def write_generated_config(content: str, config_name: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_DIR / config_name
    output_path.write_text(content, encoding="utf-8")
    yaml.safe_load(content)
    logger.info("Rendered job config: %s", output_path)
    return output_path


def execute_job(
    project_id: str,
    config_path: Path,
    *,
    async_mode: bool = True,
    async_output: Path | None = None,
) -> None:
    datasphere_cli = resolve_datasphere_cli()
    cmd = [
        datasphere_cli,
        "project",
        "job",
        "execute",
        "-p",
        project_id,
        "-c",
        str(config_path),
    ]
    if async_mode:
        output_path = async_output or (GENERATED_DIR / "last_job_execute.json")
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--async", "-o", str(output_path)])
        logger.info("Submitting job asynchronously; metadata will be written to %s", output_path)
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if async_mode:
        logger.info(
            "Job submitted. Track progress in DataSphere UI → Jobs → Run history, "
            "or inspect %s",
            async_output or (GENERATED_DIR / "last_job_execute.json"),
        )


def main() -> None:
    args = parse_args()

    if args.template is None:
        args.template = EXTRACT_TEMPLATE if args.job == "extract" else DEFAULT_TEMPLATE
    if args.config_name == "job_train.yaml" and args.job == "extract":
        args.config_name = "job_extract_frames.yaml"
    if args.params == REPO_ROOT / "jobs" / "inputs" / "train_input.json" and args.job == "extract":
        extract_params = REPO_ROOT / "jobs" / "inputs" / "extract_input.json"
        if extract_params.exists():
            args.params = extract_params

    if not args.template.exists():
        raise FileNotFoundError(f"Template not found: {args.template}")
    if not ENV_FILE.exists():
        logger.warning(
            "No .env file at %s — copy .env.example and fill in Yandex Cloud values",
            ENV_FILE,
        )

    validate_inputs(args.params)
    if args.job == "train" and args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    context = build_context(args)
    rendered = render_template(args.template, context)
    config_path = write_generated_config(rendered, args.config_name)

    if args.dry_run and not args.execute:
        logger.info("Dry run complete. Review %s and run with --execute", config_path)
        return

    if args.execute:
        try:
            resolve_datasphere_cli()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        params_for_mode = load_params(materialize_train_params(args.params, args.batch_size))
        if args.sync and args.async_submit:
            raise SystemExit("Cannot use --sync and --async together")
        if args.sync:
            wait_for_completion = True
        elif args.async_submit:
            wait_for_completion = False
        else:
            wait_for_completion = args.job == "train" and is_smoke_train_params(params_for_mode)
        execute_job(
            context["DATASPHERE_PROJECT_ID"],
            config_path,
            async_mode=not wait_for_completion,
            async_output=args.async_output,
        )
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
