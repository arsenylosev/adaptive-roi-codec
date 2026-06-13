"""Environment variable helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"


def load_project_env() -> None:
    """Load `.env` from the repository root if present."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Set it in {ENV_FILE} (see .env.example)."
        )
    return value


def optional_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def s3_mount_root(connector_id: str) -> Path:
    """Return the Datasphere S3 mount path for a connector."""
    return Path("/job/s3") / connector_id
