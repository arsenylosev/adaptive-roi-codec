"""Device selection for training (local and Datasphere Jobs)."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def resolve_training_device(require_cuda: bool | None = None) -> torch.device:
    """Pick training device; fail fast on GPU jobs when CUDA is unavailable."""
    if require_cuda is None:
        require_cuda = os.getenv("TRAIN_REQUIRE_CUDA", "").lower() in {"1", "true", "yes"}

    prefer = os.getenv("TRAIN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(prefer)

    if device.type == "cuda" and not torch.cuda.is_available():
        cuda_error = getattr(torch.cuda, "_lazy_init_error", None)
        if require_cuda:
            raise RuntimeError(
                "TRAIN_REQUIRE_CUDA is set but PyTorch cannot use CUDA. "
                f"torch.cuda.is_available()=False. {cuda_error or ''} "
                "On Datasphere Jobs, pin PyTorch to a CUDA wheel matching the VM driver "
                "(see jobs/requirements-datasphere-gpu.txt)."
            )
        logger.warning(
            "TRAIN_DEVICE=cuda requested but CUDA unavailable (%s); falling back to CPU",
            cuda_error,
        )
        device = torch.device("cpu")

    if require_cuda and device.type != "cuda":
        raise RuntimeError(
            f"TRAIN_REQUIRE_CUDA is set but training device is {device}. "
            "Use a GPU instance type (g1.1, gt4.1) and compatible PyTorch CUDA wheels."
        )

    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (driver %s)", torch.cuda.get_device_name(0), _driver_version())
    elif require_cuda:
        raise RuntimeError("CUDA required but device is CPU")
    return device


def _driver_version() -> str:
    try:
        return str(torch._C._cuda_getDriverVersion())  # noqa: SLF001
    except Exception:
        return "unknown"
