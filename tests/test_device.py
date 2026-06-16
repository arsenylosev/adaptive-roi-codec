"""Tests for training device resolution."""

import pytest

from adaptive_roi_codec.utils import device as device_module


def test_require_cuda_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIN_REQUIRE_CUDA", "1")
    monkeypatch.setenv("TRAIN_DEVICE", "cuda")
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_module.torch.cuda, "_lazy_init_error", "driver too old", raising=False)

    with pytest.raises(RuntimeError, match="TRAIN_REQUIRE_CUDA"):
        device_module.resolve_training_device(require_cuda=True)


def test_cpu_fallback_when_cuda_unavailable_without_require(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIN_REQUIRE_CUDA", raising=False)
    monkeypatch.setenv("TRAIN_DEVICE", "cuda")
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)

    resolved = device_module.resolve_training_device(require_cuda=False)
    assert resolved.type == "cpu"
