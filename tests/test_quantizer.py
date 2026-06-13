"""Tests for adaptive quantizer tensor shapes."""

import torch

from adaptive_roi_codec.model.quantizer import AdaptiveQuantizer


def test_quantize_supports_batch_size_greater_than_one() -> None:
    quantizer = AdaptiveQuantizer()
    batch_size = 4
    mask = torch.rand(batch_size, 3, 64, 64)
    latent = torch.randn(batch_size, 192, 8, 12)

    quantized = quantizer.quantize(latent, mask)

    assert quantized.shape == latent.shape


def test_global_step_is_scalar_per_batch_item() -> None:
    quantizer = AdaptiveQuantizer()
    mask = torch.rand(3, 3, 32, 32)

    steps = quantizer.global_step(mask)

    assert steps.shape == (3,)
