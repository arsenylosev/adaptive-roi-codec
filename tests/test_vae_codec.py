"""Tests for VAE codec spatial alignment."""

import torch

from adaptive_roi_codec.model.vae_codec import VAECodec


def test_forward_reconstruction_matches_input_resolution() -> None:
    codec = VAECodec(latent_channels=32)
    frame = torch.rand(2, 3, 384, 640)

    outputs = codec(frame)

    assert outputs["recon"].shape == frame.shape
