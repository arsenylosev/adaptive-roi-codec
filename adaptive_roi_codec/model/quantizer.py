"""Adaptive quantizer (formulas 4–6 from paper [3])."""

from __future__ import annotations

import torch


class AdaptiveQuantizer:
    """Spatially adaptive latent quantization."""

    def __init__(
        self,
        q_min: float = 0.1,
        q_max: float = 2.0,
        kappa: float = 2.0,
        alpha_spatial: float = 0.5,
    ) -> None:
        self.q_min = q_min
        self.q_max = q_max
        self.kappa = kappa
        self.alpha_spatial = alpha_spatial

    def roi_fraction(self, mask: torch.Tensor) -> torch.Tensor:
        # E_ROI(t): average ROI activation over spatial dimensions (and channels).
        return mask.flatten(1).mean(dim=1)

    def global_step(self, mask: torch.Tensor) -> torch.Tensor:
        e_roi = self.roi_fraction(mask)
        return self.q_min + (self.q_max - self.q_min) * torch.pow(e_roi, self.kappa)

    def spatial_step(self, mask: torch.Tensor) -> torch.Tensor:
        q_t = self.global_step(mask).view(-1, 1, 1, 1)
        mask_spatial = mask.mean(dim=1, keepdim=True)
        return q_t * (1.0 + self.alpha_spatial * mask_spatial)

    def quantize(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        step = self.spatial_step(mask)
        if step.shape[-2:] != latent.shape[-2:]:
            step = torch.nn.functional.interpolate(
                step,
                size=latent.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.round(latent / step) * step
