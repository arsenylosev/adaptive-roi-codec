"""Clinically-oriented loss function (formula 14, paper [2])."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from adaptive_roi_codec.losses.ssim import ssim_soft
from adaptive_roi_codec.utils.metrics import psnr


@dataclass
class LossWeights:
    alpha: float = 0.5
    lambda_roi: float = 1.5
    lambda_rate: float = 0.01
    lambda_temp: float = 0.1
    beta: float = 0.01


class ClinicalLoss(nn.Module):
    """L_total = L_base + λ_ROI·L_ROI + λ_rate·L_rate + λ_temp·L_temp."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def l_base(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        psnr_term = -psnr(pred, target)
        ssim_term = 1.0 - ssim_soft(pred, target)
        return self.weights.alpha * psnr_term + (1.0 - self.weights.alpha) * ssim_term

    def l_roi(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_mean = mask.mean(dim=1, keepdim=True)
        diff = (pred - target) ** 2
        weighted = (mask_mean * diff).sum(dim=(1, 2, 3))
        denom = mask_mean.sum(dim=(1, 2, 3)) + 1e-6
        return (weighted / denom).mean()

    def l_rate(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3))
        return kl.mean()

    def l_temp(
        self,
        recon: torch.Tensor,
        prev_recon: torch.Tensor | None,
        frame: torch.Tensor,
        prev_frame: torch.Tensor | None,
        warped_prev: torch.Tensor | None,
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prev_recon is None or prev_frame is None or warped_prev is None:
            return torch.zeros((), device=recon.device)
        motion_term = ((recon - prev_recon) - (frame - prev_frame)).pow(2).mean(dim=(1, 2, 3))
        warp_term = (warped_prev - prev_recon).pow(2).mean(dim=(1, 2, 3))
        per_sample = motion_term + warp_term
        if temporal_mask is not None:
            mask = temporal_mask.float()
            return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
        return per_sample.mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor,
        prev_frame: torch.Tensor | None = None,
        prev_recon: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        recon = outputs["recon"]
        base = self.l_base(recon, target)
        roi = self.l_roi(recon, target, mask)
        rate = self.l_rate(outputs["mu"], outputs["logvar"])
        temp = self.l_temp(
            recon,
            prev_recon,
            target,
            prev_frame,
            outputs.get("warped_prev"),
            temporal_mask=temporal_mask,
        )
        total = (
            base
            + self.weights.lambda_roi * roi
            + self.weights.lambda_rate * self.weights.beta * rate
            + self.weights.lambda_temp * temp
        )
        return {
            "total": total,
            "base": base.detach(),
            "roi": roi.detach(),
            "rate": rate.detach(),
            "temp": temp.detach(),
        }
