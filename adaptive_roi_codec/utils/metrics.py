"""Evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(torch.tensor(max_val**2, device=pred.device) / mse.clamp_min(1e-8))


def dice_roi(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    return (2 * intersection + eps) / (union + eps)
