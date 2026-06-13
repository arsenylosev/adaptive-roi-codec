"""Differentiable SSIM soft approximation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)


def ssim_soft(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> torch.Tensor:
    channel = pred.size(1)
    window = _gaussian_window(window_size, sigma, pred.device)
    window = window.expand(channel, 1, window_size, window_size)

    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (numerator / denominator).mean()
