"""ROI detector: U-Net with MobileNetV3-large backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class ROIDetector(nn.Module):
    """Predicts a soft 3-channel significance mask in [0, 1]."""

    def __init__(
        self,
        input_size: int = 336,
        out_channels: int = 3,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.encoder = backbone.features
        self.head = nn.Sequential(
            nn.Conv2d(960, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            frame,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        features = self.encoder(resized)
        logits = self.head(features)
        mask = torch.sigmoid(
            F.interpolate(logits, size=frame.shape[-2:], mode="bilinear", align_corners=False)
        )
        return mask
