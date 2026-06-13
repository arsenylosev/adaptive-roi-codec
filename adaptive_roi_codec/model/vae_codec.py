"""VAE codec with motion compensation (paper [1] skeleton)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 192) -> None:
        super().__init__()
        self.down1 = ConvBlock(in_channels, 64, stride=2)
        self.down2 = ConvBlock(64, 128, stride=2)
        self.down3 = ConvBlock(128, 256, stride=2)
        self.down4 = ConvBlock(256, 512, stride=2)
        self.mu = nn.Conv2d(512, latent_channels, kernel_size=1)
        self.logvar = nn.Conv2d(512, latent_channels, kernel_size=1)
        self.skips: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        self.skips = []
        x = self.down1(x)
        self.skips.append(x)
        x = self.down2(x)
        self.skips.append(x)
        x = self.down3(x)
        self.skips.append(x)
        x = self.down4(x)
        self.skips.append(x)
        return self.mu(x), self.logvar(x), self.skips


class VAEDecoder(nn.Module):
    def __init__(self, out_channels: int = 3, latent_channels: int = 192) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(latent_channels, 512, kernel_size=4, stride=2, padding=1)
        self.block1 = ConvBlock(512 + 512, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.block2 = ConvBlock(256 + 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.block3 = ConvBlock(128 + 128, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.block4 = ConvBlock(64 + 64, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, z: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = self.up1(z)
        x = self.block1(torch.cat([x, self._resize_to(x, skips[3])], dim=1))
        x = self.up2(x)
        x = self.block2(torch.cat([x, self._resize_to(x, skips[2])], dim=1))
        x = self.up3(x)
        x = self.block3(torch.cat([x, self._resize_to(x, skips[1])], dim=1))
        x = self.up4(x)
        x = self.block4(torch.cat([x, self._resize_to(x, skips[0])], dim=1))
        return torch.sigmoid(self.out(x))

    @staticmethod
    def _resize_to(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[-2:] == target.shape[-2:]:
            return target
        return F.interpolate(target, size=source.shape[-2:], mode="bilinear", align_corners=False)


class MotionCompensator(nn.Module):
    """Predicts a simple affine warp between consecutive reconstructions."""

    def __init__(self, latent_channels: int = 192) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(latent_channels * 2, 6)

    def forward(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([self.pool(z_prev), self.pool(z_curr)], dim=1).flatten(1)
        theta = self.fc(pooled).view(-1, 2, 3)
        identity = torch.tensor([1, 0, 0, 0, 1, 0], device=z_curr.device, dtype=z_curr.dtype)
        return theta + identity.view(1, 2, 3)


class VAECodec(nn.Module):
    """End-to-end VAE codec with optional temporal warping."""

    def __init__(self, latent_channels: int = 192) -> None:
        super().__init__()
        self.encoder = VAEEncoder(latent_channels=latent_channels)
        self.decoder = VAEDecoder(latent_channels=latent_channels)
        self.motion = MotionCompensator(latent_channels=latent_channels)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        frame: torch.Tensor,
        prev_recon: torch.Tensor | None = None,
        prev_z: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mu, logvar, skips = self.encoder(frame)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, skips)

        warped_prev = prev_recon
        if prev_recon is not None and prev_z is not None:
            theta = self.motion(prev_z, z)
            grid = F.affine_grid(theta, prev_recon.size(), align_corners=False)
            warped_prev = F.grid_sample(prev_recon, grid, align_corners=False)

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "warped_prev": warped_prev,
        }
