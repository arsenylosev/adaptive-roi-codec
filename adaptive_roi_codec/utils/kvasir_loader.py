"""Kvasir-Capsule dataset loader (Object Storage / local debug paths)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def discover_videos(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. "
            "Upload Kvasir-Capsule to Object Storage and mount it via s3-mounts."
        )
    videos = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No video files found under {root}")
    return videos


class SyntheticFrameDataset(Dataset[torch.Tensor]):
    """Lightweight dataset for smoke tests when real frames are unavailable."""

    def __init__(self, length: int = 64, height: int = 1080, width: int = 1920) -> None:
        self.length = length
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(index)
        return torch.rand(3, self.height, self.width, generator=generator)


class VideoIndexDataset(Dataset[Path]):
    """Index of capsule endoscopy videos for frame extraction pipelines."""

    def __init__(self, root: Path) -> None:
        self.videos = discover_videos(root)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Path:
        return self.videos[index]


def iter_video_paths(root: Path) -> Iterator[Path]:
    for path in discover_videos(root):
        yield path
