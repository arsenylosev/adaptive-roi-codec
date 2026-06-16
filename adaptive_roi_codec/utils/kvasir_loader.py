"""Kvasir-Capsule dataset loader for 336×336 capsule endoscopy videos."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_FRAME_SIZE = 336
PREPROCESSED_MANIFEST = "frames_manifest.jsonl"


def discover_videos(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(
            f"Video directory not found: {root}. "
            "Upload Kvasir-Capsule to Object Storage under kvasir-capsule/raw/labelled_videos/."
        )
    videos = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No video files found under {root}")
    return videos


def resolve_video_dir(dataset_root: Path, video_subdir: str | None = None) -> Path:
    """Resolve labelled video directory across local and S3 raw layouts."""
    candidates: list[Path] = []
    if video_subdir:
        candidates.append(dataset_root / video_subdir)
    candidates.extend(
        [
            dataset_root / "raw" / "labelled_videos",
            dataset_root / "labelled_videos",
            dataset_root / "processed" / "videos",
        ]
    )
    for path in candidates:
        if path.is_dir() and any(path.glob("*.mp4")):
            return path
    raise FileNotFoundError(
        f"No labelled videos found under {dataset_root}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def resolve_splits_dir(dataset_root: Path, splits_subdir: str = "splits") -> Path:
    return dataset_root / splits_subdir


def load_video_ids(split_file: Path) -> list[str]:
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. "
            "Run: uv run build-dataset-manifest --dataset-root <path>"
        )
    ids: list[str] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.append(value)
    return ids


def filter_videos(videos: list[Path], video_ids: list[str] | None) -> list[Path]:
    if not video_ids:
        return videos
    allowed = set(video_ids)
    filtered = [path for path in videos if path.stem in allowed]
    if not filtered:
        raise FileNotFoundError("No videos matched the requested split IDs")
    return filtered


def frame_to_tensor(frame_bgr, height: int, width: int) -> torch.Tensor:
    if frame_bgr is None:
        raise ValueError("Received empty frame from video decoder")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != height or rgb.shape[1] != width:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0


@dataclass(frozen=True)
class FrameSample:
    frame: torch.Tensor
    prev_frame: torch.Tensor | None
    video_id: str
    frame_index: int


class SyntheticFrameDataset(Dataset[torch.Tensor]):
    """Lightweight dataset for smoke tests when real frames are unavailable."""

    def __init__(
        self,
        length: int = 64,
        height: int = DEFAULT_FRAME_SIZE,
        width: int = DEFAULT_FRAME_SIZE,
    ) -> None:
        self.length = length
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(index)
        return torch.rand(3, self.height, self.width, generator=generator)


class KvasirVideoFrameDataset(IterableDataset[FrameSample]):
    """Stream consecutive 336×336 frames from Kvasir-Capsule videos."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        split: str = "train",
        frame_stride: int = 30,
        max_frames_per_video: int | None = None,
        height: int = DEFAULT_FRAME_SIZE,
        width: int = DEFAULT_FRAME_SIZE,
        video_subdir: str | None = None,
        splits_subdir: str = "splits",
        video_ids: list[str] | None = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.frame_stride = max(frame_stride, 1)
        self.max_frames_per_video = max_frames_per_video
        self.height = height
        self.width = width
        self.video_dir = resolve_video_dir(dataset_root, video_subdir)
        if video_ids is None and split:
            split_file = resolve_splits_dir(dataset_root, splits_subdir) / f"{split}_videos.txt"
            if split_file.exists():
                video_ids = load_video_ids(split_file)
        self.videos = filter_videos(discover_videos(self.video_dir), video_ids)

    def __iter__(self) -> Iterator[FrameSample]:
        worker_info = torch.utils.data.get_worker_info()
        videos = self.videos
        if worker_info is not None:
            videos = videos[worker_info.id :: worker_info.num_workers]

        for video_path in videos:
            yield from self._iter_video(video_path)

    def _iter_video(self, video_path: Path) -> Iterator[FrameSample]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            logger.warning("Skipping unreadable video: %s", video_path)
            return

        prev_tensor: torch.Tensor | None = None
        frame_index = 0
        yielded = 0
        next_yield = 0

        try:
            while True:
                if frame_index < next_yield:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, next_yield)
                    frame_index = next_yield

                ok, frame_bgr = capture.read()
                if not ok:
                    break

                if frame_index == next_yield:
                    tensor = frame_to_tensor(frame_bgr, self.height, self.width)
                    yield FrameSample(
                        frame=tensor,
                        prev_frame=prev_tensor,
                        video_id=video_path.stem,
                        frame_index=frame_index,
                    )
                    prev_tensor = tensor.detach()
                    yielded += 1
                    next_yield = frame_index + self.frame_stride
                    if self.max_frames_per_video and yielded >= self.max_frames_per_video:
                        break

                frame_index += 1
        finally:
            capture.release()


def collate_frame_samples(batch: list[FrameSample]) -> dict[str, torch.Tensor | list[str]]:
    frames = torch.stack([sample.frame for sample in batch])
    prev_frames = torch.stack(
        [
            sample.prev_frame
            if sample.prev_frame is not None
            else torch.zeros_like(sample.frame)
            for sample in batch
        ]
    )
    has_prev = torch.tensor(
        [sample.prev_frame is not None for sample in batch],
        dtype=torch.bool,
    )
    return {
        "frame": frames,
        "prev_frame": prev_frames,
        "has_prev": has_prev,
        "video_id": [sample.video_id for sample in batch],
        "frame_index": torch.tensor([sample.frame_index for sample in batch]),
    }


@dataclass(frozen=True)
class FrameRecord:
    video_id: str
    frame_index: int
    path: Path
    prev_path: Path | None = None


def load_preprocessed_manifest(manifest_path: Path) -> list[FrameRecord]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Preprocessed manifest not found: {manifest_path}. "
            "Run extract-frames (stage 1) or set data.source=video."
        )
    records: list[FrameRecord] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prev_raw = row.get("prev_path")
            records.append(
                FrameRecord(
                    video_id=row["video_id"],
                    frame_index=int(row["frame_index"]),
                    path=Path(row["path"]),
                    prev_path=Path(prev_raw) if prev_raw else None,
                )
            )
    if not records:
        raise FileNotFoundError(f"No frame records in {manifest_path}")
    return records


class KvasirPreprocessedFrameDataset(Dataset[FrameSample]):
    """Load 336×336 frames from pre-extracted `.pt` tensors (stage-1 output)."""

    def __init__(
        self,
        frames_root: Path,
        *,
        split: str = "train",
        dataset_root: Path | None = None,
        splits_subdir: str = "splits",
        video_ids: list[str] | None = None,
    ) -> None:
        self.frames_root = frames_root
        manifest_path = frames_root / PREPROCESSED_MANIFEST
        records = load_preprocessed_manifest(manifest_path)

        if video_ids is None and dataset_root and split:
            split_file = resolve_splits_dir(dataset_root, splits_subdir) / f"{split}_videos.txt"
            if split_file.exists():
                video_ids = load_video_ids(split_file)

        if video_ids:
            allowed = set(video_ids)
            records = [record for record in records if record.video_id in allowed]

        if not records:
            raise FileNotFoundError(f"No preprocessed frames under {frames_root} for split={split!r}")

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> FrameSample:
        record = self.records[index]
        tensor = torch.load(record.path, map_location="cpu", weights_only=True)
        prev = None
        if record.prev_path is not None:
            prev = torch.load(record.prev_path, map_location="cpu", weights_only=True)
        return FrameSample(
            frame=tensor,
            prev_frame=prev,
            video_id=record.video_id,
            frame_index=record.frame_index,
        )


class VideoIndexDataset(Dataset[Path]):
    """Index of capsule endoscopy videos."""

    def __init__(self, root: Path) -> None:
        self.videos = discover_videos(root)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Path:
        return self.videos[index]


def iter_video_paths(root: Path) -> Iterator[Path]:
    for path in discover_videos(root):
        yield path
