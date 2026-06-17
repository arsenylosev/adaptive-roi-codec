"""Kvasir-Capsule dataset loader for 336×336 capsule endoscopy videos."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from adaptive_roi_codec.utils.frame_io import (
    PREPROCESSED_MANIFEST,
    frame_to_chw_numpy,
    load_preprocessed_frame_array,
)
from adaptive_roi_codec.utils.video_index import (
    DEFAULT_FRAME_SIZE,
    VIDEO_EXTENSIONS,
    discover_videos,
    filter_videos,
    load_video_ids,
    resolve_splits_dir,
    resolve_video_dir,
)

logger = logging.getLogger(__name__)


def frame_to_tensor(frame_bgr, height: int, width: int) -> torch.Tensor:
    return torch.from_numpy(frame_to_chw_numpy(frame_bgr, height, width))


def load_frame_tensor(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = load_preprocessed_frame_array(path)
        return torch.from_numpy(np.array(array, dtype=np.float32))
    if suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(f"Unsupported preprocessed frame format: {path}")


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


def resolve_frame_cache_path(frames_root: Path, raw_path: Path) -> Path:
    if raw_path.is_file():
        return raw_path
    relative = frames_root / raw_path
    if relative.is_file():
        return relative
    raise FileNotFoundError(f"Frame cache not found: {raw_path} (frames_root={frames_root})")


def load_preprocessed_manifest(manifest_path: Path, *, frames_root: Path | None = None) -> list[FrameRecord]:
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
            path = Path(row["path"])
            prev_path = Path(prev_raw) if prev_raw else None
            if frames_root is not None:
                path = resolve_frame_cache_path(frames_root, path)
                if prev_path is not None:
                    prev_path = resolve_frame_cache_path(frames_root, prev_path)
            records.append(
                FrameRecord(
                    video_id=row["video_id"],
                    frame_index=int(row["frame_index"]),
                    path=path,
                    prev_path=prev_path,
                )
            )
    if not records:
        raise FileNotFoundError(f"No frame records in {manifest_path}")
    return records


def stage_frame_records(
    records: list[FrameRecord],
    frames_root: Path,
    cache_root: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_every: int = 50,
) -> list[FrameRecord]:
    """Copy referenced frame files from S3 FUSE to local SSD for faster random reads."""
    cache_root.mkdir(parents=True, exist_ok=True)
    localized: dict[Path, Path] = {}
    total = len(records)

    def localize(source: Path) -> Path:
        return localize_frame_path(source, frames_root, cache_root, localized)

    staged: list[FrameRecord] = []
    for index, record in enumerate(records, start=1):
        staged.append(
            FrameRecord(
                video_id=record.video_id,
                frame_index=record.frame_index,
                path=localize(record.path),
                prev_path=localize(record.prev_path) if record.prev_path is not None else None,
            )
        )
        if progress_callback and (index == total or index % progress_every == 0):
            progress_callback(index, total)

    logger.info("Staged %s frame records under %s", len(staged), cache_root)
    return staged


def localize_frame_path(
    source: Path,
    frames_root: Path,
    cache_root: Path,
    localized: dict[Path, Path],
) -> Path:
    """Copy a single frame to local SSD on first access (deduplicated by source path)."""
    if source in localized:
        return localized[source]
    try:
        relative = source.relative_to(frames_root)
    except ValueError:
        relative = Path(source.name)
    destination = cache_root / relative
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            if exc.errno == 28:
                raise OSError(
                    exc.errno,
                    f"{exc.strerror} while staging {source} -> {destination}. "
                    "Reduce max_frames, disable stage_frames_local, or use a custom "
                    "Docker image so pip/venv does not fill the root filesystem.",
                ) from exc
            raise
    localized[source] = destination
    return destination


class KvasirPreprocessedFrameDataset(Dataset[FrameSample]):
    """Load 336×336 frames from pre-extracted `.npy` or legacy `.pt` caches (stage-1 output)."""

    def __init__(
        self,
        frames_root: Path,
        *,
        split: str = "train",
        dataset_root: Path | None = None,
        splits_subdir: str = "splits",
        video_ids: list[str] | None = None,
        max_frames: int | None = None,
        stage_frames_local: bool = False,
        local_frame_cache: Path | None = None,
        staging_progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.frames_root = frames_root
        manifest_path = frames_root / PREPROCESSED_MANIFEST
        records = load_preprocessed_manifest(manifest_path, frames_root=frames_root)

        if video_ids is None and dataset_root and split:
            split_file = resolve_splits_dir(dataset_root, splits_subdir) / f"{split}_videos.txt"
            if split_file.exists():
                video_ids = load_video_ids(split_file)

        if video_ids:
            allowed = set(video_ids)
            records = [record for record in records if record.video_id in allowed]

        if max_frames is not None:
            records = records[: int(max_frames)]

        if not records:
            raise FileNotFoundError(f"No preprocessed frames under {frames_root} for split={split!r}")

        self.stage_frames_local = bool(stage_frames_local and local_frame_cache is not None)
        self.local_frame_cache = local_frame_cache
        self._localized_paths: dict[Path, Path] = {}
        self._staging_progress_callback = staging_progress_callback
        self._staging_progress_every = max(1, len(records) // 20)
        self._staging_completed = 0

        if self.stage_frames_local and local_frame_cache is not None:
            local_frame_cache.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Lazy local staging enabled: frames copy to %s on first dataloader access",
                local_frame_cache,
            )

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_frame_path(self, source: Path) -> Path:
        if not self.stage_frames_local or self.local_frame_cache is None:
            return source
        had = len(self._localized_paths)
        resolved = localize_frame_path(
            source,
            self.frames_root,
            self.local_frame_cache,
            self._localized_paths,
        )
        if len(self._localized_paths) > had:
            self._staging_completed += 1
            callback = self._staging_progress_callback
            total = len(self.records)
            if callback and (
                self._staging_completed == 1
                or self._staging_completed == total
                or self._staging_completed % self._staging_progress_every == 0
            ):
                callback(self._staging_completed, total)
        return resolved

    def __getitem__(self, index: int) -> FrameSample:
        record = self.records[index]
        frame_path = self._resolve_frame_path(record.path)
        tensor = load_frame_tensor(frame_path)
        prev = None
        if record.prev_path is not None:
            prev_path = self._resolve_frame_path(record.prev_path)
            prev = load_frame_tensor(prev_path)
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
