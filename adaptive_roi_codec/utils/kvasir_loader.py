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

DEFAULT_MAX_STAGE_FRAMES = 4800
ROOT_DISK_STAGE_THRESHOLD = 0.85
CACHE_DISK_STAGE_THRESHOLD = 0.90
STAGE_MODE_BULK = "bulk"
STAGE_MODE_LAZY = "lazy"


def root_filesystem_staging_headroom(threshold: float = ROOT_DISK_STAGE_THRESHOLD) -> bool:
    """Return False when the job root overlay is too full for S3 FUSE staging."""
    try:
        usage = shutil.disk_usage("/")
    except OSError:
        return True
    return (usage.used / usage.total) < threshold


def cache_filesystem_staging_headroom(
    cache_path: Path,
    required_bytes: int,
    *,
    threshold: float = CACHE_DISK_STAGE_THRESHOLD,
) -> bool:
    """Return True when ``cache_path``'s filesystem has room for ``required_bytes``."""
    cache_path.mkdir(parents=True, exist_ok=True)
    try:
        usage = shutil.disk_usage(cache_path)
    except OSError:
        return False
    reserved = int(usage.total * threshold) - usage.used
    available = min(usage.free, max(reserved, 0))
    return available >= required_bytes


def resolve_stage_mode(data_cfg: dict) -> str:
    explicit = data_cfg.get("stage_mode")
    if explicit is not None:
        mode = str(explicit).lower()
        if mode not in {STAGE_MODE_BULK, STAGE_MODE_LAZY}:
            raise ValueError(f"Unsupported data.stage_mode: {explicit!r}")
        return mode
    if data_cfg.get("stage_frames_local"):
        return STAGE_MODE_BULK
    return STAGE_MODE_LAZY


def collect_unique_frame_paths(records: list[FrameRecord]) -> set[Path]:
    paths: set[Path] = set()
    for record in records:
        paths.add(record.path)
        if record.prev_path is not None:
            paths.add(record.prev_path)
    return paths


def estimate_staging_bytes(records: list[FrameRecord]) -> int:
    total = 0
    for path in collect_unique_frame_paths(records):
        try:
            total += path.stat().st_size
        except OSError:
            # Conservative fallback for unreadable S3 paths during estimation.
            total += 336 * 336 * 3 * 4
    return total


def resolve_max_staged_files(data_cfg: dict, training_cfg: dict) -> int | None:
    explicit = data_cfg.get("max_staged_files")
    if explicit is not None:
        return int(explicit)
    max_batches = training_cfg.get("max_train_batches")
    batch_size = int(training_cfg.get("batch_size", 1))
    if max_batches is not None:
        return int(max_batches) * batch_size * 2
    max_frames = data_cfg.get("max_frames")
    if max_frames is not None:
        return min(int(max_frames) * 2, DEFAULT_MAX_STAGE_FRAMES)
    return DEFAULT_MAX_STAGE_FRAMES


def should_stage_frames_locally(
    data_cfg: dict,
    training_cfg: dict,
    *,
    cache_path: Path | None,
    records: list[FrameRecord] | None = None,
) -> bool:
    if not data_cfg.get("stage_frames_local") or cache_path is None:
        return False

    stage_mode = resolve_stage_mode(data_cfg)
    max_frames = data_cfg.get("max_frames")

    if stage_mode == STAGE_MODE_LAZY and max_frames is not None and int(max_frames) > DEFAULT_MAX_STAGE_FRAMES:
        logger.warning(
            "Disabling stage_frames_local: max_frames=%s exceeds safe lazy limit %s. "
            "Use stage_mode=bulk to copy frames to extended SSD before training.",
            max_frames,
            DEFAULT_MAX_STAGE_FRAMES,
        )
        return False

    if stage_mode == STAGE_MODE_BULK and records is not None:
        required_bytes = estimate_staging_bytes(records)
        if not cache_filesystem_staging_headroom(cache_path, required_bytes):
            logger.warning(
                "Disabling stage_frames_local: need ~%.1f GB on %s but cache filesystem is too full",
                required_bytes / (1024**3),
                cache_path,
            )
            return False
        logger.info(
            "Bulk staging enabled: ~%.1f GB (%s unique files) -> %s",
            required_bytes / (1024**3),
            len(collect_unique_frame_paths(records)),
            cache_path,
        )

    return True


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


def _copy_frame_file(source: Path, destination: Path) -> None:
    """Copy without sendfile so FUSE→SSD copies fail less often on busy root disks."""
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


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
        _copy_frame_file(source, destination)
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
        max_staged_files: int | None = None,
        stage_mode: str | None = None,
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

        resolved_stage_mode = stage_mode or STAGE_MODE_LAZY
        self.stage_frames_local = bool(stage_frames_local and local_frame_cache is not None)
        self.local_frame_cache = local_frame_cache
        self._max_staged_files = max_staged_files
        self._localized_paths: dict[Path, Path] = {}
        self._staging_progress_callback = staging_progress_callback
        self._staging_progress_every = max(1, len(records) // 20)
        self._staging_completed = 0
        self._staging_fallback_logged = False

        if self.stage_frames_local and local_frame_cache is not None:
            if resolved_stage_mode == STAGE_MODE_LAZY and max_frames is not None and int(max_frames) > DEFAULT_MAX_STAGE_FRAMES:
                logger.warning(
                    "Disabling stage_frames_local: max_frames=%s exceeds safe lazy limit %s. "
                    "Set stage_mode=bulk to copy frames to extended SSD before training.",
                    max_frames,
                    DEFAULT_MAX_STAGE_FRAMES,
                )
                self.stage_frames_local = False
            elif resolved_stage_mode == STAGE_MODE_BULK:
                required_bytes = estimate_staging_bytes(records)
                if cache_filesystem_staging_headroom(local_frame_cache, required_bytes):
                    local_frame_cache.mkdir(parents=True, exist_ok=True)
                    logger.info(
                        "Bulk staging %s unique files (~%.1f GB) from S3 to %s",
                        len(collect_unique_frame_paths(records)),
                        required_bytes / (1024**3),
                        local_frame_cache,
                    )
                    records = stage_frame_records(
                        records,
                        frames_root,
                        local_frame_cache,
                        progress_callback=staging_progress_callback,
                    )
                    self.stage_frames_local = False
                else:
                    logger.warning(
                        "Disabling stage_frames_local: need ~%.1f GB on %s but cache filesystem is too full",
                        required_bytes / (1024**3),
                        local_frame_cache,
                    )
                    self.stage_frames_local = False
            else:
                local_frame_cache.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Lazy local staging enabled: frames copy to %s on first dataloader access",
                    local_frame_cache,
                )

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _disable_local_staging(self, reason: str) -> None:
        if not self._staging_fallback_logged:
            logger.warning(reason)
            self._staging_fallback_logged = True
        self.stage_frames_local = False

    def _resolve_frame_path(self, source: Path) -> Path:
        if not self.stage_frames_local or self.local_frame_cache is None:
            return source
        if self._max_staged_files is not None and len(self._localized_paths) >= self._max_staged_files:
            return source
        if not root_filesystem_staging_headroom():
            self._disable_local_staging(
                "Root filesystem above 85% capacity; reading frames directly from S3 mount"
            )
            return source
        had = len(self._localized_paths)
        try:
            resolved = localize_frame_path(
                source,
                self.frames_root,
                self.local_frame_cache,
                self._localized_paths,
            )
        except OSError as exc:
            if exc.errno != 28:
                raise
            self._disable_local_staging(
                f"No space left on device while staging {source}; "
                "reading remaining frames directly from S3 mount"
            )
            self._localized_paths[source] = source
            return source
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
