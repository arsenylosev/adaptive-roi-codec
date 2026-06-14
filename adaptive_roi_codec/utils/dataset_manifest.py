"""Build MANIFEST.json and train/val/test splits for Kvasir-Capsule."""

from __future__ import annotations

import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from adaptive_roi_codec.utils.kvasir_loader import (
    DEFAULT_FRAME_SIZE,
    discover_videos,
    resolve_video_dir,
)

DEFAULT_FRAME_SIZE_tuple = (DEFAULT_FRAME_SIZE, DEFAULT_FRAME_SIZE)


def _count_images_in_tar(archive_path: Path) -> int:
    with tarfile.open(archive_path) as handle:
        return sum(1 for name in handle.getnames() if name.lower().endswith(".jpg"))


def _probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "size_bytes": path.stat().st_size,
    }


def build_manifest(dataset_root: Path) -> dict[str, Any]:
    video_dir = resolve_video_dir(dataset_root)
    videos = discover_videos(video_dir)

    labelled_images_dir = dataset_root / "raw" / "labelled_images"
    if not labelled_images_dir.is_dir():
        labelled_images_dir = dataset_root / "labelled_images"

    class_archives: dict[str, int] = {}
    if labelled_images_dir.is_dir():
        for archive in sorted(labelled_images_dir.glob("*.tar.gz")):
            class_name = archive.name[: -len(".tar.gz")]
            class_archives[class_name] = _count_images_in_tar(archive)

    sample_meta = _probe_video(videos[0])
    video_entries = [
        {
            "video_id": path.stem,
            "path": str(path.relative_to(dataset_root)),
            **_probe_video(path),
        }
        for path in videos
    ]

    return {
        "dataset": "Kvasir-Capsule",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "native_resolution": {
            "width": sample_meta["width"],
            "height": sample_meta["height"],
        },
        "training_resolution": {
            "width": DEFAULT_FRAME_SIZE,
            "height": DEFAULT_FRAME_SIZE,
        },
        "video_count": len(videos),
        "expected_full_release_video_count": 47,
        "labeled_image_archives": class_archives,
        "labeled_image_count": sum(class_archives.values()),
        "videos": video_entries,
        "notes": [
            "Kvasir-Capsule releases native 336×336 video frames.",
            "Paper Full HD targets apply to deployment upscaling, not this dataset copy.",
        ],
    }


def write_splits(
    dataset_root: Path,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    import random

    video_dir = resolve_video_dir(dataset_root)
    video_ids = sorted(path.stem for path in discover_videos(video_dir))
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    train_count = max(1, int(len(video_ids) * train_ratio))
    val_count = max(1, int(len(video_ids) * val_ratio))
    if train_count + val_count >= len(video_ids):
        val_count = max(1, len(video_ids) - train_count - 1)

    train_ids = video_ids[:train_count]
    val_ids = video_ids[train_count : train_count + val_count]
    test_ids = video_ids[train_count + val_count :]

    splits_dir = dataset_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    mapping = {"train": train_ids, "val": val_ids, "test": test_ids}
    for split_name, ids in mapping.items():
        split_path = splits_dir / f"{split_name}_videos.txt"
        content = "\n".join(ids) + ("\n" if ids else "")
        split_path.write_text(content, encoding="utf-8")
    return mapping


def write_manifest(dataset_root: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = dataset_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
