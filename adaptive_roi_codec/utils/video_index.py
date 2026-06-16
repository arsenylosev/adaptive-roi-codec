"""Video path discovery for Kvasir-Capsule (no PyTorch dependency)."""

from __future__ import annotations

from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_FRAME_SIZE = 336


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
