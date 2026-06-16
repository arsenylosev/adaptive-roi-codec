"""Extract strided training frames from Kvasir videos (Datasphere stage-1 CPU job)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from adaptive_roi_codec.utils.config import load_yaml, merge_dicts
from adaptive_roi_codec.utils.env import load_project_env, optional_env, s3_mount_root
from adaptive_roi_codec.utils.frame_io import (
    FRAME_CACHE_SUFFIX,
    PREPROCESSED_MANIFEST,
    frame_to_chw_numpy,
    save_preprocessed_frame,
)
from adaptive_roi_codec.utils.video_index import (
    discover_videos,
    filter_videos,
    load_video_ids,
    resolve_splits_dir,
    resolve_video_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode Kvasir MP4s to .npy frame caches for fast GPU training"
    )
    parser.add_argument("--config", required=True, help="YAML config (configs/base.yaml)")
    parser.add_argument("--params", required=False, help="JSON overrides")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: FRAMES_OUTPUT_DIR env or ./frames_cache)",
    )
    return parser.parse_args()


def resolve_dataset_root(config: dict) -> Path:
    data_cfg = config.get("data", {})
    prefix = optional_env("S3_DATA_PREFIX", data_cfg.get("s3_prefix", "kvasir-capsule"))
    connector_id = optional_env("S3_CONNECTOR_ID", data_cfg.get("s3_connector_id", ""))
    if connector_id:
        mount = s3_mount_root(connector_id) / prefix
        if mount.parent.exists() and mount.exists():
            logger.info("Reading videos from S3 mount: %s", mount)
            return mount
    return Path(data_cfg.get("local_root", "kvasir-capsule"))


def resolve_output_dir(config: dict, cli_output: Path | None) -> Path:
    data_cfg = config.get("data", {})
    env_path = optional_env("FRAMES_OUTPUT_DIR", "")
    if cli_output is not None:
        return cli_output
    if env_path:
        return Path(env_path)
    return Path(data_cfg.get("frames_cache_dir", "frames_cache"))


def extract_split(
    dataset_root: Path,
    output_root: Path,
    *,
    split: str,
    frame_stride: int,
    max_frames_per_video: int | None,
    height: int,
    width: int,
    video_subdir: str | None,
    splits_subdir: str,
) -> int:
    import cv2

    video_dir = resolve_video_dir(dataset_root, video_subdir)
    split_file = resolve_splits_dir(dataset_root, splits_subdir) / f"{split}_videos.txt"
    video_ids = load_video_ids(split_file) if split_file.exists() else None
    videos = filter_videos(discover_videos(video_dir), video_ids)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / PREPROCESSED_MANIFEST
    written = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for video_path in videos:
            video_id = video_path.stem
            video_out = output_root / video_id
            video_out.mkdir(parents=True, exist_ok=True)

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                logger.warning("Skipping unreadable video: %s", video_path)
                continue

            prev_path: Path | None = None
            frame_index = 0
            next_yield = 0
            yielded = 0

            try:
                while True:
                    if frame_index < next_yield:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, next_yield)
                        frame_index = next_yield

                    ok, frame_bgr = capture.read()
                    if not ok:
                        break

                    if frame_index != next_yield:
                        frame_index += 1
                        continue

                    tensor = frame_to_chw_numpy(frame_bgr, height, width)
                    out_path = video_out / f"frame_{frame_index:06d}{FRAME_CACHE_SUFFIX}"
                    save_preprocessed_frame(tensor, out_path)

                    row = {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "path": str(out_path.resolve()),
                        "prev_path": str(prev_path.resolve()) if prev_path else None,
                    }
                    manifest.write(json.dumps(row) + "\n")
                    manifest.flush()

                    prev_path = out_path
                    written += 1
                    yielded += 1
                    next_yield = frame_index + frame_stride
                    frame_index += 1

                    if max_frames_per_video and yielded >= max_frames_per_video:
                        break
            finally:
                capture.release()

            logger.info("Extracted %s frames from %s", yielded, video_id)

    return written


def main() -> None:
    load_project_env()
    args = parse_args()
    config = load_yaml(args.config)
    if args.params:
        with Path(args.params).open(encoding="utf-8") as handle:
            config = merge_dicts(config, json.load(handle))

    data_cfg = config.get("data", {})
    model_cfg = config["model"]
    height, width = model_cfg["input_res"]
    dataset_root = resolve_dataset_root(config)
    output_root = resolve_output_dir(config, args.output)

    total = extract_split(
        dataset_root,
        output_root,
        split=data_cfg.get("split", "train"),
        frame_stride=int(data_cfg.get("frame_stride", 30)),
        max_frames_per_video=data_cfg.get("max_frames_per_video"),
        height=height,
        width=width,
        video_subdir=data_cfg.get("video_subdir"),
        splits_subdir=data_cfg.get("splits_subdir", "splits"),
    )
    logger.info("Wrote %s frame records to %s", total, output_root / PREPROCESSED_MANIFEST)


if __name__ == "__main__":
    main()
