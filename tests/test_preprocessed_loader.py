"""Tests for preprocessed frame manifest loading."""

import json
from pathlib import Path

import numpy as np
import torch

from adaptive_roi_codec.utils.frame_io import FRAME_CACHE_SUFFIX, save_preprocessed_frame
from adaptive_roi_codec.utils.kvasir_loader import (
    PREPROCESSED_MANIFEST,
    FrameRecord,
    KvasirPreprocessedFrameDataset,
    load_preprocessed_manifest,
    stage_frame_records,
)


def test_load_preprocessed_manifest(tmp_path: Path) -> None:
    frame_a = tmp_path / "v1" / "frame_000000.pt"
    frame_b = tmp_path / "v1" / "frame_000030.pt"
    frame_a.parent.mkdir(parents=True)
    torch.save(torch.zeros(3, 336, 336), frame_a)
    torch.save(torch.ones(3, 336, 336), frame_b)

    manifest = tmp_path / PREPROCESSED_MANIFEST
    rows = [
        {"video_id": "v1", "frame_index": 0, "path": str(frame_a), "prev_path": None},
        {"video_id": "v1", "frame_index": 30, "path": str(frame_b), "prev_path": str(frame_a)},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    records = load_preprocessed_manifest(manifest)
    assert len(records) == 2
    assert records[1].prev_path == frame_a

    dataset = KvasirPreprocessedFrameDataset(tmp_path, split="")
    sample = dataset[1]
    assert sample.prev_frame is not None
    assert sample.frame.shape == (3, 336, 336)


def test_preprocessed_dataset_loads_npy_frames(tmp_path: Path) -> None:
    frame_a = tmp_path / "v1" / f"frame_000000{FRAME_CACHE_SUFFIX}"
    frame_b = tmp_path / "v1" / f"frame_000030{FRAME_CACHE_SUFFIX}"
    save_preprocessed_frame(np.zeros((3, 336, 336), dtype=np.float32), frame_a)
    save_preprocessed_frame(np.ones((3, 336, 336), dtype=np.float32), frame_b)

    manifest = tmp_path / PREPROCESSED_MANIFEST
    rows = [
        {"video_id": "v1", "frame_index": 0, "path": str(frame_a), "prev_path": None},
        {"video_id": "v1", "frame_index": 30, "path": str(frame_b), "prev_path": str(frame_a)},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = KvasirPreprocessedFrameDataset(tmp_path, split="")
    sample = dataset[1]
    assert sample.frame.shape == (3, 336, 336)
    assert float(sample.prev_frame[0, 0, 0]) == 0.0


def test_preprocessed_dataset_respects_max_frames(tmp_path: Path) -> None:
    frame_paths = []
    rows = []
    for index in range(5):
        path = tmp_path / "v1" / f"frame_{index:06d}{FRAME_CACHE_SUFFIX}"
        save_preprocessed_frame(np.full((3, 336, 336), index, dtype=np.float32), path)
        frame_paths.append(path)
        rows.append(
            {
                "video_id": "v1",
                "frame_index": index,
                "path": str(path),
                "prev_path": None,
            }
        )

    manifest = tmp_path / PREPROCESSED_MANIFEST
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = KvasirPreprocessedFrameDataset(tmp_path, split="", max_frames=3)
    assert len(dataset) == 3


def test_stage_frame_records_copies_to_local_cache(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    source = frames_root / "v1" / f"frame_000000{FRAME_CACHE_SUFFIX}"
    save_preprocessed_frame(np.zeros((3, 336, 336), dtype=np.float32), source)
    records = [
        FrameRecord(
            video_id="v1",
            frame_index=0,
            path=source,
            prev_path=None,
        )
    ]
    cache_root = tmp_path / "cache"

    staged = stage_frame_records(records, frames_root, cache_root)
    assert staged[0].path != source
    assert staged[0].path.exists()
    assert staged[0].path.is_relative_to(cache_root)


def test_preprocessed_dataset_lazy_staging_on_getitem(tmp_path: Path) -> None:
    frames_root = tmp_path / "frames"
    source = frames_root / "v1" / f"frame_000000{FRAME_CACHE_SUFFIX}"
    save_preprocessed_frame(np.zeros((3, 336, 336), dtype=np.float32), source)
    manifest = frames_root / PREPROCESSED_MANIFEST
    manifest.write_text(
        json.dumps(
            {
                "video_id": "v1",
                "frame_index": 0,
                "path": str(source),
                "prev_path": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache_root = tmp_path / "cache"

    dataset = KvasirPreprocessedFrameDataset(
        frames_root,
        split="",
        stage_frames_local=True,
        local_frame_cache=cache_root,
    )
    assert not cache_root.exists() or list(cache_root.rglob("*.npy")) == []

    sample = dataset[0]
    assert sample.frame.shape == (3, 336, 336)
    assert list(cache_root.rglob("*.npy"))


def test_load_preprocessed_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    frame_a = tmp_path / "v1" / f"frame_000000{FRAME_CACHE_SUFFIX}"
    frame_b = tmp_path / "v1" / f"frame_000030{FRAME_CACHE_SUFFIX}"
    save_preprocessed_frame(np.zeros((3, 336, 336), dtype=np.float32), frame_a)
    save_preprocessed_frame(np.ones((3, 336, 336), dtype=np.float32), frame_b)

    manifest = tmp_path / PREPROCESSED_MANIFEST
    rows = [
        {"video_id": "v1", "frame_index": 0, "path": "v1/frame_000000.npy", "prev_path": None},
        {
            "video_id": "v1",
            "frame_index": 30,
            "path": "v1/frame_000030.npy",
            "prev_path": "v1/frame_000000.npy",
        },
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    records = load_preprocessed_manifest(manifest, frames_root=tmp_path)
    assert records[1].path == frame_b
    assert records[1].prev_path == frame_a
