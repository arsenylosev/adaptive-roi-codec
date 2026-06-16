"""Tests for preprocessed frame manifest loading."""

import json
from pathlib import Path

import numpy as np
import torch

from adaptive_roi_codec.utils.frame_io import FRAME_CACHE_SUFFIX, save_preprocessed_frame
from adaptive_roi_codec.utils.kvasir_loader import (
    PREPROCESSED_MANIFEST,
    KvasirPreprocessedFrameDataset,
    load_preprocessed_manifest,
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
