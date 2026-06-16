"""Tests for preprocessed frame manifest loading."""

import json
from pathlib import Path

import torch

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
