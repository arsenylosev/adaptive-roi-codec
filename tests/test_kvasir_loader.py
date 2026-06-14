"""Tests for Kvasir video loader."""

from pathlib import Path

import pytest
import torch

from adaptive_roi_codec.utils.kvasir_loader import (
    KvasirVideoFrameDataset,
    collate_frame_samples,
    discover_videos,
    frame_to_tensor,
    load_video_ids,
    resolve_video_dir,
)
from adaptive_roi_codec.utils.env import REPO_ROOT

DATASET_ROOT = REPO_ROOT / "kvasir-capsule"


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="local kvasir-capsule folder missing")
def test_resolve_video_dir_finds_labelled_videos() -> None:
    video_dir = resolve_video_dir(DATASET_ROOT)
    assert video_dir.name == "labelled_videos"
    assert discover_videos(video_dir)


@pytest.mark.skipif(
    not (DATASET_ROOT / "splits" / "train_videos.txt").exists(),
    reason="run build-dataset-manifest first",
)
def test_load_train_split_ids() -> None:
    ids = load_video_ids(DATASET_ROOT / "splits" / "train_videos.txt")
    assert ids
    assert all(len(video_id) == 16 for video_id in ids)


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="local kvasir-capsule folder missing")
def test_video_dataset_yields_336_tensors() -> None:
    dataset = KvasirVideoFrameDataset(
        DATASET_ROOT,
        split="",
        video_ids=[discover_videos(resolve_video_dir(DATASET_ROOT))[0].stem],
        frame_stride=300,
        max_frames_per_video=1,
    )
    sample = next(iter(dataset))
    assert sample.frame.shape == (3, 336, 336)
    assert sample.prev_frame is None


def test_collate_frame_samples() -> None:
    from adaptive_roi_codec.utils.kvasir_loader import FrameSample

    batch = [
        FrameSample(torch.zeros(3, 336, 336), None, "a", 0),
        FrameSample(torch.ones(3, 336, 336), torch.zeros(3, 336, 336), "a", 30),
    ]
    collated = collate_frame_samples(batch)
    assert collated["frame"].shape == (2, 3, 336, 336)
    assert collated["has_prev"].tolist() == [False, True]


def test_frame_to_tensor_resizes_to_target() -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor = frame_to_tensor(frame, height=336, width=336)
    assert tensor.shape == (3, 336, 336)
