"""Tests for torch-free preprocessed frame I/O."""

from pathlib import Path

import numpy as np

from adaptive_roi_codec.utils.frame_io import (
    FRAME_CACHE_SUFFIX,
    frame_to_chw_numpy,
    load_preprocessed_frame_array,
    save_preprocessed_frame,
)


def test_frame_to_chw_numpy_resizes_to_target() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 200:300] = 255

    array = frame_to_chw_numpy(frame, height=336, width=336)

    assert array.shape == (3, 336, 336)
    assert array.dtype == np.float32
    assert 0.0 <= float(array.max()) <= 1.0


def test_save_and_load_preprocessed_frame_roundtrip(tmp_path: Path) -> None:
    array = np.ones((3, 336, 336), dtype=np.float32) * 0.5
    path = tmp_path / f"frame_000000{FRAME_CACHE_SUFFIX}"
    save_preprocessed_frame(array, path)

    loaded = load_preprocessed_frame_array(path)
    assert np.allclose(loaded, array)
