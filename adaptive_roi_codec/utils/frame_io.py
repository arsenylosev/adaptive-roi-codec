"""Preprocessed frame encode/decode without PyTorch (stage-1 CPU jobs)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PREPROCESSED_MANIFEST = "frames_manifest.jsonl"
FRAME_CACHE_SUFFIX = ".npy"


def frame_to_chw_numpy(frame_bgr, height: int, width: int) -> np.ndarray:
    if frame_bgr is None:
        raise ValueError("Received empty frame from video decoder")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != height or rgb.shape[1] != width:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    chw = np.transpose(rgb, (2, 0, 1))
    return np.ascontiguousarray(chw, dtype=np.float32) / np.float32(255.0)


def save_preprocessed_frame(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_preprocessed_frame_array(path: Path) -> np.ndarray:
    return np.load(path)
