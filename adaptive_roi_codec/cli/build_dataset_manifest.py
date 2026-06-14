"""CLI entry point for building dataset manifest and splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adaptive_roi_codec.utils.dataset_manifest import build_manifest, write_manifest, write_splits
from adaptive_roi_codec.utils.env import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kvasir-Capsule MANIFEST.json and splits")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "kvasir-capsule",
        help="Path to kvasir-capsule directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    manifest = build_manifest(dataset_root)
    splits = write_splits(dataset_root, seed=args.seed)
    manifest["splits"] = {name: len(ids) for name, ids in splits.items()}
    manifest_path = write_manifest(dataset_root, manifest)

    payload = {"manifest": str(manifest_path), "splits": splits}
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
