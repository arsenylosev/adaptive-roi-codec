"""Training entry point for local debugging and Datasphere Jobs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from adaptive_roi_codec.losses.clinical_loss import ClinicalLoss, LossWeights
from adaptive_roi_codec.model.quantizer import AdaptiveQuantizer
from adaptive_roi_codec.model.roi_detector import ROIDetector
from adaptive_roi_codec.model.vae_codec import VAECodec
from adaptive_roi_codec.utils.config import load_yaml, merge_dicts
from adaptive_roi_codec.utils.env import load_project_env, optional_env, s3_mount_root
from adaptive_roi_codec.utils.kvasir_loader import (
    KvasirVideoFrameDataset,
    SyntheticFrameDataset,
    collate_frame_samples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive ROI neural video codec")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/base.yaml)")
    parser.add_argument(
        "--params",
        required=False,
        help="JSON file with job overrides (epoch count, experiment id, etc.)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run one batch on synthetic data")
    return parser.parse_args()


def _s3_mount_available(connector_id: str) -> bool:
    if not connector_id:
        return False
    mount_root = s3_mount_root(connector_id)
    return mount_root.parent.exists() and mount_root.exists()


def resolve_dataset_root(config: dict) -> Path:
    data_cfg = config.get("data", {})
    prefix = optional_env("S3_DATA_PREFIX", data_cfg.get("s3_prefix", "kvasir-capsule"))
    connector_id = optional_env("S3_CONNECTOR_ID", data_cfg.get("s3_connector_id", ""))

    if connector_id and _s3_mount_available(connector_id):
        root = s3_mount_root(connector_id) / prefix
        logger.info("Using S3 mount dataset root: %s", root)
        return root

    local_root = Path(data_cfg.get("local_root", "kvasir-capsule"))
    logger.info("Using local dataset root: %s", local_root.resolve())
    return local_root


def resolve_checkpoint_dir(config: dict) -> Path:
    ckpt_cfg = config.get("checkpoints", {})
    connector_id = optional_env("S3_CONNECTOR_ID", ckpt_cfg.get("s3_connector_id", ""))
    subdir = ckpt_cfg.get("subdir", "checkpoints")
    experiment_id = config.get("experiment_id", "default")

    if connector_id and _s3_mount_available(connector_id):
        path = s3_mount_root(connector_id) / subdir / experiment_id
    else:
        path = Path(ckpt_cfg.get("local_dir", "checkpoints")) / experiment_id

    path.mkdir(parents=True, exist_ok=True)
    return path


def build_dataloader(config: dict, dry_run: bool) -> DataLoader:
    training = config["training"]
    data_cfg = config.get("data", {})
    model_cfg = config["model"]
    batch_size = training["batch_size"]
    width, height = model_cfg["input_res"]

    if dry_run or optional_env("TRAIN_DRY_RUN", "").lower() in {"1", "true", "yes"}:
        dataset = SyntheticFrameDataset(
            length=data_cfg.get("dry_run_samples", 4),
            height=height,
            width=width,
        )
        logger.warning("Dry-run mode: using %sx%s synthetic frames", width, height)
        return DataLoader(dataset, batch_size=min(batch_size, 2), shuffle=True, num_workers=0)

    root = resolve_dataset_root(config)
    dataset = KvasirVideoFrameDataset(
        root,
        split=data_cfg.get("split", "train"),
        frame_stride=int(data_cfg.get("frame_stride", 30)),
        max_frames_per_video=data_cfg.get("max_frames_per_video"),
        height=height,
        width=width,
        video_subdir=data_cfg.get("video_subdir"),
        splits_subdir=data_cfg.get("splits_subdir", "splits"),
    )
    logger.info(
        "Using Kvasir video loader: split=%s stride=%s videos=%s",
        data_cfg.get("split", "train"),
        data_cfg.get("frame_stride", 30),
        len(dataset.videos),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_frame_samples,
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    codec: VAECodec,
    roi_detector: ROIDetector,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "codec": codec.state_dict(),
        "roi_detector": roi_detector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s", path)


def train(config_path: str, params_path: str | None, dry_run: bool) -> dict[str, float]:
    load_project_env()
    config = load_yaml(config_path)

    if params_path:
        with Path(params_path).open(encoding="utf-8") as handle:
            params = json.load(handle)
        config = merge_dicts(config, params)

    device_name = optional_env("TRAIN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    model_cfg = config["model"]
    training_cfg = config["training"]
    quant_cfg = config["quantizer"]
    width, height = model_cfg["input_res"]
    logger.info("Training resolution: %sx%s", width, height)

    roi_detector = ROIDetector(input_size=config["roi_detector"]["input_res"]).to(device)
    codec = VAECodec(latent_channels=model_cfg["latent_ch"]).to(device)
    quantizer = AdaptiveQuantizer(
        q_min=quant_cfg["q_min"],
        q_max=quant_cfg["q_max"],
        kappa=quant_cfg["kappa"],
        alpha_spatial=quant_cfg["alpha_spatial"],
    )
    loss_fn = ClinicalLoss(
        LossWeights(
            alpha=training_cfg["alpha"],
            lambda_roi=training_cfg["lambda_roi"],
            lambda_rate=training_cfg["lambda_rate"],
            lambda_temp=training_cfg.get("lambda_temp", 0.1),
            beta=training_cfg.get("beta_0", 0.01),
        )
    )
    params = list(codec.parameters()) + list(roi_detector.parameters())
    optimizer = torch.optim.Adam(params, lr=training_cfg["lr"])

    loader = build_dataloader(config, dry_run=dry_run)
    checkpoint_dir = resolve_checkpoint_dir(config)
    save_every = config.get("checkpoints", {}).get("save_every_epochs", 5)

    epochs = 1 if dry_run else training_cfg["epochs"]
    last_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        codec.train()
        roi_detector.train()
        epoch_loss = 0.0
        batches = 0
        prev_recon: torch.Tensor | None = None
        prev_z: torch.Tensor | None = None

        for batch in loader:
            if isinstance(batch, torch.Tensor):
                samples = [(batch[i : i + 1], None) for i in range(batch.size(0))]
            else:
                frames = batch["frame"]
                prev_frames = batch["prev_frame"]
                has_prev = batch["has_prev"]
                samples = [
                    (
                        frames[i : i + 1],
                        prev_frames[i : i + 1] if bool(has_prev[i].item()) else None,
                    )
                    for i in range(frames.size(0))
                ]

            for frame, prev_frame in samples:
                frame = frame.to(device)
                if prev_frame is not None:
                    prev_frame = prev_frame.to(device)
                else:
                    prev_recon = None
                    prev_z = None

                mask = roi_detector(frame)
                outputs = codec(frame, prev_recon=prev_recon, prev_z=prev_z)
                outputs["z"] = quantizer.quantize(outputs["z"], mask)
                losses = loss_fn(outputs, frame, mask, prev_frame, prev_recon)

                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()

                epoch_loss += float(losses["total"].item())
                batches += 1
                prev_recon = outputs["recon"].detach()
                prev_z = outputs["z"].detach()

                if dry_run:
                    break
            if dry_run:
                break

        avg_loss = epoch_loss / max(batches, 1)
        last_metrics = {"epoch": float(epoch), "loss": avg_loss}
        logger.info("Epoch %s/%s — loss=%.6f", epoch, epochs, avg_loss)

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(ckpt_path, epoch, codec, roi_detector, optimizer, last_metrics)

    metrics_path = Path(optional_env("TRAIN_METRICS_PATH", "metrics/train_metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(last_metrics, handle, indent=2)
    logger.info("Wrote metrics to %s", metrics_path)
    return last_metrics


def main() -> None:
    args = parse_args()
    train(args.config, args.params, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
