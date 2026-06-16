"""Training entry point for local debugging and Datasphere Jobs."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from adaptive_roi_codec.losses.clinical_loss import ClinicalLoss, LossWeights
from adaptive_roi_codec.model.quantizer import AdaptiveQuantizer
from adaptive_roi_codec.model.roi_detector import ROIDetector
from adaptive_roi_codec.model.vae_codec import VAECodec
from adaptive_roi_codec.utils.config import load_yaml, merge_dicts
from adaptive_roi_codec.utils.device import resolve_training_device
from adaptive_roi_codec.utils.env import load_project_env, optional_env, s3_mount_root
from adaptive_roi_codec.utils.kvasir_loader import (
    KvasirPreprocessedFrameDataset,
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


def resolve_frames_root(config: dict) -> Path:
    data_cfg = config.get("data", {})
    env_path = optional_env("FRAMES_OUTPUT_DIR", "")
    if env_path:
        return Path(env_path)
    return Path(data_cfg.get("frames_root", "frames_cache"))


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


def build_dataloader(config: dict, dry_run: bool, device: torch.device) -> DataLoader:
    training = config["training"]
    data_cfg = config.get("data", {})
    model_cfg = config["model"]
    batch_size = training["batch_size"]
    width, height = model_cfg["input_res"]
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    if dry_run or optional_env("TRAIN_DRY_RUN", "").lower() in {"1", "true", "yes"}:
        dataset = SyntheticFrameDataset(
            length=data_cfg.get("dry_run_samples", 4),
            height=height,
            width=width,
        )
        logger.warning("Dry-run mode: using %sx%s synthetic frames", width, height)
        return DataLoader(
            dataset,
            batch_size=min(batch_size, 2),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    dataset_root = resolve_dataset_root(config)
    source = data_cfg.get("source", "video")

    if source == "preprocessed":
        frames_root = resolve_frames_root(config)
        logger.info("Using preprocessed frames from %s", frames_root)
        dataset = KvasirPreprocessedFrameDataset(
            frames_root,
            split=data_cfg.get("split", "train"),
            dataset_root=dataset_root,
            splits_subdir=data_cfg.get("splits_subdir", "splits"),
        )
        logger.info("Preprocessed frame count: %s", len(dataset))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(data_cfg.get("shuffle", True)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_frame_samples,
            persistent_workers=num_workers > 0,
        )

    dataset = KvasirVideoFrameDataset(
        dataset_root,
        split=data_cfg.get("split", "train"),
        frame_stride=int(data_cfg.get("frame_stride", 30)),
        max_frames_per_video=data_cfg.get("max_frames_per_video"),
        height=height,
        width=width,
        video_subdir=data_cfg.get("video_subdir"),
        splits_subdir=data_cfg.get("splits_subdir", "splits"),
    )
    logger.info(
        "Using Kvasir video loader (S3 decode): split=%s stride=%s videos=%s",
        data_cfg.get("split", "train"),
        data_cfg.get("frame_stride", 30),
        len(dataset.videos),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_frame_samples,
        persistent_workers=num_workers > 0,
    )


def shutdown_dataloader(loader: DataLoader | None) -> None:
    """Stop DataLoader worker processes so the job exits with status 0."""
    if loader is None or loader.num_workers == 0:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None
    del loader
    gc.collect()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = optional_env("DATASPHERE_LOCAL_CHECKPOINT_DIR", "")
    if staging_root:
        local_path = Path(staging_root) / path.parent.name / path.name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, local_path)
        shutil.copy2(local_path, path)
        logger.info("Saved checkpoint to %s (via %s)", path, local_path)
        return

    torch.save(payload, path)
    logger.info("Saved checkpoint to %s", path)


def write_metrics_file(metrics_path: Path, metrics: dict[str, float]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    logger.info("Wrote metrics to %s", metrics_path)


def _train_batch(
    batch: dict,
    *,
    device: torch.device,
    roi_detector: ROIDetector,
    codec: VAECodec,
    quantizer: AdaptiveQuantizer,
    loss_fn: ClinicalLoss,
    optimizer: torch.optim.Optimizer,
    video_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> float:
    frames = batch["frame"].to(device, non_blocking=True)
    prev_frames = batch["prev_frame"].to(device, non_blocking=True)
    has_prev = batch["has_prev"]
    video_ids: list[str] = batch["video_id"]
    batch_size = frames.size(0)

    optimizer.zero_grad(set_to_none=True)
    batch_loss = 0.0

    for index in range(batch_size):
        frame = frames[index : index + 1]
        video_id = video_ids[index]
        use_prev = bool(has_prev[index].item())

        prev_frame = prev_frames[index : index + 1] if use_prev else None
        prev_recon = None
        prev_z = None
        if use_prev and video_id in video_states:
            recon_state, z_state = video_states[video_id]
            prev_recon = recon_state.unsqueeze(0)
            prev_z = z_state.unsqueeze(0)

        mask = roi_detector(frame)
        outputs = codec(frame, prev_recon=prev_recon, prev_z=prev_z)
        outputs["z"] = quantizer.quantize(outputs["z"], mask)
        losses = loss_fn(outputs, frame, mask, prev_frame, prev_recon)

        (losses["total"] / batch_size).backward()
        batch_loss += float(losses["total"].item())

        video_states[video_id] = (outputs["recon"].detach()[0], outputs["z"].detach()[0])

    optimizer.step()
    return batch_loss / batch_size


def train(config_path: str, params_path: str | None, dry_run: bool) -> dict[str, float]:
    load_project_env()
    config = load_yaml(config_path)

    if params_path:
        with Path(params_path).open(encoding="utf-8") as handle:
            params = json.load(handle)
        config = merge_dicts(config, params)

    device = resolve_training_device()

    model_cfg = config["model"]
    training_cfg = config["training"]
    quant_cfg = config["quantizer"]
    data_cfg = config.get("data", {})
    width, height = model_cfg["input_res"]
    log_every = int(training_cfg.get("log_every_batches", 20))
    logger.info("Training resolution: %sx%s data.source=%s", width, height, data_cfg.get("source", "video"))

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

    loader = build_dataloader(config, dry_run=dry_run, device=device)
    checkpoint_dir = resolve_checkpoint_dir(config)
    save_every = config.get("checkpoints", {}).get("save_every_epochs", 5)

    epochs = 1 if dry_run else training_cfg["epochs"]
    last_metrics: dict[str, float] = {}

    try:
        for epoch in range(1, epochs + 1):
            codec.train()
            roi_detector.train()
            epoch_loss = 0.0
            batches = 0
            epoch_start = time.perf_counter()
            video_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

            for batch in loader:
                if isinstance(batch, torch.Tensor):
                    batch = {
                        "frame": batch,
                        "prev_frame": torch.zeros_like(batch),
                        "has_prev": torch.zeros(batch.size(0), dtype=torch.bool),
                        "video_id": ["synthetic"] * batch.size(0),
                    }

                avg_batch_loss = _train_batch(
                    batch,
                    device=device,
                    roi_detector=roi_detector,
                    codec=codec,
                    quantizer=quantizer,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    video_states=video_states,
                )
                epoch_loss += avg_batch_loss
                batches += 1

                if batches % log_every == 0:
                    elapsed = time.perf_counter() - epoch_start
                    logger.info(
                        "Epoch %s batch %s — loss=%.6f elapsed=%.1fs",
                        epoch,
                        batches,
                        avg_batch_loss,
                        elapsed,
                    )

                if dry_run:
                    break

            avg_loss = epoch_loss / max(batches, 1)
            last_metrics = {"epoch": float(epoch), "loss": avg_loss, "batches": float(batches)}
            logger.info(
                "Epoch %s/%s complete — loss=%.6f batches=%s elapsed=%.1fs",
                epoch,
                epochs,
                avg_loss,
                batches,
                time.perf_counter() - epoch_start,
            )

            if epoch % save_every == 0 or epoch == epochs:
                ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                save_checkpoint(ckpt_path, epoch, codec, roi_detector, optimizer, last_metrics)
    finally:
        shutdown_dataloader(loader)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    metrics_path = Path(optional_env("TRAIN_METRICS_PATH", "metrics/train_metrics.json"))
    write_metrics_file(metrics_path, last_metrics)
    logger.info("Training finished successfully")
    return last_metrics


def main() -> None:
    args = parse_args()
    train(args.config, args.params, dry_run=args.dry_run)
    sys.exit(0)


if __name__ == "__main__":
    main()
