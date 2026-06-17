"""Training entry point for local debugging and Datasphere Jobs."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from adaptive_roi_codec.losses.clinical_loss import ClinicalLoss, LossWeights
from adaptive_roi_codec.model.quantizer import AdaptiveQuantizer
from adaptive_roi_codec.model.roi_detector import ROIDetector
from adaptive_roi_codec.model.vae_codec import VAECodec
from adaptive_roi_codec.utils.config import load_yaml, merge_dicts
from adaptive_roi_codec.utils.device import resolve_training_device
from adaptive_roi_codec.utils.datasphere_exit import finalize_datasphere_job
from adaptive_roi_codec.utils.env import load_project_env, optional_env, s3_mount_root
from adaptive_roi_codec.utils.job_progress import JobProgressTracker
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


def resolve_local_frame_cache(config: dict) -> Path | None:
    data_cfg = config.get("data", {})
    env_path = optional_env("DATASPHERE_LOCAL_FRAME_CACHE", "")
    if env_path:
        return Path(env_path)
    local_path = data_cfg.get("local_frame_cache")
    return Path(local_path) if local_path else None


def is_datasphere_job() -> bool:
    return bool(os.getenv("JOB_PROGRESS_FILENAME"))


def configure_multiprocessing(num_workers: int) -> None:
    """Use spawn only when DataLoader workers are enabled."""
    if num_workers <= 0:
        return
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


@contextlib.contextmanager
def redirect_stderr_to_stdout():
    """Torchvision weight downloads write progress bars to stderr."""
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    try:
        os.dup2(sys.stdout.fileno(), stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def build_dataloader(
    config: dict,
    dry_run: bool,
    *,
    pin_memory: bool,
    progress: JobProgressTracker | None = None,
) -> DataLoader:
    training = config["training"]
    data_cfg = config.get("data", {})
    model_cfg = config["model"]
    batch_size = training["batch_size"]
    width, height = model_cfg["input_res"]
    num_workers = int(data_cfg.get("num_workers", 0))
    configure_multiprocessing(num_workers)
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    local_frame_cache = resolve_local_frame_cache(config)
    stage_frames_local = bool(data_cfg.get("stage_frames_local", False)) and local_frame_cache is not None
    max_frames = data_cfg.get("max_frames")

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
            max_frames=int(max_frames) if max_frames is not None else None,
            stage_frames_local=stage_frames_local,
            local_frame_cache=local_frame_cache,
            staging_progress_callback=progress.staging if progress else None,
        )
        logger.info(
            "Preprocessed frame count: %s (max_frames=%s stage_local=%s num_workers=%s)",
            len(dataset),
            max_frames,
            stage_frames_local,
            num_workers,
        )
        loader_kwargs: dict = {
            "batch_size": batch_size,
            "shuffle": bool(data_cfg.get("shuffle", True)),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_frame_samples,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = not is_datasphere_job()
        return DataLoader(dataset, **loader_kwargs)

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
        persistent_workers=num_workers > 0 and not is_datasphere_job(),
    )


def shutdown_dataloader(loader: DataLoader | None) -> None:
    """Stop DataLoader worker processes so the job exits with status 0."""
    if loader is None:
        return

    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        try:
            iterator._shutdown_workers()
        except Exception:
            logger.warning("DataLoader worker shutdown raised; continuing cleanup")
        loader._iterator = None

    workers = list(getattr(loader, "_workers", []) or [])
    for worker in workers:
        if worker.is_alive():
            worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)

    del loader
    gc.collect()
    if is_datasphere_job():
        from adaptive_roi_codec.utils.datasphere_exit import terminate_multiprocessing_children

        terminate_multiprocessing_children()


def _release_training_models(
    *,
    codec: VAECodec | None,
    roi_detector: ROIDetector | None,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> None:
    del codec, roi_detector, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def finalize_job_exit(code: int = 0) -> None:
    """Exit cleanly so DataSphere marks the job as Success."""
    if is_datasphere_job():
        code = finalize_datasphere_job(code)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(code)


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


def _build_video_prev_tensors(
    batch: dict,
    video_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    latent_channels: int,
    latent_h: int,
    latent_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = batch["frame"]
    batch_size = frames.size(0)
    _, _, height, width = frames.shape

    prev_recon = torch.zeros(batch_size, 3, height, width, device=device, dtype=frames.dtype)
    prev_z = torch.zeros(
        batch_size,
        latent_channels,
        latent_h,
        latent_w,
        device=device,
        dtype=frames.dtype,
    )
    has_state = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for index, video_id in enumerate(batch["video_id"]):
        if not bool(batch["has_prev"][index].item()):
            continue
        state = video_states.get(video_id)
        if state is None:
            continue
        recon_state, z_state = state
        prev_recon[index] = recon_state
        prev_z[index] = z_state
        has_state[index] = True

    return prev_recon, prev_z, has_state


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
    latent_channels: int,
    latent_h: int,
    latent_w: int,
) -> float:
    frames = batch["frame"].to(device, non_blocking=True)
    prev_frames = batch["prev_frame"].to(device, non_blocking=True)
    video_ids: list[str] = batch["video_id"]

    optimizer.zero_grad(set_to_none=True)

    prev_recon, prev_z, has_state = _build_video_prev_tensors(
        batch,
        video_states,
        device,
        latent_channels=latent_channels,
        latent_h=latent_h,
        latent_w=latent_w,
    )

    mask = roi_detector(frames)
    outputs = codec(frames, prev_recon=prev_recon, prev_z=prev_z)
    outputs["z"] = quantizer.quantize(outputs["z"], mask)
    losses = loss_fn(
        outputs,
        frames,
        mask,
        prev_frame=prev_frames,
        prev_recon=prev_recon,
        temporal_mask=has_state,
    )

    losses["total"].backward()
    optimizer.step()

    for index, video_id in enumerate(video_ids):
        video_states[video_id] = (
            outputs["recon"].detach()[index],
            outputs["z"].detach()[index],
        )

    return float(losses["total"].item())


def train(config_path: str, params_path: str | None, dry_run: bool) -> dict[str, float]:
    load_project_env()
    config = load_yaml(config_path)

    if params_path:
        with Path(params_path).open(encoding="utf-8") as handle:
            params = json.load(handle)
        config = merge_dicts(config, params)

    device = resolve_training_device()
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    model_cfg = config["model"]
    training_cfg = config["training"]
    quant_cfg = config["quantizer"]
    data_cfg = config.get("data", {})
    width, height = model_cfg["input_res"]
    log_every = int(training_cfg.get("log_every_batches", 20))
    job_batch_size = optional_env("TRAIN_BATCH_SIZE", "")
    if job_batch_size:
        logger.info("Job TRAIN_BATCH_SIZE=%s", job_batch_size)
    logger.info(
        "Training resolution: %sx%s data.source=%s batch_size=%s",
        width,
        height,
        data_cfg.get("source", "video"),
        training_cfg["batch_size"],
    )

    epochs = 1 if dry_run else training_cfg["epochs"]
    max_train_batches = training_cfg.get("max_train_batches")
    if max_train_batches is not None:
        max_train_batches = int(max_train_batches)

    progress = JobProgressTracker(
        total_batches=max_train_batches if max_train_batches is not None else 1,
        epochs=epochs,
    )
    progress.setup("Initializing training pipeline")

    loader = build_dataloader(
        config,
        dry_run=dry_run,
        pin_memory=use_cuda,
        progress=progress,
    )
    if max_train_batches is not None:
        total_batches = min(len(loader), max_train_batches) if not dry_run else 1
    else:
        total_batches = len(loader) if not dry_run else 1
    progress.total_batches = max(total_batches, 1)

    skip_checkpoint = bool(training_cfg.get("skip_checkpoint", False))
    checkpoint_dir = resolve_checkpoint_dir(config)
    save_every = config.get("checkpoints", {}).get("save_every_epochs", 5)
    roi_cfg = config.get("roi_detector", {})
    pretrained_backbone = bool(roi_cfg.get("pretrained", True))

    with redirect_stderr_to_stdout():
        roi_detector = ROIDetector(
            input_size=int(roi_cfg.get("input_res", roi_cfg.get("input_size", height))),
            pretrained=pretrained_backbone,
        ).to(device)
        codec = VAECodec(latent_channels=model_cfg["latent_ch"]).to(device)
    progress.setup("Models loaded on GPU")
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

    last_metrics: dict[str, float] = {}

    data_wait_total = 0.0
    compute_total = 0.0
    batch_wait_start = time.perf_counter()

    try:
        for epoch in range(1, epochs + 1):
            codec.train()
            roi_detector.train()
            epoch_loss = 0.0
            batches = 0
            epoch_start = time.perf_counter()
            video_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

            for batch in loader:
                data_wait_total += time.perf_counter() - batch_wait_start
                compute_start = time.perf_counter()
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
                    latent_channels=model_cfg["latent_ch"],
                    latent_h=model_cfg["latent_h"],
                    latent_w=model_cfg["latent_w"],
                )
                epoch_loss += avg_batch_loss
                batches += 1
                compute_total += time.perf_counter() - compute_start

                if batches % log_every == 0:
                    elapsed = time.perf_counter() - epoch_start
                    logger.info(
                        "Epoch %s batch %s — loss=%.6f elapsed=%.1fs data_wait=%.2fs compute=%.2fs",
                        epoch,
                        batches,
                        avg_batch_loss,
                        elapsed,
                        data_wait_total / batches,
                        compute_total / batches,
                    )
                if total_batches > 0:
                    progress.training(epoch=epoch, batch=batches)

                if dry_run:
                    break
                if max_train_batches is not None and batches >= max_train_batches:
                    logger.info("Reached max_train_batches=%s", max_train_batches)
                    break

                batch_wait_start = time.perf_counter()

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

            if not skip_checkpoint and (epoch % save_every == 0 or epoch == epochs):
                ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                save_checkpoint(ckpt_path, epoch, codec, roi_detector, optimizer, last_metrics)
            elif skip_checkpoint:
                logger.info("Skipping checkpoint save (skip_checkpoint=true)")
    finally:
        shutdown_dataloader(loader)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    metrics_path = Path(optional_env("TRAIN_METRICS_PATH", "train_metrics.json"))
    progress.finalize("Writing metrics")
    write_metrics_file(metrics_path, last_metrics)
    progress.complete("Training finished successfully")
    logger.info("Training finished successfully")
    _release_training_models(
        codec=codec,
        roi_detector=roi_detector,
        optimizer=optimizer,
        device=device,
    )
    return last_metrics


def main() -> None:
    args = parse_args()
    try:
        train(args.config, args.params, dry_run=args.dry_run)
    except Exception:
        logger.exception("Training failed")
        finalize_job_exit(1)
    finalize_job_exit(0)


if __name__ == "__main__":
    main()
