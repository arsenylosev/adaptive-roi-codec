# Experiment Report: V100 Training Run (18 Epochs)

**Experiment ID:** `v100-kappa-2.0-18ep-bs24`  
**Status:** Success (DataSphere Job exit code 0)  
**Completion time (UTC):** 2026-06-18 10:31:04  
**Checkpoints:** Object Storage `checkpoints/v100-kappa-2.0-18ep-bs24/` (epochs 5, 10, 15, 18)

### Numerical data (for plots and tables)

Canonical machine-readable export (committed):

| File | Use |
|------|-----|
| [metrics.json](v100-kappa-2.0-18ep-bs24/metrics.json) | Full structured record (config, aggregates, timeline) |
| [epochs.csv](v100-kappa-2.0-18ep-bs24/epochs.csv) | **Loss vs epoch**, epoch duration, I/O timing |
| [batch_progress.csv](v100-kappa-2.0-18ep-bs24/batch_progress.csv) | Intra-epoch loss and `data_wait` / `compute` |
| [gpu_samples.csv](v100-kappa-2.0-18ep-bs24/gpu_samples.csv) | GPU utilization and VRAM time series |
| [cpu_samples.csv](v100-kappa-2.0-18ep-bs24/cpu_samples.csv) | Container CPU time series |
| [checkpoints.csv](v100-kappa-2.0-18ep-bs24/checkpoints.csv) | Checkpoint paths |

Format reference: [v100-kappa-2.0-18ep-bs24/README.md](v100-kappa-2.0-18ep-bs24/README.md).  
Raw log dump used for export: `logs_debug/` (local, not committed).

---

## Summary

First full multi-epoch GPU training run of the adaptive ROI neural video codec on preprocessed Kvasir-Capsule frames. The job completed **18 epochs** over **54 088 training frames** without failure. Total training loss improved from **−17.44** (epoch 1) to **−23.33** (epoch 18). Checkpoints were written to Object Storage at epochs 5, 10, 15, and 18.

> **Note on loss sign:** `ClinicalLoss` includes **−PSNR** in `L_base`, so more negative values indicate better reconstruction quality during training (this is expected, not a numerical error).

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Platform | Yandex DataSphere Jobs, `g1.1` (NVIDIA Tesla V100 32 GB) |
| Job params file | `jobs/inputs/train_v100_scale.json` |
| Effective `batch_size` | **36** (job env `TRAIN_BATCH_SIZE=36`; experiment name retained `bs24`) |
| Epochs | 18 |
| Learning rate | 1×10⁻⁴ (from `configs/base.yaml`) |
| Quantizer κ | 2.0 |
| ROI backbone | MobileNetV3-large, **pretrained** (ImageNet weights downloaded on first run) |
| Input resolution | 336×336×3 |
| Latent shape | 192×21×21 |
| Data source | Preprocessed `.npy` frames on S3 |
| Train split frames | 54 088 (full manifest, no `max_frames` cap) |
| Batches per epoch | 1 503 |
| DataLoader | `num_workers=8`, `prefetch_factor=4`, `shuffle=true` |
| Staging | `stage_mode=bulk` requested; **disabled at runtime** (see Infrastructure) |
| Checkpoints | Every 5 epochs + final epoch → `checkpoints/v100-kappa-2.0-18ep-bs24/` |

### Loss weights (base config)

| Component | Weight |
|-----------|--------|
| α (PSNR / SSIM) | 0.5 |
| λ_ROI | 1.5 |
| λ_rate | 0.01 |
| λ_temp | 0.1 |
| β₀ | 0.01 |

---

## Results

### Per-epoch metrics

Parsed from job stdout into [epochs.csv](v100-kappa-2.0-18ep-bs24/epochs.csv). Summary:

| Epoch | Avg loss | Batches | Epoch time |
|------:|---------:|--------:|-----------:|
| 1 | −17.443 | 1 503 | 29.1 min |
| 2 | −20.089 | 1 503 | 27.1 min |
| 3 | −20.632 | 1 503 | 27.1 min |
| 4 | −21.019 | 1 503 | 27.1 min |
| 5 | −21.266 | 1 503 | 27.1 min |
| 6 | −21.470 | 1 503 | 27.1 min |
| 7 | −21.734 | 1 503 | 27.1 min |
| 8 | −21.930 | 1 503 | 27.1 min |
| 9 | −22.090 | 1 503 | 27.1 min |
| 10 | −22.258 | 1 503 | 27.3 min |
| 11 | −22.428 | 1 503 | 27.1 min |
| 12 | −22.613 | 1 503 | 28.6 min |
| 13 | −22.755 | 1 503 | 27.1 min |
| 14 | −22.844 | 1 503 | 27.5 min |
| 15 | −23.019 | 1 503 | 27.1 min |
| 16 | −23.137 | 1 503 | 27.1 min |
| 17 | −23.202 | 1 503 | 27.1 min |
| 18 | −23.327 | 1 503 | 27.8 min |

**Aggregate**

| Metric | Value |
|--------|-------|
| Total epoch time (sum) | **8.21 h** ([metrics.json](v100-kappa-2.0-18ep-bs24/metrics.json)) |
| Mean epoch duration | **27.37 min** |
| Wall-clock job duration | **8.23 h** (includes weight download + epoch 1 cold start) |
| Loss change (epoch 1 → 18) | **−5.88** (improving) |
| Final epoch avg loss | **−23.327** |

### Job output (`train_metrics.json`)

The DataSphere job output file contained **only the final epoch** (legacy format at run time). The committed [metrics.json](v100-kappa-2.0-18ep-bs24/metrics.json) restores the full **18-epoch history** from stdout. New jobs write `epochs[]` directly via `build_train_metrics_report` in `adaptive_roi_codec/train.py`.

### Checkpoints (S3)

| Epoch | Path |
|------:|------|
| 5 | `…/checkpoints/v100-kappa-2.0-18ep-bs24/epoch_005.pt` |
| 10 | `…/checkpoints/v100-kappa-2.0-18ep-bs24/epoch_010.pt` |
| 15 | `…/checkpoints/v100-kappa-2.0-18ep-bs24/epoch_015.pt` |
| 18 | `…/checkpoints/v100-kappa-2.0-18ep-bs24/epoch_018.pt` |

---

## Performance and resource usage

Diagnostics sampled during the run ([gpu_samples.csv](v100-kappa-2.0-18ep-bs24/gpu_samples.csv), [cpu_samples.csv](v100-kappa-2.0-18ep-bs24/cpu_samples.csv)):

| Metric | Value | Comment |
|--------|-------|---------|
| GPU utilization (avg) | **97.3%** | Strong saturation vs earlier smoke runs (~17–56%) |
| GPU memory (avg) | **83.7%** | ≈ **27 GB / 32 GB** VRAM at `batch_size=36` |
| Container CPU (avg, Docker) | **156%** | Multi-core reporting; ≈ **1.56 cores** busy on 8 vCPU |
| Epoch 1 `data_wait` at batch 50 | **6.04 s** | S3 FUSE cold start (staging off) |
| Epoch 1 `data_wait` at batch 1500 | **0.20 s** | FUSE / cache warmed up |
| Epoch 18 `compute` at batch 1500 | **19.48 s** | Large batch forward+backward |
| Extended SSD used | **5.3 GB** | Checkpoints + torch hub cache |
| Root overlay | 65% → 69% used | No ENOSPC failure |

### Bulk staging

Log at startup:

```text
Disabling stage_frames_local: need ~68.2 GB on …/frame_cache but cache filesystem is too full
```

Extended SSD had **~140 GB free** (`df_before.out`); the headroom check incorrectly rejected bulk copy on this build. Training proceeded with **direct S3 reads**. A fix to the disk check is in the repository (`resolve_staging_disk_path`); enabling bulk staging on the next run should further reduce `data_wait` at epoch start.

---

## Timeline (UTC)

| Event | Time |
|-------|------|
| Job start / CUDA init | 2026-06-18 02:17:17 |
| Dataset init (54 088 frames) | 02:18:00 |
| Epoch 1 complete | 02:47:09 |
| Checkpoint epoch 5 | 04:35:41 |
| Checkpoint epoch 10 | 06:51:27 |
| Checkpoint epoch 15 | 09:08:56 |
| Epoch 18 complete | 10:30:58 |
| Metrics + job success | 10:31:04 |

Wall-clock job duration: **8 h 14 min** (8.23 h in [metrics.json](v100-kappa-2.0-18ep-bs24/metrics.json); includes MobileNet weight download and epoch 1 cold start).

---

## Conclusions

1. **Pipeline validated:** 18-epoch V100 training is stable end-to-end (metrics, checkpoints, job status Success).
2. **Scaling effective:** `batch_size=36` uses most of V100 VRAM (~27 GB) and achieves ~97% average GPU utilization.
3. **Loss still decreasing at epoch 18:** No clear plateau; longer runs or validation metrics (PSNR, ROI Dice) are needed before drawing quality conclusions.
4. **Follow-ups for paper-grade evaluation:**
   - Enable bulk SSD staging and re-measure epoch-1 I/O.
   - Run validation on `val` split with frozen metrics (`PSNR`, MS-SSIM, bitrate).
   - Per-epoch metrics JSON is implemented for future runs; this run’s history is in [metrics.json](v100-kappa-2.0-18ep-bs24/metrics.json).
   - Compare κ ablations (1.0, 1.5, 2.5) per [experiment plan](../opisanie-eksperimenta.md).

---

## References

- Experiment plan (RU): [docs/opisanie-eksperimenta.md](../opisanie-eksperimenta.md)
- Job params: [jobs/inputs/train_v100_scale.json](../../jobs/inputs/train_v100_scale.json)
- Metrics data: [v100-kappa-2.0-18ep-bs24/](v100-kappa-2.0-18ep-bs24/)
- Raw logs: `logs_debug/` (local export used to build metrics; not committed)
