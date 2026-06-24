# Experiment data: `v100-kappa-2.0-18ep-bs24`

Machine-readable metrics exported from the DataSphere V100 training run (2026-06-18). Use these files for paper plots and cross-experiment comparison.

**Narrative reports:** [English](../v100-kappa-2.0-18ep.en.md) · [Russian](../v100-kappa-2.0-18ep.ru.md)

**Source:** local `logs_debug/` export (stdout, diagnostics TSV, job JSON). Raw logs are not committed; this folder is the canonical in-repo record.

---

## Files

| File | Description |
|------|-------------|
| `metrics.json` | Full export: config, aggregates, per-epoch table, checkpoints, resource summaries, timeline |
| `epochs.csv` | One row per epoch — **primary table for loss vs epoch plots** |
| `batch_progress.csv` | Logged batches (every 50) — loss, `data_wait_s`, `compute_s` |
| `gpu_samples.csv` | Time series: GPU util % and VRAM fraction (~4 s sampling) |
| `cpu_samples.csv` | Time series: Docker container CPU % and memory % |
| `checkpoints.csv` | S3 and local paths for saved checkpoints |

---

## Key columns

### `epochs.csv`

| Column | Meaning |
|--------|---------|
| `epoch` | 1…18 |
| `avg_loss` | Mean `ClinicalLoss` over the epoch (includes −PSNR; more negative = better) |
| `elapsed_s` / `elapsed_min` | Wall time for the epoch |
| `batch_50_data_wait_s` | I/O wait at start of epoch (batch 50) |
| `batch_1500_compute_s` | GPU compute time near end of epoch |
| `completed_at_utc` | Epoch completion timestamp |

### `metrics.json` → `aggregate`

| Field | Value (this run) |
|-------|------------------|
| `loss_first` | −17.443 |
| `loss_last` | −23.327 |
| `total_training_h` | 8.21 |
| `mean_epoch_min` | 27.37 |

### `metrics.json` → `resource_utilization`

| Field | Value |
|-------|-------|
| `gpu_util_pct.mean` | 97.3% |
| `gpu_mem_pct.mean_percent_vram` | 83.7% (~27 GB on V100 32 GB) |
| `cpu_util_pct.mean` | 156% (Docker multi-core; ≈1.56 cores busy) |

---

## Plotting examples (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

epochs = pd.read_csv("epochs.csv")
plt.plot(epochs["epoch"], epochs["avg_loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average training loss")
plt.savefig("loss_curve.pdf")
```

```python
gpu = pd.read_csv("gpu_samples.csv")
gpu["timestamp_utc"] = pd.to_datetime(gpu["timestamp_utc"])
gpu.plot(x="timestamp_utc", y="gpu_util_pct")
```

---

## Checkpoints on Object Storage

Relative to bucket mount: `checkpoints/v100-kappa-2.0-18ep-bs24/epoch_{005,010,015,018}.pt`

Full paths in `checkpoints.csv` and `metrics.json`.
