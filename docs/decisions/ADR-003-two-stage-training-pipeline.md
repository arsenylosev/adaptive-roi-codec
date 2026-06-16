# ADR-003: Two-stage Datasphere pipeline and GPU training fixes

## Status

Accepted

## Date

2026-06-15

## Context

The first cloud smoke job (`epochs: 1`, `g1.1` V100) ran for **~13 hours** with **0% GPU utilization** and ~400% CPU load. Logs in `logs_debug/` showed:

1. **`Device: cpu`** in `stdout.log` — training never used the GPU.
2. **`CUDA initialization: NVIDIA driver on your system is too old (found version 12020)`** in `stderr.log` — `python: auto` installed **PyTorch 2.12** with bundled **CUDA 13**, incompatible with DataSphere `g1.1` host drivers.
3. **No epoch completion lines** in stdout — the job was not idle; it was slowly training on **CPU** while decoding MP4s over **S3 FUSE** with OpenCV.
4. Per-frame training loop with effective batch size 1 and no step logging made progress invisible.

Billing impact: a full **GPU instance** was charged while compute was **CPU-bound** on video I/O and incompatible PyTorch wheels.

Related: [ADR-001](ADR-001-datasphere-jobs-training.md), [ADR-002](ADR-002-native-336-training-resolution.md).

## Decision

### 1. Pin GPU PyTorch for Datasphere Jobs

Replace `env.python: auto` on GPU jobs with **manual** environment and `jobs/requirements-datasphere-gpu.txt`:

- `torch==2.4.1+cu121` / `torchvision==0.19.1+cu121` from the PyTorch cu121 index
- Compatible with DataSphere V100/T4 driver CUDA 12.x

Set `TRAIN_REQUIRE_CUDA=1` so jobs **fail fast** if CUDA is unavailable instead of silently falling back to CPU on a GPU VM.

### 2. Two-stage cloud pipeline

| Stage | Job | Instance | Purpose |
|-------|-----|----------|---------|
| **1 — Extract** | `uv run launch-train --job extract --execute` | `c1.8` (CPU) | Decode MP4s once; write `336×336` `.pt` frames + `frames_manifest.jsonl` to S3 `processed/frames/` |
| **2 — Train** | `uv run launch-train --execute` with `data.source: preprocessed` | `gt4.1` (smoke) / `g1.1` (full runs) | Load `.pt` tensors; GPU training with pinned CUDA stack |

Stage 1 amortizes decode cost across all training runs and keeps GPU jobs off S3+OpenCV hot paths.

### 3. Training loop improvements (same repo, stage 2)

- **Per-video temporal state** (`video_states`) for correct motion-compensation across batches
- **Gradient accumulation** over batch items (optimizer step per loaded batch)
- **`log_every_batches`** progress logging (default 20)
- **OpenCV seek** (`CAP_PROP_POS_FRAMES`) instead of decoding every frame between strides
- **`num_workers` + `pin_memory`** when `device=cuda`
- **`KvasirPreprocessedFrameDataset`** for stage-1 output

### 4. Instance type policy

| Config | GPU | vCPU | Use |
|--------|-----|------|-----|
| `c1.8` | — | 8 | Stage-1 frame extraction |
| `gt4.1` | T4 16 GB | 4 | Smoke tests, ablations |
| `g1.1` | V100 32 GB | 8 | Full 50-epoch runs after pipeline validated |
| `g2.1` | A100 80 GB | 28 | **Not default** — only if profiling shows GPU saturation |

V100 cannot be fully utilized until the data path delivers batches faster than the model forward pass; stage-1 + preprocessed tensors is the primary lever.

### 5. Smoke-test parameters

Use `jobs/inputs/train_smoke.json`:

- `data.source: preprocessed`
- `epochs: 1`
- `log_every_batches: 10`

Run stage 1 once per dataset revision; reuse `processed/frames/` for multiple training jobs.

## Alternatives considered

### Single GPU job decoding MP4 from S3 (status quo)

- **Rejected:** 13 h CPU-bound run; FUSE + OpenCV + incompatible PyTorch; wastes GPU billing.

### `python: auto` with local torch 2.12

- **Rejected:** Auto mode snapshots operator venv; PyTorch 2.12 CUDA 13 wheels break on DataSphere drivers.

### Pre-encode to VAE latents offline

- **Deferred:** Smaller storage and faster epochs, but ties preprocessing to a fixed encoder checkpoint; frame tensors are checkpoint-agnostic and sufficient for phase 1.

### Larger `frame_stride` only

- **Insufficient:** Reduces steps but does not fix CPU fallback or S3 decode latency per step.

## Consequences

- Operators run **two job submissions** for the full cloud path (extract → train).
- `jobs/requirements-datasphere-gpu.txt` must be kept in sync with DataSphere driver capabilities.
- Local development: `uv run extract-frames` then `data.source: preprocessed` or continue `source: video` for small local folders.
- ADR-001 remains valid for S3 mounts and launcher pattern; GPU job env section is superseded by manual requirements here.

## Verification checklist

After deploying these changes:

- [ ] GPU job stdout shows `Device: cuda` and GPU name
- [ ] `TRAIN_REQUIRE_CUDA=1` aborts if CUDA missing
- [ ] Stage-1 writes `…/processed/frames/frames_manifest.jsonl` on S3
- [ ] Stage-2 smoke completes epoch 1 in minutes (not hours) with batch logs
- [ ] `gpu_stats.tsv` shows non-zero utilization during training steps

## References

- [DataSphere job runtime environment](https://yandex.cloud/en/docs/datasphere/concepts/jobs/environment)
- [Computing resource configurations](https://yandex.cloud/en/docs/datasphere/concepts/configurations)
- Debug logs: `logs_debug/` (2026-06-15 failed run)
