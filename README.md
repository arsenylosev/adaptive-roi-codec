# Adaptive ROI Neural Video Codec

Neural video codec for capsule endoscopy with adaptive bitrate allocation across clinically significant regions of interest (ROI). See [MCE-Experiment-Requirements-EN.md](MCE-Experiment-Requirements-EN.md) for the full experiment specification.

Training uses **native Kvasir-Capsule 336×336** video frames ([ADR-002](docs/decisions/ADR-002-native-336-training-resolution.md)). Heavy training runs on **Yandex Datasphere Jobs** (GPU). Local execution is for manifest generation, smoke tests, and debugging.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [Yandex Cloud CLI](https://yandex.cloud/en/docs/cli/quickstart) (`yc`) authenticated
- DataSphere project with an S3 connector to Object Storage (Kvasir-Capsule dataset)
- `datasphere` Python package for launching jobs (installed via optional dependency)

## Setup

```bash
uv sync --extra cloud --extra dev
cp .env.example .env
# Edit .env: DATASPHERE_PROJECT_ID, S3_CONNECTOR_ID, bucket credentials
yc init
```

## Dataset manifest and splits

With Kvasir-Capsule in `kvasir-capsule/`:

```bash
uv run build-dataset-manifest --dataset-root kvasir-capsule
```

Creates `MANIFEST.json` and `splits/{train,val,test}_videos.txt`.

## Project layout

```
adaptive_roi_codec/     # VAE, ROI detector, losses, training, CLI
configs/                # base.yaml — 336×336 defaults
docs/guides/            # cloud launch instructions
jobs/                   # Datasphere Jobs template + per-run JSON params
kvasir-capsule/         # local dataset (gitignored)
```

## Local smoke test

```bash
uv run pytest
uv run python -m adaptive_roi_codec.train \
  --config configs/base.yaml \
  --params jobs/inputs/train_input.json \
  --dry-run
```

## Launch training on Yandex Cloud

Full step-by-step guide: **[docs/guides/cloud-launch.md](docs/guides/cloud-launch.md)**

Quick path:

```bash
# 1. Upload kvasir-capsule/{MANIFEST.json,splits/,raw/} to Object Storage
# 2. Create DataSphere S3 connector → set S3_CONNECTOR_ID in .env
uv run launch-train --dry-run    # review generated YAML
uv run launch-train --execute    # submit GPU job
```

Checkpoints: `s3://…/checkpoints/<experiment_id>/epoch_*.pt`

## Experiment overrides

Edit `jobs/inputs/train_input.json`:

```json
{
  "experiment_id": "baseline-kappa-2.0",
  "training": { "epochs": 50 },
  "quantizer": { "kappa": 2.0 },
  "data": { "split": "train", "frame_stride": 30 }
}
```

## Architecture

ADRs in `docs/decisions/`:

- [ADR-001](docs/decisions/ADR-001-datasphere-jobs-training.md) — Datasphere Jobs + S3 mounts
- [ADR-002](docs/decisions/ADR-002-native-336-training-resolution.md) — 336×336 training resolution

## Security

- Store API keys and cloud IDs only in `.env` (gitignored).
- Do not commit checkpoints or the dataset.

## Contributing

Commit messages follow [Conventional Commits](CONTRIBUTING.md) (`feat:`, `fix:`, `docs:`, etc.).

## References

- [Cloud launch guide](docs/guides/cloud-launch.md)
- [DataSphere S3 connector setup](docs/guides/datasphere-s3-connector.md)
- [MCE-Experiment-Requirements-EN.md](MCE-Experiment-Requirements-EN.md)
- [DataSphere Jobs](https://yandex.cloud/en/docs/datasphere/concepts/jobs/)
