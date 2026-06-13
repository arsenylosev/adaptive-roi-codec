# Adaptive ROI Neural Video Codec

Neural video codec for capsule endoscopy with adaptive bitrate allocation across clinically significant regions of interest (ROI). See [MCE-Experiment-Requirements-EN.md](MCE-Experiment-Requirements-EN.md) for the full experiment specification.

Heavy training runs on **Yandex Datasphere Jobs** (GPU). Local execution is limited to debugging and smoke tests.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [Yandex Cloud CLI](https://yandex.cloud/en/docs/cli/quickstart) (`yc`) authenticated
- DataSphere project with an S3 connector to Object Storage (Kvasir-Capsule dataset)
- `datasphere` Python package for launching jobs (installed via optional dependency)

## Setup

```bash
# Install project + cloud launcher dependencies
uv sync --extra cloud

# Configure secrets locally
cp .env.example .env
# Edit .env: DATASPHERE_PROJECT_ID, S3_CONNECTOR_ID, bucket credentials
```

Authenticate with Yandex Cloud before launching jobs:

```bash
yc init
```

## Project layout

```
adaptive_roi_codec/     # Python package (VAE, ROI detector, losses, training)
configs/                # Training hyperparameters (base.yaml, quantizer.yaml)
jobs/
  configs/              # Datasphere Jobs YAML template
  inputs/               # Per-run JSON overrides for training
metrics/                # Job output metrics (downloaded after run)
```

## Local smoke test (CPU, synthetic data)

```bash
uv run python -m adaptive_roi_codec.train \
  --config configs/base.yaml \
  --params jobs/inputs/train_input.json \
  --dry-run
```

## Launch training on Datasphere Jobs

1. Upload Kvasir-Capsule to Object Storage (see spec §6.3).
2. Attach the bucket to your DataSphere project as an **S3 connector**.
3. Put connector and project IDs into `.env`.
4. Render and review the generated job config:

```bash
uv run launch-train --dry-run
```

Generated config path: `jobs/configs/.generated/job_train.yaml`

5. Launch the job (uses `env.python: auto` — dependencies are collected from your local venv):

```bash
uv run launch-train --execute
```

Equivalent manual command:

```bash
datasphere project job execute \
  -p "$DATASPHERE_PROJECT_ID" \
  -c jobs/configs/.generated/job_train.yaml
```

### Job configuration highlights

| Setting | Value | Notes |
|---------|-------|-------|
| GPU | `g1.1` (V100), fallback `g2.1` | Priority list in job YAML |
| Working storage | 150 GB SSD | Dataset cache / temp files |
| S3 mount | `${S3_CONNECTOR_ID}` | Dataset + checkpoints at `/job/s3/<id>/` |
| Checkpoints | every 5 epochs | Written to Object Storage via S3 mount |
| Outputs | `metrics/train_metrics.json` | Downloaded to local `metrics/` after job |

Track progress in the DataSphere UI (**Jobs → Launch history**) or via `job_progress.jsonl` in the job log directory.

## Experiment overrides

Edit `jobs/inputs/train_input.json` to change κ, λ_ROI, epoch count, or experiment ID without modifying `configs/base.yaml`:

```json
{
  "experiment_id": "kappa-2.0",
  "training": { "epochs": 50 },
  "quantizer": { "kappa": 2.0 }
}
```

## Security

- Store API keys and cloud IDs only in `.env` (gitignored).
- Use `.env.example` as a template for other machines.
- Do not commit checkpoints or the 61 GB dataset.

## References

Specification and paper targets: [MCE-Experiment-Requirements-EN.md](MCE-Experiment-Requirements-EN.md)

Yandex Cloud docs:

- [DataSphere Jobs](https://yandex.cloud/en/docs/datasphere/concepts/jobs/)
- [Running jobs](https://yandex.cloud/en/docs/datasphere/operations/projects/work-with-jobs)

Documentation references during development:

- **Yandex Cloud MCP** — Datasphere Jobs, Object Storage, CLI
- **Context7 MCP** — PyTorch, uv, and other library docs
