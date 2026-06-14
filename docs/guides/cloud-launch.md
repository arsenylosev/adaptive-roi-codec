# Launch experiment on Yandex Cloud

Step-by-step guide for uploading Kvasir-Capsule, wiring DataSphere Jobs, and starting GPU training at **native 336×336** resolution.

Related decisions:

- [ADR-001: Datasphere Jobs training](../decisions/ADR-001-datasphere-jobs-training.md)
- [ADR-002: Native 336×336 training](../decisions/ADR-002-native-336-training-resolution.md)

## 1. Prepare the dataset locally

Your folder should match:

```
kvasir-capsule/
├── MANIFEST.json                 # generated
├── splits/
│   ├── train_videos.txt
│   ├── val_videos.txt
│   └── test_videos.txt
├── raw/
│   ├── labelled_videos/
│   │   └── {video_id}.mp4        # 336×336, 30 fps
│   └── labelled_images/
│       └── {class}.tar.gz
├── labelled_videos/              # acceptable legacy layout (local only)
└── labelled_images/
```

Generate manifest and splits from the repository root:

```bash
uv sync
uv run build-dataset-manifest --dataset-root kvasir-capsule
```

This writes `MANIFEST.json` (video count, native 336×336 resolution, class counts) and an 80/10/10 split over available video IDs.

Optional: reorganize into `raw/` before upload:

```bash
mkdir -p kvasir-capsule/raw
mv kvasir-capsule/labelled_videos kvasir-capsule/raw/
mv kvasir-capsule/labelled_images kvasir-capsule/raw/
```

## 2. Create Object Storage bucket

Console: **Object Storage → Create bucket** (authorized access only).

```bash
yc storage bucket create --name adaptive-roi-codec-data
yc iam access-key create --service-account-name <service-account>
```

Put credentials in `.env` (see `.env.example`).

Assign the service account **`storage.editor`** (or `storage.admin`) on the folder or bucket. Without this role, uploads fail even with valid keys.

## 3. Upload to Object Storage

Yandex Object Storage is S3-compatible. You can upload with **`yc storage s3`** (recommended) or **AWS CLI**.

### Option A — `yc storage s3` (recommended)

Uses your `yc init` login — no static keys in the shell, fewer signing mistakes.

Preflight (check bucket access):

```bash
yc storage bucket list
yc storage s3api list-objects --bucket "$S3_BUCKET" --max-keys 5
```

Upload dataset (legacy local layout `labelled_videos/` → S3 prefix `raw/labelled_videos/`):

```bash
export S3_BUCKET=adaptive-roi-codec-data   # or from .env

# Small metadata first
yc storage s3 cp kvasir-capsule/MANIFEST.json \
  s3://$S3_BUCKET/kvasir-capsule/MANIFEST.json

yc storage s3 cp kvasir-capsule/splits/ \
  s3://$S3_BUCKET/kvasir-capsule/splits/ --recursive

# Videos (~31 GB)
yc storage s3 cp kvasir-capsule/labelled_videos/ \
  s3://$S3_BUCKET/kvasir-capsule/raw/labelled_videos/ --recursive

# Label archives (~521 MB)
yc storage s3 cp kvasir-capsule/labelled_images/ \
  s3://$S3_BUCKET/kvasir-capsule/raw/labelled_images/ --recursive
```

If you already moved folders under `raw/` locally, change the source paths to `kvasir-capsule/raw/labelled_videos/`, etc.

Docs: [AWS CLI tools for Object Storage](https://yandex.cloud/en/docs/storage/tools/aws-cli) (same S3 API; `yc storage s3` is a built-in wrapper).

### Option B — AWS CLI with static access keys

**Do not use `uvx --with awscli` for large uploads** unless you explicitly export all three variables below — `uvx` ignores `aws configure` and often omits the region, which causes:

`SignatureDoesNotMatch … CreateMultipartUpload`

Required environment (from `yc iam access-key create`):

```bash
set -a && source .env && set +a

# All three are mandatory for Yandex Object Storage:
echo "$AWS_ACCESS_KEY_ID" | head -c 8    # should print first chars of key ID
echo "$AWS_DEFAULT_REGION"               # MUST print: ru-central1
# AWS_SECRET_ACCESS_KEY must be the *secret* field, not the key ID
```

One-time configure (persists in `~/.aws/`):

```bash
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region ru-central1
aws configure set endpoint_url https://storage.yandexcloud.net
```

Preflight — upload a tiny test object:

```bash
echo test > /tmp/yc-upload-test.txt
aws s3 cp /tmp/yc-upload-test.txt \
  s3://$S3_BUCKET/kvasir-capsule/_preflight/upload-test.txt
aws s3 rm s3://$S3_BUCKET/kvasir-capsule/_preflight/upload-test.txt
```

If preflight succeeds, upload the dataset (`--endpoint-url` before `s3` is the form Yandex documents):

```bash
aws --endpoint-url=https://storage.yandexcloud.net s3 cp \
  kvasir-capsule/MANIFEST.json \
  s3://$S3_BUCKET/kvasir-capsule/MANIFEST.json

aws --endpoint-url=https://storage.yandexcloud.net s3 cp \
  kvasir-capsule/splits/ \
  s3://$S3_BUCKET/kvasir-capsule/splits/ --recursive

aws --endpoint-url=https://storage.yandexcloud.net s3 cp \
  kvasir-capsule/labelled_videos/ \
  s3://$S3_BUCKET/kvasir-capsule/raw/labelled_videos/ --recursive

aws --endpoint-url=https://storage.yandexcloud.net s3 cp \
  kvasir-capsule/labelled_images/ \
  s3://$S3_BUCKET/kvasir-capsule/raw/labelled_images/ --recursive
```

### Verify upload

```bash
yc storage s3api list-objects \
  --bucket "$S3_BUCKET" \
  --prefix kvasir-capsule/raw/labelled_videos/ \
  --max-keys 1000 | grep -c key
# Expect: 43

yc storage s3api list-objects \
  --bucket "$S3_BUCKET" \
  --prefix kvasir-capsule/raw/labelled_images/ \
  --max-keys 100 | grep -c key
# Expect: 14
```

### Upload path mapping

| Local (your folder) | S3 key prefix |
|---------------------|---------------|
| `kvasir-capsule/MANIFEST.json` | `kvasir-capsule/MANIFEST.json` |
| `kvasir-capsule/splits/` | `kvasir-capsule/splits/` |
| `kvasir-capsule/labelled_videos/` | `kvasir-capsule/raw/labelled_videos/` |
| `kvasir-capsule/labelled_images/` | `kvasir-capsule/raw/labelled_images/` |

Training resolves videos at `/job/s3/<connector>/kvasir-capsule/raw/labelled_videos/` (or legacy `labelled_videos/`).

## 4. DataSphere S3 connector

1. Open your DataSphere project → **Create resource → S3 connector**
2. Endpoint: `https://storage.yandexcloud.net`
3. Bucket: your bucket name (no dots in the name)
4. Mode: **Read and write** (checkpoints)
5. **Activate** the connector
6. Copy connector ID → `S3_CONNECTOR_ID` in `.env`

On the worker, data appears at:

```
/job/s3/<S3_CONNECTOR_ID>/kvasir-capsule/raw/labelled_videos/
/job/s3/<S3_CONNECTOR_ID>/checkpoints/<experiment_id>/
```

Docs: [S3 connector](https://yandex.cloud/en/docs/datasphere/operations/data/s3-connectors)

## 5. Configure environment

```bash
cp .env.example .env
```

Required values:

| Variable | Example |
|----------|---------|
| `DATASPHERE_PROJECT_ID` | DataSphere project ID |
| `S3_CONNECTOR_ID` | Connector ID from step 4 |
| `S3_DATA_PREFIX` | `kvasir-capsule` |
| `S3_CHECKPOINT_SUBDIR` | `checkpoints` |
| `S3_BUCKET` | `adaptive-roi-codec-data` |

Install tooling:

```bash
uv sync --extra cloud --extra dev
yc init
```

## 6. Validate locally before GPU spend

```bash
# Unit tests
uv run pytest

# CPU smoke test (synthetic 336×336, one batch)
uv run python -m adaptive_roi_codec.train \
  --config configs/base.yaml \
  --params jobs/inputs/train_input.json \
  --dry-run

# Optional: one real video batch locally (slow on CPU)
TRAIN_DRY_RUN=0 uv run python -m adaptive_roi_codec.train \
  --config configs/base.yaml \
  --params jobs/inputs/train_input.json
```

Limit local real-video testing with overrides in `jobs/inputs/train_input.json`:

```json
{
  "experiment_id": "local-smoke",
  "data": { "max_frames_per_video": 2, "frame_stride": 300 }
}
```

## 7. Configure experiment

Edit `jobs/inputs/train_input.json`:

```json
{
  "experiment_id": "baseline-kappa-2.0",
  "training": { "epochs": 50 },
  "quantizer": { "kappa": 2.0 },
  "data": { "split": "train", "frame_stride": 30 }
}
```

Training defaults (`configs/base.yaml`):

| Parameter | Value |
|-----------|-------|
| Input resolution | 336×336 |
| Latent (16× down) | 21×21×192 |
| ROI detector input | 336 |
| Batch size | 4 |
| Checkpoint interval | every 5 epochs |

## 8. Launch Datasphere Job

Render job config:

```bash
uv run launch-train --dry-run
# Review jobs/configs/.generated/job_train.yaml
```

Submit:

```bash
uv run launch-train --execute
```

First run recommendation: set `"epochs": 1` in `train_input.json` to validate the full cloud path before a 50-epoch job.

## 9. Monitor and collect artifacts

| Location | Content |
|----------|---------|
| DataSphere UI → Jobs → Launch history | Progress, logs |
| Local job directory | `job_progress.jsonl` |
| Job output | `metrics/train_metrics.json` |
| S3 | `checkpoints/<experiment_id>/epoch_*.pt` |

## 10. Experiment matrix (spec §7.2)

Run separate jobs with distinct `experiment_id` values:

| Experiment | Parameter | Values |
|------------|-----------|--------|
| 1 | κ | 1.0, 1.5, 2.0, 2.5 |
| 2 | λ_ROI | 1.0, 1.5, 2.0 |
| 3 | Target bitrate | 1.5, 2.0, 2.5 Mbps (future entropy stage) |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| **`SignatureDoesNotMatch` on upload** | Missing/wrong `AWS_DEFAULT_REGION`, swapped key ID/secret, or `uvx` without env | Use **`yc storage s3 cp`** (Option A), or export `AWS_DEFAULT_REGION=ru-central1` + run preflight test (Option B) |
| **`AccessDenied`** | Service account lacks `storage.editor` | Assign role on folder/bucket in IAM |
| `Video directory not found` | Wrong S3 prefix layout | Upload under `kvasir-capsule/raw/labelled_videos/` |
| `Split file not found` | Missing splits on S3 | Upload `splits/` or run manifest builder before upload |
| Job fails on import cv2 | OpenCV missing in venv | Run `uv sync` before `launch-train --execute` (`python: auto`) |
| OOM on V100 | batch 4 at full stride | Lower `batch_size` or increase `frame_stride` |

## Cost note

Spec estimate: ~11,600 ₽ for ~20 h V100 training. Reduce cost with higher `frame_stride`, fewer epochs for ablations, and `epochs: 1` smoke jobs first.
