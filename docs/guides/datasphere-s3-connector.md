# DataSphere S3 connector and service account setup

How to connect a Yandex Object Storage bucket to a DataSphere project, obtain `S3_CONNECTOR_ID`, and use a **service account static access key** correctly.

This guide is written for this repository’s training workflow ([cloud launch](cloud-launch.md), [ADR-001](../decisions/ADR-001-datasphere-jobs-training.md)). Official references:

- [Connecting to an S3 storage](https://yandex.cloud/en/docs/datasphere/operations/data/s3-connectors)
- [S3 connector concept](https://yandex.cloud/en/docs/datasphere/concepts/s3-connector)
- [DataSphere Jobs — `s3-mounts`](https://yandex.cloud/en/docs/datasphere/concepts/jobs/)
- [Creating a service account](https://yandex.cloud/en/docs/iam/operations/sa/create)
- [Static access keys](https://yandex.cloud/en/docs/iam/operations/authentication/manage-access-keys)
- [Object Storage IAM roles](https://yandex.cloud/en/docs/storage/security/)
- [DataSphere secrets](https://yandex.cloud/en/docs/datasphere/operations/data/secrets)
- [Tutorial: Object Storage → DataSphere](https://yandex.cloud/en/docs/tutorials/ml-ai/s3-to-datasphere)

## What `S3_CONNECTOR_ID` is

An **S3 connector** is a DataSphere project resource that stores connection settings (endpoint, bucket, access key ID, encrypted secret, read/write mode). DataSphere assigns each connector a **unique resource ID** — that string is `S3_CONNECTOR_ID`.

| Identifier | What it is | Used for |
|------------|------------|----------|
| **`S3_CONNECTOR_ID`** | Connector **resource ID** (e.g. `bt1xxxxxxxxxx`) | `s3-mounts` in job YAML, paths `/job/s3/<id>/…` |
| **Mount name** | Human-readable volume label you choose at creation (lowercase, hyphens) | JupyterLab file browser under `/s3/` |
| **Bucket name** | Object Storage bucket (e.g. `adaptive-roi-codec-data`) | S3 keys inside the mount |
| **Service account ID** | IAM identity (e.g. `aje6o61dvog2…`) | Creating keys and assigning roles — **not** `S3_CONNECTOR_ID` |
| **`key_id` from access key** | AWS-compatible access key ID | Connector field “Static access key ID”; also `AWS_ACCESS_KEY_ID` for local CLI upload |

Do not confuse the IAM access-key record `id` (internal) with `key_id` (the value S3 and the connector expect).

## Where credentials are used (two contexts)

```
┌─────────────────────────────────────────────────────────────────┐
│  Your laptop (dataset upload)                                    │
│  • yc storage s3 cp …  → uses your `yc init` user session       │
│    OR                                                            │
│  • aws / yc with static keys → .env AWS_ACCESS_KEY_ID/SECRET    │
└────────────────────────────┬────────────────────────────────────┘
                             │ same bucket
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Yandex Object Storage bucket                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ mounted via connector
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  DataSphere project                                              │
│  • S3 connector stores key_id + secret (in a DataSphere Secret)  │
│  • Jobs read/write: /job/s3/<S3_CONNECTOR_ID>/…                │
└─────────────────────────────────────────────────────────────────┘
```

You can use the **same service account** (and the same static key pair) for both local upload and the DataSphere connector. They are configured in **different places**:

- **Local:** `.env` → `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (see [cloud-launch.md §3](cloud-launch.md#3-upload-to-object-storage))
- **DataSphere:** connector form → “Static access key ID” + a **DataSphere Secret** holding the secret part

Training jobs never read your laptop’s `.env`; they only see the bucket through the activated connector.

## Prerequisites

1. **DataSphere project** in a community with an active billing account.
2. **Object Storage bucket** without dots in the name ([connector limitation](https://yandex.cloud/en/docs/datasphere/operations/data/s3-connectors)).
3. **NAT gateway** on the subnet used by the project’s community. The subnet must be in the **same availability zone** as the community ([S3 connector note](https://yandex.cloud/en/docs/datasphere/operations/data/s3-connectors)). See [NAT gateway setup](https://yandex.cloud/en/docs/vpc/operations/create-nat-gateway).
4. **Folder-level IAM permission** to create service accounts and access keys (`iam.serviceAccounts.admin` or equivalent).

## Step 1 — Create a service account

Console: **Identity and Access Management → Service accounts → Create**.

CLI (default folder):

```bash
yc iam service-account create \
  --name adaptive-roi-codec-s3 \
  --description "Object Storage access for Kvasir dataset and training checkpoints"
```

List accounts:

```bash
yc iam service-account list
```

Save the service account **ID** (`aje…`) for role assignment and key creation.

## Step 2 — Assign Object Storage roles

The connector’s service account must read the dataset and write checkpoints. For this project, assign at folder or bucket scope:

| Goal | Role | Notes |
|------|------|-------|
| Read + write (recommended) | `storage.editor` | Create/delete objects; used in [Yandex tutorial](https://yandex.cloud/en/docs/tutorials/ml-ai/s3-to-datasphere) |
| Read + upload only | `storage.uploader` | Includes `storage.viewer`; cannot delete objects |
| Read only | `storage.viewer` | **Insufficient** for checkpoint writes |

CLI example (folder):

```bash
SA_ID=$(yc iam service-account get adaptive-roi-codec-s3 --format json | jq -r .id)
FOLDER_ID=$(yc config get folder-id)

yc resource-manager folder add-access-binding "$FOLDER_ID" \
  --role storage.editor \
  --subject "serviceAccount:${SA_ID}"
```

Role reference: [Object Storage access management](https://yandex.cloud/en/docs/storage/security/).

## Step 3 — Create a static access key

Static keys are **created for the service account** ([docs](https://yandex.cloud/en/docs/iam/operations/authentication/manage-access-keys)).

```bash
yc iam access-key create \
  --service-account-name adaptive-roi-codec-s3 \
  --description "DataSphere S3 connector + local upload"
```

Example output:

```text
access_key:
  id: aje6t3vsbj8l********      # IAM record ID — do NOT use as S3 key
  service_account_id: ajepg0mjt06s********
  created_at: "2026-01-01T12:00:00Z"
  key_id: YCAJExxxxxxxxxxxx      # → Static access key ID / AWS_ACCESS_KEY_ID
secret: YCMxxxxxxxxxxxxxxxxxxxx   # → DataSphere Secret / AWS_SECRET_ACCESS_KEY
```

**Save `key_id` and `secret` immediately.** The secret is shown only once.

Mapping for this repository:

| Source field | DataSphere connector | `.env` (local upload) |
|--------------|----------------------|------------------------|
| `key_id` | **Static access key ID** | `AWS_ACCESS_KEY_ID` |
| `secret` | **DataSphere Secret** value | `AWS_SECRET_ACCESS_KEY` |
| — | — | `AWS_DEFAULT_REGION=ru-central1` (required for AWS CLI) |

## Step 4 — (Recommended) Create a DataSphere Secret

The connector stores the **secret part** of the key in an encrypted [DataSphere Secret](https://yandex.cloud/en/docs/datasphere/concepts/secrets), not in plain text in the UI.

1. Open your **DataSphere project**.
2. **Project resources → Secret → Create**.
3. **Name:** e.g. `s3-adaptive-roi-codec-key` (2–63 chars; letter first).
4. **Value:** paste the `secret` from step 3 (the `YCM…` string).
5. Click **Create**.

You will select this secret when creating the S3 connector. Secret values are shown as `***` afterward and are not exposed in job logs if used correctly ([secrets scope](https://yandex.cloud/en/docs/datasphere/concepts/secrets)).

## Step 5 — Create the S3 connector

1. Open the same **DataSphere project**.
2. **Project resources → S3 Connector → Create** (or **Create resource → S3 connector**).
3. Fill the form ([field reference](https://yandex.cloud/en/docs/datasphere/operations/data/s3-connectors)):

| Field | Value for this project |
|-------|-------------------------|
| **Name** | e.g. `Kvasir capsule bucket` (display name; 3–63 chars) |
| **Description** | Optional |
| **Endpoint** | `https://storage.yandexcloud.net/` |
| **Bucket** | `adaptive-roi-codec-data` (your bucket; no dots) |
| **Mount name** | e.g. `kvasir-data` (lowercase letters, numbers, hyphens only) |
| **Static access key ID** | `key_id` from step 3 (`YCAJE…`) |
| **Static access key** | Select the secret from step 4, or **Create** a new secret inline |
| **Mode** | **Read and write** (training writes checkpoints under `checkpoints/`) |

4. Click **Create**.

The connector’s static key is stored encrypted ([S3 connector concept](https://yandex.cloud/en/docs/datasphere/concepts/s3-connector)).

## Step 6 — Activate the connector

1. Open the connector’s resource page.
2. Click **Activate**.

After activation:

- **JupyterLab:** bucket appears under `/s3/` in the file browser.
- **DataSphere Jobs:** bucket is available when the connector ID is listed in `s3-mounts` and the job runs.

To detach later: **Deactivate** on the connector page ([docs](https://yandex.cloud/en/docs/datasphere/operations/data/s3-connectors#detach-an-s3-storage)).

## Step 7 — Copy `S3_CONNECTOR_ID`

You need the connector’s **unique resource ID**, not the mount name or bucket name.

**Console**

1. **Project resources → S3 Connector**.
2. Open your connector.
3. Copy the **ID** from the resource details page (often labeled **ID** or shown in the URL / resource header). It is a short alphanumeric string assigned by DataSphere.

**Sanity check in JupyterLab (optional)**

1. **Activate** the connector and open the project in JupyterLab.
2. Open the **S3 Mounts** tab, browse to e.g. `kvasir-capsule/MANIFEST.json`, right-click → **Copy path**.
3. Paths in notebooks use the `/s3/` prefix; **Jobs use a different prefix** (see below).

Put the ID in `.env`:

```bash
S3_CONNECTOR_ID=bt1xxxxxxxxxxxxxxxx
```

This repo passes it into the job template (`jobs/configs/job_train.yaml.template`):

```yaml
s3-mounts:
  - ${S3_CONNECTOR_ID}
```

## Step 8 — Paths on the training worker

[DataSphere Jobs](https://yandex.cloud/en/docs/datasphere/concepts/jobs/) mount connectors at:

```text
/job/s3/<S3_CONNECTOR_ID>/<object_key_inside_bucket>
```

For this repository (bucket prefix `kvasir-capsule`):

| Purpose | Path on job VM |
|---------|----------------|
| Training videos | `/job/s3/<S3_CONNECTOR_ID>/kvasir-capsule/raw/labelled_videos/` |
| Split lists | `/job/s3/<S3_CONNECTOR_ID>/kvasir-capsule/splits/train_videos.txt` |
| Checkpoints | `/job/s3/<S3_CONNECTOR_ID>/checkpoints/<experiment_id>/` |

`S3_DATA_PREFIX` and `S3_CHECKPOINT_SUBDIR` in `.env` must match these key prefixes.

Verify the mount before a long GPU job:

```bash
uv run launch-train --dry-run
# Confirm jobs/configs/.generated/job_train.yaml lists your connector under s3-mounts
```

## End-to-end checklist

- [ ] Service account created
- [ ] `storage.editor` (or sufficient read/write role) assigned
- [ ] Static access key created; `key_id` + `secret` saved
- [ ] Bucket exists; dataset uploaded (see [cloud-launch.md §3](cloud-launch.md#3-upload-to-object-storage))
- [ ] DataSphere Secret created with the `secret` value
- [ ] S3 connector created with correct endpoint, bucket, `key_id`, secret, **Read and write**
- [ ] NAT gateway configured for the community subnet
- [ ] Connector **Activated**
- [ ] `S3_CONNECTOR_ID` copied to `.env`
- [ ] `DATASPHERE_PROJECT_ID` set in `.env`
- [ ] `uv run launch-train --dry-run` shows the connector in `s3-mounts`

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Connector creation fails / no S3 access | Wrong `key_id` or secret; used IAM record `id` instead of `key_id` | Recreate key; use `key_id` + `secret` per step 3 |
| `AccessDenied` in job or Jupyter | Service account lacks `storage.editor` / `storage.uploader` | Assign role on folder or bucket |
| Connector inactive / empty mount | Not activated | **Activate** on connector page |
| Job cannot see dataset | Wrong `S3_CONNECTOR_ID`; connector not in `s3-mounts` | Match ID from connector page; rerun `launch-train` |
| `Video directory not found` | S3 key layout mismatch | Upload to `kvasir-capsule/raw/labelled_videos/` ([mapping](cloud-launch.md#upload-path-mapping)) |
| Timeouts / no egress to Object Storage | Missing NAT gateway on subnet | [Configure NAT gateway](https://yandex.cloud/en/docs/vpc/operations/create-nat-gateway) in the community AZ |
| Local upload works, job does not | Laptop uses your user credentials; job uses connector keys | Fix connector’s service account roles and key, not only `.env` |
| Poor performance listing many files | FUSE on flat prefixes | Expected for large single-level folders ([note](https://yandex.cloud/en/docs/datasphere/operations/data/s3-connectors)); use prefix layout with subfolders |

## Rotating keys

1. Create a new static access key for the same service account.
2. Update the DataSphere Secret (or create a new secret and edit the connector).
3. Update `.env` if you use the same key for local upload.
4. Delete the old static access key in IAM when nothing uses it ([delete access key](https://yandex.cloud/en/docs/iam/operations/authentication/manage-access-keys#delete)).

## Related `.env` variables

```bash
# DataSphere project (UI) — not the same as connector ID
DATASPHERE_PROJECT_ID=

# Connector resource ID from step 7
S3_CONNECTOR_ID=

# Keys below are for LOCAL upload only (AWS CLI / optional)
AWS_ACCESS_KEY_ID=      # key_id from yc iam access-key create
AWS_SECRET_ACCESS_KEY=  # secret from the same command
AWS_DEFAULT_REGION=ru-central1
S3_BUCKET=adaptive-roi-codec-data
```

See [.env.example](../../.env.example) for the full template.
