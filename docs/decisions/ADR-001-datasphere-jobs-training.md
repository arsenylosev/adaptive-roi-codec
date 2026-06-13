# ADR-001: GPU training via Yandex Datasphere Jobs with S3-mounted dataset

## Status

Accepted

## Date

2026-06-13

## Context

The experiment in [MCE-Experiment-Requirements-EN.md](../../MCE-Experiment-Requirements-EN.md) targets:

- Kvasir-Capsule (~61 GB) stored in Yandex Object Storage
- GPU training on V100-class hardware (~20 h per run)
- Local development machine with limited disk (40 GB) and no GPU

Training must not download the full dataset or checkpoints to the local machine. Secrets (API keys, project IDs) must stay out of version control. The team uses `uv` for Python dependency management and Cursor agents with Yandex Cloud MCP for infrastructure documentation.

## Decision

1. **Package and dependencies:** Manage the project with `uv` (`pyproject.toml`, `uv.lock`). Heavy dependencies (PyTorch) are declared in the main package; the `datasphere` CLI is an optional `[cloud]` extra used only on the operator machine.

2. **Training execution:** Submit training as a **Datasphere Job** from the local machine via `launch-train`, which renders `jobs/configs/job_train.yaml.template` using values from `.env` and runs `datasphere project job execute`.

3. **Data and checkpoints:** Mount Object Storage through a DataSphere **S3 connector** (`s3-mounts`). Training reads Kvasir-Capsule from `/job/s3/<connector_id>/kvasir-capsule/` and writes checkpoints to `/job/s3/<connector_id>/checkpoints/`. The job config requests **150 GB SSD working storage** and GPU instance types `g1.1` (V100) with fallback `g2.1`.

4. **Environment on the worker:** Use `env.python: auto` so Datasphere collects dependencies from the operator's activated `uv` virtualenv when the job is submitted.

5. **Local execution scope:** Local runs are limited to **smoke tests** (`--dry-run`, reduced resolution synthetic frames on CPU). Full-resolution training is intentionally outscoped locally.

6. **Secrets:** Store `DATASPHERE_PROJECT_ID`, `S3_CONNECTOR_ID`, and Object Storage credentials only in `.env` (gitignored). Ship `.env.example` as a reproducibility template.

7. **Job outputs:** Persist run metrics to `metrics/train_metrics.json`, declared as a Datasphere Job **output** so results are downloaded after completion.

## Alternatives Considered

### Train locally with dataset subset

- **Pros:** Simpler loop, no cloud setup.
- **Cons:** Violates spec (full dataset, V100 budget); 40 GB local disk is insufficient.
- **Rejected:** Cannot meet experiment requirements.

### Yandex DataProc / custom VM + SSH

- **Pros:** Full control over the VM image and startup scripts.
- **Cons:** More operational overhead (image build, monitoring, cost control) than Datasphere Jobs for a single training pipeline.
- **Rejected:** Datasphere Jobs already integrates S3 mounts, GPU quotas, and dependency packaging.

### Commit generated `job_train.yaml` with real IDs

- **Pros:** One-step `datasphere project job execute -c ...` without a launcher.
- **Cons:** Risk of committing secrets or stale project-specific IDs; harder to reuse across machines.
- **Rejected:** Render from template + `.env` at launch time; gitignore `jobs/configs/.generated/`.

### Docker-only worker environment (manual `env.docker`)

- **Pros:** Reproducible runtime independent of local venv.
- **Cons:** Extra build/publish step; `python: auto` is the documented default for PyTorch projects and matches the Datasphere quickstart.
- **Deferred:** Revisit if dependency drift between local venv and worker becomes a problem.

## Consequences

- Operators must configure Yandex Cloud CLI, DataSphere project, and S3 connector before the first job.
- Generated job configs live under `jobs/configs/.generated/` and must not be committed.
- Video frame extraction and full 1920×1080 training loop remain follow-up work; the current scaffold validates job submission and end-to-end tensor plumbing.
- Future ADRs should cover ROI detector supervision, entropy coding, and CPU inference benchmarking separately.

## References

- [DataSphere Jobs concept](https://yandex.cloud/en/docs/datasphere/concepts/jobs/)
- [Running DataSphere Jobs](https://yandex.cloud/en/docs/datasphere/operations/projects/work-with-jobs)
- [MCE-Experiment-Requirements-EN.md](../../MCE-Experiment-Requirements-EN.md) §6, §8
