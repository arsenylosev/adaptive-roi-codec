# DataSphere Docker image build (project storage only)

Guide for building `adaptive-roi-gpu:cu121` inside a DataSphere project (location **DataSphere**, not YCR).

Official reference: [Working with Docker images](https://yandex.cloud/en/docs/datasphere/operations/user-images)

---

## 1. Upload / sync the repository

Project disk in JupyterLab:

```text
/home/jupyter/datasphere/project/
```

Clone or copy this repo so these paths exist (relative to project root):

```text
jobs/requirements-datasphere-gpu.txt
adaptive_roi_codec/
configs/
docker/datasphere-gpu/Dockerfile
```

### Verify mount context (run in JupyterLab terminal)

```python
from pathlib import Path

PROJECT = Path("/home/jupyter/datasphere/project")
for root in [PROJECT, PROJECT / "adaptive-roi-codec"]:
    req = root / "jobs" / "requirements-datasphere-gpu.txt"
    pkg = root / "adaptive_roi_codec"
    print(root.name or ".", "requirements:", req.exists(), "package:", pkg.exists())
```

| Output | **Mount point** in UI |
|--------|------------------------|
| Both `True` under `…/project/adaptive-roi-codec/` | `adaptive-roi-codec` |
| Both `True` under `…/project/` | `.` |

**Invalid mount points:** `project/adaptive-roi-codec`, `~/project/…`, `/home/jupyter/…` — the builder only sees paths **inside** project storage.

---

## 2. Create the Docker resource

1. Project → **Create resource** → **Docker image**
2. **Location:** **DataSphere**
3. **Disk size:** 20 GB (CUDA base + PyTorch cu121 + OpenCV)
4. **Name / tag:** e.g. `adaptive-roi-gpu` / `cu121`
5. **Project storage mounting point:** from table above (`adaptive-roi-codec` or `.`)
6. **Docker file:** paste contents of `docker/datasphere-gpu/Dockerfile` from the repo (do not edit `COPY` paths unless your mount point is not the repo root)
7. Optional: Docker Hub credentials if base image pull is slow ([NAT + subnet in project settings](https://yandex.cloud/en/docs/datasphere/operations/user-images))
8. Click **Build**

---

## 3. Platform tests (what DataSphere checks)

After `Successfully built`, DataSphere runs:

| # | Check | Requirement |
|---|--------|-------------|
| 1 | `jupyter` user | `useradd --uid 1000 jupyter` |
| 2–3 | Python 3 | `python3` on PATH (symlink to 3.11) |
| 4–5 | Kernel + hello | DataSphere kernel starts |
| 6 | **pip install** | **`jupyter` user can install a package with pip** |

### Fix: test #6 failed — “Failed to install package with pip”

**Cause:** The previous Dockerfile used a **root-owned venv** at `/opt/venv`. Post-build test runs **`pip install` as user `jupyter`**, which cannot write into that venv.

**Fix (in current Dockerfile):**

- Install training deps with **`python3 -m pip`** into **system Python** (same pattern as [official TensorFlow example](https://yandex.cloud/en/docs/datasphere/operations/user-images))
- Symlinks required by docs:

  ```dockerfile
  ln -sf /usr/bin/python3.11 /usr/bin/python3
  ln -sf /usr/bin/python3 /usr/local/bin/python3
  ```

- Ensure `jupyter` home is writable for `--user` installs:

  ```dockerfile
  RUN mkdir -p /home/jupyter/.local \
      && chown -R jupyter:jupyter /home/jupyter
  ```

If test #6 still fails with **timeout / ConnectTimeoutError**, the build VM lacks internet — add **subnet + NAT gateway** in project settings ([troubleshooting](https://yandex.cloud/en/docs/troubleshooting/datasphere/known-issues/error-connect-timeout-when-installing-via-pip)).

---

## 4. Activate and use in Jobs

1. Project resources → **Docker** → **Activate** on the new image
2. Copy resource id (`bxxxxxxxxxxxxxxxxxxx`)
3. Use in job YAML (see `job_train.docker.yaml.template`):

```yaml
env:
  docker: bxxxxxxxxxxxxxxxxxxx
  vars:
    - PYTHONPATH: /job
    - TRAIN_REQUIRE_CUDA: "1"
    # … same vars as pip-based job template
inputs:
  - configs/base.yaml: CONFIG
  - ${PARAMS_INPUT}: PARAMS
  - adaptive_roi_codec: adaptive_roi_codec   # optional: override baked code
```

Launch:

```bash
uv run launch-train --execute --async \
  --params jobs/inputs/train_v100_scale.json \
  --template docker/datasphere-gpu/job_train.docker.yaml.template
```

(Replace template path after you copy it to `jobs/configs/` and set your docker resource id.)

---

## 5. Rebuild checklist (after Dockerfile fix)

- [ ] Sync latest `docker/datasphere-gpu/Dockerfile` to project storage
- [ ] Mount point unchanged and still resolves `jobs/requirements-datasphere-gpu.txt`
- [ ] Build log: tests 1–6 all **OK**
- [ ] Activate image
- [ ] Short GPU smoke job with `env.docker` confirms `Device: cuda` in stdout

---

## Job reference

`docker/datasphere-gpu/job_train.docker.yaml.template`
