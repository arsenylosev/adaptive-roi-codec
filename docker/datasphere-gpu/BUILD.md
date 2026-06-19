# DataSphere Docker image build (project storage only)

## Where project files live in JupyterLab

DataSphere mounts **project disk** at:

```text
/home/jupyter/datasphere/project/
```

When you click **Home** in JupyterLab and go one level up, you see `/home/jupyter/datasphere/` — that parent folder is **not** the Docker build context. The build context is always **inside** `project/`.

## Mount point in “Create Docker image” UI

The field **“Project storage mounting point”** is a path **relative to project disk root**, not an absolute path.

| What you see in JupyterLab File Browser | Mount point value |
|----------------------------------------|-------------------|
| Repo cloned as `adaptive-roi-codec/` under project | `adaptive-roi-codec` |
| Repo contents copied directly into project root (`jobs/`, `adaptive_roi_codec/` at top level) | `.` |

**Do not use:** `~/project/adaptive-roi-codec`, `project/adaptive-roi-codec`, `/home/jupyter/...` — these fail because the builder resolves paths only under project storage.

## Verify before building

Run in a JupyterLab notebook or terminal **inside the project VM**:

```python
from pathlib import Path

PROJECT = Path("/home/jupyter/datasphere/project")

candidates = [
    PROJECT,
    PROJECT / "adaptive-roi-codec",
]

for root in candidates:
    req = root / "jobs" / "requirements-datasphere-gpu.txt"
    pkg = root / "adaptive_roi_codec"
    print(root.name or str(root), "requirements:", req.exists(), "package:", pkg.exists())
```

Use the row where **both** are `True` as your mount point (relative to `project/`):

- If the row is `/home/jupyter/datasphere/project/adaptive-roi-codec` → mount point **`adaptive-roi-codec`**
- If the row is `/home/jupyter/datasphere/project` → mount point **`.`**

## Build steps

1. Project → **Create resource** → **Docker image** → location **DataSphere**
2. **Disk size:** 15–20 GB
3. **Mount point:** from table above (usually `adaptive-roi-codec`)
4. **Docker file:** paste `docker/datasphere-gpu/Dockerfile` from the repo (paths assume repo root as context)
5. **Build** → **Activate** → use resource id in job YAML: `env: docker: bxxxxxxxxxxxxxxxxxxx`

## DataSphere post-build validation

After `docker build`, DataSphere runs platform tests on the image. One required check is:

```text
Checking jupyter user existence .. FAILED
Unable to find user `jupyter`
```

**Fix:** create user `jupyter` with UID **1000** in the Dockerfile (required by [DataSphere user-images docs](https://yandex.cloud/en/docs/datasphere/operations/user-images)):

```dockerfile
RUN useradd -ms /bin/bash --uid 1000 jupyter
```

This is included in `docker/datasphere-gpu/Dockerfile`. Jobs may still run as root inside the container; the user must exist in `/etc/passwd` for the platform test to pass.

## Job reference

See `docker/datasphere-gpu/job_train.docker.yaml.template`.
