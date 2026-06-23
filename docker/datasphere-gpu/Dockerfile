# syntax=docker/dockerfile:1
# GPU training image for Yandex DataSphere Jobs (V100 / CUDA 12.1).
#
# Build in DataSphere project UI (location: DataSphere, not YCR).
# Full instructions: docker/datasphere-gpu/BUILD.md
#
# Mount point (relative to project disk): adaptive-roi-codec  OR  .
# Jobs: env: docker: bxxxxxxxxxxxxxxxxxxx

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/job

# DataSphere platform tests require user jupyter (uid 1000) and a working pip for
# that user (post-build test #6: "Trying to install package with pip").
# Follow the official user-images example: system python3 + pip, not a root-only venv.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0; \
    useradd -ms /bin/bash --uid 1000 jupyter; \
    ln -sf /usr/bin/python3.11 /usr/bin/python3; \
    ln -sf /usr/bin/python3 /usr/local/bin/python3; \
    python3.11 -m ensurepip --upgrade; \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /job

COPY jobs/requirements-datasphere-gpu.txt /tmp/requirements-datasphere-gpu.txt
RUN python3 -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r /tmp/requirements-datasphere-gpu.txt

COPY adaptive_roi_codec /job/adaptive_roi_codec
COPY configs /job/configs

# jupyter must be able to `pip install --user` during DataSphere image validation.
RUN mkdir -p /home/jupyter/.local \
    && chown -R jupyter:jupyter /home/jupyter

ENV PATH="/home/jupyter/.local/bin:/usr/local/bin:/usr/bin:${PATH}"

CMD ["python3", "-m", "adaptive_roi_codec.jobs.run_train", "--help"]
