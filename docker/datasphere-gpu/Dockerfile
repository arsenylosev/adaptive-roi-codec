# syntax=docker/dockerfile:1
# GPU training image for Yandex DataSphere Jobs (V100 / CUDA 12.1).
#
# Build in DataSphere project UI (location: DataSphere, not YCR).
# See docker/datasphere-gpu/BUILD.md for mount point troubleshooting.
#
# Typical mount point (repo cloned into project disk):
#   adaptive-roi-codec
# If jobs/ and adaptive_roi_codec/ sit at project root:
#   .
#
# Then reference the resource in Jobs: env: docker: bxxxxxxxxxxxxxxxxxxx

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip

WORKDIR /job

COPY jobs/requirements-datasphere-gpu.txt /tmp/requirements-datasphere-gpu.txt
RUN pip install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r /tmp/requirements-datasphere-gpu.txt

COPY adaptive_roi_codec /job/adaptive_roi_codec
COPY configs /job/configs

ENV PYTHONPATH=/job

CMD ["python3", "-m", "adaptive_roi_codec.jobs.run_train", "--help"]
