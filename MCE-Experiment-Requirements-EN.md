# Project Specification: Adaptive Bitrate Compression of Endoscopic Video Stream

> **Based on:** Losev A. P. and Buchatsky A. N. papers (NTO RRS 2026)  
> **Date:** June 2026  
> **Goal:** practical experiment — implementation, training, inference  
> **Budget:** ~87,850 ₽ (Yandex Cloud grant 100,000 ₽)

---

## 1. Goal and Tasks

**Goal:** implement a neural video codec for capsule endoscopy compression with adaptive bitrate allocation across clinically significant regions of interest (ROI).

**Tasks:**

1. Implement VAE codec with motion compensation (paper [1])
2. Implement clinically-oriented loss function (paper [2])
3. Add ROI detector and adaptive quantization (paper [3])
4. Train model on Kvasir-Capsule in Yandex Cloud
5. Run inference on CPU and measure power consumption

---

## 2. Source Data

### 2.1 Papers (Research Base)

| # | Title | Key Contribution |
|---|---|---|
| [1] | Development of Neural Network-based Encoder for Endoscopic Video Stream Compression | VAE architecture: 120×68×192 latent, 16× compression, skip connections |
| [2] | Development of Clinically-Oriented Loss Function for Neural Compression of Endoscopic Video Stream | L_base (α-PSNR + (1−α)-SSIM), L_ROI, L_rate, L_temp; adaptive β(t) |
| [3] | Adaptive Bitrate Allocation in Neural Network Encoder of Endoscopic Video Stream Based on ROI Detection | ROI detector (U-Net + MobileNetV3), κ=2.0, adaptive quantization |

### 2.2 Dataset

| Parameter | Value |
|---|---|
| Name | Kvasir-Capsule |
| Source | simula.no |
| Size | ~61 GB |
| Content | 47 capsule endoscopy videos, >44 hours |
| Frames | ~23 million (estimated) |
| License | CC BY-SA 4.0 |
| **Storage location** | **Yandex Object Storage** (server has only 40 GB — insufficient) |

### 2.3 Target Platform

| Parameter | Value |
|---|---|
| Inference | Intel Core i5/i7 12th-14th gen, 15–28 W |
| Target power consumption | 28 W |
| Target FPS | 30 fps real-time |
| Latency budget | 33 ms/frame (ROI det: 12 ms + quant: 3 ms + decode: 18 ms) |
| Training | Yandex Datasphere Jobs (GPU V100) |

---

## 3. Model Architecture

### 3.1 Three Modules

```
Input Frame (1920×1080)
       │
       ▼
┌──────────────────┐
│   ROI Detector   │  MobileNetV3-backbone U-Net
│  (512×512 input) │  Output: M_t ∈ [0,1]^(H×W)
└────────┬─────────┘
         │ M_t
         ▼
┌──────────────────┐
│   VAE Codec      │  Encoder: 4-5 blocks, stride conv
│  latent: 120×68×192 │  Decoder: transposed conv + skip connections
│  16× compression │  Motion compensation between frames
└────────┬─────────┘
         │ z_t
         ▼
┌──────────────────┐
│ Adaptive Quantizer│  q_t = q_min + (q_max−q_min)·E_ROI(t)^κ
│  (formulas 4–6)   │  Q_t(p,q) = q_t·(1+α·M_t^e(p,q))
└──────────────────┘
         │
         ▼
  Quantized z_t → entropy coding → bitstream
```

### 3.2 Latent Space

```
Input:  1920 × 1080 × 3
Output:  120 ×  68  × 192   (16× spatial compression)
Compression: 120·68·192·8 bit ≈ 125 KB/frame → at 30 fps ≈ 30 Mbps
Target bitrate: 2 Mbps (significant compression via entropy coding)
```

### 3.3 ROI Detector

- Architecture: U-Net with MobileNetV3-large backbone (ImageNet pretrained)
- Input: 512×512 (resize from original frame)
- Output: 3-channel soft significance mask [0,1]
- ROI classes: inflammation/ulcer boundaries, vascular pattern, local color changes

### 3.4 Adaptive Quantization Parameters

| Parameter | Formula | Value |
|---|---|---|
| E_ROI(t) | (1/HW)·Σ M_t(i,j) | ROI fraction in frame |
| q_t | q_min + (q_max−q_min)·E_ROI(t)^κ | quantization step |
| Q_t(p,q) | q_t·(1+α·M_t^e(p,q)) | spatial mask |
| κ | — | **2.0** (optimal from paper [3]) |
| α | — | 0.5 (spatial selectivity) |
| δ | — | bitrate feedback coefficient |

---

## 4. Loss Function

### 4.1 Formula (14) from paper [2]

```
L_total = L_base + λ_ROI·L_ROI + λ_rate·L_rate + λ_temp·L_temp
```

### 4.2 Components

| Component | Formula | Description |
|---|---|---|
| L_base | α·(−PSNR) + (1−α)·(1−SSIM_soft) | Base distortion, α=0.5 |
| L_ROI | (1/HW)·Σ M_t·‖x−x̂‖² / (Σ M_t + ε) | Weighted distortion in ROI |
| L_rate | β·KL(q_φ(z|x) ‖ N(0,I)) | KL regularization of latent space |
| L_temp | ‖(x̂_t−x̂_{t−1}) − F_t·(x_t−x_{t−1})‖² | Temporal consistency |

### 4.3 Weights (from paper [2])

| Parameter | Value |
|---|---|
| α (PSNR/SSIM balance) | 0.5 |
| λ_ROI | 1.5 |
| λ_rate | 0.01 |
| λ_temp | as needed |
| β(t) | adaptive (subgradient method, η=0.5) |

---

## 5. Implementation Requirements

### 5.1 Stack

```
Python         ≥ 3.10
PyTorch        ≥ 2.0  (CPU + CUDA)
TorchVision    for MobileNetV3 backbone
Albumentations for augmentations
```

### 5.2 Project Structure

```
capsule-compression/
├── model/
│   ├── roi_detector.py      # U-Net + MobileNetV3 backbone
│   ├── vae_codec.py         # VAE-encoder, VAE-decoder, motion compensation
│   ├── quantizer.py         # Adaptive quantizer (formulas 4-6)
│   └── __init__.py
├── losses/
│   ├── clinical_loss.py     # L_total (formula 14)
│   ├── ssim.py              # differentiable SSIM soft approximation
│   └── __init__.py
├── utils/
│   ├── kvasir_loader.py     # DataLoader for Kvasir-Capsule dataset
│   ├── metrics.py           # PSNR, MS-SSIM, Dice ROI
│   └── __init__.py
├── train.py                 # Training loop
├── inference.py             # CPU inference + power measurement
├── requirements.txt
├── README.md
└── configs/
    ├── base.yaml            # hyperparameters
    └── quantizer.yaml       # κ ∈ {1.0, 1.5, 2.0, 2.5}
```

### 5.3 Key Hyperparameters (from papers)

```yaml
# base.yaml
model:
  latent_h: 68
  latent_w: 120
  latent_ch: 192
  input_res: [1920, 1080]

training:
  batch_size: 4          # for V100 80GB
  lr: 1.0e-4
  epochs: 50
  warmup_epochs: 5
  alpha: 0.5             # PSNR/SSIM balance
  lambda_roi: 1.5
  lambda_rate: 0.01
  beta_0: 0.01           # initial KL weight
  eta: 0.5               # beta update rate

quantizer:
  q_min: 0.1
  q_max: 2.0
  kappa: 2.0             # nonlinearity (target: 2.0)
  alpha_spatial: 0.5     # α — spatial selectivity

roi_detector:
  input_res: 512
  backbone: mobilenet_v3_large
  pretrained: true
```

---

## 6. Yandex Cloud Infrastructure

### 6.1 Components

| Service | Purpose | Cost (estimate) |
|---|---|---|
| Object Storage | Kvasir-Capsule storage (~61 GB) | ~182 ₽/month |
| Datasphere Jobs | GPU training V100 | ~11,610 ₽ |
| Total | — | ~11,800 ₽ |

### 6.2 Recommended Datasphere Job Config

```yaml
# job_train.yaml
computable_name: vae-capsule-train
description: VAE codec training on Kvasir-Capsule
resource:
  cpu: 8
  memory: 64
  gpu: 1          # V100 or A100
  runtime: 20h
environment:
  PYTHONPATH: /workspace
```

### 6.3 Dataset Upload (via YC CLI)

```bash
# Assumes: Kvasir-Capsule downloaded locally
# Create Object Storage bucket via web UI or:
yc iam access-key create --folder-name <FOLDER_ID>

# Upload
aws configure --profile yc
aws s3 cp --recursive ./kvasir-capsule/ s3://<BUCKET>/kvasir-capsule/ \
  --endpoint-url https://storage.yandexcloud.net
```

---

## 7. Metrics and Validation

### 7.1 Metrics (targets from paper [3])

| Metric | Target Value |
|---|---|
| Dice ROI | ≥ 0.89 |
| PSNR | ≥ 32.7 dB |
| Average bitrate | ≤ 1.81 Mbps |
| Subjective score | ≥ 4.6 / 5.0 |
| CPU latency | ≤ 33 ms/frame |
| CPU power | ≤ 28 W |

### 7.2 Experiments

| Experiment | Parameter | Variants |
|---|---|---|
| 1 | κ (nonlinearity) | 1.0, 1.5, 2.0, 2.5 |
| 2 | λ_ROI | 1.0, 1.5, 2.0 |
| 3 | Target bitrate | 1.5, 2.0, 2.5 Mbps |

---

## 8. Workflow

1. **Implementation** (Cursor, locally) — all code from spec above
2. **Dataset** — download Kvasir-Capsule → upload to Object Storage
3. **Training** — Datasphere Job with `job_train.yaml` (~20 hours on V100)
4. **Checkpoints** — save to Object Storage every 5 epochs
5. **Inference test** — run `inference.py` on CPU (target: 28 W, 30 fps)
6. **Metrics** — compare with Table 1 from paper [3]

---

## 9. What NOT to Do on the Server

- ❌ Download Kvasir-Capsule dataset to server (40 GB — insufficient)
- ❌ Run training on server CPU (no GPU, 4 GB RAM)
- ❌ Store checkpoints locally

---

## 10. References

1. Losev A. P., Buchatsky A. N. Development of Neural Network-based Encoder for Endoscopic Video Stream Compression // NTO RRS 2026.
2. Losev A. P., Buchatsky A. N. Development of Clinically-Oriented Loss Function for Neural Compression of Endoscopic Video Stream // NTO RRS 2026.
3. Losev A. P., Buchatsky A. N. Adaptive Bitrate Allocation in Neural Network Encoder of Endoscopic Video Stream Based on ROI Detection // NTO RRS 2026.
4. Smedsrud P. H. et al. Kvasir-Capsule, a video capsule endoscopy dataset // Sci Data. 2021. doi:10.1038/s41597-021-00917-6
5. Kingma D. P., Welling M. Auto-Encoding Variational Bayes // ICLR 2014.
6. Wang Z. et al. Image quality assessment: from error visibility to structural similarity // IEEE TIP. 2004.