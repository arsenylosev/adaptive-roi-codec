"""End-to-end pipeline demo (эпизод 3 + 5 сценария).

Полный пайплайн на одном MP4:
    1. Генерация манифеста (если нет).
    2. Извлечение кадров → .npy (если нет).
    3. Inference на одном кадре → side-by-side PNG.

Использование:
    uv run python scripts/demo_pipeline.py \\
        --video kvasir-capsule/raw/labelled_videos/$(ls kvasir-capsule/raw/labelled_videos | head -1) \\
        --workdir kvasir-capsule \\
        --output docs/screencast/pipeline_demo.png
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_roi_codec.model.quantizer import AdaptiveQuantizer
from adaptive_roi_codec.model.roi_detector import ROIDetector
from adaptive_roi_codec.model.vae_codec import VAECodec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Путь к одному MP4")
    p.add_argument("--workdir", default="kvasir-capsule", help="Корень датасета для manifest/frames")
    p.add_argument("--frame-idx", type=int, default=0)
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--output", default="docs/screencast/pipeline_demo.png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--skip-manifest", action="store_true", help="Не пересоздавать manifest")
    p.add_argument("--skip-extract", action="store_true", help="Не извлекать кадры")
    return p.parse_args()


def step(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd, check=False)
    if res.returncode != 0:
        print(f"  ! Команда вернула {res.returncode}; продолжаем")


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # ---- Шаг 1: манифест
    manifest = workdir / "MANIFEST.json"
    if not manifest.exists() and not args.skip_manifest:
        step("[1/3] Генерация манифеста")
        run(["uv", "run", "build-dataset-manifest", "--dataset-root", str(workdir)])
    else:
        print(f"[1/3] манифест уже существует: {manifest}")

    # ---- Шаг 2: извлечение кадров
    frames_dir = workdir / "processed" / "frames"
    if not frames_dir.exists() and not args.skip_extract:
        step("[2/3] Извлечение кадров → .npy (336×336)")
        run([
            "uv", "run", "extract-frames",
            "--video", str(video_path),
            "--output", str(frames_dir),
            "--height", "336",
            "--width", "336",
        ])
    else:
        print(f"[2/3] кадры уже извлечены: {frames_dir}")

    # ---- Шаг 3: inference
    step("[3/3] Inference: ROI → quantizer → VAE → reconstruction")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device = {device}")

    # Импортируем cv2 только здесь — чтобы шаги 1-2 работали даже без него
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = min(args.frame_idx, max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        print(f"  ! не удалось прочитать кадр {target} из {video_path}")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (336, 336), interpolation=cv2.INTER_AREA)
    import numpy as np
    chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.float32) / 255.0
    frame = torch.from_numpy(chw).unsqueeze(0).to(device)

    roi_detector = ROIDetector(input_size=336, pretrained=True).to(device).eval()
    codec = VAECodec(latent_channels=192).to(device).eval()
    quantizer = AdaptiveQuantizer(kappa=args.kappa, alpha_spatial=0.5)

    with torch.no_grad():
        mask = roi_detector(frame)
        outputs = codec(frame)
        z_q = quantizer.quantize(outputs["z"], mask)
        _, _, skips = codec.encoder(frame)
        recon = codec.decoder(z_q, skips)

    # Визуализация 1×4
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig = frame[0].cpu().permute(1, 2, 0).numpy()
    roi_heat = mask[0].mean(dim=0).cpu().numpy()
    roi_heat = (roi_heat - roi_heat.min()) / (roi_heat.max() - roi_heat.min() + 1e-8)
    latent = z_q[0].abs().mean(dim=0).cpu()
    latent = torch.nn.functional.interpolate(
        latent[None, None], size=(336, 336), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    recon_img = recon[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(orig); axes[0].set_title("1. Original"); axes[0].axis("off")
    axes[1].imshow(roi_heat, cmap="jet"); axes[1].set_title("2. ROI detector"); axes[1].axis("off")
    axes[2].imshow(latent, cmap="inferno"); axes[2].set_title(f"3. Quantized z (κ={args.kappa})"); axes[2].axis("off")
    axes[3].imshow(recon_img); axes[3].set_title("4. VAE decoder"); axes[3].axis("off")
    fig.suptitle("End-to-end pipeline: manifest → extract → ROI → quantize → decode", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ PNG сохранён: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
