"""Live inference demo for the adaptive ROI neural video codec.

Использование:
    # На одном MP4-видео (рекомендуется)
    uv run python scripts/demo_inference.py \\
        --video path/to/kvasir_capsule.mp4 \\
        --output docs/screencast/inference_demo.png

    # С уже обученными весами
    uv run python scripts/demo_inference.py \\
        --video path/to/video.mp4 \\
        --checkpoint checkpoints/v100-kappa-2.0-18ep/epoch_18.pt

    # Без видео (синтетический кадр — для dry-run демо)
    uv run python scripts/demo_inference.py \\
        --output docs/screencast/inference_demo_synthetic.png

Скрипт выводит PNG 2×2: Original | ROI mask | Quantized latent map | Reconstruction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_roi_codec.model.quantizer import AdaptiveQuantizer
from adaptive_roi_codec.model.roi_detector import ROIDetector
from adaptive_roi_codec.model.vae_codec import VAECodec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=str, default=None, help="Путь к MP4-видео (опц.)")
    p.add_argument("--frame-idx", type=int, default=0, help="Индекс кадра в видео (по умолчанию 0)")
    p.add_argument("--checkpoint", type=str, default=None, help="Путь к .pt чекпоинту (опц.)")
    p.add_argument("--no-pretrained", action="store_true", help="Не загружать MobileNetV3 pretrained")
    p.add_argument("--kappa", type=float, default=2.0, help="Параметр κ квантизатора")
    p.add_argument("--alpha-spatial", type=float, default=0.5, help="α_spatial квантизатора")
    p.add_argument("--output", type=str, default="docs/screencast/inference_demo.png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_frame_from_video(path: str, frame_idx: int, size: int = 336) -> torch.Tensor:
    """Читает один кадр из MP4 и возвращает тензор [1, 3, H, W] в [0, 1]."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = min(frame_idx, max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"Не удалось прочитать кадр {target} из {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.float32) / 255.0
    return torch.from_numpy(chw).unsqueeze(0)


def make_synthetic_frame(size: int = 336, seed: int = 42) -> torch.Tensor:
    """Генерирует синтетический «эндоскопический» кадр: розовый фон + структуры.

    Используется как fallback, если видео нет. Не претендует на клиническую
    достоверность, но даёт визуально правдоподобный вход для пайплайна.
    """
    rng = np.random.default_rng(seed)
    h = w = size
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Розовый фон с радиальным градиентом
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r /= r.max()
    img[..., 0] = 0.65 + 0.25 * (1 - r)  # R
    img[..., 1] = 0.20 + 0.20 * (1 - r)  # G
    img[..., 2] = 0.30 + 0.20 * (1 - r)  # B
    # Складки и текстуры
    for _ in range(6):
        ay = rng.integers(0, h)
        ax = rng.integers(0, w)
        sigma = rng.uniform(15, 45)
        blob = np.exp(-((yy - ay) ** 2 + (xx - ax) ** 2) / (2 * sigma ** 2))
        intensity = rng.uniform(-0.15, 0.25)
        img[..., 0] = np.clip(img[..., 0] + intensity * blob, 0, 1)
        img[..., 1] = np.clip(img[..., 1] + intensity * blob * 0.7, 0, 1)
        img[..., 2] = np.clip(img[..., 2] + intensity * blob * 0.5, 0, 1)
    # Центральная «патология» — яркая структура
    sigma = 35
    blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    img[..., 0] = np.clip(img[..., 0] + 0.4 * blob, 0, 1)
    img[..., 1] = np.clip(img[..., 1] + 0.1 * blob, 0, 1)
    img[..., 2] = np.clip(img[..., 2] - 0.1 * blob, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    chw = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def load_checkpoint_if_any(path: str | None, codec: VAECodec, roi_detector: ROIDetector) -> str:
    if not path:
        return "random init"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    codec.load_state_dict(ckpt["codec"], strict=False)
    if "roi_detector" in ckpt:
        roi_detector.load_state_dict(ckpt["roi_detector"], strict=False)
    epoch = ckpt.get("epoch", "?")
    return f"epoch={epoch}"


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] или [1, C, H, W] → [H, W, C] в [0, 1]."""
    if t.dim() == 4:
        t = t[0]
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def mask_to_heatmap(mask: torch.Tensor) -> np.ndarray:
    """[1, 3, H, W] → [H, W] в [0, 1] (среднее по каналам → серый + color-map)."""
    m = mask[0].mean(dim=0).detach().cpu().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m


def latent_to_image(z: torch.Tensor) -> np.ndarray:
    """Латент [1, C, H_lat, W_lat] → нормированная визуализация по каналам."""
    z = z[0].detach().cpu()
    # Берём среднее и std по каналам, делаем «heat» картинку
    energy = z.abs().mean(dim=0)
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    # Апскейлим до 336×336 для наглядности
    energy_img = torch.nn.functional.interpolate(
        energy[None, None], size=(336, 336), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return energy_img


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo_inference] device={device}")

    # 1. Загрузить кадр
    if args.video:
        try:
            frame = load_frame_from_video(args.video, args.frame_idx)
            print(f"[demo_inference] loaded frame from {args.video} (idx={args.frame_idx})")
        except Exception as exc:
            print(f"[demo_inference] WARN: {exc}; fallback to synthetic")
            frame = make_synthetic_frame()
    else:
        print("[demo_inference] no --video, using synthetic frame")
        frame = make_synthetic_frame()
    frame = frame.to(device)

    # 2. Построить модели
    roi_detector = ROIDetector(input_size=336, pretrained=not args.no_pretrained).to(device).eval()
    codec = VAECodec(latent_channels=192).to(device).eval()
    quantizer = AdaptiveQuantizer(kappa=args.kappa, alpha_spatial=args.alpha_spatial)

    ckpt_info = load_checkpoint_if_any(args.checkpoint, codec, roi_detector)
    print(f"[demo_inference] codec: {ckpt_info}, kappa={args.kappa}, alpha_spatial={args.alpha_spatial}")

    # 3. Forward pass
    with torch.no_grad():
        mask = roi_detector(frame)
        outputs = codec(frame)
        z_q = quantizer.quantize(outputs["z"], mask)
        # Прогоняем квантизованный латент через декодер ещё раз, чтобы получить
        # реконструкцию при использовании адаптивного квантования
        _, _, skips = codec.encoder(frame)
        recon_q = codec.decoder(z_q, skips)

    # 4. Подготовить визуализации
    orig_img = tensor_to_image(frame)
    roi_heat = mask_to_heatmap(mask)
    latent_img = latent_to_image(z_q)
    recon_img = tensor_to_image(recon_q)

    # Метрики
    mse = float(((frame - recon_q) ** 2).mean().item())
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
    roi_mean = float(roi_heat.mean())
    q_global = float(quantizer.global_step(mask).mean().item())
    print(
        f"[demo_inference] PSNR(recon,orig)={psnr:.2f} dB  "
        f"E_ROI={roi_mean:.3f}  q_t={q_global:.3f}  kappa={args.kappa}"
    )

    # 5. Рисуем 2×2
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title("Оригинальный кадр (336×336)", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(roi_heat, cmap="jet")
    axes[0, 1].set_title(f"ROI-карта (E_ROI={roi_mean:.3f})", fontsize=12)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(latent_img, cmap="inferno")
    axes[1, 0].set_title(f"Квантизованный латент (q_t={q_global:.3f}, κ={args.kappa})", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(recon_img)
    axes[1, 1].set_title(f"Реконструкция (PSNR={psnr:.2f} dB)", fontsize=12)
    axes[1, 1].axis("off")

    fig.suptitle(
        f"Adaptive ROI Neural Video Codec — live inference\n"
        f"checkpoint: {ckpt_info} · device: {device}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo_inference] saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
