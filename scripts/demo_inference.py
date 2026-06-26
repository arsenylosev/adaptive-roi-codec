"""Live inference demo for the adaptive ROI neural video codec.

Использование:
    # На одном MP4-видео (рекомендуется)
    uv run python scripts/demo_inference.py \\
        --video path/to/kvasir_capsule.mp4 \\
        --output docs/screencast/inference_demo.png

    # С обученным VAE (ROI — pretrained backbone, см. --load-roi-checkpoint)
    uv run python scripts/demo_inference.py \\
        --video path/to/video.mp4 \\
        --auto-frame \\
        --checkpoint checkpoints/epoch_018.pt

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
    p.add_argument("--frame", type=str, default=None, help="Путь к .npy кадру [3,H,W] (опц.)")
    p.add_argument("--frame-idx", type=int, default=0, help="Индекс кадра в видео (по умолчанию 0)")
    p.add_argument("--auto-frame", action="store_true", help="Выбрать кадр с наибольшим контрастом ROI")
    p.add_argument("--checkpoint", type=str, default=None, help="Путь к .pt чекпоинту (опц.)")
    p.add_argument(
        "--load-roi-checkpoint",
        action="store_true",
        help="Загружать ROI-детектор из чекпоинта (по умолчанию — только VAE)",
    )
    p.add_argument("--no-pretrained", action="store_true", help="Не загружать MobileNetV3 pretrained")
    p.add_argument("--kappa", type=float, default=2.0, help="Параметр κ квантизатора")
    p.add_argument("--alpha-spatial", type=float, default=0.5, help="α_spatial квантизатора")
    p.add_argument("--output", type=str, default="docs/screencast/inference_demo.png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_frame_from_npy(path: str) -> torch.Tensor:
    """Читает кадр из .npy [3, H, W] и возвращает [1, 3, H, W] в [0, 1]."""
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise RuntimeError(f"Ожидался массив [3, H, W], получен {arr.shape}")
    chw = np.ascontiguousarray(arr, dtype=np.float32)
    if chw.max() > 1.0:
        chw = chw / 255.0
    return torch.from_numpy(chw).unsqueeze(0)


def find_best_frame_idx(path: str, size: int = 336, samples: int = 12) -> int:
    """Ищет кадр с максимальным пространственным контрастом ROI (pretrained probe)."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    probe = ROIDetector(input_size=size, pretrained=True).eval()

    if total <= 1:
        cap.release()
        return 0

    indices = np.linspace(0, total - 1, num=min(samples, total), dtype=int)
    best_idx = 0
    best_contrast = -1.0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.float32) / 255.0
        frame = torch.from_numpy(chw).unsqueeze(0)
        with torch.no_grad():
            mask = probe(frame)[0].mean(dim=0)
        contrast = float(mask.max() - mask.min())
        if contrast > best_contrast:
            best_contrast = contrast
            best_idx = int(idx)
    cap.release()
    return best_idx


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


def load_checkpoint_if_any(
    path: str | None,
    codec: VAECodec,
    roi_detector: ROIDetector,
    *,
    load_roi: bool,
) -> tuple[str, str]:
    if not path:
        return "random init", "random init"

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    codec.load_state_dict(ckpt["codec"], strict=False)
    epoch = ckpt.get("epoch", "?")
    codec_info = f"epoch={epoch}"

    roi_info = "pretrained backbone"
    if load_roi and "roi_detector" in ckpt:
        roi_detector.load_state_dict(ckpt["roi_detector"], strict=False)
        roi_info = f"epoch={epoch}"
    return codec_info, roi_info


def roi_mask_is_usable(mask: torch.Tensor, *, min_span: float = 0.02, min_peak: float = 0.05) -> bool:
    """Проверяет, что ROI-карта не вырождена (не «нулевая» и не чистый шум от float)."""
    values = mask[0].mean(dim=0)
    span = float(values.max() - values.min())
    peak = float(values.max())
    return span >= min_span and peak >= min_peak


def mask_spatial_mean(mask: torch.Tensor) -> torch.Tensor:
    return mask[0].mean(dim=0).detach().cpu()


def mask_to_display(mask: torch.Tensor) -> tuple[np.ndarray, float, float, bool]:
    """Готовит ROI для отображения: [H,W] в [0,1], E_ROI, контраст, усилен ли контраст."""
    m = mask_spatial_mean(mask).numpy()
    e_roi = float(m.mean())
    span = float(m.max() - m.min())
    std = float(m.std())

    enhanced = False
    if span < 0.08 or std < 0.015:
        z = (m - m.mean()) / (std + 1e-6)
        m_disp = np.clip(0.5 + z * 0.22, 0, 1)
        enhanced = True
    else:
        lo, hi = np.percentile(m, [3, 97])
        m_disp = np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)

    return m_disp, e_roi, span, enhanced


def roi_overlay_image(orig_img: np.ndarray, m_disp: np.ndarray) -> np.ndarray:
    """Накладывает ROI-теплокарту на оригинал для наглядности в эндоскопии."""
    heat = plt.cm.magma(m_disp)[..., :3]
    return np.clip(0.52 * orig_img + 0.48 * heat, 0, 1)


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] или [1, C, H, W] → [H, W, C] в [0, 1]."""
    if t.dim() == 4:
        t = t[0]
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


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
    frame_idx = args.frame_idx
    if args.frame:
        try:
            frame = load_frame_from_npy(args.frame)
            print(f"[demo_inference] loaded frame from {args.frame}")
        except Exception as exc:
            print(f"[demo_inference] WARN: {exc}; fallback to synthetic")
            frame = make_synthetic_frame()
    elif args.video:
        try:
            if args.auto_frame:
                frame_idx = find_best_frame_idx(args.video)
                print(f"[demo_inference] auto-frame selected idx={frame_idx}")
            frame = load_frame_from_video(args.video, frame_idx)
            print(f"[demo_inference] loaded frame from {args.video} (idx={frame_idx})")
        except Exception as exc:
            print(f"[demo_inference] WARN: {exc}; fallback to synthetic")
            frame = make_synthetic_frame()
    else:
        print("[demo_inference] no --video/--frame, using synthetic frame")
        frame = make_synthetic_frame()
    frame = frame.to(device)

    # 2. Построить модели
    roi_detector = ROIDetector(input_size=336, pretrained=not args.no_pretrained).to(device).eval()
    codec = VAECodec(latent_channels=192).to(device).eval()
    quantizer = AdaptiveQuantizer(kappa=args.kappa, alpha_spatial=args.alpha_spatial)

    codec_info, roi_info = load_checkpoint_if_any(
        args.checkpoint,
        codec,
        roi_detector,
        load_roi=args.load_roi_checkpoint,
    )
    if args.checkpoint and not args.load_roi_checkpoint:
        roi_info = "pretrained backbone (VAE from checkpoint)"

    # Деградировавший ROI в чекпоинте (epoch 18) даёт ~0; откатываемся на pretrained.
    with torch.no_grad():
        probe_mask = roi_detector(frame)
    if not roi_mask_is_usable(probe_mask):
        if args.checkpoint and args.load_roi_checkpoint:
            print(
                "[demo_inference] WARN: ROI из чекпоинта вырожден; "
                "используем pretrained MobileNetV3 для ROI"
            )
            roi_detector = ROIDetector(input_size=336, pretrained=True).to(device).eval()
            roi_info = "pretrained (checkpoint ROI degenerate)"
        elif not args.no_pretrained:
            print("[demo_inference] WARN: ROI-карта почти константа; контраст усилен для отображения")

    print(
        f"[demo_inference] codec: {codec_info}, roi: {roi_info}, "
        f"kappa={args.kappa}, alpha_spatial={args.alpha_spatial}"
    )

    # 3. Forward pass
    with torch.no_grad():
        mask = roi_detector(frame)
        outputs = codec(frame)
        z_q = quantizer.quantize(outputs["z"], mask)
        _, _, skips = codec.encoder(frame)
        recon_q = codec.decoder(z_q, skips)

    # 4. Подготовить визуализации
    orig_img = tensor_to_image(frame)
    roi_disp, e_roi, roi_span, roi_enhanced = mask_to_display(mask)
    roi_panel = roi_overlay_image(orig_img, roi_disp)
    latent_img = latent_to_image(z_q)
    recon_img = tensor_to_image(recon_q)

    mse = float(((frame - recon_q) ** 2).mean().item())
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
    q_global = float(quantizer.global_step(mask).mean().item())
    print(
        f"[demo_inference] PSNR(recon,orig)={psnr:.2f} dB  "
        f"E_ROI={e_roi:.3f}  q_t={q_global:.3f}  roi_span={roi_span:.3f}  kappa={args.kappa}"
    )

    # 5. Рисуем 2×2
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title("Оригинальный кадр (336×336)", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(roi_panel)
    roi_title = f"ROI-карта (E_ROI={e_roi:.3f})"
    if roi_enhanced:
        roi_title += "\nконтраст усилен для отображения"
    axes[0, 1].set_title(roi_title, fontsize=11)
    axes[0, 1].axis("off")
    levels = np.linspace(0.55, 0.85, 4)
    axes[0, 1].contour(roi_disp, levels=levels, colors="white", linewidths=0.7, alpha=0.65)

    axes[1, 0].imshow(latent_img, cmap="inferno")
    axes[1, 0].set_title(f"Квантизованный латент (q_t={q_global:.3f}, κ={args.kappa})", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(recon_img)
    axes[1, 1].set_title(f"Реконструкция (PSNR={psnr:.2f} dB)", fontsize=12)
    axes[1, 1].axis("off")

    fig.suptitle(
        f"Adaptive ROI Neural Video Codec — live inference\n"
        f"VAE: {codec_info} · ROI: {roi_info} · device: {device}",
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
