"""Демонстрация адаптивного распределения битрейта (эпизод 6 сценария).

Строит семейство кривых:
    q_t(E_ROI; κ) = q_min + (q_max - q_min) * E_ROI^κ
для разных κ и сравнивает с baseline (фиксированный шаг).

Использование:
    uv run python scripts/demo_quantization.py \\
        --output docs/screencast/quantization_curves.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_roi_codec.model.quantizer import AdaptiveQuantizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--q-min", type=float, default=0.1, help="q_min (минимум шага квантования)")
    p.add_argument("--q-max", type=float, default=2.0, help="q_max (максимум шага квантования)")
    p.add_argument("--kappas", type=str, default="0.5,1.0,2.0,4.0", help="Список κ через запятую")
    p.add_argument("--output", type=str, default="docs/screencast/quantization_curves.png")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    kappas = [float(x) for x in args.kappas.split(",")]
    e_roi = np.linspace(0.0, 1.0, 500)

    quantizer = AdaptiveQuantizer(q_min=args.q_min, q_max=args.q_max)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Левый график: q_t(E_ROI; κ)
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for i, kappa in enumerate(kappas):
        q = quantizer.global_step(torch.tensor(e_roi).float().view(1, -1, 1, 1)).numpy().ravel()
        color = cmap(i / max(1, len(kappas) - 1))
        ax.plot(e_roi, q, label=f"κ = {kappa}", linewidth=2.5, color=color)
    # Baseline — фиксированный шаг = 1
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.8, label="Baseline (const = 1)")
    ax.set_xlabel("E_ROI — средняя активация ROI", fontsize=11)
    ax.set_ylabel("q_t — шаг квантования (меньше ⇒ больше бит)", fontsize=11)
    ax.set_title(f"Адаптивный квантизатор\nq_t = q_min + (q_max − q_min)·E_ROI^κ", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, args.q_max * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    # --- Правый график: относительный битрейт
    # Битрейт обратно пропорционален log2(1 + 1/q^2) — для наглядности нормируем
    ax = axes[1]
    for i, kappa in enumerate(kappas):
        q = quantizer.global_step(torch.tensor(e_roi).float().view(1, -1, 1, 1)).numpy().ravel()
        # Относительный битрейт = 1/q (грубое приближение)
        bitrate_rel = 1.0 / q
        # Нормируем к baseline (q=1 ⇒ bitrate_rel = 1)
        bitrate_rel = bitrate_rel / 1.0
        color = cmap(i / max(1, len(kappas) - 1))
        ax.plot(e_roi, bitrate_rel, label=f"κ = {kappa}", linewidth=2.5, color=color)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.8, label="Baseline")
    ax.set_xlabel("E_ROI — средняя активация ROI", fontsize=11)
    ax.set_ylabel("Относительный битрейт (1/q_t)", fontsize=11)
    ax.set_title("Распределение битрейта по ROI\nчем выше кривая — тем больше бит в ROI-зоне", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    fig.suptitle(
        "Adaptive ROI Neural Video Codec — распределение битрейта\n"
        f"q_min={args.q_min}, q_max={args.q_max}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo_quantization] saved → {out_path}")

    # Краткая сводка в stdout
    print("\nШаг квантования q_t при E_ROI = 0.5 (ROI-зона):")
    for kappa in kappas:
        q_mid = quantizer.global_step(torch.tensor([[[[0.5]]]]).float()).item()
        print(f"  κ = {kappa:>4}: q_t = {q_mid:.3f}  (в {1.0 / q_mid:.2f}× больше бит, чем baseline)")
    print("\nШаг квантования q_t при E_ROI = 0.05 (фон):")
    for kappa in kappas:
        q_bg = quantizer.global_step(torch.tensor([[[[0.05]]]]).float()).item()
        print(f"  κ = {kappa:>4}: q_t = {q_bg:.3f}  (в {1.0 / q_bg:.2f}× больше бит, чем baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
