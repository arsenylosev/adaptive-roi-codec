# Демо-скрипты для видеозаписи ЭВМ

Скрипты, иллюстрирующие работу кодека. Все работают на CPU, GPU опциональна.

| Скрипт | Эпизод сценария | Что показывает |
|--------|-----------------|----------------|
| [`demo_inference.py`](demo_inference.py) | 5 | Один кадр → ROI → quantizer → VAE → reconstruction (PNG 2×2) |
| [`demo_quantization.py`](demo_quantization.py) | 6 | Семейство кривых `q_t(E_ROI; κ)`, сравнение с baseline |
| [`demo_pipeline.py`](demo_pipeline.py) | 3 + 5 | Полный пайплайн: manifest → extract → inference (PNG 1×4) |

## Запуск

Все скрипты используют окружение, поднятое через `uv` в корне репозитория:

```bash
cd /path/to/adaptive-roi-codec
uv sync --extra dev
```

Минимальные зависимости: `torch`, `torchvision`, `opencv-python`, `numpy`, `matplotlib`, `Pillow` (см. `pyproject.toml`).

### demo_inference.py

```bash
# На реальном видео
uv run python scripts/demo_inference.py \
    --video kvasir-capsule/raw/labelled_videos/$(ls kvasir-capsule/raw/labelled_videos | head -1) \
    --output docs/screencast/inference_demo.png

# Без видео (синтетика)
uv run python scripts/demo_inference.py \
    --output docs/screencast/inference_demo_synthetic.png

# С чекпоинтом
uv run python scripts/demo_inference.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/<experiment>/epoch_18.pt
```

### demo_quantization.py

```bash
uv run python scripts/demo_quantization.py \
    --output docs/screencast/quantization_curves.png
```

### demo_pipeline.py

```bash
uv run python scripts/demo_pipeline.py \
    --video kvasir-capsule/raw/labelled_videos/<one>.mp4 \
    --workdir kvasir-capsule \
    --output docs/screencast/pipeline_demo.png
```

## Замечания

- Все скрипты используют `matplotlib.use("Agg")` — не требуется дисплей.
- MobileNetV3-pretrained веса подгружаются при первом запуске; если интернета нет — добавьте `--no-pretrained` в `demo_inference.py` и `demo_pipeline.py`.
- Размер PNG регулируется флагом `--dpi` (по умолчанию 150).
- Все выходы по умолчанию кладутся в `docs/screencast/`.
