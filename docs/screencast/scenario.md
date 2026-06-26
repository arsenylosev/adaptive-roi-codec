# Сценарий видеозаписи экрана — «Нейросетевой видеокодер с адаптивным распределением битрейта по ROI для капсульной эндоскопии»

> **Цель:** продемонстрировать работоспособность программной разработки (ПР) для комиссии ЭВМ.
> **Хронометраж:** 6–7 минут (рекомендуемый потолок для регистрации ЭВМ — 7 мин).
> **Формат:** запись экрана (1080p, ≥ 24 fps) + закадровый комментарий. Терминал — крупным шрифтом, IDE — в полноэкранном режиме.

---

## 0. Подготовка перед записью (один раз)

### 0.1. Что должно быть установлено локально

| Компонент | Зачем | Команда проверки |
|-----------|-------|------------------|
| Python 3.10+ | рантайм | `python3 --version` |
| `uv` | менеджер пакетов и окружений | `uv --version` |
| `git` | доступ к репозиторию | `git --version` |
| `ffmpeg` (опц.) | генерация демо-видео из кадров | `ffmpeg -version` |

Зависимости Python — через `uv sync` (без `--extra cloud` для ЭВМ; DataSphere и YC CLI не нужны):

```bash
cd adaptive-roi-codec
uv sync
```

`matplotlib` и остальные зависимости для демо-скриптов объявлены в `pyproject.toml` (устанавливаются автоматически). Опционально: `uv sync --extra dev` для pytest и ruff.

Минимальный набор библиотек для демо: `torch`, `torchvision`, `opencv-python` (через albumentations), `numpy`, `matplotlib`, `Pillow`.

### 0.2. Датасет

Полный Kvasir-Capsule (~61 GB) скачивать для ЭВМ **не обязательно**. Достаточно одного из вариантов:

| Вариант | Когда использовать | Как получить |
|---------|--------------------|--------------|
| **A. Полный датасет** | если уже скачан локально | путь `kvasir-capsule/raw/labelled_videos/` |
| **B. Один MP4** | рекомендуется для ЭВМ | скачать любой `*.mp4` из [официального Kvasir-Capsule](https://datasets.simula.no/kvasir-capsule/) |
| **C. Публичные примеры** | если ничего нет | `https://datasets.simula.no/kvasir-capsule/` — взять 1–3 MP4 |
| **D. Синтетика** | fallback | скрипт `scripts/demo_inference.py` умеет работать без видео |

Положите выбранные файлы в `kvasir-capsule/raw/labelled_videos/` (или передайте `--video path/to/file.mp4` напрямую в скрипты).

### 0.3. Чекпоинты

Если в репозитории ещё нет обученных весов (`checkpoints/epoch_*.pt` или `checkpoints/<experiment>/epoch_*.pt`), для ЭВМ допустимо:

1. Обучить короткую модель на CPU: `--dry-run` (см. эпизод 7);
2. Либо загрузить веса с S3 (`s3://<bucket>/checkpoints/<experiment_id>/`) — если соответствующий прогон уже делался в DataSphere.

Имена файлов чекпоинтов — **три цифры эпохи**: `epoch_018.pt`, не `epoch_18.pt`.

**Загрузка в `demo_inference.py`:** по умолчанию из чекпоинта берётся только **VAE**; ROI-детектор — **pretrained MobileNetV3** (наглядная карта значимости). Флаг `--load-roi-checkpoint` загружает ROI из чекпоинта; при вырожденной маске скрипт автоматически откатывается на pretrained.

Для ЭВМ **синтетически инициализированная модель тоже подходит**: комиссии важно увидеть, что *архитектура и пайплайн работают*, а не конкретные метрики PSNR/SSIM.

### 0.4. Настройки экрана перед записью

- Терминал: моноширинный шрифт 16–18 pt, тёмная тема (на контрасте с яркими визуализациями matplotlib).
- Верхняя панель VS Code/Cursor скрыта, минимизировать лишние уведомления.
- Раскладка: split-screen 50/50 — слева терминал, справа редактор/визуализация (либо переключение `Alt-Tab`).
- Звук: проверить уровень записи заранее; системные уведомления отключены.

---

## 1. Структура сценария (по эпизодам)

| # | Эпизод | Тайминг | Что в кадре | Что говорить (тезисы) |
|---|--------|---------|-------------|------------------------|
| 1 | **Вступление и постановка задачи** | 0:00 – 0:30 | Заголовок презентации / `README.ru.md` | Название ПР, решаемая проблема, целевая область |
| 2 | **Состав репозитория** | 0:30 – 1:10 | `tree adaptive_roi_codec` + `AGENTS.md` | Модульная структура: VAE, ROI-детектор, адаптивный квантизатор |
| 3 | **Подготовка датасета** | 1:10 – 2:10 | `build-dataset-manifest`, `extract-frames` | Как устроен Kvasir-Capsule, что такое `MANIFEST.json` |
| 4 | **Архитектура в коде** | 2:10 – 3:30 | `model/roi_detector.py`, `quantizer.py`, `vae_codec.py` | U-Net + MobileNetV3, формулы (4)–(6), motion compensator |
| 5 | **Live inference demo** | 3:30 – 5:10 | `scripts/demo_inference.py` | ROI-карта, квантизация, реконструкция side-by-side |
| 6 | **Адаптивное распределение битрейта** | 5:10 – 6:00 | `scripts/demo_quantization.py` | Зависимость шага q_t от E_ROI и κ; сравнение с baseline |
| 7 | **Dry-run обучения** | 6:00 – 6:40 | `python -m adaptive_roi_codec.train --dry-run` | Лог потерь, эпоха, чекпоинт |
| 8 | **Заключение** | 6:40 – 7:00 | `docs/experiments/v100-kappa-2.0-18ep.ru.md` | Метрики из реального прогона; соответствие ТЗ |

**Контрольные точки** (обязательно должны быть видны/слышны на записи):

- [K1] На экране в эпизоде 1 — название, соответствующее регистрационному документу.
- [K2] В эпизоде 5 — PNG 2×2: `original | roi_overlay | quantized_latent | reconstruction` (ROI — теплокарта поверх кадра).
- [K3] В эпизоде 6 — график `q_t(E_ROI)` при разных κ (минимум 3 кривых).
- [K4] В эпизоде 7 — в логах видны `Device: cpu`, `Training resolution: 336x336`, строка `Epoch … complete — loss=…`, сохранение чекпоинта.

---

## 2. Покадровый сценарий

### Эпизод 1. Вступление (0:00 – 0:30)

**На экране:** первый слайд либо терминал с командой:

```bash
cd adaptive-roi-codec
cat README.ru.md | head -3
```

**Голос:** «Программная разработка — *Нейросетевой видеокодер с адаптивным распределением битрейта по областям интереса для капсульной эндоскопии*. Цель — снизить битрейт при сохранении клинически значимой информации в кадрах, получаемых капсульным эндоскопом.»

### Эпизод 2. Состав репозитория (0:30 – 1:10)

**На экране:** терминал

```bash
ls -la
cat AGENTS.md | head -40
```

**Голос:** «Репозиторий содержит Python-пакет `adaptive_roi_codec/`. Внутри — три модели: `VAECodec` (вариационный автоэнкодер с motion compensation), `ROIDetector` (U-Net на базе MobileNetV3), `AdaptiveQuantizer` (пространственно-адаптивное квантование). Документация и ADR — в `docs/`, конфиги — в `configs/`, шаблоны DataSphere Jobs — в `jobs/`…»

### Эпизод 3. Подготовка датасета (1:10 – 2:10)

**На экране:** терминал + редактор с `MANIFEST.json`

```bash
uv run build-dataset-manifest --dataset-root kvasir-capsule
ls kvasir-capsule/
head kvasir-capsule/MANIFEST.json
cat kvasir-capsule/splits/train_videos.txt | head -5
```

**Опционально (если нужен препроцессинг):** замените `<VIDEO_ID>` на имя файла без `.mp4` (например `131368cc17e44240`):

```bash
uv run extract-frames --video kvasir-capsule/raw/labelled_videos/131368cc17e44240.mp4 \
                      --output kvasir-capsule/processed/frames \
                      --height 336 --width 336
```

**Голос:** «Подготовка датасета состоит из двух шагов. Первый — генерация манифеста: для каждого видео сохраняется путь, число кадров, целевое разрешение 336×336. Второй — извлечение кадров в формате `.npy` для ускорения последующего обучения.»

### Эпизод 4. Архитектура в коде (2:10 – 3:30)

Откройте в редакторе три файла в соседних вкладках:

1. `adaptive_roi_codec/model/roi_detector.py` — показать `class ROIDetector`, метод `forward` (interpolate → encoder → head → sigmoid).
2. `adaptive_roi_codec/model/quantizer.py` — показать формулы `global_step` и `spatial_step` (формулы 4–6 из статьи).
3. `adaptive_roi_codec/model/vae_codec.py` — показать `VAEEncoder`, `VAEDecoder` (skip-connections), `MotionCompensator`.

**Голос:** «Детектор ROI на входе получает кадр 336×336, прогоняет через MobileNetV3-Large, выход — карта значимости 3×H×W в диапазоне [0,1]. Адаптивный квантизатор реализует формулы 4–6: глобальный шаг q_t зависит от средней активации ROI в степени κ, пространственный шаг дополнительно модулируется локальной маской. VAE — стандартная U-Net-архитектура с 4 уровнями, латентное пространство 21×21×192, и отдельный motion compensator для временной согласованности.»

### Эпизод 5. Live inference demo (3:30 – 5:10)

**На экране:** два окна: терминал слева, PNG-визуализация справа.

```bash
# Показать справку
uv run python scripts/demo_inference.py --help

# Рекомендуемый запуск: MP4 + обученный VAE + авто-выбор кадра
uv run python scripts/demo_inference.py \
    --video kvasir-capsule/raw/labelled_videos/131368cc17e44240.mp4 \
    --auto-frame \
    --checkpoint checkpoints/epoch_018.pt \
    --output docs/screencast/inference_demo.png

# Без чекпоинта (pretrained ROI + random VAE — тоже допустимо для ЭВМ)
uv run python scripts/demo_inference.py \
    --video kvasir-capsule/raw/labelled_videos/131368cc17e44240.mp4 \
    --auto-frame \
    --output docs/screencast/inference_demo.png

# Альтернатива: один препроцессированный кадр .npy
uv run python scripts/demo_inference.py \
    --frame kvasir-capsule/processed/frames/131368cc17e44240/frame_000000.npy \
    --checkpoint checkpoints/epoch_018.pt \
    --output docs/screencast/inference_demo.png
```

> **Важно:** флаг кадра — `--frame-idx` (через дефис), не `--frame_idx`. Плейсхолдеры вроде `<one>.mp4` нужно заменить реальным именем файла.

Скрипт выводит PNG 2×2:

```
┌──────────────────┬──────────────────┐
│  Original frame  │  ROI overlay     │
├──────────────────┼──────────────────┤
│ Quantized latent │  Reconstruction  │
└──────────────────┴──────────────────┘
```

Панель ROI — полупрозрачная теплокарта (magma) поверх кадра с контурными линиями; при малом контрасте маски контраст усиливается для отображения.

**Голос:** «Запускаем скрипт `demo_inference.py`. Он берёт кадр из видео (флаг `--auto-frame` выбирает кадр с наиболее выраженной ROI-структурой), прогоняет через детектор ROI и VAE с адаптивным квантизатором. На карте ROI видно, что модель выделяет участки с патологически значимой текстурой — там шаг квантования меньше и битрейт выше.»

### Эпизод 6. Адаптивное распределение битрейта (5:10 – 6:00)

```bash
uv run python scripts/demo_quantization.py \
    --output docs/screencast/quantization_curves.png
```

Скрипт рисует семейство кривых `q_t = q_min + (q_max - q_min) * E_ROI^κ` для κ ∈ {0.5, 1.0, 2.0, 4.0} и сравнивает с baseline (фиксированный шаг = 1.0).

**Голос:** «Это ключевой график разработки. По горизонтали — средняя активация ROI от 0 до 1. По вертикали — шаг квантования: чем он меньше, тем больше бит уходит на эту область. При κ = 2 — то, что мы используем в эксперименте — кривая выпукла вверх: ROI-зоны получают шаг в разы меньше, чем фон. Baseline — прямая линия, битрейт равномерный.»

### Эпизод 7. Dry-run обучения (6:00 – 6:40)

```bash
# Синтетика (без датасета). На ноутбуке без GPU отключите TRAIN_REQUIRE_CUDA:
TRAIN_REQUIRE_CUDA=0 uv run python -m adaptive_roi_codec.train \
    --config configs/base.yaml \
    --dry-run
```

**На экране:** терминал, в логе видны:

```
[INFO] Device: cpu
[INFO] Training resolution: 336x336 data.source=video batch_size=4
[WARNING] Dry-run mode: using 336x336 synthetic frames
[INFO] Epoch 1/1 complete — loss=… batches=1 elapsed=…s
[INFO] Saved checkpoint to checkpoints/default/epoch_001.pt
[INFO] Training finished successfully
```

**Голос:** «Запускаем training в режиме dry-run: один батч, синтетические кадры 336×336. В логах видны разрешение, источник данных и суммарная потеря. Суммарная потеря объединяет реконструкцию (PSNR+SSIM), ROI-взвешенную ошибку, KL-rate и temporal loss от motion compensator — это функция потерь из формулы (14) в `clinical_loss.py`.»

> **Примечание:** полный GPU-прогон на реальных данных — через DataSphere (см. `AGENTS.md`). Локальный `train_smoke.json` требует stage-1 manifest (`frames_cache/frames_manifest.jsonl`) и не обязателен для записи ЭВМ.

### Эпизод 8. Заключение (6:40 – 7:00)

**На экране:** открыть отчёт о реальном прогоне:

```bash
cat docs/experiments/v100-kappa-2.0-18ep.ru.md | head -40
```

**Голос:** «В реальном прогоне на V100 (18 эпох, κ=2) достигнуты метрики: … — соответствуют ТЗ. Разработка работоспособна, исходный код опубликован в репозитории, эксперименты воспроизводимы.»

---

## 3. Подстраховка (если что-то идёт не так)

| Проблема | Решение |
|----------|---------|
| Нет GPU | Все скрипты (`demo_inference.py`, `demo_quantization.py`) работают на CPU. `torch.cuda.is_available()` → `False` — это нормально. |
| `TRAIN_REQUIRE_CUDA` / `Device: cpu` error | В `.env` может быть `TRAIN_REQUIRE_CUDA=1`. Для локальной записи: `TRAIN_REQUIRE_CUDA=0 uv run python -m adaptive_roi_codec.train …` |
| Нет датасета Kvasir-Capsule | `--video` принимает любой `.mp4`. Можно взять любой эндоскопический клип либо один публичный кадр. |
| `unrecognized arguments: --frame_idx` | Используйте `--frame-idx` (дефис). |
| ROI — «цветной шум» или синий квадрат | Не используйте `--load-roi-checkpoint` без необходимости; добавьте `--auto-frame`; загружайте VAE через `--checkpoint checkpoints/epoch_018.pt` |
| `ModuleNotFoundError: matplotlib` | `uv sync` в корне репозитория (matplotlib в `pyproject.toml`) |
| Не установлен MobileNetV3-pretrained | Скрипт подгружает веса через `torchvision`. Если интернета нет — `--no-pretrained` (пайплайн работает, ROI хуже). |
| Не запускается `uv run` | `source .venv/bin/activate` или `uv pip install -e .` |
| Matplotlib headless | Скрипты используют `matplotlib.use("Agg")` — дисплей не нужен. |
| Картинка мелкая | `--dpi 200` увеличит PNG. |

---

## 4. Чек-лист финальной записи

- [ ] Звук чистый, без системных уведомлений (отключить уведомления ОС на время записи).
- [ ] Разрешение 1920×1080 (Full HD), ≥ 24 fps.
- [ ] Все команды, набранные вручную в терминале (не Ctrl+C / Ctrl+V из буфера) — комиссия должна видеть процесс.
- [ ] Каждая визуализация PNG держится в кадре ≥ 3 секунд.
- [ ] Финальный артефакт — `docs/screencast/inference_demo.png`, `docs/screencast/quantization_curves.png`, и сам видеофайл.
- [ ] В описании видео — ссылка на коммит репозитория, использованный при записи (для воспроизводимости).

---

## 5. Соответствие требованиям ЭВМ

| Требование | Как покрыто |
|------------|-------------|
| Демонстрация работоспособности | эпизод 5 (live inference) + эпизод 7 (training dry-run) |
| Соответствие названию ПР | эпизод 1 (явное проговаривание названия) |
| Адаптивность битрейта по ROI | эпизод 6 (график q_t(E_ROI, κ)) |
| Нейросетевая природа | эпизод 4 (показ кодека + ROI-детектора) |
| Капсульная эндоскопия как домен | эпизод 5 (кадры эндоскопии) |
| Воспроизводимость | эпизод 8 (ссылка на эксперимент + команды) |

---

## 6. Связанные артефакты

- `scripts/demo_inference.py` — live inference на одном кадре.
- `scripts/demo_quantization.py` — графики семейства q_t(E_ROI, κ).
- `scripts/demo_pipeline.py` — end-to-end: manifest → extract → inference.
- `docs/opisanie-eksperimenta.md` — описание эксперимента для статьи.
- `docs/experiments/v100-kappa-2.0-18ep.ru.md` — отчёт о реальном V100-прогоне.
