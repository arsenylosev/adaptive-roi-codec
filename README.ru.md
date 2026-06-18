# Адаптивный нейросетевой видеокодек ROI для капсульной эндоскопии

Нейросетевой видеокодек для капсульной эндоскопии с **адаптивным распределением битрейта** по клинически значимым областям интереса (ROI). Полное описание эксперимента для научной статьи: **[docs/opisanie-eksperimenta.md](docs/opisanie-eksperimenta.md)**.

Обучение выполняется в **нативном разрешении Kvasir-Capsule 336×336** ([ADR-002](docs/decisions/ADR-002-native-336-training-resolution.md)). Полные прогоны — на **Yandex DataSphere Jobs** (GPU). Локально — генерация манифеста, smoke-тесты и отладка.

> English documentation: [README.md](README.md)

## Требования

- Python 3.10+
- Менеджер пакетов [uv](https://docs.astral.sh/uv/)
- [Yandex Cloud CLI](https://yandex.cloud/ru/docs/cli/quickstart) (`yc`), аутентификация
- Проект DataSphere с S3-коннектором к Object Storage (датасет Kvasir-Capsule)
- Пакет `datasphere` для запуска job-ов (опциональная зависимость `[cloud]`)

## Установка

```bash
uv sync --extra cloud --extra dev
cp .env.example .env
# Заполните .env: DATASPHERE_PROJECT_ID, S3_CONNECTOR_ID, ключи бакета
yc init
```

## Манифест датасета и сплиты

При локальной копии Kvasir-Capsule в `kvasir-capsule/`:

```bash
uv run build-dataset-manifest --dataset-root kvasir-capsule
```

Создаёт `MANIFEST.json` и `splits/{train,val,test}_videos.txt`.

## Структура проекта

```
adaptive_roi_codec/     # VAE, детектор ROI, функции потерь, обучение, CLI
configs/                # base.yaml — параметры 336×336
docs/
  opisanie-eksperimenta.md   # описание эксперимента для статьи
  guides/                    # инструкции по облаку
  decisions/                 # ADR
jobs/                   # шаблоны DataSphere Jobs + JSON-параметры прогонов
kvasir-capsule/         # локальный датасет (в .gitignore)
docker/datasphere-gpu/  # Dockerfile для образа проекта DataSphere
```

## Локальный smoke-тест

```bash
uv run pytest
uv run python -m adaptive_roi_codec.train \
  --config configs/base.yaml \
  --params jobs/inputs/train_input.json \
  --dry-run
```

## Запуск обучения в Yandex Cloud

Пошаговое руководство: **[docs/guides/cloud-launch.md](docs/guides/cloud-launch.md)**

Краткий путь:

```bash
# 1. Загрузить kvasir-capsule/{MANIFEST.json,splits/,raw/} в Object Storage
# 2. Создать S3-коннектор DataSphere → указать S3_CONNECTOR_ID в .env
# 3. (Рекомендуется) Этап 1: извлечение кадров на CPU
uv run launch-train --job extract --execute

# 4. Этап 2: GPU-обучение
uv run launch-train --dry-run    # проверить сгенерированный YAML
uv run launch-train --execute    # отправить GPU job

# Длинный мониторинговый прогон на V100:
uv run launch-train --execute --async \
  --params jobs/inputs/train_smoke_v100_monitor.json \
  --template jobs/configs/job_train_v100.yaml.template
```

Чекпоинты: `s3://…/checkpoints/<experiment_id>/epoch_*.pt`

### Ускорение I/O: bulk staging на SSD job-а

В параметрах прогона (`jobs/inputs/*.json`):

```json
"data": {
  "source": "preprocessed",
  "stage_frames_local": true,
  "stage_mode": "bulk",
  "num_workers": 4
}
```

Перед обучением кадры копируются с S3 на extended SSD (`150 GB`), что снимает латентность FUSE во время итераций.

## Переопределение параметров эксперимента

Файл `jobs/inputs/train_input.json`:

```json
{
  "experiment_id": "baseline-kappa-2.0",
  "training": { "epochs": 50, "batch_size": 12 },
  "quantizer": { "kappa": 2.0 },
  "data": {
    "split": "train",
    "frame_stride": 30,
    "stage_frames_local": true,
    "stage_mode": "bulk",
    "num_workers": 4
  }
}
```

## Архитектура и решения

ADR в `docs/decisions/`:

- [ADR-001](docs/decisions/ADR-001-datasphere-jobs-training.md) — DataSphere Jobs + S3
- [ADR-002](docs/decisions/ADR-002-native-336-training-resolution.md) — разрешение 336×336
- [ADR-003](docs/decisions/ADR-003-two-stage-training-pipeline.md) — двухэтапный пайплайн, pinned CUDA

## Docker-образ для DataSphere (опционально)

Сборка образа **внутри проекта DataSphere** (без YCR): см. `docker/datasphere-gpu/Dockerfile` и шаблон job `docker/datasphere-gpu/job_train.docker.yaml.template`.

Контекст сборки загружается в хранилище проекта (рекомендуется **Git → Clone a Repository** в JupyterLab).

## Безопасность

- Ключи и ID облака — только в `.env` (не коммитить).
- Не коммитить чекпоинты и датасет.

## Участие в разработке

Сообщения коммитов — [Conventional Commits](CONTRIBUTING.md) (`feat:`, `fix:`, `docs:` и т.д.).

## Ссылки

- [Описание эксперимента (RU)](docs/opisanie-eksperimenta.md)
- [Руководство по облачному запуску](docs/guides/cloud-launch.md)
- [Настройка S3-коннектора DataSphere](docs/guides/datasphere-s3-connector.md)
- [DataSphere Jobs](https://yandex.cloud/ru/docs/datasphere/concepts/jobs/)
