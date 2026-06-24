# Experiment reports and data

This directory holds write-ups and **committed numerical exports** for training runs used in the scientific paper.

| Experiment ID | Report | Data (JSON/CSV) | Notes |
|---------------|--------|-----------------|-------|
| `v100-kappa-2.0-18ep-bs24` | [EN](v100-kappa-2.0-18ep.en.md) · [RU](v100-kappa-2.0-18ep.ru.md) | [v100-kappa-2.0-18ep-bs24/](v100-kappa-2.0-18ep-bs24/) | First full 18-epoch V100 run; κ=2.0; batch 36 |

## Adding a new experiment

1. Run training; collect DataSphere stdout, diagnostics, and job outputs locally.
2. Export metrics to `docs/experiments/<experiment_id>/` (`metrics.json`, `epochs.csv`, …).
3. Add EN/RU narrative reports and a row in the table above.
4. Keep raw log dumps outside the repo (or gitignored); commit only structured metrics.

See [v100-kappa-2.0-18ep-bs24/README.md](v100-kappa-2.0-18ep-bs24/README.md) for file format details.
