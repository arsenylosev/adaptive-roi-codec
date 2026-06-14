# ADR-002: Train at native Kvasir-Capsule 336×336 resolution

## Status

Accepted

## Date

2026-06-14

## Context

The project specification ([MCE-Experiment-Requirements-EN.md](../../MCE-Experiment-Requirements-EN.md)) describes a Full HD (1920×1080) deployment target and paper-scale latent tensors (120×68×192). The **actual Kvasir-Capsule release** used in this repository stores:

- **43 labelled MP4 videos** at **336×336**, 30 fps (~31 GB in the current partial download)
- **14 labelled image archives** with 336×336 JPEG frames (`{video_id}_{frame_idx}.jpg`)

Upsampling 336→1920 before training would multiply compute and memory (~32× pixels) without adding information present in the dataset. The VAE latent shape must follow input resolution under 16× spatial downsampling (336÷16 = **21×21**, not 120×68).

## Decision

1. **Training and evaluation resolution:** Use **336×336×3** as the canonical training input size across config, dataloaders, ROI detector, and smoke tests.

2. **Latent documentation values:** Set `model.latent_h` / `model.latent_w` to **21** in `configs/base.yaml` to reflect the encoder output at 16× compression. The channel width (`latent_ch: 192`) is unchanged.

3. **ROI detector input:** Set `roi_detector.input_res` to **336** (native size; no upscale to 512).

4. **Data layout on Object Storage:** Store raw downloads under `kvasir-capsule/raw/labelled_videos/` and `kvasir-capsule/raw/labelled_images/`, with `MANIFEST.json` and `splits/*.txt` at the dataset prefix root. Training reads videos via OpenCV streaming (`KvasirVideoFrameDataset`).

5. **Frame sampling:** Default `frame_stride: 30` (~1 frame/s per 30 fps video) to keep epoch time manageable on V100; override per experiment in config or job params.

6. **Paper Full HD targets:** Treat 1920×1080 metrics (PSNR, latency, power) as **deployment-phase** goals with upscaled inference, documented separately from cloud training on Kvasir native resolution.

## Alternatives Considered

### Upsample all frames to 1920×1080 before training

- **Pros:** Matches paper diagram literally.
- **Cons:** ~32× compute/memory; bilinear upsampling does not recover high-frequency detail; misaligned with available labels.
- **Rejected:** Not justified for Kvasir-Capsule training.

### Keep 1920×1080 in config and resize inside the model only at loss time

- **Pros:** Config unchanged from spec.
- **Cons:** Confusing for operators; latent 120×68 incompatible with actual encoder geometry unless architecture changes.
- **Rejected:** Chose explicit 336×336 config aligned with data.

### Extract all labelled JPEGs as primary training source instead of videos

- **Pros:** Direct pathology labels per frame.
- **Cons:** Sparse temporal continuity for motion-compensation loss; video stream is the codec primary modality.
- **Deferred:** Labelled frames will feed ROI supervision in a follow-up task; video loader remains primary for codec training.

## Consequences

- Paper table targets must be reinterpreted or re-benchmarked at 336×336 until a deployment upscaling stage exists.
- `build-dataset-manifest` generates reproducible splits and records native resolution in `MANIFEST.json`.
- Datasphere Jobs must mount the S3 prefix containing `raw/labelled_videos/` and `splits/`.
- Future ADR may cover Full HD inference benchmarking on CPU without changing this training decision.

## References

- [ADR-001](ADR-001-datasphere-jobs-training.md) — cloud training infrastructure
- Kvasir-Capsule: Smedsrud et al., Sci Data 2021
- Local inventory: `kvasir-capsule/MANIFEST.json` (generated)
