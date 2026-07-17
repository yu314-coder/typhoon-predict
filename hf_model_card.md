---
license: mit
library_name: pytorch
pipeline_tag: time-series-forecasting
tags:
- tropical-cyclone
- weather-forecasting
- pytorch
- era5
- ibtracs
---

# StormFusion-MT & TrackFormer — tropical-cyclone forecasting

Two research checkpoints for tropical-cyclone forecasting. Each predicts, at 20 six-hourly lead
times (6–120 h), a 17-dim state per lead: east/north storm motion (km), max wind (kt), central
pressure (hPa), radius of max wind (km), and 34/50/64-kt wind radii in four quadrants.

**Research models — not an operational warning system. Do not use for evacuation, aviation,
maritime, or emergency decisions.**

| model | params | inputs | training data |
|---|---|---|---|
| **TrackFormer v8** | 15M (fp16, 30MB) | **track history only, protected dual-stream** | all basins, 1980+, 193k partial-lead windows |
| StormFusion-MT v2 | 3.3M (fp16, 6.7MB) | ERA5 patches + track history | WP, 2000+, 1,337 storm-centered windows |
| TrackFormer (v1) | 21M (fp16, 43MB) | track history only (single-stream) | all basins, 1980+, 84,150 windows |

Weights and full reproducible code (dataset builders, training, eval) are in the GitHub repo:
**https://github.com/yu314-coder/typhoon-predict** (`models/`).

## Results — WP 2020+ held-out test (lower is better)

| model | track km | vmax kt | pres hPa | rmw km | radius km |
|---|---|---|---|---|---|
| StormFusion-MT v2 (3.3M, ERA5) | 729 | 24.2 | 21.6 | 16.2 | 31.8 |
| TrackFormer v1 (21M, single-stream) | 720 | 22.1 | 21.2 | 11.8 | 31.5 |
| TrackFormer v3 (15M, dual-stream) | 659 | 21.6 | 18.1 | 11.8 | 28.8 |
| **TrackFormer v8 (15M, +partial-lead data)** | **649** | **20.7** | **15.9** | **11.3** | **27.8** |

**Key findings.** (1) A track-only model that never sees ERA5 **matches or beats** the full ERA5
model, so **data diversity > engineered features > parameters** (a 17.7M ERA5 model overfit and did
*worse* than the 3.3M one). (2) Naively adding motion-dynamics features to a single-stream model
improves intensity but hurts track through **negative transfer**; **TrackFormer v3** fixes this with a
protected dual-stream architecture (separate kinematic/thermodynamic encoders, gradient routing, a
zero-init gated thermo→track adapter, and a persistence-residual track head), cutting WP-2020+ track
error to 659 km (−61, storm-bootstrap 95% CI [−103, −16] km, p≈0.995) while keeping the intensity
gains. Full architecture and derivation (incl. a random-matrix block-covariance uncertainty head) in
`paper/trackformer.pdf` in the GitHub repo.

## Architectures

- **StormFusion-MT v2** — separate inner/outer ERA5 conv encoders keeping a 3×3 grid of spatial
  tokens, track/environment token encoders, a temporal Transformer context, learned + sinusoidal
  lead-time queries, cross-attention decoding, and multi-task state / log-scale heads.
- **TrackFormer** — the same decoder design, track-only: a 40-dim track-history projection →
  Transformer context (d_model 384, 8 heads, 4+6 layers) → lead queries → dual heads. No
  atmospheric inputs.

## Usage

See the GitHub repo for `model_v2.py` / `train_track.py`, the checkpoints, and normalization
stats. Inputs are per-feature standardized (stats saved with each checkpoint / dataset);
multiply predictions by `TARGET_SCALE = [100,100,35,20,50] + [50]*12` for physical units.

## Data

IBTrACS v04r01 best tracks (NOAA NCEI) and ERA5 reanalysis (Copernicus/ECMWF). Obtain the source
data under its own access and licensing terms.

## Limitations

- Absolute track error (~720 km averaged over 6–120 h) is far from operational quality.
- The real ceiling is storm **diversity** (~13k storms have ever existed); larger models overfit.
- Wind-radius labels are sparse; no calibration or comparison against official agency forecasts.
- Pre-satellite track/intensity labels are lower quality.
