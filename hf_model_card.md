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

# Typhoon Predict — tropical-cyclone forecasting

**Research models — not an operational warning system. Do not use for evacuation, aviation,
maritime, or emergency decisions.**

## Current best model: TrackFormer v23

TrackFormer v23 predicts the atmospheric steering flow that carries a storm as an explicit
chain-of-thought (CoT) intermediate step, conditions that estimate on how the flow has been
evolving over the previous day (t-24h, t-12h, now), and derives track from it. Result:
**434.96 km RMS track error** (10-seed ensemble), WP+EP 2020+, full 20-lead-horizon test set
(3,763 windows).

This is the best-performing model in the whole project, reached through a longer architecture
progression:

1. **v10–v20** — a small CNN encoder reads a deep-layer-mean steering-wind patch around the storm.
2. **v21, v22** — chain-of-thought: predict the steering flow itself, then derive track from it
   (v22 adds a latent CoT with weight-tied feedback rounds).
3. **v23** — add a temporal history of the CoT steering representation. Best result: **434.96 km**.
4. **v24–v29** — four further environmental additions on top of v23 (an environmental token, an
   ocean-heat CNN patch, a drift adapter, raw ERA5 steering wind) all came back **null**: once a
   CoT representation already extracts the steering signal that matters, handing the model the raw
   field again is redundant.
5. **v31–v34** — land/terrain-interaction correction, motivated by real-world reports of typhoons
   stalling at mountainous coastlines (Typhoon Gaemi, 2024, at Taiwan) and terrain-deflection
   literature (AOT-TCNet, arXiv 2603.29200):

| model | aggregate track (km) | Typhoon Tip 1979 (km) | Typhoon Noul 2026, ocean/landfall (km) |
|---|---|---|---|
| v23 (baseline) | **434.96** | 939 | 267 / 356 |
| v31 — LandDrag, uniform training | 443.07 (+8.11) | — | — |
| v32 — LandDrag, window-oversampled | 460.48 (+25.52, backfired) | — | — |
| v33 — LandDrag, storm-normalized | 442.33 (+7.37) | **876** | **243 / 319** |
| v34 — LandGate, **frozen v23 backbone** | 460.52 (−0.33 vs. own backbone) | **795** | 287 / 382 |

v34 is the methodologically important result: v31–v33 each retrained the *entire* architecture
from scratch, so their deltas vs. v23 include ~19 km/seed of ordinary retrain noise on top of
whatever the land correction did. v34 instead freezes a real, already-trained v23 checkpoint and
trains only a new ~437-parameter gated correction — a true same-backbone-plus-one-addition
comparison. Result: essentially null everywhere, including the mountainous-near-land regime every
earlier attempt targeted. On the two real out-of-training storms available, v33 and v34 each split
1–1 against v23 — a small-n disagreement with the aggregate test set, not a reliable effect.

**Methodological lessons:** retrain-to-retrain seed noise (~19 km/seed) is large enough to
manufacture or hide most small version-to-version deltas; freezing a real backbone and training
only a small addition isolates a causal effect that comparing two from-scratch runs cannot; and a
large in-distribution aggregate test set does not always agree with genuinely out-of-training
real-storm validation.

These are research checkpoints, not yet converted to this card's release format — see below for
checkpoints you can load and run today. Full write-up, architecture equations, and every
intermediate result: `paper/trackformer.pdf`, "Phase II: chain-of-thought steering and
land-interaction testing," in the GitHub repo
(**https://github.com/yu314-coder/typhoon-predict**).

## Released checkpoints: StormFusion-MT & TrackFormer v1–v9

The earlier, fully released and locally-runnable line. Each predicts, at 20 six-hourly lead times
(6–120 h), a 17-dim state per lead: east/north storm motion (km), max wind (kt), central pressure
(hPa), radius of max wind (km), and 34/50/64-kt wind radii in four quadrants.

| model | params | inputs | training data |
|---|---|---|---|
| **TrackFormer v9** | 17M (fp16, 33MB) | **track history + IBTrACS environment, protected triple-stream** | all basins, 1980+, 193k partial-lead windows |
| TrackFormer v8 | 15M (fp16, 30MB) | track history only, protected dual-stream | all basins, 1980+, 193k partial-lead windows |
| StormFusion-MT v2 | 3.3M (fp16, 6.7MB) | ERA5 patches + track history | WP, 2000+, 1,337 storm-centered windows |
| TrackFormer (v1) | 21M (fp16, 43MB) | track history only (single-stream) | all basins, 1980+, 84,150 windows |

Weights and full reproducible code (dataset builders, training, eval) are in the GitHub repo
(`models/`).

### Results — WP 2020+ held-out test (lower is better)

| model | track km | vmax kt | pres hPa | rmw km | radius km |
|---|---|---|---|---|---|
| StormFusion-MT v2 (3.3M, ERA5) | 729 | 24.2 | 21.6 | 16.2 | 31.8 |
| TrackFormer v1 (21M, single-stream) | 720 | 22.1 | 21.2 | 11.8 | 31.5 |
| TrackFormer v3 (15M, dual-stream) | 659 | 21.6 | 18.1 | 11.8 | 28.8 |
| TrackFormer v8 (15M, +partial-lead data) | 649 | 20.7 | 15.9 | **11.3** | 27.8 |
| **TrackFormer v9 (17M, +IBTrACS environment)** | **618** | **18.6** | 15.8 | 11.5 | **27.2** |

**Key findings.** (1) A track-only model that never sees ERA5 **matches or beats** the full ERA5
model, so **data diversity > engineered features > parameters** (a 17.7M ERA5 model overfit and did
*worse* than the 3.3M one). (2) Naively adding motion-dynamics features to a single-stream model
improves intensity but hurts track through **negative transfer**; **TrackFormer v3** fixes this with a
protected dual-stream architecture (separate kinematic/thermodynamic encoders, gradient routing, a
zero-init gated thermo→track adapter, and a persistence-residual track head), cutting WP-2020+ track
error to 659 km (−61, storm-bootstrap 95% CI [−103, −16] km, p≈0.995) while keeping the intensity
gains. Full architecture and derivation (incl. a random-matrix block-covariance uncertainty head) in
`paper/trackformer.pdf`, Phase I, in the GitHub repo.

### Architectures

- **StormFusion-MT v2** — separate inner/outer ERA5 conv encoders keeping a 3×3 grid of spatial
  tokens, track/environment token encoders, a temporal Transformer context, learned + sinusoidal
  lead-time queries, cross-attention decoding, and multi-task state / log-scale heads.
- **TrackFormer** — the same decoder design, track-only: a 40-dim track-history projection →
  Transformer context (d_model 384, 8 heads, 4+6 layers) → lead queries → dual heads. No
  atmospheric inputs.

### Usage

See the GitHub repo for `model_v2.py` / `train_track.py`, the checkpoints, and normalization
stats. Inputs are per-feature standardized (stats saved with each checkpoint / dataset);
multiply predictions by `TARGET_SCALE = [100,100,35,20,50] + [50]*12` for physical units.

## Data

IBTrACS v04r01 best tracks (NOAA NCEI) and ERA5 reanalysis (Copernicus/ECMWF). Obtain the source
data under its own access and licensing terms.

## Limitations

- Research models throughout — not operational quality in either line. TrackFormer v23 (434.96 km
  RMS track error, 6–120 h) and the released StormFusion-MT/TrackFormer v1–v9 checkpoints
  (~618–730 km, different test split) are both far from operational.
- The v10–v34 line's checkpoints are not yet packaged for this card's load/run format.
- The real ceiling is storm **diversity** (~13k storms have ever existed); larger models overfit.
- Wind-radius labels are sparse; no calibration or comparison against official agency forecasts.
- Pre-satellite track/intensity labels are lower quality.
