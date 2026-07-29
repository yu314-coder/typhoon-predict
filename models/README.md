# Trained models — TrackFormer and StormFusion-MT

Research models for Western-Pacific-and-beyond tropical-cyclone forecasting. **Not** an
operational warning system.

## Current best model: TrackFormer v23 (v10–v34 research line)

A separate, later line of experiments (`colab_train_v17.ipynb` onward, `colab_v2*`–`colab_v39_train.py`)
iterates a WP-focused TrackFormer on a fuller test set (WP+EP, 2020+, full 20-lead windows,
3,763 windows) than the released-checkpoint table below. **v23 is the current best model in the
whole project**, and its 10-seed ensemble is released in this directory
(`v23_seed0.pt`–`v23_seed9.pt`, 52.6 MB each, fp32) with a standalone architecture module and
CLI — see "Usage — v23" below.

### Architecture progression

1. **v10–v20 — CNN steering-field encoder.** A small convolutional encoder (conv, GELU, strided
   downsampling) reads a 17×17 patch of deep-layer-mean steering wind (a weighted blend of
   850/500/200-hPa u/v) centered on the storm.
2. **v21, v22 — chain-of-thought (CoT) steering.** Rather than regressing track displacement
   directly from an opaque decoder, the model first predicts the ambient steering flow itself as
   an explicit, inspectable intermediate representation, then derives track from it. v22 adds a
   latent CoT with two weight-tied feedback rounds.
3. **v23 — temporal steering-history stack, the best model in the project.** Conditions the CoT
   steering estimate on its own history (t-24h, t-12h, and now), not just the current instant, so
   recurvature responds to how the steering flow has been evolving. **434.96 km** RMS track error
   (10-seed ensemble), WP+EP 2020+, full 20-lead-horizon test set.
4. **v24, v25, v27, v28, v29 — further environmental additions on top of v23: all null or
   marginal.** A basin-map/augmentation ablation (v24), an environmental token of ocean heat +
   shear + humidity (v25), an ocean-heat CNN patch on the intensity head (v27), a meridional drift
   adapter whose matched-retrain control (v28abl) exposed the apparent gain as ordinary run-to-run
   noise (v28), and — most tellingly — raw 0.25° ERA5 steering wind fed directly on top of v23
   (v29): 438.70 km, +3.74 km worse, every seed above the bar. Once a CoT representation already
   extracts the steering signal that matters, handing the model the raw field again is redundant.
5. **v31–v34 — land-interaction testing.** Motivated by real-world reports of typhoons stalling at
   mountainous coastlines (Typhoon Gaemi, 2024, at Taiwan's Central Mountain Range) and
   terrain-deflection literature (AOT-TCNet, arXiv 2603.29200):

| model | aggregate WP+EP track (km) | Typhoon Tip 1979 (km) | Typhoon Noul 2026, ocean/landfall (km) |
|---|---|---|---|
| v23 (baseline) | **434.96** | 939 | 267 / 356 |
| v31 — LandDrag, uniform training | 443.07 (+8.11) | — | — |
| v32 — LandDrag, window-level oversampled | 460.48 (+25.52, backfired) | — | — |
| v33 — LandDrag, storm-normalized oversampled | 442.33 (+7.37, closest from-scratch attempt) | **876** | **243 / 319** |
| v34 — LandGate, **frozen v23 backbone** | 460.52 (−0.33 vs. its own frozen backbone) | **795** | 287 / 382 |

**Why v34 is the methodologically cleanest of the land-drag attempts.** v31–v33 all retrained the
*entire* architecture from scratch, so each number above also carries ~19 km/seed of ordinary
retrain noise on top of whatever the land correction did. v34 instead loads a real, already-trained
v23 checkpoint, freezes every parameter except a new ~437-param gated correction, and trains only
that — a true same-backbone-plus-one-addition comparison. Result: **−0.33 km vs its own v23
backbone, essentially null in every distance/terrain bucket tested, including the
mountainous-near-land regime the whole line of experiments targeted.** The three architecturally
different attempts at a small along-track land add-on (uniform sampling, oversampling, gated MoE)
have now converged on no clean aggregate win, though v33 and v34 each beat v23 on one of the two
real out-of-training storms tested (Tip 1979, Noul 2026) and lose on the other — a small-n,
storm-to-storm result, not a reliable effect.

### Methodological lessons (apply to any future version)

1. **Seed noise dominates small version deltas.** A single retrained v23 seed scored 460.85 km on
   this same test set — 25.89 km worse than the 10-seed ensemble mean, from ordinary retrain
   variance alone. Never trust a claimed improvement under ~20 km from a single from-scratch
   retrain without a matched-seed-count check.
2. **Freezing a real, already-trained backbone isolates a true causal effect** far better than
   comparing two independently-retrained models — the technique that let v34 give a clean,
   definitive answer after three inconclusive from-scratch attempts (v31/v32/v33).
3. **The in-distribution aggregate test set does not always agree with real-storm validation.**
   v33 and v34 are both null-to-slightly-negative in aggregate but each beat v23 on one of the two
   individually-checked real out-of-training storms.

Full narrative, architecture equations, and every intermediate result: **`paper/trackformer.pdf`**,
"Chain-of-thought steering and land-interaction testing." Map of every model tested
against all six storms: [`paper/all_storms_v23_v33_v34_mean_map.html`](../paper/all_storms_v23_v33_v34_mean_map.html).
Training scripts: `colab_train_v17.ipynb` (base), `colab_v26_train.py` (v21 CoT),
`colab_v28_train.py` (v23 temporal history), `colab_v36`–`v39_train.py` (v31–v34).

### Usage — v23 (this directory)

Files: `v23_seed0.pt`–`v23_seed9.pt` (checkpoints, average their output for the ensemble),
`trackformer_v23.py` (the architecture — copied verbatim from the training scripts, no
notebook/exec tricks), `v23_norm_stats.npz` + `v23_terrain_wp.npz` (small companion data, a few
hundred KB total), `run_v23.py` (CLI).

v23 runs in two modes, controlled by whether you pass `--steering`:

**IBTrACS-only** (default) — give it nothing but the storm's own recent track: position, max wind,
central pressure. This is what any best-track record (IBTrACS itself, or any agency's advisory)
gives you for a storm, nothing more. The steering field and its 12h/24h history are zero-filled
with an explicit availability flag — the same "unavailable == exact zeros, not fabricated"
convention used throughout this project, not a degraded/broken input.

```bash
python run_v23.py --track my_storm.json --out forecast.json
```

**Full data** — additionally give it a real deep-layer-mean steering-wind patch (850/500/200 hPa
u/v, 2.5° resolution, ±20° box centered on the storm) for the current fix and, ideally, the two
fixes 12h/24h before it. This is what the project's headline **434.96 km** result requires.

```bash
python run_v23.py --track my_storm.json --steering my_steering.npz --out forecast.json
```

`my_storm.json` is a list of fixes, oldest→newest, spaced 6h apart, ending at the fix to forecast
from — see `run_v23.py`'s module docstring for the exact schema and an example. `my_steering.npz`
is keyed by the same ISO timestamps; `_fetch_dolphin_steering.py` (repo root) is a complete, working
example of building one from NOAA/NOMADS GFS analysis fields for a live storm (ERA5 works the same
way for a past one).

**How much does the steering field actually matter?** Tested directly on Typhoon Dolphin (2026,
active as of this writing): with real fetched GFS steering, v23's 120h forecast was 14.7°N,150.7°E /
99 kt / 951 hPa; with the steering field zeroed out (IBTrACS-only) it was 17.5°N,156.3°E / 90 kt /
957 hPa — the track moved by several hundred km while the intensity forecast only softened modestly.
So the steering field mainly earns its keep on **track**, not intensity, at least on this storm —
see `paper/dolphin_pressure_vmax_chart.html` and `paper/overlay_real_vs_models.html` for the full
comparison (both also show v10, JTWC, and JMA for context).

## Released checkpoints: StormFusion-MT and TrackFormer v1–v9

The earlier, fully released and locally-runnable line. Weights are stored with Git LFS. Both
predict, at 20 six-hourly lead times (6–120 h), a 17-dim state per lead: east/north storm motion
(km), max wind (kt), central pressure (hPa), radius of max wind (km), and 34/50/64-kt wind radii in
four quadrants.

| file | params | size | inputs | training data |
|---|---|---|---|---|
| `trackformer_v9_17M_fp16.pt` | 17M | 33 MB (fp16) | **track history + IBTrACS environment (protected triple-stream)** | all basins, 1980+, 193k partial-lead windows |
| `trackformer_v8_15M_fp16.pt` | 15M | 30 MB (fp16) | track history only (protected dual-stream) | all basins, 1980+, 193k partial-lead windows |
| `stormfusion_v2_era5_3.3M_fp16.pt` | 3.3M | 6.7 MB (fp16) | ERA5 patches + track history | WP, 2000+, 1,337 storm-centered windows |
| `trackformer_21M_fp16.pt` | 21M | 43 MB (fp16) | track history only (single-stream) | all basins, 1980+, 84,150 windows |

**`trackformer_v9` is the best model in this released line** (though `v23` above beats it on a
different, larger test set). It adds a third, **environmental** stream to v8's protected
architecture — absolute latitude/longitude, distance-to-land, and a lat+month climatological SST
proxy, all **derived from IBTrACS** (no ERA5, still deployable on a fast CSV). v8 was position-blind
(translation-invariant); giving it absolute latitude — which drives Coriolis, recurvature, and SST —
cut WP-2020+ track to **618 km (−31 vs v8)** and all-basin to **543 km (−37)**, with markedly better
wind (vmax −2 kt). It also **halved the error on the erratic Co-may (2025)** case and improved Bavi
(2026). See `paper/trackformer.pdf`, Appendix A, for the architecture and derivation.

Both checkpoints store weights in fp16 (half the size, identical metrics) and are
inference-only (optimizer state stripped). The track-only model predicts the **full 17-dim
state** (motion, wind, pressure, RMW, all 12 wind radii) — not just track.

### Architectures

**StormFusion-MT v2** (`model_v2.py`): separate inner/outer ERA5 conv encoders that keep a 3×3
grid of spatial tokens (not global-pooled), track and environment token encoders, a temporal
Transformer context, learned + sinusoidal lead-time queries, cross-attention decoding, and
multi-task state / log-scale heads.

**TrackFormer** (`train_track.py`, `TrackModel`): the same decoder design but track-only —
a 40-dim track-history projection → Transformer context (d_model 384, 8 heads, 4+6 layers) →
lead queries → dual heads. No atmospheric inputs at all.

### Results — WP 2020+ held-out test (lower is better)

| model | track km | vmax kt | pres hPa | rmw km | radius km |
|---|---|---|---|---|---|
| single-stream 40-feat (21M) | 720 | 22.1 | 21.2 | 11.8 | 31.5 |
| single-stream 48-feat (21M) | 737 | 21.5 | 17.7 | 11.8 | 30.9 |
| TrackFormer v3 (dual-stream, full-lead data) | 659 | 21.6 | 18.1 | 11.8 | 28.8 |
| TrackFormer v8 (dual-stream, +partial-lead data) | 649 | 20.7 | 15.9 | **11.3** | 27.8 |
| **TrackFormer v9 (triple-stream, +IBTrACS environment)** | **618** | **18.6** | 15.8 | 11.5 | **27.2** |

TrackFormer v9 is best on track, wind, and radius. The largest single jump came from making the model
**position-aware**: v8 saw only relative motion, so it could not use latitude (Coriolis, recurvature,
SST). Adding absolute lat/lon + distance-to-land + a lat/month SST proxy (all IBTrACS-derived) cut
track error 649→618 km (WP) / 580→543 km (all-basin) and improved wind by ~2 kt.

**Key findings:** (1) a track-only model that never sees ERA5 **matches or beats** a full ERA5 model,
so **data diversity > engineered features > parameters** (a 17.7M ERA5 model overfit and did *worse*
than the 3.3M one). (2) Adding motion-dynamics features to a single-stream model improves intensity
but hurts track via **negative transfer**; the fix is architectural — a **protected dual-stream** that
routes kinematic and thermodynamic gradients separately. (3) The largest remaining gains came from
**data, not architecture**: keeping the partial-lead windows (storm-end / short storms, masked instead
of discarded) doubled the clean training set and improved *every* metric. Full analysis in
`paper/trackformer.pdf`, Appendix A.

### Loading

```python
import torch, numpy as np
# --- TrackFormer v8 (best of this released line; protected dual-stream, track-only) ---
# The TrackFormerV3 class lives in train_track_v3.py; full inference example in eval_v8.py.
# Inputs: standardized 9x48 history + v0 (current 6-h motion in km, from the last history step
# de-standardized: track[:, -1, 2:4] * track_std[2:4] + track_mean[2:4]).
ckpt = torch.load("models/trackformer_v8_15M_fp16.pt", map_location="cpu", weights_only=False)
# model = TrackFormerV3(); model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
# state, logscale = model(track_tensor, v0_tensor)   # state[..., :2] = per-step motion (÷ scale)
# ckpt["track_mean"], ckpt["track_std"] are the per-feature standardization stats.

# --- StormFusion-MT v2 (ERA5) ---
import importlib.util
spec = importlib.util.spec_from_file_location("m", "model_v2.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ckpt = torch.load("models/stormfusion_v2_era5_3.3M_fp16.pt", map_location="cpu", weights_only=False)
model = m.StormFusionMT("recommended", lead_count=20)
model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()}); model.eval()  # fp16 -> fp32
```

Inputs must be normalized with the stored stats (ERA5: the `*_mean`/`*_std` keys saved in the
window npz; TrackFormer: `track_mean`/`track_std` in the checkpoint). Multiply predictions by
`TARGET_SCALE = [100,100,35,20,50] + [50]*12` to get physical units.

## Limitations

- Research baselines, not operational forecasts, for every version in either line.
- The real ceiling is storm **diversity** (~13k storms exist); bigger models overfit.
- Sparse/missing wind-radius labels; no calibration or comparison to official agencies, in either line.
- Pre-satellite (older) track/intensity labels are lower quality.
- v24–v34 (the further additions and land-interaction line built on top of v23) remain research
  artifacts only, no packaged local-inference format — only v23 itself is released, see "Usage —
  v23" above.
