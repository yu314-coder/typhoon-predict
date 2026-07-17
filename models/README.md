# Trained models — StormFusion-MT and TrackFormer

Two trained PyTorch checkpoints for Western-Pacific-and-beyond tropical-cyclone forecasting.
Both predict, at 20 six-hourly lead times (6–120 h), a 17-dim state per lead: east/north
storm motion (km), max wind (kt), central pressure (hPa), radius of max wind (km), and
34/50/64-kt wind radii in four quadrants. Research models — **not** an operational warning system.

Weights are stored with Git LFS.

| file | params | size | inputs | training data |
|---|---|---|---|---|
| `trackformer_v9_17M_fp16.pt` | 17M | 33 MB (fp16) | **track history + IBTrACS environment (protected triple-stream)** | all basins, 1980+, 193k partial-lead windows |
| `trackformer_v8_15M_fp16.pt` | 15M | 30 MB (fp16) | track history only (protected dual-stream) | all basins, 1980+, 193k partial-lead windows |
| `stormfusion_v2_era5_3.3M_fp16.pt` | 3.3M | 6.7 MB (fp16) | ERA5 patches + track history | WP, 2000+, 1,337 storm-centered windows |
| `trackformer_21M_fp16.pt` | 21M | 43 MB (fp16) | track history only (single-stream) | all basins, 1980+, 84,150 windows |

**`trackformer_v9` is the best model.** It adds a third, **environmental** stream to v8's protected
architecture — absolute latitude/longitude, distance-to-land, and a lat+month climatological SST
proxy, all **derived from IBTrACS** (no ERA5, still deployable on a fast CSV). v8 was position-blind
(translation-invariant); giving it absolute latitude — which drives Coriolis, recurvature, and SST —
cut WP-2020+ track to **618 km (−31 vs v8)** and all-basin to **543 km (−37)**, with markedly better
wind (vmax −2 kt). It also **halved the error on the erratic Co-may (2025)** case and improved Bavi
(2026). See [`paper/trackformer.pdf`](../paper/trackformer.pdf) for the architecture and derivation.

Both checkpoints store weights in fp16 (half the size, identical metrics) and are
inference-only (optimizer state stripped). The track-only model predicts the **full 17-dim
state** (motion, wind, pressure, RMW, all 12 wind radii) — not just track.

## Architectures

**StormFusion-MT v2** (`model_v2.py`): separate inner/outer ERA5 conv encoders that keep a 3×3
grid of spatial tokens (not global-pooled), track and environment token encoders, a temporal
Transformer context, learned + sinusoidal lead-time queries, cross-attention decoding, and
multi-task state / log-scale heads.

**TrackFormer** (`train_track.py`, `TrackModel`): the same decoder design but track-only —
a 40-dim track-history projection → Transformer context (d_model 384, 8 heads, 4+6 layers) →
lead queries → dual heads. No atmospheric inputs at all.

## Results — WP 2020+ held-out test (lower is better)

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
[`paper/trackformer.pdf`](../paper/trackformer.pdf).

## Loading

```python
import torch, numpy as np
# --- TrackFormer v8 (best; protected dual-stream, track-only) ---
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

- Research baselines; absolute track error (~720 km avg over 6–120 h) is far from operational.
- The real ceiling is storm **diversity** (~13k storms exist); bigger models overfit.
- Sparse/missing wind-radius labels; no calibration or comparison to official agencies.
- Pre-satellite (older) track/intensity labels are lower quality.
