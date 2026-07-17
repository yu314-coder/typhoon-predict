# Trained models — StormFusion-MT and TrackFormer

Two trained PyTorch checkpoints for Western-Pacific-and-beyond tropical-cyclone forecasting.
Both predict, at 20 six-hourly lead times (6–120 h), a 17-dim state per lead: east/north
storm motion (km), max wind (kt), central pressure (hPa), radius of max wind (km), and
34/50/64-kt wind radii in four quadrants. Research models — **not** an operational warning system.

Weights are stored with Git LFS.

| file | params | size | inputs | training data |
|---|---|---|---|---|
| `trackformer_v8_15M_fp16.pt` | 15M | 30 MB (fp16) | **track history only (protected dual-stream)** | all basins, 1980+, 193k partial-lead windows |
| `stormfusion_v2_era5_3.3M_fp16.pt` | 3.3M | 6.7 MB (fp16) | ERA5 patches + track history | WP, 2000+, 1,337 storm-centered windows |
| `trackformer_21M_fp16.pt` | 21M | 43 MB (fp16) | track history only (single-stream) | all basins, 1980+, 84,150 windows |

**`trackformer_v8` is the best model.** It uses a protected dual-stream architecture (separate
kinematic/thermodynamic encoders, gradient routing, a zero-init gated thermo→track adapter, and a
persistence-residual track head) that removes the negative transfer the single-stream models suffer,
trained on a 2×-larger clean dataset (partial-lead windows kept via target masking, not discarded).
On WP-2020+ it reaches 649 km track — **−71 vs the single-stream baseline** and −10 vs the same
architecture on full-lead-only data — while being **best-in-class on every intensity/structure metric
too**. See [`paper/trackformer.pdf`](../paper/trackformer.pdf) for the full architecture and derivation.

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
| **TrackFormer v8 (dual-stream, +partial-lead data)** | **649** | **20.7** | **15.9** | **11.3** | **27.8** |

TrackFormer v8 is **best on every column**. Its WP-2020+ track win over the single-stream baseline is
significant (storm-bootstrap 95% CI [−103, −16] km, p≈0.995); the −10 km over v3 is near-significant
(CI [−36, +2], p≈0.95) and the intensity/structure gains are consistent across both WP and all-basin.

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
