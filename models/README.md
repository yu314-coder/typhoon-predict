# Trained models — StormFusion-MT and TrackFormer

Two trained PyTorch checkpoints for Western-Pacific-and-beyond tropical-cyclone forecasting.
Both predict, at 20 six-hourly lead times (6–120 h), a 17-dim state per lead: east/north
storm motion (km), max wind (kt), central pressure (hPa), radius of max wind (km), and
34/50/64-kt wind radii in four quadrants. Research models — **not** an operational warning system.

Weights are stored with Git LFS.

| file | params | size | inputs | training data |
|---|---|---|---|---|
| `stormfusion_v2_era5_3.3M.pt` | 3.3M | 38 MB | ERA5 patches + track history | WP, 2000+, 1,337 storm-centered windows |
| `trackformer_21M_fp16.pt` | 21M | 43 MB (fp16) | **track history only (no ERA5)** | all basins, 1980+, 84,150 windows |

The track-only model predicts the **full 17-dim state** (motion, wind, pressure, RMW, all 12
wind radii) — not just track. Its weights are stored in fp16 (half the size, identical metrics).

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
| ERA5 v2 (3.3M) | 729 | 24.2 | 21.6 | 16.2 | 31.8 |
| **TrackFormer (21M, no ERA5)** | **720** | **22.1** | 21.2 | **12.9** | 31.5 |

**Key finding:** a track-only model that never sees ERA5 **matches or beats** the full
ERA5 model on every metric. The expensive ERA5 atmospheric patches add little over
past-track history + more storms. Across experiments, **data diversity > engineered features
> parameters** (a 17.7M ERA5 model overfit and did *worse* than the 3.3M one).

## Loading

```python
import torch, numpy as np
# --- TrackFormer (track-only) ---
from train_track import TrackModel          # class defined at module import
ckpt = torch.load("models/trackformer_21M_fp16.pt", map_location="cpu", weights_only=False)
model = TrackModel()
model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})  # fp16 -> fp32 for CPU
model.eval()
# ckpt["track_mean"], ckpt["track_std"] are the per-feature standardization stats.

# --- StormFusion-MT v2 (ERA5) ---
import importlib.util
spec = importlib.util.spec_from_file_location("m", "model_v2.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ckpt = torch.load("models/stormfusion_v2_era5_3.3M.pt", map_location="cpu", weights_only=False)
model = m.StormFusionMT("recommended", lead_count=20)
model.load_state_dict(ckpt["model"]); model.eval()
```

Inputs must be normalized with the stored stats (ERA5: the `*_mean`/`*_std` keys saved in the
window npz; TrackFormer: `track_mean`/`track_std` in the checkpoint). Multiply predictions by
`TARGET_SCALE = [100,100,35,20,50] + [50]*12` to get physical units.

## Limitations

- Research baselines; absolute track error (~720 km avg over 6–120 h) is far from operational.
- The real ceiling is storm **diversity** (~13k storms exist); bigger models overfit.
- Sparse/missing wind-radius labels; no calibration or comparison to official agencies.
- Pre-satellite (older) track/intensity labels are lower quality.
