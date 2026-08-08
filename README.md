# Trackformer1.1

Trackformer1.1 is a research model for causal tropical-cyclone track,
pressure, intensity, and wind-structure inference over the western Pacific.
It is not an operational warning system and must not be used for evacuation,
aviation, maritime, emergency-management, or other safety-critical decisions.

## Release names

- **Current public release:** `Trackformer1.1`
- **Historical public releases:** `Trackformer1.0.x`, where `x` identifies the
  historical patch or experiment in that family.

The current public package and model card use `Trackformer1.1` only. Historical
experiments are described as `Trackformer1.0.x` so old internal experiment
labels are not presented as current models.

## Demo and forecast grids

Open the interactive demonstration on the project GitHub Pages site:

**[Trackformer1.1 live demo](https://yu314-coder.github.io/typhoon-tracks.html)**

The repository also contains the forecast maps, pressure maps, and intensity
plots used for the two public case snapshots:

- [Dolphin world map](https://github.com/yu314-coder/typhoon-predict/blob/main/paper/dolphin_trackformer_1_1_pacific_domain_world_map.html)
- [Dolphin pressure map](https://github.com/yu314-coder/typhoon-predict/raw/refs/heads/main/paper/dolphin_trackformer_1_1_pacific_pressure_forecast.png)
- [Dolphin intensity and structure plot](https://github.com/yu314-coder/typhoon-predict/raw/refs/heads/main/paper/dolphin_trackformer_1_1_intensity_structure.png)
- [Tip world map](https://github.com/yu314-coder/typhoon-predict/blob/main/paper/tip_trackformer_1_1_pacific_domain_world_map.html)
- [Tip pressure map](https://github.com/yu314-coder/typhoon-predict/raw/refs/heads/main/paper/tip_trackformer_1_1_pacific_pressure_forecast.png)
- [Tip intensity and structure plot](https://github.com/yu314-coder/typhoon-predict/raw/refs/heads/main/paper/tip_trackformer_1_1_intensity_structure.png)

These tables are compact forecast grids from the checked-in case artifacts.
They are examples for inspecting the output, not a skill score and not a
replacement for an official forecast. Wind is maximum 1-minute wind in knots;
pressure is central pressure in hPa; radius is the radius of maximum wind in
kilometres.

### Dolphin, issue time 2026-07-31 15:00 UTC

| Lead | Latitude | Longitude | Max wind (kt) | Pressure (hPa) | RMW (km) |
|---:|---:|---:|---:|---:|---:|
| +6 h | 19.856 | 158.902 | 124.8 | 924.5 | 25.1 |
| +24 h | 20.723 | 156.159 | 118.3 | 931.5 | 22.0 |
| +48 h | 21.335 | 152.621 | 106.0 | 949.4 | 21.6 |
| +72 h | 22.093 | 149.456 | 95.6 | 963.9 | 22.8 |
| +96 h | 23.162 | 146.200 | 88.1 | 973.5 | 23.4 |
| +120 h | 24.242 | 142.811 | 80.1 | 980.5 | 25.1 |

### Tip, issue time 1979-10-12 00:00 UTC

| Lead | Latitude | Longitude | Max wind (kt) | Pressure (hPa) | RMW (km) |
|---:|---:|---:|---:|---:|---:|
| +6 h | 16.860 | 137.717 | 161.1 | 892.6 | 19.0 |
| +24 h | 18.005 | 135.849 | 152.1 | 898.8 | 17.8 |
| +48 h | 18.794 | 133.396 | 135.7 | 913.8 | 16.9 |
| +72 h | 19.569 | 131.390 | 118.6 | 931.5 | 18.1 |
| +96 h | 20.614 | 129.499 | 108.2 | 942.7 | 18.7 |
| +120 h | 21.668 | 127.791 | 97.8 | 952.6 | 21.6 |

## Data used and how to get it

Trackformer1.1 uses observed or analyzed weather available at the forecast
issue time and earlier. The model does not use another agency's predicted
track or positive-lead weather forecast as an input.

| Data | Role in Trackformer1.1 | How to obtain it |
|---|---|---|
| IBTrACS | Observed storm track history, intensity history, training labels, and post-run verification | [NOAA/NCEI IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) |
| Historical atmospheric analysis | Current, -12 h, and -24 h pressure, height, and wind states for causal hindcasts | [NOAA Climate Forecast System data and documentation](https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system) |
| Current atmospheric analysis | The latest available pressure and multi-level wind state for a live-style run | [NOAA NOMADS](https://nomads.ncep.noaa.gov/) using an analysis/`f000` product whose valid time is no later than the issue time |
| Derived arrays and case artifacts | Compact tensors, manifests, maps, and plots consumed by the public inference bundle | Build from the repository scripts locally or in GitHub Actions; raw archives are intentionally not stored in this model repository |

For historical coverage, CFSR and CFSv2 are different sources: CFSR does not
cover the full modern period, so later cases must use the matching CFSv2/CDAS
analysis product when that is the source selected for the run. The source and
valid time must be recorded in the preprocessing manifest.

The public GitHub Pages site serves generated JSON, PNG, HTML, and model-demo
assets. It should not hold private credentials or try to download large raw
archives in a browser. A GitHub Actions job can fetch public analysis files,
reject files newer than the issue time, build compact derived arrays, and
publish the resulting artifacts for the [live demo](https://yu314-coder.github.io/typhoon-tracks.html).

### Causal data boundary

At inference time, the model uses:

- the observed storm track and intensity history up to the issue time;
- current, 12-hour-old, and 24-hour-old analysis fields;
- the full western-Pacific domain covering China, Japan, Taiwan, the
  Philippines, and the surrounding Pacific;
- derived pressure minima, pressure deficit, wind flow, steering layers, and
  nearby-system context computed from those analysis fields.

It does **not** use JMA or JTWC forecast tracks, any official forecast track,
positive-lead GFS/GEFS/CFS forecast fields, or observations that became
available after the issue time. Later data may be used only to score the
forecast.

## Model structure

1. A causal regional state module extrapolates pressure, 500 hPa height, and
   850/500/200 hPa wind from three analysis snapshots.
2. A weighted steering ensemble samples inner, deep-layer, ridge, trough, and
   jet views across the full western-Pacific domain and integrates the route.
3. Three primary neural experts predict maximum wind, central pressure, RMW,
   and directional 34/50/64 kt radii from a nine-step track window and the
   current four-channel analysis patch.
4. Three structure experts provide a validation-selected secondary blend.
5. Three temporal experts read only same-storm 12-hour and 24-hour analysis
   patches. Their residual branch is enabled only for the validated wind
   adjustment; pressure remains on the calibrated primary path.
6. The final intensity output is coupled to the causal pressure-map minimum,
   pressure deficit, 850 hPa wind, and quadrant anomaly extent. Radius outputs
   remain ordered as `R34 >= R50 >= R64`.

The public bundle contains frozen inference checkpoints and calibration state;
it does not include raw weather archives or a training service.

## Intensity upgrade plan using the same data

The current intensity head is the weakest part of the public model. The next
improvement should keep the existing causal data boundary and change only the
target formulation, feature extraction, and calibration:

1. Build a lead-wise feature vector from data already available to Trackformer1.1:
   current wind and pressure, 6/12/24-hour observed tendencies, pressure-map
   minimum and deficit, 850/500/200 hPa flow, vertical-shear proxies, latitude,
   and the pressure-anomaly area around the storm.
2. Predict changes from the current observed intensity (`delta wind`, `delta
   pressure`, and `delta radius`) rather than predicting every future absolute
   value from scratch. Use separate short-, medium-, and long-lead heads so a
   120-hour estimate cannot dominate the 6-hour behavior.
3. Fit the calibration and regime gate only on the existing storm-held-out
   training/validation arrays. Keep complete storms in one split, and keep the
   test storms untouched until the final comparison against persistence and
   the current Trackformer1.1 head.
4. Add physical consistency constraints: pressure deepening must agree with
   wind strengthening, weakening must not create an implausible pressure fall,
   radii must be non-negative and ordered, and changes must be bounded by the
   observed analysis tendency and map confidence.
5. Report wind MAE, pressure MAE, RMW/radius MAE, bias by lead, and uncertainty
   coverage for every split. Publish new checkpoints only if the same-data
   model improves both all-lead and 120-hour test metrics without degrading
   track error.

The current calibration reference is 16.40 kt test wind MAE across leads and
14.52 hPa test pressure MAE at 120 hours. Those numbers are a baseline for the
next experiment, not a guarantee of operational accuracy. No new intensity
weights are claimed by this README update.

## Files

- `trackformer_1_1.py` - public loader and route API.
- `trackformer_1_1_route.py` - whole-domain causal pressure-state route.
- `trackformer_1_1_intensity.py` - intensity and wind-structure inference.
- `trackformer_1_1_temporal.py` - inference-only temporal expert architecture.
- `models/trackformer_1_1/` - three expert groups, calibration, and manifest.
- `track_build/` - checked-in Dolphin and Tip forecast-grid source artifacts.
- `paper/` - map, pressure-map, and intensity/structure visualizations.

## Quick start

```bash
python -m pip install -r requirements.txt
python - <<'PY'
import numpy as np
from trackformer_1_1 import load_intensity

model = load_intensity(device="cpu")
track = np.zeros((9, 54), dtype="float32")
field = np.zeros((4, 17, 17), dtype="float32")
current_structure = np.full(13, np.nan, dtype="float32")
rows, metadata = model.predict(
    track,
    field,
    current_wind=65.0,
    current_pressure=980.0,
    previous_wind=60.0,
    previous_pressure=985.0,
    current_structure=current_structure,
)
print(rows[0])
print(metadata["model"])
PY
```

The intensity contract is `track=(9,54)`, `field=(4,17,17)`, and optional
`history_field=(8,17,17)` for the 12-hour and 24-hour analysis patches. The
route contract is `fields=(3,7,latitude,longitude)` and
`pressure=(3,latitude,longitude)`, ordered current, 12 hours before, and 24
hours before. See `models/trackformer_1_1/manifest.json` for the complete
contract.

## Requirements and limits

CPU inference works with Python 3.10+, NumPy, PyTorch, and SciPy. A modern
Mac with 8 GB RAM is a practical minimum; 16 GB RAM is preferable when
decoding large analysis grids. The checkpoints are native PyTorch files.
GGUF is not an appropriate format for this CNN/Transformer weather model;
keep the native PyTorch weights or convert to another format only after
numerical parity testing.

The model is experimental. Track and intensity errors vary by storm, basin,
analysis source, and data quality. Use official meteorological agencies for
real-world forecasts and warnings.
