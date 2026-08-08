# Trackformer1.1

Trackformer1.1 is a research model for causal tropical-cyclone track,
pressure, intensity, and wind-structure inference over the western Pacific.
It is not an operational warning system and must not be used for evacuation,
aviation, maritime, emergency-management, or other safety-critical decisions.

## What it uses

Inference accepts the observed storm history and weather analyses available at
the issue time. It uses the current, 12-hour, and 24-hour analysis states to
extrapolate a bounded western-Pacific pressure and flow state. The route sees
a domain covering China, Japan, Taiwan, the Philippines, and the open western
Pacific, so nearby lows, subtropical ridges, troughs, jets, and other storms
can affect the steering field.

No positive-lead weather field, official agency forecast, or post-issue
observation is passed to the model. Later observations can be used only for
verification after a forecast has been generated.

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
   remain ordered as R34 >= R50 >= R64.

The public bundle contains frozen inference checkpoints and calibration state;
it does not include raw weather archives or a training service.

## Files

- `trackformer_1_1.py` - public loader and route API.
- `trackformer_1_1_route.py` - whole-domain causal pressure-state route.
- `trackformer_1_1_intensity.py` - intensity and wind-structure inference.
- `trackformer_1_1_temporal.py` - inference-only temporal expert architecture.
- `models/trackformer_1_1/` - three expert groups, calibration, and manifest.

## Quick start

```bash
python -m pip install -r requirements.txt
python - <<'PY'
import numpy as np
from trackformer_1_1 import load_intensity, forecast_pacific_state

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
