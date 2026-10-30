# TrackFormer 1.2 · model input and output guide

[Project overview](../../README.md) · [Complete training manual (PDF)](../../docs/TrackFormer_1.2_Training_and_Reproducibility.pdf) · [Protocols and reference](../../docs/REFERENCE.md)

This guide covers **one clean forecast from the released pressure-field model**. It explains how to supply a prepared input packet, export the predicted pressure/track arrays and render them. The model does not download or construct raw-weather inputs for you. The training process is documented in the single PDF above.

## 1. Set up and identify the weights

From the repository root:

```bash
python -m pip install -r requirements.txt
curl --fail --location \
  "https://huggingface.co/euler314/typhoon-predict/resolve/main/models/trackformer_1_2_field/weights.pt" \
  --output models/trackformer_1_2_field/weights.pt
```

Use a compatible PyTorch environment with CPU, Apple MPS or NVIDIA CUDA. Matplotlib is needed only for PNG rendering. The original training environment is not fully recorded; minimum dependency versions do not establish a bitwise reproduction environment.

| Identity | Value |
| --- | --- |
| Public model | TrackFormer 1.2 |
| Internal source checkpoint | 1.2.73, completed epoch 4 |
| Architecture | `v173-moving-core-multiscale-attention` |
| Inference-weight SHA-256 | `db49f36e85a3766defc4c172746897a1f783705d1ce8e6f9dfb8e87ae1d902cb` |
| Forecast horizon | Twenty six-hour steps, +6 through +120 h |

`predict.py` verifies the weights against [manifest.json](manifest.json) and loads the state strictly. The three neural source modules are identical to the released training implementation. A changed checkpoint is a new candidate, not the released model.

## 2. Prepare one causal issue packet

Save an `.npz` containing **exactly** the following eleven keys, with no pickle/object arrays. Numeric model inputs must be finite and have the unbatched shapes shown; the wrapper adds a batch dimension. Issue/history times are Unix nanoseconds represented as integers.

| Key | Shape | Value / units |
| --- | --- | --- |
| `global_history` | `(9,8,25,33)` | Normalized eight-channel basin analyses at −48, −42, …, 0 h. |
| `regional_history` | `(9,1,121,121)` | Normalized native MSLP history when available; the training loader uses zeros plus a false availability flag otherwise. |
| `global_static` | `(4,25,33)` | Latitude/90, longitude/180−1, land fraction, elevation metres/8000. |
| `regional_static` | `(4,121,121)` | The same geographic features on the fixed issue-centred regional grid. |
| `detail_available` | `(1,)` | 1 for native history, 0 if absent. The model samples the basin history when this flag is false. |
| `center` | `(2,)` | Issue-time latitude, longitude in degrees N/E. |
| `motion` | `(2,)` | Previous six-hour displacement: east, north, in km. |
| `issue_intensity` | `(2,)` | Issue-time maximum wind in knots; central pressure in hPa. |
| `issue_mask` | `(2,)` | Independent validity flags for the two intensity values. Use finite storage values with explicit masks for missing intensity. |
| `issue_time_ns` | scalar | Forecast initialization time, UTC Unix nanoseconds. |
| `history_time_ns` | `(9,)` | Exact six-hour sequence ending at `issue_time_ns`. |

**Channel order:** MSLP, HGT500, U850, V850, U500, V500, U200, V200. The u/v pairs carry wind direction and speed. Use `(physical_value − mean) / std` with the channel statistics in `manifest.json`; do not normalize a second time. Fit years for these statistics are 2000–2021.

**Geography:** the basin is 25 north-to-south latitudes from 60°N to 0°N and 33 longitudes from 100°E to 180°E at 2.5° spacing. The fixed 121×121 regional grid uses the manifest's 0.25° offsets around the issue-centred patch. Static coordinates must match the actual arrays, not just their shape. The moving 65×65 core is reconstructed by the model at approximately 20 km spacing.

The wrapper checks exact keys, finite shapes and consecutive causal timestamps. It cannot authenticate the upstream archive or detect a mislabeled future field. Future analyses/observations and agency forecast fields/routes must never be substituted into inputs. Missing native history is not a claim of newly resolved pressure detail.

## 3. Export a numerical forecast

```bash
python models/trackformer_1_2_field/predict.py \
  causal_issue_packet.npz forecast.npz --device cpu
```

Use `--device mps` or `--device cuda` for an available accelerator. This command runs **one clean member**, with frozen weights and no optimizer updates. It does not reproduce the separately published 50-member input-perturbation mean.

| Output | Shape | Interpretation |
| --- | --- | --- |
| `lead_hours` | `(20,)` | 6, 12, …, 120 h. |
| `track_lat_lon` | `(20,2)` | Pressure-associated forecast centre, latitude/longitude. |
| `central_pressure_hpa` | `(20,)` | Bilinear pressure from the same moving field at that centre. |
| `track_valid` | `(20,)` | Track support mask. |
| `basin_mslp_hpa` | `(20,25,33)` | Predicted basin pressure in physical hPa. |
| `basin_latitude_deg`, `basin_longitude_deg` | `(25,33)` each | Basin coordinates. |
| `regional_mslp_hpa` | `(20,121,121)` | Predicted fixed issue-relative composite; detail coverage changes as the core moves. |
| `regional_latitude_deg`, `regional_longitude_deg` | `(121,121)` each | Fixed regional coordinates. |
| `regional_valid` | `(20,121,121)` | Geometric support; does not guarantee native-core detail across the whole patch. |
| `core_mslp_hpa` | `(20,65,65)` | Actual predicted moving-core pressure. |
| `core_latitude_deg`, `core_longitude_deg` | `(20,65,65)` each | Per-lead coordinates of the moving core. |
| `core_valid` | `(20,65,65)` | Moving-core support mask. |
| `issue_time_ns`, `valid_time_ns` | scalar, `(20,)` | Issue and exact forecast-valid timestamps. |
| `issue_center_lat_lon`, `member_count` | `(2,)`, scalar | Initialization centre; member count is 1. |
| `core_information_spacing_km` | scalar | Approximately 20 km computational reconstruction. |
| `pressure_export_json` | string scalar | Schema, units, weight/checkpoint hashes, coverage and ensemble policies. |

Pressure and coordinates must travel with their masks. Finite values outside a false support mask are not valid forecasts; zero padding is not zero physical pressure. The associated central pressure need not equal the discrete grid minimum or a drawn contour level. No forecast field is moved onto the observed route or replaced by a display vortex.

### Read the saved arrays

```python
import numpy as np

with np.load("forecast.npz", allow_pickle=False) as forecast:
    lead_index = int(np.flatnonzero(forecast["lead_hours"] == 24)[0])
    pressure = np.where(
        forecast["core_valid"][lead_index],
        forecast["core_mslp_hpa"][lead_index],
        np.nan,
    )
    latitude = forecast["core_latitude_deg"][lead_index]
    longitude = forecast["core_longitude_deg"][lead_index]
```

## Render a pressure map directly

With Matplotlib installed in your own environment:

```bash
python models/trackformer_1_2_field/predict.py \
  causal_issue_packet.npz forecast.npz --device cpu \
  --pressure-map pressure_120h.png --map-lead 120 --isobar-interval 4

# Render another saved lead without rerunning the network:
python models/trackformer_1_2_field/plot_pressure.py \
  forecast.npz pressure_24h.png --lead 24 --interval 2
```

The renderer shows the basin and actual moving core using one scale: blue low pressure, red high pressure. Apply per-lead geographic coordinates and masks. When the cyclone leaves the fixed regional patch, use the moving-core fields rather than extending the old patch by hand. For an ensemble, register physical member fields onto a common geographic grid before averaging; never average unaligned moving-core indices.

## Wind and radius diagnostics

`maximum_wind_auxiliary_kt` is an existing supervised scalar, accompanied by a numerical/domain `_valid` mask. That mask is not a demonstration of calibrated surface-wind skill.

Experimental `pressure_wind_estimate_kt`, `rmw_estimate_km`, and `r34_estimate_km` / `r50_estimate_km` / `r64_estimate_km` are pressure-derived, circular-equivalent diagnostics, not a trained radius head or observed quadrant radii. Each has a matching `_valid` mask. If it is false, stored zero is padding. `wind_estimation_json` records assumptions and unavailable reasons. [Diagnostic method and validation limits](../../docs/REFERENCE.md#models-trackformer-1-2-field-wind-estimation-md).

## Common problems

| Symptom | Check |
| --- | --- |
| Weight identity error | Use the exact 1.2 inference weight file, not a full optimizer checkpoint or later rainfall model. |
| Packet-key or shape error | Supply the eleven keys above with unbatched shapes and integer timestamps. |
| History-time error | All nine times must be six hours apart and end exactly at issue time. |
| Native detail unavailable | Keep the explicit false flag; do not insert future pressure or claim interpolated detail as observations. |
| Finite field looks wrong outside the basin | Apply `core_valid`, `regional_valid` and `track_valid`; inspect geography and normalization. |
| Different from the website mean | Match checkpoint, issue time, data source and one-member versus 50-member policy first. |
| Need rainfall | Released 1.2 has no rainfall head or humidity input. Later rainfall experiments are separate. |

For measured development results use the [main scorecard](../../README.md#results-in-the-research-report), [common-support benchmark](../../evaluation/deepmind_daily/benchmark.json) and [reference](../../docs/REFERENCE.md). For fitting cutoffs, batch-size evidence, losses and reproduction use the single training PDF. Research results do not establish operational warning reliability or newly resolved 20 km observations.
