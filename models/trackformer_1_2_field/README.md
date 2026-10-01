# Trackformer 1.2 field-model inference

**Documentation revision: 1 October 2026.** This describes the current GitHub/Hugging Face wrapper. The 29 September release tarball remains an earlier snapshot; its optional wind/radius diagnostics may differ. Released neural weights, architecture, causal input schema and route/pressure predictions are unchanged.

This folder contains the exact source, input contract and inference-only weights for the research model in the [main README](../../README.md). `model.py`, `baseline_model.py` and `v165_base.py` are source-identical to the selected training implementation. The weight file is distributed via the [GitHub release](https://github.com/yu314-coder/typhoon-predict/releases/tag/trackformer-1.2) and [Hugging Face](https://huggingface.co/euler314/typhoon-predict/tree/main/models/trackformer_1_2_field), not Git.

Use Python with PyTorch and NumPy. The export environment used PyTorch 2.13.0; the source was evaluated on macOS MPS. Example:

```bash
python models/trackformer_1_2_field/predict.py causal_issue_packet.npz forecast.npz --device cpu
```

The `.npz` packet must contain exactly these arrays, with no object/pickle content:

| Key | Shape | Meaning |
| --- | --- | --- |
| `global_history` | `(9, 8, 25, 33)` | Past/issue-time basin analyses in manifest channel order, normalized with manifest means/stds. |
| `regional_history` | `(9, 1, 121, 121)` | Native MSLP patches when present, otherwise benchmark's coarse fallback; normalized using MSLP statistics. |
| `global_static` | `(4, 25, 33)` | Latitude/90, longitude/180−1, land fraction, elevation metres/8000. |
| `regional_static` | `(4, 121, 121)` | Same four static channels on the regional grid. |
| `detail_available` | `(1,)` | One when native history exists, otherwise zero. |
| `center` | `(2,)` | Issue-time latitude and longitude in degrees. |
| `motion` | `(2,)` | Previous six-hour east/north displacement in km. |
| `issue_intensity` | `(2,)` | Issue-time maximum wind in knots and central pressure in hPa. |
| `issue_mask` | `(2,)` | Validity flags for issue-time wind and pressure. |
| `issue_time_ns` | scalar | Forecast issue timestamp in Unix nanoseconds. |
| `history_time_ns` | `(9,)` | Nine consecutive six-hour timestamps ending at issue time. |

Output `.npz` contains `lead_hours`, `track_lat_lon`, `central_pressure_hpa`, `basin_mslp_hpa` and `regional_mslp_hpa`. Pressure arrays are physical hPa. The regional grid is issue-relative and must be located using the static coordinate channels, not interpreted as a fixed global map.

It also exposes the existing `maximum_wind_auxiliary_kt` scalar and experimental
pressure-derived `pressure_wind_estimate_kt`, `rmw_estimate_km` and
`r34_estimate_km` / `r50_estimate_km` / `r64_estimate_km`. Every pressure-derived
array has a matching `_valid` mask: zero padding where the mask is false is
**not a zero wind or radius forecast**. `wind_estimation_json` is a JSON string
with assumptions, member counts and unavailable reasons, not pickled objects.
The auxiliary wind also has `maximum_wind_auxiliary_kt_valid`, a numerical/domain
validity mask, not a claim of calibrated wind skill.
These are uncalibrated circular-equivalent ocean estimates, not resolved
surface winds, agency-compatible sustained winds or quadrant radii. See
[the estimation method and validation requirements](WIND_ESTIMATION.md).
No model weights, learned architecture or existing benchmark results changed.

**Causality:** Never populate arrays from positive-lead analysis, later best track, agency forecast or future pressure field. The wrapper checks timestamp order and shapes, but cannot authenticate the upstream archive. Do not silently replace missing values with future data. This is not a raw-weather downloader or live warning service.

The published 50-member mean is a separate evaluation policy: 50 deterministic seeds perturb only normalized historical basin/regional fields with smooth zero-centred noise, then average routes, central pressures and common-grid fields. `predict.py` computes one clean forecast; do not label it a 50-member mean.

**Pressure-display coverage:** `regional_mslp_hpa` is the original fixed issue-relative composite, not an automatically extended moving-core map. When the forecast centre leaves that patch, a scalar central-pressure readout must not be inserted into the image to make it appear consistent. The corrected Mangkhut film separately exports the actual moving model cores and registers member fields before averaging; [the current model card](https://huggingface.co/euler314/typhoon-predict) links the verified pressure arrays and shows which other films retain their older fixed patches. The automatic History archive and its separate core recovery expose exact geography and coverage through [the public API](https://trackformer-weatherlab.rudin-euler-8253.chatgpt.site/history-api).
