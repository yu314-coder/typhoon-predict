# Trackformer 1.2 field-model inference

The released pressure-field model predicts Western Pacific storm tracks, central pressure and spatial MSLP through +120 h. This is the inference reference; see the [model announcement](../../README.md) for capabilities, the pressure-map showcase and development results.

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

**Matched development evaluation:** the released 1.1 and 1.2 route comparison uses the same frozen **1,473 daily starts / 270 equal-weight storms**, +6 to +120 h. The completed native-pressure comparison has **134 common starts / 40 storms**; the other 1,339 starts lack valid frozen 1.1 intensity inputs and are not zero-scored. The Site defaults to central-pressure MAE: **13.53 / 12.84 hPa** against JMA and **12.84 / 12.55 hPa** against USA (1.1 / 1.2), slight mean improvements with paired whole-storm uncertainty including zero. The optional curve-similarity diagnostic is **(1 + centred cosine) / 2 at exact common valid times**, without time shifting or warping. JMA similarity is **0.7074 / 0.7118**; USA is **0.7237 / 0.6637** on 131 eligible non-flat curves. Removing level/amplitude makes this a shape diagnostic, not proof of correct pressure levels. Actual unshifted hPa timelines, agency masks and uncertainty remain separate. See the [shared snapshot](../../evaluation/released_daily/released_daily_benchmark.json), [verification receipt](../../evaluation/released_daily/released_daily_verification.json), [pressure protocol](../../docs/intensity_benchmark.md) and [public benchmark API](https://trackformer-weatherlab.rudin-euler-8253.chatgpt.site/api/benchmarks/released). A new matched WeatherNext Cyclones Mini run is in progress on these exact daily starts; results remain pending. See the [frozen protocol](../../docs/deepmind_daily_benchmark.md); no old scores are transferred.

**Pressure-display coverage:** `regional_mslp_hpa` is the original fixed issue-relative composite, not an automatically extended moving-core map. When the forecast centre leaves that patch, a scalar central-pressure readout must not be inserted into the image to make it appear consistent. The Mangkhut film separately exports the actual moving model cores and registers member fields before averaging; [the current model card](https://huggingface.co/euler314/typhoon-predict) links the verified pressure arrays and links older fixed-patch films in a separate archive. The automatic History archive and its separate core recovery expose exact geography and coverage through [the public API](https://trackformer-weatherlab.rudin-euler-8253.chatgpt.site/history-api).
