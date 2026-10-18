# Trackformer 1.2 field-model inference

The released pressure-field model predicts Western Pacific storm tracks, central pressure and spatial MSLP through +120 h. This is the inference reference; see the [model announcement](../../README.md) for capabilities, the pressure-map showcase and development results.

This folder contains the exact source, input contract and inference-only weights for the research model in the [main README](../../README.md). `model.py`, `baseline_model.py` and `v165_base.py` are source-identical to the selected training implementation. The weight file is distributed via the [GitHub release](https://github.com/yu314-coder/typhoon-predict/releases/tag/trackformer-1.2) and [Hugging Face](https://huggingface.co/euler314/typhoon-predict/tree/main/models/trackformer_1_2_field), not Git.

**Training-data cutoff:** fitting and weather normalization use **2000–2021**; validation/checkpoint selection uses **2022–2023**; the original test partition uses **2024–2025**. Inputs come from IBTrACS-derived storm records, NOAA PSL NCEP/NCAR Reanalysis basin fields, ARCO-ERA5 regional pressure and static land/elevation. Newer 2026 archive/live inputs are inference, not retraining. [Sources and exact split coverage](../../docs/trackformer_1_2_training_data.md) · [Verified training-data receipt](../../evaluation/training_data/trackformer_1_2_provenance.json).

Use Python with PyTorch and NumPy. The export environment used PyTorch 2.13.0; the source was evaluated on macOS MPS. Example:

```bash
python models/trackformer_1_2_field/predict.py causal_issue_packet.npz forecast.npz --device cpu
```

The `.npz` packet must contain exactly these arrays, with no object/pickle content:

The nine history timestamps span **−48 to 0 h**, including both endpoints. This is the release's fixed two-day input format, not evidence that two days outperforms one or three. The eight basin channels are MSLP, HGT500, U850, V850, U500, V500, U200 and V200: each u/v pair represents **both horizontal wind direction and speed**. Static coordinates, land fraction and elevation remain part of the actual input contract below. See the [plain-language Chinese report](../../docs/research_report_zh.md) for the research motivation and household-computer equipment.

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

Output `.npz` contains twenty exact +6, +12, …, +120-hour forecasts. Pressure arrays are physical hPa. The moving core is exported by default, not just used internally for a central-pressure number:

| Output | Shape | Meaning |
| --- | --- | --- |
| `basin_mslp_hpa` | `(20,25,33)` | Unchanged coarse Western Pacific pressure fields. |
| `basin_latitude_deg`, `basin_longitude_deg` | `(25,33)` each | Basin geography from the actual input grid. |
| `regional_mslp_hpa` | `(20,121,121)` | Unchanged fixed issue-relative pressure composites. |
| `regional_latitude_deg`, `regional_longitude_deg` | `(121,121)` each | Fixed regional geography. |
| `regional_valid` | `(20,121,121)` | Original model domain-support masks; not a guarantee of moving-core detail throughout this fixed patch. |
| `core_mslp_hpa` | `(20,65,65)` | Actual learned moving-core fields, converted from model normalization to hPa. |
| `core_latitude_deg`, `core_longitude_deg` | `(20,65,65)` each | Original moving-frame geographic coordinates at every forecast lead. |
| `core_valid` | `(20,65,65)` | Original moving-core coverage masks. Mask false cells even when their stored pressure is finite. |
| `track_lat_lon`, `central_pressure_hpa`, `track_valid` | `(20,2)`, `(20,)`, `(20,)` | Unchanged associated track and bilinear core-pressure readout, with domain support. |
| `issue_time_ns`, `valid_time_ns` | scalar, `(20,)` | UTC Unix nanoseconds for the issue and each exact forecast lead. |
| `issue_center_lat_lon`, `member_count` | `(2,)`, scalar | Initialization location and the honest count of one clean member. |
| `pressure_export_json` | string scalar | Export schema, units, frozen checkpoint/weight hashes, reconstruction and coverage policy. |

No field is shifted onto the forecast or observed route, and no central-pressure scalar is inserted into a display. The moving-core information spacing is **20 km computational reconstruction**, not new native high-resolution observations. Coordinates and masks must travel with the fields; an invalid cell is not a zero-pressure forecast. `central_pressure_hpa` is sampled from this same moving field at the associated centre, so it need not equal a discrete cell minimum or a labelled contour level.

### Render a pressure map directly

Install Matplotlib in your inference environment only if you want PNG rendering. NumPy and PyTorch suffice for the numerical export.

```bash
python models/trackformer_1_2_field/predict.py causal_issue_packet.npz forecast.npz --device cpu --pressure-map pressure_120h.png --map-lead 120 --isobar-interval 4
```

The optional image shows the whole Western Pacific basin and the actual moving core side by side, on one pressure colour scale: **blue low / red high**. It keeps original geography and masks unsupported core cells. To draw another saved lead without repeating inference:

```bash
python models/trackformer_1_2_field/plot_pressure.py forecast.npz pressure_24h.png --lead 24 --interval 2
```

All original track, central-pressure, basin/regional and auxiliary-wind outputs retain their existing definitions. The exporter verifies the downloaded weight SHA-256 against `manifest.json`; the learned model modules, weights and forward equations are unchanged.

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

The published 50-member mean is a separate evaluation policy: 50 deterministic seeds perturb only normalized historical basin/regional fields with smooth zero-centred noise, then average routes, central pressures and common-grid fields. The weights are the same for every member; this is **not 50 separately trained models**. Issue-time centre, motion, intensity, masks and static geography are not perturbed. `predict.py` computes one clean forecast; do not label it a 50-member mean. Neither this member count nor the 48-hour history has been established as optimal by a matched ablation; spread is not automatically calibrated uncertainty.

**Report's matched development evaluation:** route uses the frozen **1,473 daily starts / 270 equal-weight storms**, +6 to +120 h. Central pressure uses **40 of those 270 storms**, **134 common starts / 2,637 valid lead records**, comparing each model separately with JMA at identical valid times. This is an inherited common-scoring subset, not evidence that all other storms lack pressure observations. Missing labels and unsupported outputs are not zero-scored. TrackFormer 1.2's mean route error is **471.20 km**, +120 h error **1031.63 km**, six-hour direction error **34.96°**, and JMA central-pressure MAE **12.84 hPa**. Central-pressure error is not whole-map error. See the [report figures](../../docs/research_report_zh.md), [frozen common scores](../../evaluation/released_daily/released_daily_benchmark.json), [pressure protocol](../../docs/intensity_benchmark.md) and [detailed evaluation notes](../../docs/trackformer_1_2_evaluation.md).

**Completed DeepMind comparison:** Google's official **WeatherNext Cyclones Mini `<2024`**, run with **WeatherNext software v0.3.0**, completed all **1,473 daily starts / 270 storms** on an **NVIDIA GeForce RTX 3070 using CUDA**. This is the 1° Mini checkpoint trained through 2023, not the full-sized WeatherNext model. Mini uses one seeded member; TrackFormer 1.2 uses a 50-member mean. Mini's report values are **498.37 km** mean track error, **1031.91 km** at +120 h, **39.76°** direction error and **10.20 hPa** JMA central-pressure MAE on the **134 starts / 40 storms / 2,637 common valid records**. The combined cohort favours 1.2 for position and heading, but Mini has lower pressure MAE. Mini also has lower recent-only track error: **258.7 versus 469.4 km** on the strict 2025–2026 subset (61 starts / 18 storms). Different input/member policies and prior case inspection prevent general superiority, equal-compute or untouched-holdout claims. [Completed results and model identity](../../docs/deepmind_daily_benchmark.md) · [CUDA completion receipt](../../evaluation/deepmind_daily/verification.json) · [Independent publication audit](../../evaluation/deepmind_daily/publication_audit.json). No older benchmark's DeepMind scores were substituted.

**Rainfall and radius scope:** ongoing Taiwan rainfall work is separate from these weights. Released 1.2 has no rainfall head or humidity input. The report's three R34 diagrams are **observed IBTrACS quadrant radii**, not native 1.2 radius forecasts. Experimental pressure-derived radius outputs described above remain separately labelled diagnostics.

**Pressure-display coverage:** `regional_mslp_hpa` remains the original fixed issue-relative composite. Use the separately exported `core_mslp_hpa` and its per-lead coordinates/mask when the storm moves away from that fixed patch. Do not extend it by inventing a vortex or moving it onto a route. If the moving core leaves supported basin geography, the corresponding masked cells remain unavailable. For an ensemble, register each physical member field onto a common geographic grid **before** averaging; do not average moving-frame array indices. The Mangkhut film uses that separate 50-member policy. Older fixed-patch films retain their original data in [the showcase archive](../../docs/showcase_archive.md). The automatic History archive and its separate core recovery expose geography and coverage through [the public API](https://trackformer-weatherlab.rudin-euler-8253.chatgpt.site/history-api).
