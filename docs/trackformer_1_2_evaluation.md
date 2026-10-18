# Trackformer 1.2 evaluation notes

## Completed equal-storm benchmark

The current comparison uses **1,473 daily forecasts across 270 Western Pacific storms**, with 20 six-hour leads from +6 through +120 h. Scores average leads within issues, daily issues within each storm, then storms equally. The completed results and all forecast SHA-256 checks are in [the final report](../evaluation/daily_storm_final.json); the [daily-storm protocol](daily_storm_benchmark.md) defines the frozen cohort and metrics.

The current [Chinese research-report view](research_report_zh.md) compares **released 1.2 with DeepMind Mini**, using unchanged saved results. TrackFormer uses the mean of 50 distinct causal input perturbations of one model, not 50 trained networks; Mini uses one seeded member. These are matched outputs with different inputs and member policies, not a controlled architecture ablation, equal-compute test or untouched generalization test.

| Report metric | TrackFormer 1.2 | DeepMind Mini |
| --- | ---: | ---: |
| Mean track error | 471.20 km | 498.37 km |
| +120 h track error | 1031.63 km | 1031.91 km |
| Six-hour direction error | 34.96° | 39.76° |
| Central-pressure MAE · JMA | 12.84 hPa | 10.20 hPa |

Direction, centred shape, geographic path and same-time position error measure different aspects of a forecast. A translated or mistimed route can have high shape similarity without overlapping the observed route. No combined weighted ranking is introduced.

**Central-pressure coverage is a subset of route coverage:** **40 of the 270 storms / 134 starts / 2,637 common valid lead records**. Each model's pressure is compared separately with the JMA record at the identical forecast time. This is not a separate 270-storm pressure evaluation. The original 1.2-only pressure mean of **12.62 hPa** uses a different mask and must not replace the report's common-support **12.84 hPa**. The original basin MSLP mean of **2.72 hPa on 270 storms** is a whole-map measure, not central-pressure error. [Common scores](../evaluation/released_daily/released_daily_benchmark.json) · [Pressure protocol](intensity_benchmark.md).

The inherited shared-case eligibility excludes **1,339 starts with unavailable frozen legacy issue-time intensity inputs**. That does **not** mean all excluded storms lack pressure observations. Original masks remain unchanged; unsupported scores are unavailable, not zero. The cohort contains 40 recent and 230 historical storms from 1980–1999. Retrospective analyses, complete-five-day eligibility and prior model selection limit interpretation.

The [report's day-1–5 chart](report_figures/day_1_to_5.png) shows **exact 24, 48, 72, 96 and 120 h** leads rather than daily-bin averages. Track covers 270 storms at each displayed lead. JMA pressure has 40, 40, 40, 40 and 39 storms, with 131, 133, 134, 133 and 127 common starts respectively. All-lead pressure totals and per-lead counts describe different aggregations. [Exact chart values](report_figures/day_1_to_5.json).

## Completed DeepMind comparison

Google's official **WeatherNext Cyclones Mini `<2024`**, using **WeatherNext software v0.3.0**, completed the same **1,473 daily starts / 270 storms** on an **NVIDIA GeForce RTX 3070 with CUDA**. It is a 1° model trained through 2023. Mini is one seeded member, not the full-sized WeatherNext model; 1.2 remains a 50-member mean. The common-support results are **498.37 km** mean track error, **1031.91 km** at +120 h, **39.76°** direction error and **10.20 hPa** JMA central-pressure MAE on **134 starts / 40 storms / 2,637 valid records**. Its JMA pressure-curve similarity is **0.8088**.

Mini's recent-only mean track error is **288.5 km** versus **477.2 km** for 1.2. Historical fitting-year overlap and different input/member policies limit the combined-cohort comparison; no general superiority claim is justified. The [completed protocol and period breakdowns](deepmind_daily_benchmark.md), [CUDA receipt](../evaluation/deepmind_daily/verification.json) and [independent three-model publication audit](../evaluation/deepmind_daily/publication_audit.json) establish completion. The older two-model receipt remains the original Trackformer audit, not the hash receipt for the subsequently merged snapshot. No missing or unsupported score is treated as zero.

That **recent-only** group includes 2024. The report-facing strict **after-2024** group is different: 2025–2026, 61 starts / 18 storms, with **258.7 km Mini versus 469.4 km 1.2**. Calendar 2024 is not part of that strict subset. Do not interchange these period scores.

## Report showcases and preserved earlier records

The [three report examples](research_report_zh.md#三個颱風展示) use unchanged released-1.2 weights: Fung-wong and Mangkhut are saved Metal forecasts; Jebi was run on CPU. All use 50 input-perturbed members and exact +6 to +120 h times. Mangkhut and Jebi are within the fitting years. Their R34 panels show **IBTrACS observations only**, not model radius skill. NCEP is still an analysis reference even where the figure label is simplified to “reference pressure”.

Earlier release comparisons and their pressure-curve diagnostics remain in the [archived shared snapshot](../evaluation/released_daily/released_daily_benchmark.json), [full original report](../evaluation/daily_storm_final.json) and [intensity protocol](intensity_benchmark.md). They are not silently substituted for the current report's two-model table. Centred curve similarity removes level and amplitude; read it with hPa MAE, masks and original timelines rather than treating it as accurate pressure levels.

## Recent pressure-line example

### Fung-wong MP4: selected route-and-pressure example

The [Fung-wong video](https://yu314-coder.github.io/typhoon-predict/trackformer_1_2_fung_wong.mp4) uses the unchanged saved 50-member pressure fields and mean route for case 262, issued 2025-11-07 00 UTC. It is a user-selected development example after inspection of full routes and pressure curves, not an untouched test or a representative sample. The geographic mean error is 130.9 km, the +120 h error is 142.3 km, and maximum error is 198.3 km over 20 leads. These are great-circle distances to exact observed coordinates, separate from the main benchmark's saved local-coordinate scores. The route follows the broad observed curve but is not perfectly overlapping.

The pressure MAE is **7.36 hPa** against IBTrACS TOKYO_PRES (JMA best track), with 20 valid labels. Intensification and weakening still differ from observations. Fields are physical model means on fixed common grids, without shifting them onto the observed track. Native-detail history was unavailable; 0.25° regional output sampling is not native resolution. From +66 h the forecast centre leaves the regional patch: only saved coarse basin pressure is shown there, while member-mean central pressure remains the moving-core readout. No display-only core is added. Observed future routes and pressures are verification overlays only. [Data and provenance](../evaluation/release_data/fung_wong_video.json) record hashes and member policy. Reproduce with `python release_tools/build_fung_wong_video.py --work /path/to/frames` using NumPy, Matplotlib and ffmpeg with VideoToolbox on macOS; this renders saved outputs without model inference. The former interactive showcase has been removed.

### Recent single-member example

The Surigae archive issued **2026-09-27 12 UTC** supplies real model regional MSLP fields at +6, +24 and +36 h. Isobars are drawn every 4 hPa, with selected labels every 12 hPa and one common colour scale. The magenta curve is the saved model route. Fields are neither relocated nor altered to force the map minimum onto the route. The regional map is a resampled composite and can differ slightly from central pressure sampled in the moving core.

This example is one deterministic member from the released checkpoint. Inputs contain nine GFS analyses ending at 06 UTC (six hours before issue), issue-time JMA centre/central pressure, and static geography. Issue-time wind, prior motion and native regional history were unavailable and masked/zeroed. The stored +0 field is an input analysis; it is excluded from these forecast panels. The GFS transfer and input lag remain limitations. The selected leads keep the centre within the fixed 114–144°E, 11–41°N map. Names, times, member count and checkpoint/source hashes are in `evaluation/release_data/surigae_provenance.json`.

## Reproduce the benchmark chart

```bash
python release_tools/plot_daily_storm_final.py
```

This plots the saved completed equal-storm results; it performs no new inference or weather retrieval. Original research arrays and reproduction tools remain preserved for provenance, but the retired smaller-cohort comparison is not presented as the current benchmark.

## Validation boundary

Training used whole-storm year partitions: 2000–2021 training, 2022–2023 validation, 2024–2025 test in the original pipeline, with 13,949 / 1,041 / 1,195 windows respectively; native-pressure windows total 800 / 100 / 100. Repeatedly viewed benchmarks, TIP and calibration cases have become development evidence. Freeze a genuinely unused storm-level holdout with matched issue-time data before claiming generalization. Evaluate route, central pressure and pressure fields at every lead on common cases and masks; a lower map average alone is not a model win.

The fitting sources are IBTrACS-derived storm records, NOAA PSL NCEP/NCAR basin analyses, ARCO-ERA5 regional MSLP and static ERA5 land fraction/elevation. Newer 2026 archive records are not part of the released model's gradient fitting. See [training-source documentation](trackformer_1_2_training_data.md) and [checksum-verified split receipt](../evaluation/training_data/trackformer_1_2_provenance.json).
