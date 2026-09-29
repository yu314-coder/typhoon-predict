# Trackformer 1.2 evaluation notes

## Completed equal-storm benchmark

The current comparison uses **1,473 daily forecasts across 270 Western Pacific storms**, with 20 six-hour leads from +6 through +120 h. Scores average leads within issues, daily issues within each storm, then storms equally. The completed results and all forecast SHA-256 checks are in [the final report](../evaluation/daily_storm_final.json); the [daily-storm protocol](daily_storm_benchmark.md) defines the frozen cohort and metrics.

Trackformer 1.2 uses the mean of 50 distinct causal input perturbations, not 50 trained networks. Trackformer 1.1 uses its archived causal route pipeline. These are matched forecast outputs with different input pipelines, not a controlled architecture ablation or an untouched generalization test.

| Equal-storm metric | 1.1 | 1.2 mean of 50 |
| --- | ---: | ---: |
| Mean track error | 798.4 km | 471.2 km |
| +120 h track error | 1,646.4 km | 1,031.6 km |
| Six-hour direction error | 51.58° | 34.96° |
| Centred shape similarity | 0.7544 | 0.8837 |
| Geographic path similarity | 0.5345 | 0.6560 |

Direction, centred shape, geographic path and same-time position error measure different aspects of a forecast. A translated or mistimed route can have high shape similarity without overlapping the observed route. No combined weighted ranking is introduced.

The 1.2 central-pressure MAE is **12.62 hPa on 40 storms with valid labels**; area-weighted basin MSLP MAE is **2.72 hPa on 270 storms**. No matched 1.1 pressure output is available. A basin-wide average does not establish cyclone-core accuracy. The frozen cohort contains 40 recent and 230 historical storms from 1980–1999; retrospective analyses, complete-five-day eligibility and prior model selection limit interpretation.

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
