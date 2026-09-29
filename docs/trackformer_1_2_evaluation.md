# Trackformer 1.2 evaluation notes

These are archived development results, not an untouched generalization test.

## Primary 270-case route comparison

The main chart compares 270 issue cases from 90 storms and all 20 leads (+6 through +120 h). Preparation asserts exact equality of source-row order and truth arrays between the saved 1.1 cohort and the 1.2 ensemble archive. All position and shape scores use all 270 cases. The 1.2 forecasts are means of 50 input-perturbation members; 1.1 uses the archived causal route pipeline. The forecasts use different input pipelines, so the comparison is not a controlled architecture ablation.

### Coordinate correction

`benchmark_100_storms_3_days_gpu.py::build_v11_routes` returned absolute latitude/longitude, but `benchmark_strict_causal_post2021.py` treated this return value as local kilometres and saved it under `v11_local`. The new reproduction script reads that original array as degrees and projects it using the same definition as the 1.2 benchmark: east = wrapped longitude difference × 111.2 × max(cos(issue latitude), 0.2); north = latitude difference × 111.2. A round trip back to the archived latitude/longitude is asserted. The 1.2 arrays and observed local displacements are unchanged. This correction replaces the old 1.1 mean track error of 1,036.3 km with **902.3 km**. It also replaces the old direction/shape comparisons. The old reports are not overwritten.

### Separate direction, shape and path measures

- **Direction error:** absolute wrapped angle between predicted and observed six-hour displacement vectors, with the issue origin prepended. Both models use the same mask: each model and truth must move more than 1 km. There are **5,382 valid steps out of 5,400**. The mean is 56.2241° for 1.1 and 44.8241° for 1.2. Lower is better.
- **Route-shape similarity:** centre each 20-point route, flatten its east/north coordinates, calculate cosine similarity to the centred truth and map it to [0, 1] using `(1 + cosine)/2`. Translation and a uniform scale factor do not affect this measure; orientation and the ordering of points do. It is not an accuracy percentage. Means: **0.727339 for 1.1; 0.824862 for 1.2**.
- **Path similarity:** calculate discrete Fréchet distance on the ordered 20-point forecast/truth curves, then `exp(-distance / observed path length)`, and average over cases. The observed length includes the step from the issue origin. Means: **0.498254 for 1.1; 0.572741 for 1.2**. This responds to geographic displacement as well as the curve.
- **Position error:** mean Euclidean error in the saved benchmark's local-kilometre projection, **902.3001 vs 714.4444 km**. At +120 h it is **1,833.2481 vs 1,433.3863 km**. These are local-coordinate distances, not newly calculated geodesic distances.

No combined weighted ranking is introduced. Cases 1, 136 and 270 are shown for visual comparison, chosen by row position without filtering on forecast quality. Their axes retain actual kilometre displacements from the shared issue origin.

The full 270 cohort includes cases beyond the intended Western Pacific model domain. Its aggregate is a legacy development comparison, not a basin-specific validation claim. The previously inspected 133-case subset is separate and is not substituted for the requested 270 cases.

## 270-case field benchmark

The saved `benchmark_ensemble50/v173_e4/report.json` has 270 development cases (90 storms with three issue times each), 20 six-hour leads, checkpoint SHA-256 `f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0`, 50 distinct input-perturbation member IDs and deterministic seed base 2043. Each member uses separate Gaussian fields smoothed with kernel sizes 5 (basin) and 9 (regional), centred and unit-scaled per map, with normalized amplitudes 0.025 and 0.015 respectively. Only `global_history` and `regional_history` are perturbed. Means are taken of routes, central pressures and fields on the same grids. These are neither 50 trained models nor learned latent members, and their spread is not a calibrated probability.

The mean-of-50 result is 714.4 km all-lead mean track error, 1433.4 km +120 h track error, 15.73 hPa central-pressure MAE and 2.536 hPa area-weighted regional MSLP MAE. The regional field metric is on a fixed grid; only 2627 of 5400 case-leads (48.6%) had valid in-domain centres for the separately reported centre-local metric. Do not conflate low area-weighted field error with accurate cyclone-core placement. The TIP pressure error is much higher than the 270-case aggregate.

No same-270 1.1 central-pressure prediction array was verified in the route artifacts, and 1.1 has no model-generated MSLP field there; neither receives an invented comparison bar. The separate TIP ten-start diagnostic still gives 383.9 vs 370.5 km track error and 20.1 vs 30.7 hPa central-pressure MAE for 1.1 vs 1.2. TIP's overlapping starts are one repeatedly inspected storm; those pressure scores are not mixed into the 270-case chart.

## Recent pressure-line example

The Surigae archive issued **2026-09-27 12 UTC** supplies real model regional MSLP fields at +6, +24 and +36 h. Isobars are drawn every 4 hPa, with selected labels every 12 hPa and one common colour scale. The magenta curve is the saved model route. Fields are neither relocated nor altered to force the map minimum onto the route. The regional map is a resampled composite and can differ slightly from central pressure sampled in the moving core.

This example is one deterministic member from the released checkpoint. Inputs contain nine GFS analyses ending at 06 UTC (six hours before issue), issue-time JMA centre/central pressure, and static geography. Issue-time wind, prior motion and native regional history were unavailable and masked/zeroed. The stored +0 field is an input analysis; it is excluded from these forecast panels. The GFS transfer and input lag remain limitations. The selected leads keep the centre within the fixed 114–144°E, 11–41°N map. Names, times, member count and checkpoint/source hashes are in `paper/release_data/surigae_provenance.json`.

## Reproduce the figures

The public repository includes the small numerical plot inputs under `paper/release_data/`, including original 1.1 latitude/longitude, corrected routes, 1.2 mean routes, truth, the three pressure grids and map boundaries. After installing NumPy and Matplotlib, run:

```bash
python release_tools/plot_release_270.py
```

This regenerates the four-panel benchmark, per-lead curve, fixed route examples, isobar figure and metrics JSON. `--source-root` is optional and is only used to rebuild the small plot inputs from the original local research archives. No inference, fitting or new weather retrieval is performed by this plotting script.

## Validation boundary

Training used whole-storm year partitions: 2000–2021 training, 2022–2023 validation, 2024–2025 test in the original pipeline, with 13,949 / 1,041 / 1,195 windows respectively; native-pressure windows total 800 / 100 / 100. Repeatedly viewed benchmarks, TIP and calibration cases have become development evidence. Freeze a genuinely unused storm-level holdout with matched issue-time data before claiming generalization. Evaluate route, central pressure and pressure fields at every lead on common cases and masks; a lower map average alone is not a model win.
