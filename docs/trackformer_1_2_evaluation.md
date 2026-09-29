# Trackformer 1.2 evaluation notes

These are archived development results, not an untouched generalization test.

## TIP 1.1 comparison

The bars in the [main README](../README.md) were rendered from saved local reports `benchmark_tip_ten/v11.json` and `benchmark_ensemble50/v173_e4/tip_ten/v173_e4_ensemble50.json`. The plotting script asserts that both reports have the same ten issue timestamps. The 1.1 pipeline is one released forecast; 1.2 is a mean of 50 runs of one weight set with different smoothly perturbed causal historical inputs. Both use +6 to +120 h outputs. Unlike 1.2, 1.1 has no native MSLP-field forecast to compare in the pressure-map figure.

The TIP cohort consists of one storm and ten overlapping starts. It has been repeatedly inspected and is development evidence only. Central-pressure error regresses for 1.2 even though mean route error is slightly lower.

## 270-case field benchmark

The saved `benchmark_ensemble50/v173_e4/report.json` has 270 development cases (90 storms with three issue times each), 20 six-hour leads, checkpoint SHA-256 `f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0`, 50 distinct input-perturbation member IDs and deterministic seed base 2043. Each member uses separate Gaussian fields smoothed with kernel sizes 5 (basin) and 9 (regional), centred and unit-scaled per map, with normalized amplitudes 0.025 and 0.015 respectively. Only `global_history` and `regional_history` are perturbed. Means are taken of routes, central pressures and fields on the same grids. These are neither 50 trained models nor learned latent members, and their spread is not a calibrated probability.

The mean-of-50 result is 714.4 km all-lead mean track error, 1433.4 km +120 h track error, 15.73 hPa central-pressure MAE and 2.536 hPa area-weighted regional MSLP MAE. The regional field metric is on a fixed grid; only 2627 of 5400 case-leads (48.6%) had valid in-domain centres for the separately reported centre-local metric. Do not conflate low area-weighted field error with accurate cyclone-core placement. The TIP pressure error is much higher than the 270-case aggregate.

The basin-pressure example is a saved Dolphin forecast from `benchmark_ensemble50/v173_e4/showcase_four.npz` and is model-generated. It is not a paired 1.1 map. Both figures can be reproduced with `release_tools/plot_field_1_2.py` if the original local reports are available.

## Validation boundary

Training used whole-storm year partitions: 2000–2021 training, 2022–2023 validation, 2024–2025 test in the original pipeline, with 13,949 / 1,041 / 1,195 windows respectively; native-pressure windows total 800 / 100 / 100. Repeatedly viewed benchmarks, TIP and calibration cases have become development evidence. Freeze a genuinely unused storm-level holdout with matched issue-time data before claiming generalization. Evaluate route, central pressure and pressure fields at every lead on common cases and masks; a lower map average alone is not a model win.
