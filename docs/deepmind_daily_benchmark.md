# Completed WeatherNext Cyclones Mini daily benchmark

## Model and frozen plan

**Complete and verified:** all **1,473 daily forecasts / 270 Western Pacific storms** finished on an **NVIDIA GeForce RTX 3070 using JAX CUDA in Windows/WSL**. The final worker audit is dated **4 October 2026, 12:55 UTC**. Publication independently rechecked all returned case hashes, physical grids, unchanged Trackformer reference arrays and exact common-support scores. No new inference was performed during this review.

The reference is Google's official [WeatherNext Cyclones Mini](https://github.com/google-deepmind/weathernext#weathernext-cyclones-mini): **`WeatherNextCyclones_Mini_<2024`**, trained through 2023 at native **1° resolution**. The pinned [software package is v0.3.0](https://github.com/google-deepmind/weathernext/blob/89c4b2a77a1c57b328b909c575550fd2e5aadc9c/setup.py); the package version and checkpoint generation identify different things. This is not a custom imitation, full-sized WeatherNext 2/Cyclones, or a WeatherNext 3 checkpoint. Mini's scores do not stand in for those larger models.

- Cohort: **1,473 daily starts / 270 Western Pacific storms**, identical to the frozen Trackformer 1.1/1.2 route comparison.
- Forecast: twenty six-hour steps, **+6 to +120 h**, with the exact observed issue-time centre as the initialized-storm anchor.
- Inputs: exact ERA5 analyses at **issue−6 h and issue**, calendar forcings only after issue. No future weather, storm labels or official forecast tracks enter the worker.
- Weights SHA-256: `a1bb151457077248d70a458b1a7b19deacd2926fbcb46eee4ff2691444e3596e`.
- Cohort SHA-256: `965184f1e5ab3fcb52e304f39b38583a14f753870aa2295f2d01942a441dcfd6`.
- Official code commit: `89c4b2a77a1c57b328b909c575550fd2e5aadc9c`.
- CUDA protocol SHA-256: `91b31582f24ffdd4f2309eddb0cc68498a2c31cab7cbabaf06f43dfbff413606`.
- Runtime: JAX/JAXlib 0.4.38, Python 3.12.3, highest float32 matmul precision, TF32 disabled.

## Results

| Matched development metric | Trackformer 1.1 | Trackformer 1.2 · mean of 50 | Mini · one member | Coverage |
| --- | ---: | ---: | ---: | --- |
| Mean track error · km, lower is better | 798.41 | 471.20 | 498.37 | 1,473 days / 270 storms |
| +120 h track error · km, lower is better | 1,646.40 | 1,031.63 | 1,031.91 | 1,473 days / 270 storms |
| Direction error · degrees, lower is better | 51.58 | 34.96 | 39.76 | 1,473 days / 270 storms |
| Centred route-shape similarity · higher is better | 0.7544 | 0.8837 | 0.9010 | 1,473 days / 270 storms |
| JMA central-pressure MAE · hPa, lower is better | 13.53 | 12.84 | 10.20 | 134 days / 40 storms / 2,637 leads |
| JMA pressure-curve similarity · higher is better | 0.7074 | 0.7118 | 0.8088 | 134 days / 40 storms |
| USA central-pressure MAE · hPa, lower is better | 12.84 | 12.55 | 11.55 | 134 days / 40 storms / 2,325 leads |
| USA pressure-curve similarity · higher is better | 0.7237 | 0.6637 | 0.7445 | 131 days / 40 storms |

Pressure measures central-pressure intensity, not map-grid error. The lower USA curve count excludes flat/short curves; it is not zero-filled. All three models' pressure-curve case IDs were verified equal for each agency. DeepMind geographic path/Fréchet scores and whole-storm confidence intervals are not supplied by this export and remain unavailable.

### Historical and recent groups

| Period | Daily starts / storms | Mean track error · 1.1 / 1.2 / Mini, km | Direction error · 1.1 / 1.2 / Mini, degrees |
| --- | --- | --- | --- |
| 1980–1999 | 1,336 / 230 | 780.50 / 470.17 / 534.86 | 51.36 / 35.01 / 41.26 |
| Storms beginning in 2024+ | 137 / 40 | 901.45 / 477.16 / 288.51 | 52.90 / 34.68 / 31.16 |

Mini has better recent-only track scores even though 1.2 has lower position/heading errors in the combined cohort. Mini also has lower JMA/USA pressure MAE and higher pressure-curve similarity. Historical fitting-year overlap, different weather inputs and different ensemble sizes prevent a general or equal-compute superiority claim. All pressure comparisons here belong to the recent group; no historical pressure score is inferred from missing native 1.1 inputs.

[Full results, period breakdowns and per-lead track errors](../evaluation/deepmind_daily/benchmark.json) · [Shared snapshot and existing 1.1/1.2 uncertainty](../evaluation/released_daily/released_daily_benchmark.json) · [Completion receipt](../evaluation/deepmind_daily/verification.json) · [Independent review](../evaluation/deepmind_daily/publication_audit.json)

### Strict forecast-date partition and separate charts

Both README charts use **`WeatherNextCyclones_Mini_<2024`**, the same official checkpoint trained through 2023, software v0.3.0, native 1° and one CUDA member. A post-2024 forecast is not a post-2024-trained model. The following proportions refer to this evaluation cohort, not the model's training dataset.

| UTC issue year | Starts / share | Storms / share |
| --- | --- | --- |
| `<2024` (1980–1999 here) | 1,336 / 90.7% | 230 / 85.2% |
| Calendar 2024 | 76 / 5.2% | 22 / 8.1% |
| `>2024` (2025–2026 here) | 61 / 4.1% | 18 / 6.7% |

The original `recent_2024_onward` aggregate includes 2024 and is retained unchanged. The new **strict `after_2024`** derivative excludes 2024: track position MAE is **963.02 / 469.40 / 258.67 km**, direction error **57.88 / 33.81 / 28.73°**, and JMA pressure MAE **12.59 / 11.12 / 9.55 hPa** for 1.1 / 1.2 / Mini. Track uses 61 starts / 18 storms; pressure uses 59 starts / 18 storms / 1,174 exact common leads. The total chart still uses the original scores. Its historical 230 storms contribute 85.2% of equal-storm track weight, not 90.7%.

[UTC partitions, exact case IDs, all metrics and source hashes](../evaluation/deepmind_daily/period_comparison.json) · [Total chart](../evaluation/released_daily/model_1_2_benchmark.png) · [Post-2024 chart](../evaluation/released_daily/model_1_2_after_2024_benchmark.png)

Reproduce these derivative summaries without inference or raw-weather downloads:

```bash
python release_tools/build_deepmind_period_comparison.py --archive /path/to/DeepMind_RTX3070_results_20261004_125651_467960.zip
python release_tools/plot_model_announcement.py
```

The helper checks the exact archive hash against the existing publication audit and every one of the 1,473 case-score JSON hashes against the frozen manifest. It rechecks identities, causal-time metadata, CUDA completion and common pressure masks, reproduces the original total and period scores, then recomputes only the UTC subsets using the unchanged equal-storm aggregation. It never rewrites the original worker export, changes a forecast array or claims a new raw-input audit. Chart sidecars identify their derivative source hash, checkpoint, period, exact values and missing-data policy.

## Like-for-like scoring

All three models use the frozen starts, exact future label times and original issue-relative kilometre projection. Routes are unshifted. Direction is recomputed on **common moving steps across truth and all three models**, not compared using different masks.

Pressure uses the original 134 eligible native-intensity starts / 40 storms, restricted to common valid leads across three predictions and the selected reference. JMA and USA remain separate. MAE and centred time-curve similarity are separate; missing, nonphysical or failed outputs are not zeros. Flat or shorter-than-six-point curves have unavailable shape similarity.

Average valid leads within an issue, days within a storm, then storms equally. Partial aggregates contain only fully completed storms and explicit coverage. Recent 2024+ and historical 1980–1999 groups are separate because the latter overlap WeatherNext fitting years. Different input pipelines and the one-member Mini versus 50-input-member 1.2 policy make this an output comparison, not an equal-compute architecture ablation or a certified unused test.

The official six-hour direct tracker uses a frozen initialized-storm continuity policy: no cyclogenesis, no dissipation pruning and no nearby-cyclone pruning. This is disclosed rather than called the operational default. Native 1° WP MSLP grids are saved in physical hPa; cyclone-head central pressure is not the basin-grid minimum.

## Execution and verification

The original [Mac CPU runner and scoring definitions](../release_tools/deepmind_daily_benchmark.py) provided the frozen plan. The completed run used the separately pinned, resumable RTX 3070 CUDA package; the old CPU launch instructions are not the execution record of these results. The [frozen CUDA protocol](../evaluation/deepmind_daily/protocol.json), [2,946-file case manifest](../evaluation/deepmind_daily/case-manifest.json) and [CPU/CUDA canary](../evaluation/deepmind_daily/canary.json) identify the completed run.

The strict same-runtime CPU/CUDA canary passed unchanged 0.01 hPa mean / 0.1 hPa maximum field tolerances. Its older cross-runtime Mac comparison failed the maximum-field gate and remains explicitly recorded as failed; that diagnostic is not numerical-equivalence certification. No CPU forecast was counted or relabelled as a CUDA case.

The publication review checks all 1,473 IDs, original reference hashes, exact issue−6 h / issue input-time metadata and saved input hashes, all twenty output times, finite native pressure grids, and recomputed three-model route/pressure metrics. **Raw ERA5 input states are not in the returned result-only ZIP**: this review preserves the original workers' causal-input receipts and hashes, rather than claiming to independently reopen those raw states.

To repeat the read-only archive review, use the existing frozen project reference forecasts and intensity labels, NumPy, and the returned result archive. It does not run a model or download weather:

```bash
python release_tools/import_deepmind_release_results.py \
  --archive /path/to/DeepMind_RTX3070_results_20261004_125651_467960.zip \
  --project /Volumes/D/typhoon_predict \
  --output /Volumes/D/typhoon_predict/remote_typhoon_predict_repo/evaluation/deepmind_daily
```

Completed scores are also exposed by the existing [public benchmark API](https://trackformer-weatherlab.rudin-euler-8253.chatgpt.site/api/benchmarks/released). Documentation publication does not replace immutable forecasts, change either model's weights, restart the benchmark, modify active training, or redeploy the website. The original two-model verification receipt remains unchanged; the new publication audit identifies the merged snapshot's exact hash.

The returned worker's `benchmark.json` is retained byte-for-byte for its audit hashes. Its `published_to_site: false` and `integration_note` describe the handoff before import, not the current benchmark status. The completed CUDA receipt, independent publication audit and shared public snapshot above are the current completion/publication evidence; those historical worker fields do not mean the comparison is unfinished.
