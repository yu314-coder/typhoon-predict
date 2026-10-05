# Daily forecasts, equal-storm benchmark

Status: **complete** at 2026-09-29T08:22:39Z. All **1,473 daily cases / 270 storms** finished; the SHA-256 of every saved forecast was verified. [Final equal-storm results and all 270 storm scores](../evaluation/daily_storm_final.json).

Mean track error is 798.4 km for 1.1 and 471.2 km for 1.2 (50-member mean); +120 h error is 1,646.4 versus 1,031.6 km. Direction error is 51.58° versus 34.96°, centred shape similarity 0.7544 versus 0.8837, and geographic path similarity 0.5345 versus 0.6560. The original track run reported 1.2-only central-pressure MAE of 12.62 hPa on 40 pressure-labelled storms and basin MSLP MAE of 2.72 hPa on 270 storms. It did not produce 1.1 intensity outputs.

The completed [native pressure follow-up](intensity_benchmark.md) now provides a matched 1.1 comparison on 134 daily starts / 40 storms from the same frozen plan. JMA MAE is 13.53 / 12.84 hPa for 1.1 / 1.2; USA MAE is 12.84 / 12.55 hPa. Each uses exact common agency-specific labels. The 1,339 remaining issue inputs are unavailable to the frozen 1.1 intensity pipeline, never zero-scored or filled with future observations. Paired whole-storm intervals include zero. These are not all-270-storm pressure scores.

On 1 October, `release_tools/build_release_benchmark.py` independently rechecked all forecast hashes, coordinate conversions and daily/storm route aggregates, reproduced the native pressure postprocessor and generated the [shared snapshot](../evaluation/released_daily/released_daily_benchmark.json) and [original Trackformer audit receipt](../evaluation/released_daily/released_daily_verification.json). No inference or weight changes were necessary.

The [completed WeatherNext Cyclones Mini comparison](deepmind_daily_benchmark.md) now covers exactly the same 1,473 starts / 270 storms. Mini's combined mean track error is 498.4 km, direction error 39.76°, and JMA pressure MAE 10.20 hPa on 134 shared intensity starts. Its recent-only mean track error is 288.5 km versus 477.2 km for 1.2. The [independent three-way publication audit](../evaluation/deepmind_daily/publication_audit.json) rechecks the CUDA archive, unchanged original Trackformer arrays and recomputed common-support scores; the older 1 October receipt remains the audit of the original two-model snapshot, not a hash receipt for the subsequently merged JSON. No older benchmark's DeepMind score has been transferred into this comparison.

## What counts as a case and a score

One case is one typhoon at one issue time on one UTC day. Choose the earliest eligible issue on that day, once per typhoon. Each forecast predicts all 20 six-hour leads, +6 through +120 h, from information at or before issue time.

For position error, daily score = mean of the 20 lead errors; storm score = mean of that storm's daily scores; benchmark score = mean of the 270 storm scores. A storm with ten available daily starts has exactly the same final weight as a storm with three. The per-lead comparison also averages daily values within storm before averaging storms.

Direction error, route-shape similarity, Fréchet distance and path similarity remain separate metrics. Their definitions match the [evaluation notes](trackformer_1_2_evaluation.md); direction uses only common non-stationary leads. Missing direction scores are reported, not replaced by zero. There is no invented combined weighted quality mark.

## Frozen cohort

- 270 distinct storms, 1,473 daily issue cases, selected from 523 locally eligible storms.
- All 40 eligible recent storms beginning in 2024 or later, plus 230 historical storms selected at evenly spaced positions in the chronological 1980–1999 storm list.
- The entire storm must lie outside the selected 1.2 checkpoint's 2000–2021 fitting and 2022–2023 validation years, with the history/target boundary also outside those years.
- Issue centre within 0–60°N / 100–180°E; exact six-hour weather history through issue; complete route labels through +120 h; matching pressure-atlas coverage for scoring.
- Use all eligible daily starts for each chosen storm. Never drop a case because its forecast looks poor.
- [Cohort manifest](../evaluation/release_data/daily_storm_cohort.json) freezes rows, dates, storm IDs, source hashes and checkpoint identity before inference. Forecast artifacts are new; some recent storms overlap earlier development comparisons.

Requiring complete five-day truth excludes short remaining lifetimes. This is a declared coverage limitation, not an all-storm operational sample. Historical cases are retrospective hindcasts from a model trained on later years; reanalysis and best-track issue states do not establish real-time data availability. This cohort has not been certified untouched across all earlier experiments.

## Models and input boundaries

Trackformer **1.2** uses the unchanged released checkpoint and 50 independently seeded, smooth input-perturbation members. Seeds are `2043 + source_row * 100 + member_id`, with member IDs 0–49. Normalized perturbation amplitudes remain 0.025 for basin history and 0.015 for regional history. These are not learned latent samples or 50 separate trained models. Member routes and central pressures are averaged; physical fields are averaged on fixed common geographic grids. Missing native regional inputs retain the model's missing-detail mask.

Trackformer **1.1** uses the released causal weighted route builder with issue, −12 h and −24 h analyses and historical motion. Its absolute route is converted into the same issue-centred local-km coordinate system as 1.2. The pipelines differ, so this is a released-system comparison, not an architecture-only ablation.

Future weather/track/pressure are used for eligibility coverage and scoring only, never inference. No retraining, coefficient tuning, new weather downloads or checkpoint substitution occurs in this run.

## Pressure and failures

For the original 1.2 run, central-pressure MAE uses valid labels and cosine-area-weighted basin MSLP MAE uses the native basin grid, daily then equally per storm. Keep pressure label coverage visible. The separate native-pressure follow-up uses a paired mask for both models at each agency's exact valid times; it does not substitute its smaller coverage into all-storm track results. Regional/core field scores need their own valid coverage and are not implied by basin-wide error. No same-cohort 1.1 basin-field score is claimed.

Completed daily forecasts are saved atomically with hashes. A nonfinite forecast or source mismatch stops the run visibly rather than silently deleting the case. Resume skips completed cases without rerunning them. Partial aggregate scores include only fully completed storms, identify their counts, and must not be presented as the final 270-storm result. Final uncertainty should resample whole storms, not overlapping daily issues independently.
