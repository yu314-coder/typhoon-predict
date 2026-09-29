# Daily forecasts, equal-storm benchmark

Status: inference in progress. This document specifies the frozen evaluation, not its final results.

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

For 1.2, report central-pressure MAE over valid labels and cosine-area-weighted basin MSLP MAE on the native basin grid, daily then equally per storm. Keep pressure label coverage visible. No matched 1.1 pressure score is fabricated; this run does not establish superiority on pressure. Regional/core field scores need their own valid coverage and are not implied by basin-wide error.

Completed daily forecasts are saved atomically with hashes. A nonfinite forecast or source mismatch stops the run visibly rather than silently deleting the case. Resume skips completed cases without rerunning them. Partial aggregate scores include only fully completed storms, identify their counts, and must not be presented as the final 270-storm result. Final uncertainty should resample whole storms, not overlapping daily issues independently.
