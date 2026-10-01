# Matched daily intensity benchmark

Completed GPU inference and verified aggregation on 1 October 2026 (Taipei).
The released weights, original forecasts and original calibration are unchanged.

## Same hierarchy as track

The plan is the frozen 270-storm / 1,473-daily-issue track cohort. One storm on
one UTC day is one case, at its earliest eligible issue. Forecast errors at
valid +6, +12, …, +120 h leads are averaged per daily issue, then across its
storm's daily issues, then across storms with equal weights. Long-lived storms
do not gain extra final weight. Per-lead errors use the same daily → storm
averaging. Both models use exactly the same native observation mask for each
wind/pressure comparison. Wind and each agency's pressure are separate metrics;
there is no arbitrary weighted combined rank.

Native 1.1 intensity needs valid issue-time current wind and pressure. Its
frozen input archive provides them for 134 daily issues / 40 storms, not the
other 1,339 daily starts. These missing starts are reported, never repaired
with future observations or relabeled as zero error. Thus the result shares
the **method and planned cohort** of track, but does not claim the same valid
intensity coverage. Missing future observations also remain unavailable.

## Model identity and inference

1.2 is the released checkpoint SHA-256
`f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0`.
Every original pressure forecast file is rehashed. To recover previously
unsaved auxiliary wind, the MPS runner replays the identical 50 input members
with seeds `2043 + source_row * 100 + member_id`, nine exact causal analyses,
original detail masks and original issue inputs. Every replay has 50 distinct
input hashes, finite 50 × 20 wind and pressure outputs and verified means.
The maximum pressure difference from original forecasts is **0.0 hPa**.
No weights, preprocessing or coefficients are fitted here.

1.1 uses the original public intensity pipeline: frozen primary and structure
experts, validation-fit anchor/calibration coefficients, and the frozen
same-storm −12/−24 h temporal branch where its current-wind gate permits.
Expert count/activation is preserved in each case receipt; this is not a
50-member 1.1 ensemble. Issue states and causal weather only enter inference;
future labels are loaded separately for scoring. The two systems' input
pipelines differ: this is not an architecture-controlled ablation.

## Native observation definitions

- Central pressure: exact storm SID and valid UTC; native `USA_PRES` and
  `TOKYO_PRES` are scored separately, in hPa, on a common mask for each agency.
- Wind: `USA_WIND` nominal one-minute reference. The 1.2 scalar is an auxiliary
  output, not a certified ten-metre sustained surface-wind product. JMA
  `TOKYO_WIND` is ten-minute and is never converted or pooled in this score.
- Native 1.1 radius: `USA_RMW` and `USA_R34/R50/R64_NE/SE/SW/NW`, nautical miles
  multiplied by 1.852 to kilometres. They are radii, not diameters. RMW must be
  positive. A reported zero wind-radius extent remains valid; blanks remain
  unavailable. JMA R30 longest/shortest, eye size and ROCI are not substitutes.
- Native 1.2 radius: **not available**, because it has no radius head. The
  experimental pressure-scaled candidate is separate; its strict definition
  guard prevents scoring it as calibrated official surface-wind radii.

Radius errors are averaged within the daily issue over valid leads for each
component, then equally across available quadrant components, then across
days within storm and equally across storms. Coverage differs by threshold;
all counts are printed. Observational RMW has reporting/provisional limitations.

## Curve comparison, analogous to route direction and shape

The released Site defaults to **central-pressure MAE**, with slight descriptive mean improvements (JMA 13.53 / 12.84 hPa; USA 12.84 / 12.55 hPa, for 1.1 / 1.2) and paired uncertainty including zero. Its secondary **similarity** view is the exact complement of the shape-error metric below: `(1 + centred cosine) / 2 = 1 − shape_error`. This is a presentation transform, not a new fit or score. JMA similarity is 0.7074 / 0.7118 on 40 storms and 134 starts; USA similarity is 0.7237 / 0.6637 on 40 storms and 131 non-flat starts. Unshifted physical hPa timelines remain visible separately. The public snapshot includes every input-eligible pressure timeline.

Magnitude MAE is retained alongside time-curve diagnostics, without optimizing
time lag, warping time, shifting observations or adjusting forecasts to truth.
These are separate, versioned post-hoc descriptive metrics, not a preregistered
untouched-holdout hypothesis test.

1. Curve-shape error = `(1 - centred cosine similarity) / 2`, lower is better.
   Subtract each curve's common-lead mean, normalize its scalar magnitude and
   keep original time order and sign. At least six common valid points and
   non-flat truth and both forecasts are required. A flat/unavailable curve
   does not receive a perfect score. This measure ignores level and amplitude,
   so it does not replace MAE or demonstrate matching absolute intensity.
2. Trend-direction mismatch = fraction of adjacent six-hour changes whose
   signs differ (increasing / flat / decreasing). Deadband is exactly zero;
   flat-versus-changing is a mismatch. Missing leads break adjacency, never
   form twelve-hour changes disguised as six-hour steps.
3. Six-hour tendency MAE = mean absolute error of those adjacent changes in
   kt/6 h, hPa/6 h or km/6 h. It remains separate from shape error.

All daily curve scores use the same valid model/reference masks before the
storm aggregation. Uncertainty uses 2,000 whole-storm bootstrap replicates
with seed 4712, not overlapping issue resampling. Paired 1.2-minus-1.1
differences and their intervals are included in the full JSON report.

## Results and honest boundaries

USA pressure MAE: 1.1 **12.84** vs 1.2 **12.55 hPa**. JMA pressure MAE:
**13.53** vs **12.84 hPa**. Auxiliary wind diagnostic MAE: **16.98** vs
**25.03 kt**. Wind curve-shape error: **0.271** vs **0.385**. Thus wind is
worse for 1.2 despite the slight mean-pressure improvement. At +120 h USA
pressure is worse for 1.2, whereas JMA pressure is better; no all-lead/agency
win is claimed. Radius 1.1 errors are reported, but native 1.2 is N/A.

The old 1980–1999 cohort overlaps original 1.1 fitting, and 1.2 hindcasts are
retrospective predictions by a model trained on later years. Some recent
storms have been inspected in development. Complete five-day route labels
were required by the inherited cohort. This is not untouched generalization,
an operational comparison or proof of no overfitting. Scientific safety and
wind/radius definition limitations apply to all illustrated examples.

## Files and reproduction

[Verified aggregate and per-storm metrics](../evaluation/intensity/intensity_final.json)
· [Chart verification receipt](../evaluation/intensity/verification.json)
· [GPU runner](../release_tools/benchmark_intensity_v12_v11.py)
· [Independent verifier and plotter](../release_tools/plot_intensity_benchmark.py).

```sh
MPLCONFIGDIR=/Volumes/D/typhoon_predict/.cache/matplotlib \
  /Volumes/D/typhoon_predict/.venv/bin/python -B \
  release_tools/benchmark_intensity_v12_v11.py \
  --project /Volumes/D/typhoon_predict \
  --output /Volumes/D/typhoon_predict/output/intensity-v12-v11-20261001 --chunk 10

MPLCONFIGDIR=/Volumes/D/typhoon_predict/.cache/matplotlib \
  /Volumes/D/typhoon_predict/.venv/bin/python -B \
  release_tools/plot_intensity_benchmark.py \
  --source /Volumes/D/typhoon_predict/output/intensity-v12-v11-20261001 \
  --project /Volumes/D/typhoon_predict \
  --output /Volumes/D/typhoon_predict/remote_typhoon_predict_repo/evaluation/intensity
```

The runner locks one process and resumes immutable completed cases. It never
silently restarts training. Large original forecasts/raw members remain on D;
public reports include exact source hashes and complete verified aggregates.
