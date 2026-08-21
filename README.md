# Trackformer 1.2

Trackformer 1.2 is an experimental, causal western-Pacific tropical-cyclone
forecasting model. It forecasts storm motion, maximum sustained wind, central
pressure, radius of maximum wind, and R34/R50/R64 wind radii over six-hour
leads through +120 h.

This is a research release, not an operational warning system. Do not use it
for evacuation, aviation, maritime, emergency-management, or other
safety-critical decisions.

## Release Highlights

- A whole western-Pacific route state with land, coastline, terrain, and
  ocean-context features.
- A causal intensity and structure head for maximum wind, pressure, RMW, and
  quadrant wind radii.
- Current and lagged atmospheric analysis plus SST context, nearby-storm
  context, and static geography.
- Validation-gated calibration and storm-disjoint chronological evaluation.
- Native PyTorch and joblib artifacts in a GitHub Release and the Hugging Face
  model repository.

The public release name is **Trackformer 1.2**. The internal experiment history
is intentionally not used as the public product name.

## Availability

- [GitHub Release: Trackformer 1.2](https://github.com/yu314-coder/typhoon-predict/releases/tag/trackformer-1.2)
- [Hugging Face model repository](https://huggingface.co/euler314/typhoon-predict)
- [Live interactive demo](https://yu314-coder.github.io/typhoon-tracks.html)

The main branch contains source and documentation. The model weights are
distributed as the release archive rather than committed into the Git history.

## What Changed

Trackformer 1.2 keeps the causal route design and adds a promoted
land/ocean-context intensity candidate. The structure head uses pooled
log-radius groups, lead-wise residual calibration, non-negative radius
constraints, and a validation-selected ocean-availability gate. The route and
intensity components are packaged together so the same release can expose the
forecast track and the surrounding storm structure.

The release deliberately excludes the rejected exploratory multi-variant
ensemble. It is not used to select a favorable result after looking at the
test cases.

## Data Boundary

The model input at forecast issue time is limited to information available at
that time or earlier:

- observed IBTrACS history;
- current, -12 h, and -24 h atmospheric analysis summaries;
- current and lagged sea-surface-temperature summaries;
- causal nearby-storm and basin/global pressure context;
- static land/sea, coastline, terrain, land-cover, and location features.

It does **not** use JMA, JTWC, ECMWF, GFS, GEFS, or another agency's forecast
track or positive-lead weather field as an input. Later observations and
best-track labels are used only after the forecast is frozen for verification.

## Inputs

Trackformer 1.2 keeps the public causal input families used by Trackformer 1.1;
the route context is expanded across the western Pacific and includes the
static land/ocean features needed near coastlines. The public tensor contract
is:

| Input | Shape or content | Boundary |
|---|---|---|
| Observed track and intensity history | `9 x 54` track window | issue time and earlier |
| Current storm-centred analysis | `4 x 17 x 17` | issue-time analysis |
| Older analysis context | `8 x 17 x 17` | -12 h and -24 h analysis |
| Whole-Pacific route state | `3 snapshots x 7 channels x latitude x longitude` | current and lagged analysis |
| SST, nearby systems, and pressure context | causal summaries and spatial features | issue time and earlier |
| Geography | land/sea, coastline, terrain, land-cover, and location features | static |

The data families are observations, analyses, reanalysis summaries, and static
geography. They are not agency forecast products. Missing radius labels are
masked rather than converted to zero.

## Outputs

For each of 20 six-hour leads from +6 h through +120 h, the model returns:

| Output | Shape or unit |
|---|---|
| Track latitude and longitude | `20 x 2`, degrees |
| Maximum sustained wind | knots, 1-minute estimate |
| Central pressure | hPa |
| Radius of maximum wind | kilometres |
| R34, R50, and R64 | four quadrant radii in kilometres |
| Pressure-state route fields | whole western-Pacific grid with pressure contours and steering context |

Radius outputs are non-negative and the structure head enforces R34 >= R50 >=
R64 for each quadrant.

## Held-Out Comparison

The canonical comparison uses the same 100 common storms, the same earliest
available test initialization per storm, the same +6 h through +120 h leads,
and the same issue-time causal boundary. The split remains chronological and
storm-disjoint:

- training: storms through 2015;
- validation: storms from 2016 through 2019;
- test: storms from 2020 onward, held out until scoring.

The earlier README comparison mixed a different release-manifest aggregate with
this matched-storm benchmark. That is why it appeared to show a win in every
category. The chart below is now the canonical comparison; lower error is
better and all values come from the matched 100-storm evaluation.

![Trackformer 1.2 and 1.1 matched 100-storm error comparison](paper/trackformer_1_2_vs_1_1_matched100_mae.png)

The source values and input policy are recorded in
[paper/trackformer_1_2_vs_1_1_matched100_metrics.json](paper/trackformer_1_2_vs_1_1_matched100_metrics.json).

| Metric | Trackformer 1.1 | Trackformer 1.2 | Better in this benchmark |
|---|---:|---:|---|
| Mean track error, all leads (km) | 802.406 | 398.419 | Trackformer 1.2 |
| Mean track error, +120 h (km) | 1,604.259 | 808.607 | Trackformer 1.2 |
| Wind MAE, +120 h (kt) | 20.657 | 18.509 | Trackformer 1.2 |
| Pressure MAE, +120 h (hPa) | 14.932 | 13.637 | Trackformer 1.2 |
| RMW MAE, +120 h (km) | 33.016 | 33.035 | Trackformer 1.1, marginally |
| R34 MAE, +120 h (km) | 69.357 | 76.112 | Trackformer 1.1 |
| Wind MAE, all leads (kt) | 18.121 | 17.638 | Trackformer 1.2 |
| Pressure MAE, all leads (hPa) | 13.044 | 12.755 | Trackformer 1.2 |
| RMW MAE, all leads (km) | 26.396 | 26.996 | Trackformer 1.1 |
| R34 MAE, all leads (km) | 57.253 | 57.079 | Trackformer 1.2 |

This benchmark supports a strong track improvement and lower wind/pressure
error. It does **not** support the claim that Trackformer 1.2 is better on
every radius metric; RMW remains slightly worse and +120 h R34 is worse. Those
limitations are kept visible instead of selecting a more favorable cohort.

## Pressure-Map Diagnostic

The route output is a whole-western-Pacific pressure-state diagnostic with
MSLP contours, 500-hPa height, wind context, and the causal route. It is a
model diagnostic, not an official forecast overlay.

![Trackformer 1.2 causal western-Pacific pressure map](paper/trackformer_1_2_dolphin_pressure_map.png)

The image is included to show the spatial output format. The quantitative
comparison above, not a single case map, is the model benchmark.

## Release-Candidate Metrics

For completeness, the released land/ocean-context head was also compared with
its frozen causal baseline on 9,994 untouched test windows at +120 h. Lower is
better.

| Metric at +120 h | Frozen baseline | Trackformer 1.2 candidate |
|---|---:|---:|
| Maximum wind MAE (kt) | 19.876 | 19.618 |
| Central pressure MAE (hPa) | 13.634 | 13.519 |
| Mean radius MAE (km) | 28.952 | 28.944 |
| R34 MAE (km) | 46.672 | 46.672 |
| R50 MAE (km) | 25.417 | 25.417 |
| R64 MAE (km) | 14.767 | 14.742 |

The candidate improves maximum wind, pressure, mean radius, and R64 at +120 h,
while R34 and R50 are unchanged at the displayed precision. This is a separate
candidate-versus-baseline diagnostic and must not be substituted for the
matched 1.1 comparison above.

## Weight Bundle

The release archive contains:

```text
models/trackformer_1_2/
  manifest.json
  trackformer_1_2_route_seed0.pt
  trackformer_1_2_route_seed1.pt
  trackformer_1_2_intensity_seed0.pt
  trackformer_1_2_intensity_seed1.pt
  trackformer_1_2_ocean_structure.joblib
  trackformer_1_2_feature_stats.npz
```

The bundle uses native PyTorch checkpoints, NumPy statistics, and a joblib
calibration artifact. GGUF is not an appropriate interchange format for this
CNN/Transformer weather model. The public source branch does not yet provide a
stable one-line Trackformer 1.2 inference wrapper; use the manifest and the
feature contract when integrating the native artifacts.

Download the archive from the [Trackformer 1.2 release](https://github.com/yu314-coder/typhoon-predict/releases/tag/trackformer-1.2),
verify its SHA-256 value shown in the release notes, and extract it outside the
Git checkout.

## Historical Releases

[Trackformer 1.1](https://github.com/yu314-coder/typhoon-predict/releases/tag/trackformer-1.1)
remains available as a historical baseline. It is not mixed into the
Trackformer 1.2 weights or ensemble.

## Limitations

Forecast error varies by storm, basin, analysis source, storm structure, land
interaction, and missing-data pattern. Radius labels are incomplete for some
historical storms and are masked during training. The model does not provide
calibrated operational warnings. Use official meteorological agencies for
real-world forecasts and warnings.

## License

Code and compact artifacts are released under the repository MIT license.
Upstream datasets remain subject to their own terms and attribution
requirements.
