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

## Evaluation

The evaluation is storm-disjoint and chronological:

- training: storms through 2015;
- validation: storms from 2016 through 2019;
- test: storms from 2020 onward, held out until scoring.

The table below reports the untouched test cohort at +120 h. Lower is better.
The baseline is the frozen causal Trackformer 1.2 route/intensity baseline;
the candidate is the released Trackformer 1.2 land/ocean-context head.

| Metric at +120 h | Frozen baseline | Trackformer 1.2 candidate |
|---|---:|---:|
| Maximum wind MAE (kt) | 19.876 | 19.618 |
| Central pressure MAE (hPa) | 13.634 | 13.519 |
| Mean radius MAE (km) | 28.952 | 28.944 |
| R34 MAE (km) | 46.672 | 46.672 |
| R50 MAE (km) | 25.417 | 25.417 |
| R64 MAE (km) | 14.767 | 14.742 |

The test set contains 9,994 windows. The candidate improves maximum wind,
pressure, mean radius, and R64 at +120 h, while R34 and R50 are unchanged at
the displayed precision. These are research diagnostics, not a claim of
universal operational superiority.

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
