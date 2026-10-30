# TrackFormer v1.2.64: field-first atmosphere and cyclone redesign

Date: 2026-09-13. Status: investigated proposal, NOT a trained or validated v164 model.
No Colab interaction and no training were performed. Diagnostics ran on this Mac's MPS GPU.

## Decision

Change the state representation, atmospheric training coverage, and field/track contract before scaling the current architecture. Do not merely widen v163, remove its tracker, or add more geometric route losses.

The recommended v164 prototype is a global-to-regional, multi-variable atmospheric predictor with a storm-object decoder tied to its generated fields. It must genuinely forecast pressure, not move or paint a pressure template onto a separately predicted route. A stochastic residual extension should follow a verified deterministic baseline, not obscure an input bug.

## What the Tip investigation actually established

Case: 1979-10-12 00 UTC, nine analyses from t-48 through t0, 20 six-hour forecast steps. Checkpoint metadata says epoch 7. The model has **27,179,604 parameters**, despite the `20m` configuration/file label.

Evidence files:

- `diagnostics/v164_tip_audit/report.json`: actual baseline and two controlled inference ablations.
- `diagnostics/v164_tip_audit/baseline.npz`: unmodified checkpoint tensor outputs.
- `diagnostics/v164_tip_audit/grib_verification.json`: issue-time channel checks against original NOAA CFSR GRIB.
- `audit_tip_v163_for_v164.py`: repeatable local diagnostic, with labels excluded from forward inputs.

### Confirmed data/display problems

1. The demo plotted six-hour truth movements as if they were cumulative displacements. All 20 corrected coordinates now match the archived time-tagged realized route within 0.0001 degrees. The true +120 location is 21.5 N, 128.1 E, not approximately 17 N, 138 E. This was a renderer defect, not proof that the training labels were wrong: training already accumulates the step targets.
2. The forecast renderer used predicted latitude to convert eastward displacement, whereas model field sampling uses an issue-latitude tangent plane. The renderer now uses the model's transform. The training step-coordinate versus fixed-grid geometry remains an architectural consistency issue to remove in v164 using one tested spherical coordinate API.
3. The displayed v1.1 and v1.2.26 legacy routes match exactly at every lead. The v1.2.26 renderer imports a legacy v62 artifact, not a freshly identified v1.2.26 checkpoint. They are not two independently verified model replays. A provenance warning was added; this audit does not certify their independent skill or modify the global benchmark.
4. Tip has no SST patch in this demo, so it receives zeros. The training loader filters on `sst_ok`; it does not train this exact missing-data condition. This is a confirmed train/inference contract gap. Its causal contribution requires a matched-source SST replay, not an invented warm ocean patch.

### Checks that passed

- All forward inputs are finite. Weather ranges are plausible: pressure about 972-1031 hPa over history; height about 5116-5922 gpm; winds in m/s. There is no demonstrated Pa/hPa or height/geopotential factor error.
- The nine weather timestamps stop exactly at the issue time.
- All eight issue-time map channels were checked against the original GRIB analysis. Pressure differs by at most 0.000059 hPa after float conversion; height and wind differences are zero on the sampled native grid. GRIB time is the requested issue and lead is zero. Older snapshots were timestamp-checked, not all independently re-decoded here.
- The issue-pressure mask is zero. The grid-pressure fallback therefore follows the training behavior. Missing pressure is not evidence of an incorrectly supplied 870-hPa observation.
- ONI is intentionally masked in both training and this replay until release-dated vintages exist. An empty ONI dictionary is not an accidental missing feature for this checkpoint.

### Measured model failure

These are forecast-structure diagnostics, not future-field accuracy scores. Core minimum means the lowest grid value within 500 km of the predicted center; the ring is 500-1000 km. That region contains only about 9-13 native grid points in the core.

| Lead | Whole-grid minimum, hPa | Core minimum, hPa | Ring mean minus core minimum, hPa |
| --- | ---: | ---: | ---: |
| 0 | 980.425 | 980.425 | 24.993 |
| 24 | 983.234 | 983.234 | 20.906 |
| 48 | 987.474 | 987.474 | 16.002 |
| 72 | 992.260 | 994.568 | 9.411 |
| 120 | 999.023 | 1008.427 | 2.562 |

The generated route ends near 34.30 N, 139.66 E. The scalar central-pressure head says 992.23 hPa, but the field near that route has no comparably deep resolved low. Some difference between a subgrid scalar and coarse reanalysis is legitimate; the combination of lost core contrast, weakening gradients, and diverging route/field location is the issue.

After correcting the displayed truth and projection, descriptive Tip mean geodesic route error is **677.7 km**, with mean heading error **54.6 degrees**. The stored DeepMind route has 140.1 km mean error on those leads, but uses a different analysis pipeline and this historical case is not established as held out for its pretrained weights. Do not call that a controlled generalization benchmark.

Controlled MPS ablations (same checkpoint/inputs, local-distance approximation for this diagnostic):

| Variant | Mean route error, km | Interpretation |
| --- | ---: | --- |
| Original model | 678.1 | Baseline |
| Learned acceleration forced to zero | 694.9 | Does not repair the failure |
| Field-tracker correction disabled | 1106.4 | Tracker helps on this case |

The baseline mean tracker correction is about 21.3 km per step. These interventions do NOT isolate every possible cause, but disprove the easy claim that deleting the tracker or acceleration is the demonstrated solution. They are diagnostics, not new tuned models.

### What remains unproved

- A future, matched-source Tip atmospheric verification sequence was not evaluated here; no whole-map Tip MAE claim is made. Some real weakening is expected, so simply requiring a deep low forever would be wrong.
- CFSR versus the training analysis source is a distribution shift. Finite/range checks do not eliminate it; compare source-matched replays and normalized distributions on many storms.
- This one storm cannot prove overfitting, an optimal parameter count, or guaranteed improvements from any proposed architecture.

## Why another v163-style network is unlikely to address the main issue

The actual code uses a 25 x 33, 2.5-degree grid with eight variables: MSLP, height at 500 hPa, and winds at 850/500/200 hPa. A multi-scale encoder cannot recover missing fine-scale or thermodynamic information from that grid. The 0.5-degree source already has a relatively weak resolved low (minimum 976.435 hPa); sampling to 2.5 degrees weakens it further to 980.425 hPa. These are grid values, not observed cyclone central pressure.

It predicts bounded pointwise state increments on a fixed regional grid without a global boundary forecast. The two pressure update branches permit at most about 3.52 hPa per six-hour step at a cell (0.50 times the checkpoint pressure standard deviation). This is a potential translation/gradient limitation, not proof that saturation is the observed cause.

It lacks a prognostic humidity/temperature column and evolving ocean coupling. The current physics loss matches derivatives, vorticity, divergence and shear against targets; it is not a numerical conservation-law solver. Increasing that weight cannot supply missing thermodynamics or guarantee physical pressure evolution.

The scalar intensity head, route proposal and field tracker can disagree. Numerous route losses coexist with comparatively small field objectives. Their numerical weights alone do not establish gradient dominance: measure shared-layer gradient norms and conflict before reweighting.

Training uses storm windows that have 120-hour targets and valid SST. That removes many partial lifecycle samples and does not teach general atmospheric evolution from every available analysis time. Parameter scaling leaves these problems intact.

## Proposed architecture and data contract

### 1. Atmospheric state, not pressure-only images

Use a real global 1-degree parent state and a genuine 0.25-degree western-Pacific regional state with a surrounding halo. Start with MSLP, surface pressure, 10-m winds, 2-m temperature, SST, and u/v/T/q/geopotential at 850, 700, 500, 300 and 200 hPa. Include land fraction, orography, solar geometry, availability and age channels. Add ocean heat-content/mixed-layer information only when a causal, consistently covered archive exists.

Use physical units and one canonical regridding/normalization implementation for training, replay and benchmarking. A larger interpolated copy of the current 2.5-degree labels is NOT a high-resolution dataset. Reanalysis does not fully resolve the eye even at 0.25 degrees; retain a separately identified observation-scale intensity output.

Train from 2000 onward on all eligible atmospheric timestamps, including non-storm periods, with predeclared storm-disjoint/year splits. Older years can supplement training only when sources and masks are consistent, and Tip remains excluded. No future reanalysis, best-track values or operational forecast product may enter a rollout input. A global parent generated internally from t0 is a prediction, not a future observation.

### 2. Global-to-regional atmospheric dynamics backbone

Use a spherical multi-resolution graph or spherical neural-operator backbone for the global state, with vertically coupled tokens. It forecasts evolving ridges, troughs and steering winds, not just a global pooled context vector. A regional windowed-attention/ConvNeXt operator consumes the parent forecast, its own prior predicted fields, and lateral boundary state at each step. Decode all atmospheric variables jointly.

Start with supervised six-hour state evolution and multi-step training. A differentiable transport operator can be tested for prognostic quantities with a learned physical-process residual, but do not treat MSLP as a passive scalar or claim conservation from a simple image warp. If implementing a full dynamics solver, use an established tested solver rather than an unverified hand-written approximation.

Do not require a moving cyclone crop for the first prototype: a fixed regional field avoids using true future centers to choose inputs and avoids route-driven template placement. Predicted-center local attention is allowed, but cannot move the output field to force agreement.

### 3. One coupled storm-object and field representation

Maintain a persistent storm-identity token initialized from the observed issue-time storm. Cross-attend to generated pressure, vorticity, wind and warm-core features. Forecast a spatial association distribution over the generated field, retaining multiple candidate lows where necessary. The final route is a subgrid center associated with a generated meteorological object, not an independent trajectory plus a pasted low.

Use a motion prior only as a soft association aid. Avoid a global minimum tracker, and avoid an expectation between two distinct lows. Model cyclone existence, weakening, transition and association confidence explicitly; do not force a tropical vortex to survive after genuine dissipation.

The pressure field remains the atmospheric decoder's raw output. Grid-scale pressure, observation-scale central pressure, and maximum surface wind have different contracts. Train the subgrid intensity decoder against available observations and calibrate its relation to the resolved vortex. Do not force central pressure to equal a coarse cell or claim a head offset creates a resolved eye.

### 4. Conditional probabilistic extension, after the deterministic baseline

Add a spatially coherent stochastic residual to the atmospheric evolution, shared across variables and time, using a conditional flow or FGN-style formulation. Each ensemble member must contain its own full field, storm association, track and intensity. The purpose is to represent uncertain storm positions without averaging several locations into a diffuse low.

Keep the deterministic baseline for ablation. Score members/distributions properly; never choose the best-looking member against future truth. In the demo distinguish a fixed-seed member from the ensemble mean. A smooth ensemble-mean field is not itself proof of a bad probabilistic forecast.

### 5. Training stages and sizes

1. Pretrain atmospheric next-state prediction on non-storm and storm times, then train genuinely free-running 48/120-hour rollouts with predicted lateral boundaries.
2. Add the storm-object and intensity decoders and joint training. Use field accuracy/structure and storm association losses with logged gradient scales, not ten overlapping route terms by default.
3. Add stochastic evolution only if deterministic members lose structure under uncertainty, and evaluate calibration as well as sharpness.

Initial trainable target: **40-60M total**, for example a roughly 25-30M global core plus 15-25M regional/object components. These are architecture budgets, not measured parameter counts or confirmed T4 fits. Test the same architecture at approximately 25M, 50M and 100M under matched data/optimization before attributing gains to size. A large global dataset can require substantially more training than previous storm-only runs.

Practical alternative: freeze an existing pretrained global weather backbone and train only the regional/object refinement. This may be the fastest skillful system, but label it a hybrid and disclose its pretrained-data coverage, license and full parameter count. Do not present its accuracy as a standalone TrackFormer win over that same parent. No parent product or remote service should be silently introduced.

## Promotion gates before calling v164 better

- Shared input tests: exact timestamps, field units, channel order, coordinate round trip, longitude wrapping, missingness, normalization and source checksums. Delete future-label tensors from the forward interface and verify predictions are unchanged when verification labels are shuffled.
- Missing SST: either fetch a verified causal patch or use a masked encoder trained with modality dropout. No unmasked all-zero fallback on an always-present training feature.
- Genuine field skill: matched-grid area-weighted MSLP MAE/RMSE plus anomaly and gradient error, spectral diagnostics, cyclone-centered pressure contrast and radial profiles, low-center displacement, and co-location with vorticity/winds. Evaluate core and background separately.
- Route: per-step direction, turning/recurvature timing, cross-track and along-track error, and full-path mean distance. Do not substitute a single terminal-distance score or optimize only appearance.
- Intensity: separate observed central-pressure/wind errors from gridded MSLP error; keep wind averaging conventions explicit.
- Multi-storm free rollouts: tropical, recurving, landfalling and dissipating cases; report failure rate and confidence. A persistence baseline must be included.
- Frozen validation selection, storm-clustered bootstrap uncertainty, and a new untouched test cohort. Tip is now a development diagnostic due repeated inspection; it cannot remain an untouched test merely because it is excluded from gradient training. The repeatedly used 270 cases likewise require an independent final test.
- Gates must not be tuned to force the DeepMind ranking. Compare the same initialization/verification sources and disclose pretrained train/test overlap where known.

## Research basis and limits

- [GraphCast](https://arxiv.org/abs/2212.12794): useful basis for multi-scale global atmospheric forecasting rather than a storm-only latent backbone.
- [WeatherNext Cyclones, official explanation](https://deepmind.google/blog/weathernext-ai-model-achieves-breakthrough-in-forecasting-cyclones/): combines global atmospheric prediction with cyclone observations and coherent ensembles. Its reported Mini performance also shows that resolution alone is not a sufficient explanation for our failure. Adopting the design principles does not reproduce its skill or training corpus.
- [StormCast](https://arxiv.org/abs/2408.10958): supports conditioning regional probabilistic evolution on synoptic variables and evaluating realistic spectra. Its short-range kilometer-scale experiments are not proof of 120-hour western-Pacific cyclone skill.
- [NeuralGCM](https://www.nature.com/articles/s41586-024-07744-y): supports investigating tested hybrid dynamics and learned physics, not adding an arbitrary pressure-balance penalty and calling it physically guaranteed.

Bottom line: the defensible v164 change is stronger atmospheric evolution plus a genuinely coupled storm representation, with verified data and evaluation contracts. Tip supports investigating this direction; it does not prove that more parameters or a particular new block will win.
