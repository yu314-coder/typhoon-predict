# Typhoon Predict

Typhoon Predict is a tropical-cyclone track, intensity, and structure forecasting research
project. It is not an operational warning system — do not use it for evacuation, aviation,
maritime, emergency-management, or other safety-critical decisions.

The current best model, **TrackFormer v23**, predicts the atmospheric steering flow that carries a
storm as an explicit intermediate step, conditions that prediction on how the flow has been
evolving over the previous day, and reaches **434.96 km RMS track error** at 6–120 h lead time on a
2020+ Western-Pacific-and-East-Pacific held-out test. This is the result of the v10–v34 research
line below, which is this project's main line of work.

## TrackFormer v23 and the v10–v34 research line

Full write-up, architecture equations, and every intermediate result (including what didn't work)
are in [`paper/trackformer.pdf`](paper/trackformer.pdf), "Chain-of-thought steering and
land-interaction testing." Summary:

1. **A CNN steering-field encoder** (v10–v20) reads a small patch of deep-layer-mean steering wind
   (a weighted blend of 850/500/200 hPa u/v) centered on the storm.
2. **Chain-of-thought (CoT) steering** (v21, v22): rather than regressing track displacement
   directly from an opaque decoder, the model first predicts the ambient steering flow itself as
   an explicit, inspectable intermediate representation, then derives track from it.
3. **A temporal steering-history stack — v23**, the best model in the project: conditions the CoT
   steering estimate on its own history (t-24h, t-12h, and now), not just the current instant, so
   recurvature can respond to how the flow is evolving. **434.96 km** RMS track error, 10-seed
   ensemble, WP+EP 2020+, full 20-lead-horizon test set (3,763 windows, 72% land-covered on the
   WP-only subset).
4. **Four further environmental additions on top of v23 — all null.** An ocean-heat/shear/humidity
   token (v25), an ocean-heat CNN patch on the intensity head (v27), a meridional drift adapter
   whose matched control exposed the apparent gain as retrain noise (v28abl/v28), and — most
   tellingly — feeding raw 0.25° ERA5 steering wind directly on top of v23 (v29): +3.74 km worse,
   every seed above the bar. Once a CoT representation already extracts the steering signal that
   matters, handing the model the raw field again is redundant.
5. **Land-interaction testing — three from-scratch attempts, then a frozen-backbone isolation
   (v31–v34).** Motivated by real-world reports of typhoons stalling at mountainous coastlines
   (Typhoon Gaemi, 2024, at Taiwan) and terrain-deflection literature (AOT-TCNet, arXiv 2603.29200):

   | model | aggregate track (km) | Typhoon Tip 1979 (km) | Typhoon Noul 2026, ocean/landfall (km) |
   |---|---|---|---|
   | v23 (baseline) | **434.96** | 939 | 267 / 356 |
   | v31 — LandDrag, uniform training | 443.07 (+8.11) | — | — |
   | v32 — LandDrag, window-oversampled | 460.48 (+25.52, backfired) | — | — |
   | v33 — LandDrag, storm-normalized | 442.33 (+7.37) | **876** | **243 / 319** |
   | v34 — LandGate, **frozen v23 backbone** | 460.52 (−0.33 vs. its own backbone) | **795** | 287 / 382 |

   v34 is the methodologically important result: v31–v33 each retrained the *entire* architecture
   from scratch, so their deltas vs. v23 include ~19 km/seed of ordinary retrain noise on top of
   whatever the land correction did. v34 instead freezes a real, already-trained v23 checkpoint and
   trains only a new ~437-parameter gated correction — a true same-backbone-plus-one-addition
   comparison. Result: essentially null, everywhere, including the mountainous-near-land regime
   every earlier attempt targeted. On the two real out-of-training storms available, both v33 and
   v34 split 1–1 against v23 — a small-n disagreement with the aggregate test set, not a reliable
   effect.
6. **Methodological lessons that apply beyond this line:** retrain-to-retrain seed noise
   (~19 km/seed) is large enough to manufacture or hide most small version-to-version deltas;
   freezing a real backbone and training only a small addition isolates a causal effect that
   comparing two from-scratch runs cannot; and a large in-distribution aggregate test set does not
   always agree with genuinely out-of-training real-storm validation.

These are research checkpoints, not yet converted to a packaged local-inference format (see
"Earlier released line" below for the checkpoint that *is* packaged for that). Training scripts:
`colab_train_v17.ipynb` (base), `colab_v26_train.py` (CoT), `colab_v28_train.py` (temporal
history), `colab_v36`–`v39_train.py` (v31–v34). Map of every model tested against all six storms:
[`paper/all_storms_v23_v33_v34_mean_map.html`](paper/all_storms_v23_v33_v34_mean_map.html).

## Earlier released line: StormFusion-MT / TrackFormer v1–v9

A predecessor line, fully released with a portable local inference script, summarized in the
paper's appendix and in the [model card](models/README.md). It reached 618 km RMS track error
(WP 2020+) with a protected dual-stream Transformer and a random-matrix uncertainty head, before
the v10–v34 research above superseded it as the project's main line of work. It remains the only
checkpoint packaged for local inference today:

```bash
cd /Volumes/D/typhoon_predict
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_inference.py --checkpoint best.pt --output forecast.json   # --device cpu to force CPU
```

The script uses a mean-normalized atmospheric proxy when no time-aligned ERA5 patch is available.
Full architecture, training configuration, system requirements, model-format notes, and the two
other released checkpoints (`models/stormfusion_v2_era5_3.3M.pt`, `models/trackformer_21M_fp16.pt`)
are documented in the [model card](models/README.md), not duplicated here.

## Data pipeline

Training requires:

- Tropical-cyclone best-track fixes for the target basin
- ERA5 atmospheric variables on a regular grid
- Storm-centered patch extraction
- Time-aligned history/target windows
- Train/validation/test splits separated by time to avoid leakage

ERA5 is produced by the Copernicus Climate Change Service implemented by ECMWF. Users must obtain the required track and ERA5 data under their own access and licensing terms.

## Forecast-error covariance experiment

`covariance_denoise.py` applies the generalized two-point spectral-support method as a post-processing experiment for forecast residuals. It compares raw covariance, diagonal covariance, Ledoit-Wolf shrinkage, classical Marcenko-Pastur/PCA cleaning, and the generalized two-point cleaner.

Residual input must be a matrix with shape `[independent_storm_cases, features]`. Features can be flattened lead-time and target variables, such as track, maximum wind, pressure, and wind-radius errors. Use independent storm cases or storm blocks; do not count overlapping windows or the 50 samples from one neural network as independent validation cases.

Run the mathematical and synthetic smoke test:

```bash
python covariance_denoise.py --demo --a 10 --beta 0.05
```

Run it on a residual matrix saved as `.npy`, `.npz`, `.csv`, or `.txt`:

```bash
python covariance_denoise.py \
  --residuals validation_residuals.npy \
  --a 2.0 \
  --beta 0.25 \
  --output covariance_results.npz
```

The script writes covariance estimates to the `.npz` file and diagnostics to the matching `.json` file. The two-point parameters `a` and `beta` are explicit because the research paper gives the limiting distribution but does not define a validated estimator for them. Estimate them on training storms and validate them on held-out storms. This module does not replace the forecaster and cannot compensate for missing or incorrectly aligned ERA5 input.

## Limitations

- This is a research project, not an operational forecast system. TrackFormer v23 (434.96 km
  RMS track error at 6–120 h) and the earlier released StormFusion-MT/TrackFormer v1–v9 checkpoints
  (~618–730 km depending on version and test split) are both far from operational quality.
- The v10–v34 research line's checkpoints are not yet packaged for local inference; only the
  earlier StormFusion-MT/TrackFormer v1–v9 line has a runnable `run_inference.py` path today.
- Forecast quality depends strongly on correct, time-aligned input fields and track preprocessing.
- Probabilistic spread, where present, is model-generated uncertainty, not a calibrated warning cone.
- Wind-radius labels are sparse; pre-satellite track/intensity labels are lower quality; no
  calibration or comparison against official agency forecasts has been done for any version.

## License and safety

Check the licenses for the source datasets and derived products before redistribution. Do not use this repository for evacuation, aviation, maritime, emergency-management, or other safety-critical decisions.
