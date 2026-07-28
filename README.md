# Typhoon Predict

Typhoon Predict is a tropical-cyclone track, intensity, and structure forecasting research
project. It is not an operational warning system — do not use it for evacuation, aviation,
maritime, emergency-management, or other safety-critical decisions.

The project has gone through two research lines. The current best model, **TrackFormer v23**,
predicts the atmospheric steering flow that carries a storm as an explicit intermediate step,
conditions that prediction on how the flow has been evolving over the previous day, and reaches
**434.96 km RMS track error** at 6–120 h lead time on a 2020+ Western-Pacific-and-East-Pacific
held-out test. An earlier, fully released and locally runnable line (StormFusion-MT / TrackFormer
v1–v9) is also included, with a portable inference script and trained checkpoints.

## Current best model: TrackFormer v23 and the v10–v34 research line

Full write-up, architecture equations, and every intermediate result (including what didn't work)
are in [`paper/trackformer.pdf`](paper/trackformer.pdf), "Phase II: chain-of-thought steering and
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

These are research checkpoints, not yet converted to a packaged local-inference format (see the
next section for the checkpoint that *is* packaged for that). Training scripts:
`colab_train_v17.ipynb` (base), `colab_v26_train.py` (CoT), `colab_v28_train.py` (temporal
history), `colab_v36`–`v39_train.py` (v31–v34). Map of every model tested against all six storms:
[`paper/all_storms_v23_v33_v34_mean_map.html`](paper/all_storms_v23_v33_v34_mean_map.html).

## Earlier released line: StormFusion-MT / TrackFormer v1–v9

The original released checkpoints, with a portable local inference script. This is the line
described by the rest of this README (local inference, system requirements, model formats) and by
[the model card](models/README.md) — it predates the v10–v34 research above but is the only line
packaged for hands-on use today.

### Model

An ensemble neural network with three parts:

1. A convolutional field encoder processes the ERA5 patch. It uses convolution, GELU activations, batch normalization, strided downsampling, and global average pooling.
2. A bidirectional GRU encodes the recent cyclone track history.
3. A fusion MLP combines the atmospheric and track embeddings. Separate heads predict the output mean and log scale, while a latent projection adds correlated ensemble variation.

The checkpoint uses four historical track steps and predicts seven lead times: 6, 12, 24, 48, 72, 96, and 120 hours. Each output step contains latitude and longitude displacement plus additional track variables used by the training target.

### Training configuration

The included checkpoint was trained with:

- Western Pacific basin data (`WP`)
- ERA5 data beginning in 1979
- 8 degree atmospheric patches at 0.5 degree resolution
- Four historical track steps
- 1,024 maximum training windows
- Batch size 64
- Learning rate `2e-4`
- Weight decay `1e-4`
- Up to 80 epochs with early stopping patience 12
- 50 ensemble members at inference time
- Random seed 42

The replacement StormFusion-MT retraining workflow is provided in [typhoon_stormfusion_mt_colab.ipynb](typhoon_stormfusion_mt_colab.ipynb). It downloads IBTrACS, requests real ERA5 patches through the CDS API, builds storm-level train/validation/test windows, trains on A100/H100 with BF16, and saves checkpoints and artifacts to Google Drive.

The checkpoint stores the model weights, training configuration, feature scalers, and ERA5 normalization statistics required for inference.

### Local inference

```bash
cd /Volumes/D/typhoon_predict
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_inference.py --checkpoint best.pt --output forecast.json
```

The script automatically selects Apple MPS on Apple Silicon, CUDA when added to the script for a compatible environment, or CPU otherwise. Use `--device cpu` to force CPU execution.

The current script is a demonstration inference path. It uses a mean-normalized atmospheric input when a matching live ERA5 patch is unavailable. For real research evaluation, replace that input with a correctly time-aligned ERA5 patch using the same variables, grid, normalization statistics, and storm-centered coordinates used during training.

### Minimum system requirements

For local inference with the included checkpoint:

- Operating system: macOS, Linux, or Windows
- Python: 3.10 or newer
- Processor: 64-bit CPU with four cores or more
- Memory: 8 GB RAM minimum; 16 GB recommended
- Storage: 2 GB free space for the repository, virtual environment, checkpoint, and generated forecasts
- GPU: not required; CPU inference is supported. Apple Silicon can use MPS when available.

For faster experimentation, training, or batch inference, use a CUDA-capable NVIDIA GPU with at least 16 GB VRAM. An A100 or H100 is suitable for the full training workflow, but is not required to run the released checkpoint.

### Output and visualization

The inference script writes JSON containing the ensemble mean and percentile bounds. `forecast_world_map.html` is an example Leaflet visualization of one generated result; it is not part of the model architecture and can be replaced with a generic forecast viewer.

### Model formats

The supported format is the original PyTorch checkpoint (`best.pt`). GGUF is intended mainly for llama.cpp-compatible language and tensor models; it is not a compatible runtime format for this custom convolutional encoder, GRU, and probabilistic ensemble head. Converting it to GGUF would not make it runnable in llama.cpp.

For production deployment, use PyTorch on CPU, Apple MPS, or CUDA. An ONNX or TorchScript export could be added for a fixed deterministic member, but it must preserve preprocessing, scaler state, output decoding, and ensemble sampling behavior and should be validated against the PyTorch implementation.

### Other released checkpoints

Two more trained checkpoints are released in [`models/`](models/) (Git LFS), with full details in
the [model card](models/README.md):

- **`models/stormfusion_v2_era5_3.3M.pt`** — StormFusion-MT v2 (3.3M params, ERA5 + track).
- **`models/trackformer_21M_fp16.pt`** — TrackFormer (21M params, fp16 ~43MB, **track-only, no ERA5**), trained on
  84,150 all-basin IBTrACS windows.

On a WP-2020+ held-out test, the **track-only** model matches or beats the ERA5 model on every
metric (track 720 vs 729 km; vmax 22.1 vs 24.2 kt; RMW 12.9 vs 16.2 km) — evidence that ERA5
patches add little over past-track history plus more storms. See [MODEL_COMPARISON](models/README.md#results--wp-2020-held-out-test-lower-is-better).

Reproducible pipeline: `build_windows.py` / `build_track_only.py` (dataset), `fix_windows.py`
(NaN-fill + normalization), `model_v2.py` / `train_track.py` (models + training), `eval_compare.py`.

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
  RMS track error at 6–120 h) and the released StormFusion-MT/TrackFormer v1–v9 checkpoints
  (~618–730 km depending on version and test split) are both far from operational quality.
- The v10–v34 research line's checkpoints are not yet packaged for local inference; only the
  earlier StormFusion-MT/TrackFormer v1–v9 line has a runnable `run_inference.py` path today.
- Forecast quality depends strongly on correct, time-aligned input fields and track preprocessing.
- Probabilistic spread, where present, is model-generated uncertainty, not a calibrated warning cone.
- Wind-radius labels are sparse; pre-satellite track/intensity labels are lower quality; no
  calibration or comparison against official agency forecasts has been done for any version.

## License and safety

Check the licenses for the source datasets and derived products before redistribution. Do not use this repository for evacuation, aviation, maritime, emergency-management, or other safety-critical decisions.
