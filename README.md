# Typhoon Predict

Typhoon Predict is a tropical-cyclone track and intensity forecasting research project. It is
**not** an operational warning system — do not use it for evacuation, aviation, maritime,
emergency-management, or other safety-critical decisions.

This repo is a minimal release: trained weights, the scripts that trained them, and the paper.
Everything else (research/analysis tooling, exploratory scripts, intermediate versions) lives in
project history and isn't part of this release.

## Contents

- **`models/v10/`** — the track-only baseline. `trackformer_v10_17M_fp16.pt` (17M params, fp16,
  inference-only) trained by `train_track_v10.py`. Reads only a storm's own recent
  position/wind/pressure history plus IBTrACS-derived kinematic/thermodynamic features — no
  atmospheric steering data at all.
- **`models/v23/`** — the best model in the project: **434.96 km RMS track error** (10-seed
  ensemble, WP+EP 2020+, full 20-lead-horizon test set). `v23_seed0.pt`–`v23_seed9.pt` are the
  10-seed ensemble (fp32, with optimizer/training metadata). Trained by `colab_v28_train.py`,
  which builds on `colab_v26_train.py` (defines the chain-of-thought steering class it extends)
  and, further back, the base data pipeline and model in **`colab_train_v17.ipynb`** at the repo
  root — run the notebook first in the same session/runtime, then the two `.py` scripts in order
  (v26, then v28), per the `!wget`/`exec()` recipe in each script's own docstring.
- **`colab_train_v17.ipynb`** — the one foundational notebook kept in this repo: data pipeline,
  base `TrackFormerV17` model, and the training loop every later Colab-script version (v18 onward)
  builds on top of within the same runtime.
- **`paper/`** — the full write-up (`trackformer.pdf`, `trackformer.tex`) plus every generated
  results map/chart referenced from it.
- **`LICENSE`** — MIT.

## What's *not* here

This is training code and raw checkpoints, not a packaged inference CLI — loading a checkpoint
requires instantiating the matching model class from its training script and calling
`load_state_dict` on the `"model"` key yourself; there's no bundled preprocessing/normalization
wrapper in this release. `requirements.txt` lists what the training scripts need
(`torch`, `numpy`, plus a couple of analysis-only extras).

## Results

| model | aggregate track (km) | notes |
|---|---|---|
| v10 | — (see `paper/trackformer.pdf`) | track-only, no steering data |
| **v23** | **434.96** (10-seed ensemble) | chain-of-thought steering + temporal history — best model in the project |

Full write-up, architecture equations, every intermediate result (including the land-interaction
line of experiments that came back null, and why), and the complete v10–v34 progression are in
[`paper/trackformer.pdf`](paper/trackformer.pdf), "Chain-of-thought steering and land-interaction
testing."
