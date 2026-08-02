# Typhoon Predict

Typhoon Predict is a tropical-cyclone track and intensity forecasting research project. It is
**not** an operational warning system — do not use it for evacuation, aviation, maritime,
emergency-management, or other safety-critical decisions.

## v62 causal western-Pacific model

**v62 is the latest route model in this repository.** It builds a broad causal state over
100-190E and 0-60N, covering China, Japan, Taiwan, the Philippines, and the western Pacific,
then integrates a route from the evolving pressure and flow field. The full route is a weighted
blend of a local multi-level analysis route (75%) and the broad Pacific route (25%). Its analysis
channels are MSLP plus 500 hPa height and 850/500/200 hPa wind. The pressure panels show dense
2 hPa MSLP contours, 500 hPa height contours and wind barbs, 850 hPa wind-speed structure, and
200 hPa jet-speed contours.

The state forecast is a bounded extrapolation of the current, t-12 h, and t-24 h analysis
tendency. It is deliberately **not** an imported numerical-weather-model forecast: no positive-
lead weather field, official forecast track, or post-issue observation is passed to inference.
Official JMA/JTWC tracks, where displayed on older comparison pages, are overlays only.

### GitHub Pages and Actions data boundary

GitHub Pages is a static host. It can serve the generated HTML, JSON, PNG, and compact data files,
but it cannot execute Python or provide a server-side API. A practical deployment is:

1. A scheduled or manually dispatched GitHub Actions workflow downloads public analysis files,
   validates that every timestamp is at or before the issue time, and converts them to a compact
   cache.
2. The workflow runs the causal v62 builder and publishes the resulting JSON/PNG/HTML as a Pages
   artifact.
3. The Pages frontend only reads the committed/deployed result. It does not fetch or use an
   official forecast product in the browser.

The current v62 comparison uses sources that an Actions runner can retrieve without a CDS key:

- [NOAA NOMADS GFS/GDAS](https://nomads.ncep.noaa.gov/) `f000` analysis for Dolphin.
- [NOAA CFSR archive](https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system)
  analysis for the historical Tip replay.
- [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) for the
  observed track history and, for Tip, post-issue verification truth only.
- [NOAA daily OISST](https://www.ncei.noaa.gov/products/optimum-interpolation-sst) is a viable
  future public SST input, but it is **not** injected into the current v62 route because the
  released v23/v62 input contract was not trained with an SST channel.

GFS/GEFS forecast fields are intentionally excluded from the v62 inference contract even though
they are publicly downloadable. Raw GRIB files and large caches are not committed; the generated
maps and manifests are. Do not put API keys or personal credentials in a workflow or notebook.

### Same-source v23 versus v62 comparison

The comparison runner uses the same issue-time track history and the same public analysis source
for all routes. v23 receives its released four-channel 17x17 patch; v62 receives the same source
over the full Pacific domain and the local multi-level cache. “Full v62” means the current 75%+
25% local/Pacific blend. v23 does not produce a full pressure-grid forecast, so the pressure map
uses v62's causal pressure state with the v23 line overlaid.

| Tip replay route | Mean track error | +120 h error |
|---|---:|---:|
| v23 same-source 17x17 patch | 581.478 km | 1,388.589 km |
| v62 Pacific-domain only | 388.355 km | 659.530 km |
| **v62 full local + Pacific** | **141.207 km** | **36.971 km** |

This is one 1979 Tip hindcast, not a claim of operational skill. Dolphin is a live issue-time
render with no post-issue truth in the local archive, so it is shown as a route comparison only.
The original v23 test result in the table below is a separate storm-held-out archive metric and
must not be mixed with this single-case replay.

Open the generated artifacts:

- [Dolphin v23/v62 interactive map](paper/dolphin_v62_v23_public_data_world_map.html)
- [Dolphin causal pressure map](paper/dolphin_v62_v23_public_data_pressure_map.png)
- [Tip v23/v62 interactive map](paper/tip_v62_v23_public_data_world_map.html)
- [Tip causal pressure map](paper/tip_v62_v23_public_data_pressure_map.png)
- [Dolphin comparison manifest](track_build/dolphin_v62_v23_public_data.json)
- [Tip comparison manifest](track_build/tip_v62_v23_public_data.json)

Re-run the comparison locally after the public analysis caches are present:

```bash
./.venv/bin/python scripts/run_public_data_v23_v62_comparison.py
```

The source implementation is [`v62_pacific_domain_route.py`](v62_pacific_domain_route.py), with
the case adapter in [`scripts/run_v62_pacific_domain_cases.py`](scripts/run_v62_pacific_domain_cases.py)
and the v23/v62 comparison in
[`scripts/run_public_data_v23_v62_comparison.py`](scripts/run_public_data_v23_v62_comparison.py).

### v62 minimum requirements

- Python 3.10+, PyTorch, NumPy, Matplotlib, xarray, `cfgrib`, and ecCodes for GRIB decoding.
- CPU inference is supported; no GPU is required for the case renderer. A current Mac with 8 GB
  RAM is the practical minimum, while 16 GB RAM and 10 GB free disk make GRIB decoding safer.
- GitHub Actions needs a Linux runner with Python and enough temporary disk for the raw analysis
  files. Pages viewers only need a modern browser; the PNG pressure maps remain viewable without
  JavaScript map tiles.
- Training the v23 checkpoint ensemble is a separate, much larger GPU workload. The v62 case
  runner is inference/evaluation code and does not retrain the model.

This repo is a minimal release: trained weights, the scripts that trained them, the v62 causal
case renderer, and the paper.
Everything else (research/analysis tooling, exploratory scripts, intermediate versions) lives in
project history and isn't part of this release.

## Historical v37G local forecast candidate

The v37G candidate is a historical local research forecast path built from the current observed
storm history, current JMA analysis positions, and cached public NOAA GFS/GEFS forecast fields.
The route is an adaptive multi-level vortex-track ensemble; the structure branch predicts wind,
central pressure, RMW, and R34/R50/R64 quadrant radii from storm-centered fields. Its pressure
maps are diagnostic reconstructions, not learned global pressure-grid forecasts.

**Source boundary:** JMA and JTWC official *forecast positions* are never passed to the v37 route,
structure model, gate, or training targets. JMA analysis positions may initialize the current
history. Official JMA/JTWC products are loaded only after inference for a clearly labeled visual
comparison and valid-time error table. The included Dolphin page is one verification case, not a
general skill claim.

Run locally after supplying the public caches and v37G checkpoints:

```bash
V37_MODEL_VERSION=v37G \
V37_ROUTE_POLICY=gfs_gefs_adaptive \
.venv/bin/python v37_protected_forecast.py
.venv/bin/python scripts/check_v37_source_policy.py
```

The rendered current-case map is [`paper/dolphin_v37_current_ibtracs_jma_official_world_map.html`](paper/dolphin_v37_current_ibtracs_jma_official_world_map.html).
It draws the v37 output in red and labels JMA/JTWC lines as comparison-only. Raw GRIB caches and
large intermediate datasets are intentionally excluded from GitHub.

### v37G minimum local requirements

- Python 3.10 or newer, PyTorch, NumPy, Matplotlib, and the packages in `requirements.txt`.
- A CPU or Apple Silicon Mac is supported; CUDA is optional. Expect roughly 8 GB RAM for the
  included inference path and additional disk space for downloaded GFS/GEFS caches.
- The included v37G spatial checkpoints are inference weights. Re-training needs substantially
  more RAM/VRAM and the prepared training windows described in `v37/V37G_DESIGN.md`.

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
