# TrackFormer 1.3.3 — direct context pilot (2026-11-03)

This is a new, unvalidated track and pressure experiment. It does not replace
released1.2 or change any rainfall training. The older local experiment named
1.3.3 is a different package and remains untouched.

## Why change 1.3.2

The saved1.3.2 E5 evaluation on24 development storms had mean track error685.65km
(constant motion691.03km) and whole-domain MSLP RMSE8.23hPa. Its saved current-issue
forecast developed broad pressure patches and a looping track. A forecast picture
alone is not verification against observations.

Code inspection found three plausible contributors, not proven causal ablations:

- One smoothed850/700/500hPa wind field transported every weather variable, and
  predicted fields were repeatedly fed back for20 steps.
- A pressure-soft-minimum position adjustment became part of the next motion.
- Dense reanalysis pressure within150km of a valid storm fix was excluded from
  map training, while the separate JMA central-pressure observation pulled on
  the same map. Reanalysis-grid and best-track central pressure are different targets.

## Architecture and objectives

Nine causal six-hour frames (-48..0h),31 fields, explicit masks, native241x321
Western Pacific nodes (0..60N,100..180E). A strided convolutional history encoder,
ConvGRU, multiscale spatial features and four regional attention layers encode
the history once. All20 lead times use that fixed encoding. A shared decoder with
continuous time conditioning predicts a residual relative to the issue field;
there is no recursive weather advection or pressure-position feedback.

A joint head predicts20 six-hour motion increments from issue-time context.
Their cumulative sum forms a continuous trajectory. Each pressure decoder sees
its own predicted centre coordinates. Central pressure is sampled bilinearly
from that exact physical map. This readout identity does not prove that the
forecast centre is a correct cyclone or pressure minimum.

The objective is normalized weather MSE + MSLP MSE +0.5 Gaussian400km-core
MSLP MSE +0.5 track Huber (250km scale) +0.1 motion Huber (100km scale)
+0.1 pressure-gradient MSE. Full original finite masks and cosine-latitude area
weights are retained. Dense pressure supervision includes the storm core.
JMA central pressure is reported as a diagnostic and is not substituted for a
reanalysis map target. Future weather and future track are supervision only.

Width64 (11,844,807 parameters), fresh random initialization, seed1332026, AdamW1e-4, weight decay1e-4,
gradient clipping5, strict FP32 CUDA with TF32 disabled. Eight epochs,186 TRAIN
starts from84 storms, effective batch4/microbatch1:376 optimizer updates.
Every update supervises all20 leads from the first epoch. Decoder gradients
accumulate at detached encoder feature leaves; their sum is then propagated
through the original encoder graph once. This is the exact chain rule, checked
against a joint backward pass, rather than temporal gradient truncation.

## Data, execution and comparison

The fixed cached1.3.2 cohort is reused read-only. No weather download, no new
full-cache audit, no overlap with the24 development storms or calendar windows.
The TRAIN-only normalizer is reused only if its hash/cohort/history times match.
Each frame's values, masks, coordinates and file receipt are checked on its first
actual read. The notebook reuses `/content/trackformer132-data` when available.

Run `python -m unittest discover -p 'test_*.py' -v` for CPU contracts.
Run `python train.py --data DATA --output NEW_OUTPUT --width 64 --epochs 8 --batch 4 --lr 1e-4 --seed 1332026`
on a CUDA runtime. The notebook supplies the exact command. No packages are
installed or upgraded. Every optimizer update and every epoch has a checkpoint
and checksum receipt on Drive. An interruption resumes only with the same
code/data/configuration. There are no memory-percentage or wall-clock kills.

Every epoch saves all24 development starts x20 leads, native pressure maps,
original truth/masks, track, map-derived central pressure, metrics and hashes.
Compare against1.3.2 on identical cases and support, including each lead, the
constant-motion baseline, pressure-map errors and centre/low-pressure alignment.
These reused development cases are not full811, an untouched holdout, or the
fixed11 released1.2 comparison. No automatic promotion is implemented.

## Research context and source attribution

[Pangu-Weather, Bi et al.2023](https://www.nature.com/articles/s41586-023-06185-3)
uses hierarchical temporal aggregation to reduce accumulated forecast error.
That motivates testing fewer recursive dependencies here; this architecture is
not Pangu and does not inherit its reported skill.
[Chattopadhyay et al.](https://arxiv.org/abs/2304.07029) discuss multiscale bias
and unstable error propagation in learned weather dynamics. This supports
checking spatial structure and stability, not a diagnosis of our model by itself.

Data: [ERA5/ECMWF](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels),
[ARCO ERA5](https://github.com/google-research/arco-era5),
[IBTrACS/JMA](https://www.ncei.noaa.gov/products/international-best-track-archive).
The copied1.3.2 data, contracts, acquisition helpers and receipt/validation
infrastructure retain the repository's license and third-party notices.
