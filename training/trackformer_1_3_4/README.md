# TrackFormer 1.3.4: evolving weather and shared storm motion

This is a new regional pressure/track experiment, not a released forecast model.
It targets the observed 1.3.3 defect: in the saved 2026-11-02 06Z example, E6's
pressure maps changed by only 0.178 hPa mean absolute difference between +6 and
+120 h, while its independently predicted track moved far from the stationary
pressure minimum. That example motivates this experiment; it is not a test set
on which the new model is fitted or selected.

## Forecast calculation

Nine causal six-hour frames (48 hours, 31 channels, native 241×321 grid, 0–60°N,
100–180°E) initialize a convolutional recurrent history encoder. Every subsequent
step reads the last two **predicted** weather states, their six-hour tendency,
the recurrent memory, initial masks, coordinates, lead time and previous own
storm motion. No future analysis or official forecast is used as an input.

A multiscale U-Net with three transformer blocks over 8×10 context tokens predicts
a dense forward displacement in east/north km per six hours. The displacement is
bounded to ±400 km per component. The same field transports the weather maps and
the recurrent memory, and advances the storm centre by bilinear sampling at its
current position. A native-grid residual predicts physical weather changes after
transport. It is scaled by the RMS six-hour changes fitted on TRAIN history pairs.
Central pressure is read directly from the resulting MSLP map at the new centre.

Transport uses two fixed-point inverse steps and bilinear sampling. The learned
displacement is **pattern motion, not atmospheric wind**. It does not make this
network a physical numerical weather solver. Residual tendencies can alter or
erase a low, so shared motion alone does not prove storm/map agreement. Validation
therefore measures pressure changes, the predicted centre's distance from the
local pressure minimum, map errors, track errors and persistence baselines.
No displayed smoothing, forced circular depression, minimum snapping, independent
track head or excluded storm core is used.

## Training and losses

The cached cohort has 186 TRAIN starts from 84 storms and 24 fixed development
starts from separate storms. The unchanged cohort generator also excludes
overlapping weather windows. Original finite-value, below-ground and humidity
masks remain in use. Mean/state scales and tendency scales use TRAIN histories
only. This experiment adds no new weather download and does not use rainfall data.

Defaults: width 64, eight epochs, effective batch four (serial accumulation),
AdamW learning rate 1e-4, weight decay 1e-4, gradient norm clip 5, seed 1342026,
FP32 CUDA with TF32 disabled. Training horizons are 2, 4, 8, 12, 20, 20, 20, 20
six-hour steps. All development epochs always use 20 leads (+6 through +120 h).
There are 47 optimizer updates per epoch, 376 in total.

Autoregression is free-running: future targets are read **after** each prediction
and enter only the loss. Backpropagation is truncated after every four steps;
states are detached but never replaced with truth. Gradients are exact within
each segment, not across the full 20-step sequence. Activation recomputation
reduces GPU memory. The objective is the sum of:

| Term | Weight |
|---|---:|
| Area-weighted normalized 31-field MSE | 1 |
| Dense normalized MSLP MSE, including storm core | 1 |
| Gaussian 250 km storm-core MSLP MSE | 1 |
| Six-hour tendency error, all fields plus MSLP | 0.5 |
| 600×600 km storm-centred pressure-patch error | 0.5 |
| Track SmoothL1 error, normalized by 250 km | 0.5 |
| Motion SmoothL1 error, normalized by 100 km/6 h | 0.2 |
| MSLP spatial-gradient MSE | 0.1 |
| Displacement spatial-gradient penalty | 0.002 |

Patch supervision compares pressure around the predicted and observed centres;
the observed centre is not supplied to the forecast. JMA scalar central pressure
is reported separately and is not forced into dense ERA5 MSLP targets. Their
different resolution and observation definitions must not be conflated.

## Run, durability and interpretation

Open `TrackFormer_1_3_4_Train.ipynb` in Colab with a T4. Run the four cells for
source, cached data, the disposable CUDA check, and production. Existing CUDA PyTorch is
retained. The small source package is pinned by commit and SHA256. Existing
cached frames are checked on actual first use, without a separate full-cache scan.

The new run writes to `MyDrive/TrackFormer/runs/134-evolving-transport-20261103-v1`.
Production starts from fresh weights and a fresh optimizer, separately from the
disposable check. Checkpoint zero and every optimizer update are committed to
Drive with a SHA256 receipt. Epoch checkpoints are retained. Resume only with
identical source, configuration and data; unfinished optimizer work is replayed.
There is a per-output process lock, no automatic restart, no memory-percentage
kill and no wall-clock cutoff. Colab disconnection, device failures or actual
resource exhaustion remain possible. Other runs are never signalled or changed.

Each epoch saves all24 development cases, original masks, pressure truth and
predictions, all20 leads, tracks, metrics and SHA256 receipts. These are reused
development examples, not full811 validation or an untouched holdout. Comparisons
against released1.2 must use matched cases, inputs, grid and support and are not
claimed by this training launch. Lower training loss or moving maps do not prove
accuracy. Released1.2 stays retained; there is no automatic promotion.

CPU tests cover transport direction, pressure/centre sampling, causal separation,
masked NaNs, recomputed gradients, optimizer/RNG recovery and all24/all20 export
receipts. A moving-low fixture tests whether the implementation can learn motion;
it is synthetic and is not evidence of real-world forecasting skill.

## Research context

[GraphCast](https://arxiv.org/abs/2212.12794) provides the relevant principle of
autoregressive six-hour weather forecasts using recent states. This much smaller
regional U-Net/transformer and its shared learned transport are custom hypotheses,
not a GraphCast reproduction or a claim of comparable skill. The unchanged
ARCO-ERA5 and JMA source lineage, units and catalogue hashes are documented in
`source_contract.py`, `contracts.py`, and the cohort/data receipts. Repository
license and upstream data terms continue to apply.
