# v1.2.64 Combined-Archive Training

## Active Workflow

The same `ai_train.ipynb` has two cells: version information, then fetching and preflight. It uses Google Drive only, installs no packages, and does not start training. The trainer remains `v164_global_regional_train.py` for command compatibility. Its active architecture is `wp-coarse8-native-pressure1-v164`, with 53,435,692 parameters. The old 31-channel global experiment and six-window pilot are not used.

Run in the Colab terminal after setup reports READY:

```sh
cd /content/v164_colab_bundle
python v164_global_regional_train.py 20 3 1
```

Numbers mean effective batch, workers, and CUDA AMP. Physical microbatch is one; workers are capped at two. Twelve epochs use rollout horizons 2, 4, 8, 12, then 20 steps. First two epochs train fields; later epochs add route and intensity objectives. Best selection starts at epoch five with the stable full-horizon objective. Validation is never training data; test is excluded from selection.

## Data Coverage

- Old archives supply 13,949 train, 1,041 validation, and 1,195 test windows with complete six-hour context/target weather coverage inside the Western Pacific domain.
- Train years 2000-2021; validation 2022-2023; test 2024-2025. Whole storms and window boundaries cannot cross splits. Tip 1979 is excluded from all three.
- Eight 2.5-degree context variables: mean sea-level pressure, 500-hPa height, and u/v winds at 850/500/200 hPa. Domain: 0-60 N, 100-180 E. These are old NCEP reanalysis fields, not high-resolution ERA5.
- New native 0.25-degree ERA5 pressure patches provide 800/100/100 detailed windows within those splits. Each patch is 121x121 over 30 degrees, fixed at its observed issue-time center, with nine history and twenty target frames. These 1,000 windows are additional fine supervision, NOT the entire old corpus.
- Static geographic channels: latitude, longitude, land fraction and elevation. Fine pressure is never synthesized by enlarging the old coarse map.
- SST, ONI and fine upper-air variables are not consumed by this configuration; they are not silently substituted or claimed present.

## Causality And Outputs

Inputs are analyses at -48,-42,...,0 hours, static geography, observed issue-time center/intensity and past six-hour motion. Future analysis fields and best-track positions/intensities are labels only. Issue-time retrospective best-track estimates are a historical hindcast assumption, not proof of real-time operational availability. Normalization uses only the training years. No official forecasts or future weather are model inputs.

The model autoregressively predicts eight basin fields and a native pressure patch when native history is available. No future recentering or route-painted vortex is used. Its track head uses generated basin pressure/winds/features. Fine data outside the parent domain receives an explicit missing-coverage mask, not extrapolated fields. Central pressure is a sampled generated field plus a separately learned subgrid correction; maximum wind is a scalar head, not a predicted 10m-wind map. No exact physical conservation is claimed.

The archive-wide training processes every eligible coarse window. Only the detailed subset trains the fine branch. A case lacking native pressure history can run the coarse branch but is not represented as a genuine high-resolution pressure forecast.

## Storage And Resume

Drive folder: `typhoon_predict/data/v164_reuse` contains the manifest and bundle descriptor referencing existing archived data plus `data/v164_pressure_geography` parts. No large archives are duplicated in Drive. Total payload about 2.13 GB. Four independent Drive HTTP clients fetch parts concurrently. Existing verified files are reused; the manifest is published after complete-file checksums pass.

Checkpoints are written directly to `MyDrive/typhoon_predict/training/v164_reuse_pressure/last.pt` and `best.pt`, with history and run manifest. Resume restores the last completed epoch, optimizer, scaler, and RNG; an interrupted epoch restarts. Architecture and dataset hashes must match. A stale lock after runtime death requires checking that no other training process is active before removing it. Pilot checkpoints are incompatible and are not auto-loaded.

## Verification Limits

Mac checks cover all archive hashes and split counts, real coarse and fine samples, tiny-model +120h finite rollouts, backward gradients, completed-epoch checkpoint resume, and full-width two-step MPS gradients. These are software checks, not model-skill validation. CUDA/T4 execution and a complete training run have not been tested; Colab was not accessed.
