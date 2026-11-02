# TrackFormer 1.3.2: transport and environmental steering

This is a new **2026-11-02 track and pressure experiment**, separate from released
1.2, historical 1.3 experiments and rainfall V52. It is not a demonstrated upgrade
until evaluated. Open `TrackFormer_1_3_2_Train.ipynb` in Colab and run its three cells.
It uses the already prepared 1.3.1 Drive cache; it downloads no new weather data.

## What the completed 1.3.1 pilot showed

On the same24 reused development starts and all20 six-hour leads, E4 mean track
error was697.91km; extending the initial observed motion gives691.03km. E4 pressure
map RMSE was5.76hPa, worse than E1's5.43hPa. These observations motivate testing
motion and weather dynamics, not merely increasing network size. Error growth or
source inspection alone does not establish the causal contribution of each change.

## Architecture and training

- 11,291,105 trainable parameters at width64, versus2,252,513 in1.3.1. Four regional
  attention layers, native-resolution convolutional skips, recurrent history.
- Nine issue-time-and-earlier frames:31 physical channels on241×321 quarter-degree
  nodes spanning0–60N,100–180E. Every forecast lead uses its own predicted weather.
- A spatially varying six-hour semi-Lagrangian transport prior and learned weather
  residual evolve the map. Transport is approximate, not a mass-conserving solver;
  border values are held and learned residuals must represent missing dynamics.
- Track steering uses masked850/700/500hPa winds averaged150–600km around the prior
  predicted centre, rather than one point inside the vortex. A learned residual
  sees local and surrounding features, both motion components and pressure context.
- Predicted track is associated on the same predicted MSLP map. Central pressure is
  bilinearly read from that exact map. This readout identity is not proof of an
  accurate cyclone or correct intensity; track support currently means in-domain.
- Weather MSE plus Huber track, displacement and JMA pressure losses, and MSLP
  gradient loss. Weights1,.3,.3,.5,.05 respectively. A150km inner-core ERA5/JMA
  pressure distinction is inherited from1.3.1; full-map validation includes the core.
- 186 TRAIN starts from84 storms reuse existing frames; all training29-frame
  windows are disjoint from the fixed24 development windows and storm IDs.
  This is a cache-conditioned pilot, not random full-corpus sampling.
- Eight epochs, effective batch4 (microbatch1),376 optimizer updates; horizon
  curriculum5,10,15,20,20,20,20,20; all development evaluations use20 leads.
- AdamW1e-4, weight decay1e-4, seed1322026, clip norm5, truncated backward length2;
  fresh initialization, strictFP32 CUDA, TF32 disabled. TRAIN-history-only
  normalization. No future observations, GFS inputs or old model weights.
- Each completed optimizer update saves model, optimizer and RNG to Drive. Rerun
  the same configuration to resume; changing the configuration needs a new run.
  No arbitrary RAM-percentage or elapsed-time stop. Colab/OS can still terminate.

## Evaluation and limitations

Each epoch saves24 native-grid prediction/truth archives and a hash receipt with
all+6..+120h leads, track errors, JMA central pressure error and area-weighted map
MAE/RMSE. Persistence and constant-motion scores use the same track labels. Review
unsupported outputs as well as mean errors. Do not treat training loss as skill.
These are reused development cases, not an untouched holdout or full811 evaluation.
A matched released1.2 comparison and fixed11 gate remain outstanding. No automatic
replacement of1.2, no rainfall training, and no local trainer changes are involved.

`prepare_colab.py` checks the small existing cohort identity, selects complete
cached windows and stages frames on the VM. It does not run another global raw-data
audit. `Data.frame` checks a frame's receipt and value hash on its first actual read.
Only cached observations and public best-track labels are used; no private data.

## Research and data sources

[TC-Next](https://arxiv.org/html/2609.02085v1) motivates learning from surrounding
forecast weather and observed motion. Its study concerns6–24h forecasts and includes
satellite imagery; this experiment does not reproduce that architecture or claim its
results at120h. [WeatherNext Cyclones](https://www.nature.com/articles/s41586-026-10953-2)
jointly models atmosphere and cyclone targets. Our transport prior and annular
steering are engineering hypotheses requiring ablation, not published skill claims.

Weather: [Copernicus/ECMWF ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels)
through [ARCO ERA5](https://github.com/google-research/arco-era5). Track and pressure:
[NOAA IBTrACS/JMA](https://www.ncei.noaa.gov/products/international-best-track-archive).
Code follows the repository MIT license; data retain upstream attribution/terms.
