# Matched WeatherNext Cyclones Mini daily benchmark

## Model and frozen plan

The reference is Google's official [WeatherNext Cyclones Mini](https://github.com/google-deepmind/weathernext), the released `<2024` checkpoint trained through 2023—not a custom imitation or full-sized WeatherNext. It runs one seeded member using the existing JAX CPU runtime on the Mac. This is a new benchmark; scores remain pending until saved forecasts and the full verification receipt exist.

- Cohort: **1,473 daily starts / 270 Western Pacific storms**, identical to the frozen Trackformer 1.1/1.2 route comparison.
- Forecast: twenty six-hour steps, **+6 to +120 h**, with the exact observed issue-time centre as the initialized-storm anchor.
- Inputs: exact ERA5 analyses at **issue−6 h and issue**, calendar forcings only after issue. No future weather, storm labels or official forecast tracks enter the worker.
- Weights SHA-256: `a1bb151457077248d70a458b1a7b19deacd2926fbcb46eee4ff2691444e3596e`.
- Cohort SHA-256: `965184f1e5ab3fcb52e304f39b38583a14f753870aa2295f2d01942a441dcfd6`.
- Official code commit: `89c4b2a77a1c57b328b909c575550fd2e5aadc9c`.

## Like-for-like scoring

All three models use the frozen starts, exact future label times and original issue-relative kilometre projection. Routes are unshifted. Direction is recomputed on **common moving steps across truth and all three models**, not compared using different masks.

Pressure uses the original 134 eligible native-intensity starts / 40 storms, restricted to common valid leads across three predictions and the selected reference. JMA and USA remain separate. MAE and centred time-curve similarity are separate; missing, nonphysical or failed outputs are not zeros. Flat or shorter-than-six-point curves have unavailable shape similarity.

Average valid leads within an issue, days within a storm, then storms equally. Partial aggregates contain only fully completed storms and explicit coverage. Recent 2024+ and historical 1980–1999 groups are separate because the latter overlap WeatherNext fitting years. Different input pipelines and the one-member Mini versus 50-input-member 1.2 policy make this an output comparison, not an equal-compute architecture ablation or a certified unused test.

The official six-hour direct tracker uses a frozen initialized-storm continuity policy: no cyclogenesis, no dissipation pruning and no nearby-cyclone pruning. This is disclosed rather than called the operational default. Native 1° WP MSLP grids are saved in physical hPa; cyclone-head central pressure is not the basin-grid minimum.

## Execution and verification

The [resumable runner](../release_tools/deepmind_daily_benchmark.py) freezes source hashes, the cohort, original baseline forecast hashes and checkpoint identity. Workers save exact input-state hashes, twenty route points, central-pressure values/masks and twenty native WP grids. Future truth is read only **after** inference for scoring; Trackformer forecasts and weights are unchanged.

Local output: `/Volumes/D/typhoon_predict/output/deepmind-daily-20261002`. `progress.json` records on-disk coverage; `logs/` holds diagnostics. `verification.json` is produced only after all 1,473 cases pass the final audit. These are local results, not automatically published Site API data. No score is uploaded before review.

Existing Mac runtime and weights are reused without package installation. Earlier CPU rollouts suggest **several days** for the full run; retrieval and compilation affect the total. All generated states, caches and outputs stay on `/Volumes/D`. Cloud backfill, training and Site deployments are untouched.

```bash
# Existing compatible WeatherNext environment and project assets required.
python release_tools/deepmind_daily_benchmark.py prepare \
  --project /Volumes/D/typhoon_predict \
  --output /Volumes/D/typhoon_predict/output/deepmind-daily-20261002

python release_tools/deepmind_daily_benchmark.py launch \
  --project /Volumes/D/typhoon_predict \
  --output /Volumes/D/typhoon_predict/output/deepmind-daily-20261002
```

`prepare` is only for a fresh directory. `launch` refuses an active run; `run` resumes completed cases without repetition. Source retrieval has bounded retries and failures remain visible. Final completion requires all planned cases, exact leads, valid routes/fields, unchanged forecast hashes and the verification receipt—not a process exit or count alone.
