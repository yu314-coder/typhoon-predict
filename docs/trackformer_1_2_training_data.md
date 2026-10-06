# Trackformer 1.2 training data

The **released 1.2 model's fitting cutoff is 2021**, not 2025 or 2026. Its selected checkpoint is development version 1.2.73, epoch 4. The dataset and normalization identities are frozen in the [release manifest](../models/trackformer_1_2_field/manifest.json). This document describes that release, not later 1.3 training or live-input downloads.

## Sources actually consumed by the released training loader

| Component | Dataset | Variables and sampling |
| --- | --- | --- |
| Storm observations / supervised labels | [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)-derived `track_windows_v13.npz` | Position, prior motion, available maximum wind and central pressure, exact storm/time identities and label masks. Only eligible Western Pacific-domain windows enter this release. |
| Basin weather | [NOAA PSL NCEP/NCAR Reanalysis 1](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html), `basin_all_int8.npz` and the exact-time SLP atlas | MSLP, 500 hPa geopotential height, zonal/meridional winds at 850/500/200 hPa. Six-hour analyses on 100–180°E, 0–60°N; 2.5° grid, 25 × 33 cells. |
| Native regional pressure | [ARCO-ERA5](https://github.com/google-research/arco-era5), `pressure_hpa.npy` with its frozen `plan.json` | Mean sea-level pressure on 0.25° fixed issue-centred 121 × 121 patches. 1,000 original windows across all three partitions, not native detail for every coarse window. |
| Static geography | Same ARCO-ERA5 source, `geography.npz` | Land fraction and surface geopotential / 9.80665 to obtain elevation in metres; latitude/longitude encodings. Static context is not an additional temporal fitting period. |

The eight weather channels are **MSLP, HGT500, U850, V850, U500, V500, U200, V200**. No satellite cloud imagery, humidity/vapour, SST or ocean-heat-content channel is present in this released input contract. Such features in other experiments must not be attributed to 1.2.

The selected dataset manifest identifies the **derived IBTrACS archive by SHA-256**, not by an independently recorded raw IBTrACS release/download ID. A newer downloaded IBTrACS file elsewhere is not evidence of which raw snapshot built these frozen windows. Available wind labels and the learned auxiliary head do not establish a harmonized agency averaging period or a validated surface-wind/radius forecast.

## Temporal partitions

| Role | Whole-storm years | Eligible windows / storms | Native-pressure windows | First → last eligible UTC issue |
| --- | --- | --- | ---: | --- |
| Fitting and normalization | 2000–2021 | 13,949 / 611 | 800 | 2000-05-05 18:00 → 2021-12-20 12:00 |
| Validation / checkpoint selection | 2022–2023 | 1,041 / 46 | 100 | 2022-03-31 00:00 → 2023-12-17 18:00 |
| Original test / evaluation | 2024–2025 | 1,195 / 56 | 100 | 2024-05-22 00:00 → 2025-12-02 12:00 |

These are loader-eligible window counts, not independent storms or the number of gradient updates. Native-pressure windows are a **subset** of the corresponding eligible windows, not additional cases. Storm membership uses each storm's first year in the frozen archive. Windows whose −48 h history or +120 h targets cross a partition boundary are excluded. Weather normalization uses the 2000–2021 basin frames only.

Every forecast history comprises nine consecutive six-hour analyses at **−48, −42, …, 0 h**. Supervised fields and storm labels at **+6, +12, …, +120 h** are targets only; they cannot be inference inputs. Missing native detail and missing intensity labels retain their masks. Validation is not gradient fitting, but it does inform checkpoint selection. Previously inspected original tests and showcases are development evidence, not a new unused holdout.

## Newer data do not change the fitting cutoff

The checksum-pinned caches contain records beyond the fitting years:

- Derived storm-window archive: **1980-01-03 00 UTC to 2026-07-13 00 UTC**.
- Coarse NCEP weather archive: **1980-01-01 00 UTC to 2026-03-17 18 UTC**.
- Original native-pressure plan: fitting, validation and test windows in **2000–2025**, separated as above.

Those are **source-cache endpoints**, not years all used to fit 1.2. The 2026 live/backfill pipeline uses the unchanged released model with later causal inputs; GFS transfer is experimental, and generating a new forecast does not retrain the weights. Retrospective reanalysis availability is not proof that the same inputs were available operationally at each historical issue time.

## Verified identity and reproduction

The [training-data receipt](../evaluation/training_data/trackformer_1_2_provenance.json) rechecks all six derived dataset file hashes, the released dataset-manifest hash, exact split counts, storm separation and native-patch membership. It records the original cache endpoints and exact earliest histories/latest targets. This is not a new raw-source-download audit.

| Frozen identity | SHA-256 |
| --- | --- |
| Dataset manifest | `04fd0d30620137aad20173dcd59e22da2a6c20fae2f17de593e91a59517459c1` |
| Selected training checkpoint | `f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0` |
| Inference-only weights | `db49f36e85a3766defc4c172746897a1f783705d1ce8e6f9dfb8e87ae1d902cb` |

With the original frozen dataset available, reproduce the small receipt without inference, downloads or refreshing the training cache:

```bash
python release_tools/audit_training_data.py --dataset-root /Volumes/D/typhoon_predict/data/v164_reuse --output /Volumes/D/typhoon_predict/remote_typhoon_predict_repo/evaluation/training_data/trackformer_1_2_provenance.json
```

The original full training arrays are not newly bundled by this documentation update. Model weights, neural source, forecasts and training processes are unchanged.
