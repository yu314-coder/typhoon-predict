# v37G Local Forecast Design

This document records the current local-only v37 path. The saved comparison
baseline is **v36A**, not v6.

## Forecast path

1. **Track route:** one deterministic NOAA GFS vortex track plus 31 GEFS
   member vortex tracks. The GEFS members use a weighted 850/500/200 hPa
   wind field and the route is the prediction-level mean: 50% deterministic
   GFS and 50% GEFS-member mean. The 850 hPa circulation remains the center
   anchor; 500 and 200 hPa tracks provide a limited deep-layer correction.
2. **Intensity/structure:** three independently trained v37G spatial experts
   read the nine-step track history and a four-channel storm-centered field
   tensor. They predict maximum wind, central pressure, RMW, and the R34/R50/
   R64 quadrant radii. The current wind and pressure are used as residual
   anchors.
3. **Post-processing:** coefficients are fitted on the 2016-2019 validation
   storms only. A lead-dependent current-wind blend and a joint pressure
   consistency layer are evaluated on the untouched 2020+ test storms before
   being used in inference.
4. **Visualization:** the pressure panels are a diagnostic Holland-like
   reconstruction from predicted center pressure, wind, RMW, and radii. They
   are not a learned global pressure grid. The companion GFS synoptic chart
   uses the cached public GFS MSLP field and weighted 850/500/200 hPa winds
   as a physical background, with v37G and v36A overlaid.

## Free data path

The current forecast uses cached public NOAA NOMADS GFS/GEFS GRIB files and
local IBTrACS/JMA-derived analysis history. No CDS credential, API key, or
Colab runtime is required. The route cache is created by:

- `scripts/fetch_dolphin_gfs_forecast_no_api.py`
- `scripts/fetch_dolphin_gefs_ensemble_no_api.py`
- `scripts/fetch_dolphin_gfs_prmsl_no_api.py`

The official comparison captures are saved locally from JMA and JTWC. They
are used for evaluation and display, not as future inputs to the route.

## Local held-out evidence

The v37G spatial experts use 67,283 train, 6,873 validation, and 9,994 test
windows with storm-level year splits. On the test split, the validation-only
wind blend changes 120-hour wind MAE from 20.668 to 20.547 kt. The joint
pressure layer changes clipped 120-hour pressure MAE from 16.937 raw to
14.605 hPa. These are research metrics, not operational skill claims.

The separate v37H persistence-anchored neural route was trained on MPS. It
improved over persistence but scored 1,042.32 km at 120 h on the held-out
test, so it is deliberately not blended into v37G. The existing v23 route
benchmark remains stronger for the learned-route comparison.

## Dolphin verification

Issue time: `2026-07-31T12:00:00Z`.

- v37G +120 h: `25.5202N, 134.9746E`, `76.03 kt`, `974.57 hPa`.
- JMA same-valid-time route error at +120 h: `3.4 km`.
- JTWC same-valid-time route error at +120 h: `72.83 km`.
- v36A is drawn as the saved reference line in the map.

JTWC wind is 1-minute sustained and JMA wind is 10-minute sustained; the
model output is labeled as a 1-minute estimate. Pressure and intensity
conventions therefore must not be compared as if they were identical. One
Dolphin case does not establish generalized JMA-level skill, and the current
pressure estimate remains a known limitation.

## Artifacts

- `paper/dolphin_v37g_world_map.html`
- `paper/dolphin_v37g_world_map.png`
- `paper/dolphin_v37g_pressure_maps.png`
- `paper/dolphin_v37g_gfs_synoptic.png`
- `paper/dolphin_v37g_intensity.png`
- `track_build/dolphin_v37g_forecast.json`
- `track_build/dolphin_v37g_gfs_synoptic_manifest.json`
- `v37/structure_spatial/v37g_structure_spatial_manifest.json`
- `v37/structure_spatial/v37g_intensity_calibration.json`
- `v37/route_persistence/v37h_route_manifest.json`
