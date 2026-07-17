# v9 plan — StormFusion triple-stream + IBTrACS environment + RMT uncertainty (overnight)

## Goal
Make v8 better using ONLY IBTrACS-derivable data (no ERA5, stays deployable). The user asked for
"sea water temp" and "moving direction". Moving direction is already in the 48-feature set
(heading sin/cos, speed, turn-rate). IBTrACS has NO SST, but it has strong environment signals:
absolute position and distance-to-land. So we add an ENVIRONMENTAL feature group + a climatological
SST proxy, and give the model a third protected stream.

## New features (indices 48-53; INPUT_DIM 48 -> 54), all IBTrACS-derived
- 48 lat_signed   (deg)                     -- Coriolis, recurvature latitude
- 49 abs_lat      (|lat|)
- 50 sin_lon, 51 cos_lon                     -- position / basin (wrap-safe)
- 52 dist2land    (km, from IBTrACS DIST2LAND) -- weakening/steering near land
- 53 sst_proxy    -- climatological SST from (lat, month):
      delta = 23.44*sin(2pi(m-3)/12); thermal_lat = 0.5*delta
      sst = clip(30 - 0.30*|lat - thermal_lat|^1.4, 0, 31)
  (Not real SST -- IBTrACS has none -- but a lat+season warmth signal the intensity head can use.)

## Architecture (v9): protected TRIPLE-stream
- Kinematic encoder (motion, heading, speed, turn, season, dt) -> track decoder [track grads only]
- Thermodynamic encoder (vmax, pres, gust, rmw, 12 radii, trends, validity) -> intensity decoder [intensity grads]
- Environmental encoder (lat, |lat|, lon sin/cos, dist2land, sst_proxy) -> feeds BOTH decoders
    (position/dist2land inform track steering-climatology; sst_proxy informs intensity)
- Keep persistence-residual track head + zero-init gated thermo->track adapter.
- Same losses (masked Huber + Gaussian NLL + monotone-radii penalty).
Rationale: v8 is position-BLIND (translation-invariant, only relative motion). Absolute latitude is
the single biggest missing IBTrACS signal (recurvature/Coriolis/SST-latitude). This is the
IBTrACS-only stand-in for the ERA5 steering stream I designed earlier.

## RMT usage
After training, fit the generalized-MP-cleaned 40-dim track residual covariance (the Yau paper's
two-point edge) as the calibrated ensemble/uncertainty head -- already validated to give the best
energy score (26.5% better than deterministic). Use it for the storm tests.

## Train / test
Train on MPS (Mac GPU), monitor. Then test on Bavi (2026), Wayne (1986), Co-may (2025); compare v9 vs v8.

## Honest expectation
Absolute position + SST proxy should help INTENSITY and typical-storm track. It will NOT fix erratic
loopers (Wayne/Co-may) -- those need real-time steering winds, not climatology. Measure it truthfully.
