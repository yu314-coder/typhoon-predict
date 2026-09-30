# Any-storm wind and radius plots

This is a local diagnostic/plotting extension. It does not change the released
Trackformer 1.2 weights, existing pressure maps, videos, cloud archive, or Site.
It is **not** a new validated wind/radius forecast model.

## What works

- The renderer accepts any exact SID in the local Western Pacific IBTrACS file.
  The October 1 local snapshot contains 1,910 storms since 1970. This is an
  observational coverage catalogue, **not** a claim that 1,910 forecasts ran.
- Every storm can produce a chart layout. Pressure, wind, RMW, and quadrant
  plots are drawn where reports exist. An empty panel explains missing data.
- Forecast plotting uses genuine, hashed physical member archives. All four
  examples now have 20 candidate wind and R34/R50/R64 quadrant outputs through
  +120 h, diagnosed from all 50 members before taking the mean.
- The historical MPS runner additionally accepts arbitrary `--case SLUG SID
  ISSUE_UTC`. It still requires nine exact six-hour analyses in the existing
  local atlas and the released WP domain. It neither downloads missing weather
  nor changes the independently authorized cloud backfill.

## Candidate assumptions and official definitions

IBTrACS `USA_WIND` is a 1-minute reference. USA R34/R50/R64 are maximum extents
in the NE/SE/SW/NW quadrants, in nautical miles; multiply by 1.852 for **radius**
in km, never by two. JMA wind is a separate 10-minute reference. JMA R30 cannot
be relabeled R34, and longest/shortest JMA radii are not USA quadrant radii.
Blank values remain missing; a native reported zero remains zero. USA RMW is
not universally reanalyzed best track, especially in WP. Fung-Wong's selected
2025 USA reports are provisional TCVitals, not final JTWC observations.
See the [NOAA IBTrACS column documentation](https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf).

The new experimental extension starts with only the **exact issue-time** USA
wind, RMW and available quadrant isotachs. No future wind/radius is fitted.
The neural runner itself continues to use its original JMA issue-intensity
inputs; USA structure is a separate diagnostic input, not a silent neural
input substitution. Reference and model centers can differ, and both are
recorded in the initialization receipt.

For each forecast member, ambient pressure is the median actual basin cells
600–1000 km from that member's predicted center, with at least two available
cells in each quadrant. This uses coarse pressure for the **ambient value**,
not to claim a resolved eyewall. Let `D0 = Pambient0 - Pmodel_issue` and
`Dm = Pambient_member - Pcentral_member`. The candidate uses

```text
wind_member = USA_wind_issue * sqrt(Dm / D0)
```

The square-root deficit scaling is motivated by gradient/Holland balance,
but this extension is **not PaHM/GAHM**. It builds a decreasing outer envelope
from consistent issue-time quadrant isotachs, interpolates in log-radius and
log-wind, and extrapolates with a Holland-inspired `V ~ r^(-B/2)` tail. A fixed
literature-motivated B range 1–2.5 is used, without showcase tuning. Radius
solutions beyond 1,500 km remain unavailable rather than becoming a grid-edge
radius. Nonphysical pressure, unresolved backgrounds, out-of-domain tracks and
inconsistent zero isotachs remain explicit failures.
The [NOAA PaHM theory](https://noaa-ocs-modeling.github.io/PaHM/html/models.html)
also explains why gradient wind is not automatically a validated 10-m wind.

RMW is currently **issue-time persistence**. The flat dashed RMW curve is not
a model-measured future RMW, an eye diameter, or proof of radius skill.
The candidate's nominal threshold/units/quadrant geometry target the native
USA definition, but its 1-minute surface-wind averaging behavior has not been
validated. The strict scorer therefore rejects wind/radius skill claims.
Plots are exploratory reference comparisons, not a fair validated benchmark.
All four are development examples; no unused generalization holdout is claimed.

## Reusable commands

Use the existing Python environment; no installation is needed. All outputs,
temporary files and Matplotlib caches must stay on `/Volumes/D`.

Draw a full observed storm, even without a model-member archive:

```sh
/Volumes/D/typhoon_predict/.venv/bin/python -B \
  /Volumes/D/typhoon_predict/remote_typhoon_predict_repo/release_tools/plot_storm_wind_radius.py \
  --ibtracs /Volumes/D/typhoon_predict/data/ibtracs/ibtracs.WP.list.v04r01.csv \
  --output /Volumes/D/typhoon_predict/output/nari_observed_plots \
  --observed-storm 2001248N23125
```

Generate the data-coverage catalogue with `--catalogue-only`. To draw a saved
model ensemble, pass `--forecast-folder` (repeatable), pointing to one immutable
folder containing `verification.json`, `*_video.json`, `*_video.npz`,
`ensemble-members.npz`, `weather-history.npz` and the original input receipt.

Create an explicit additional historical ensemble on the Mac GPU only when
the required nine local analyses and exact issue report exist:

```sh
/Volumes/D/typhoon_predict/.venv/bin/python -B \
  /Volumes/D/typhoon_predict/remote_typhoon_predict_repo/release_tools/forecast_historical_video50_mac.py \
  --project /Volumes/D/typhoon_predict \
  --output /Volumes/D/typhoon_predict/output/nari_ensemble50 \
  --case nari 2001248N23125 2001-09-08T00:00:00Z --chunk 5
```

SID/date in a command is an explicit requested issue, not a nearest-date fill.
If unavailable, choose an actually reported issue after checking the catalogue.
The runner refuses duplicate issues, unsafe slugs, three-hour dates, changed
completed slugs and unsupported causal input histories.

## October 1 verification

Local results: `/Volumes/D/typhoon_predict/output/all-storm-wind-radius.uSvmWY/plots`.
Fung-Wong needed a fresh 50-member MPS replay (14.33 seconds inference). Its
inputs/output differ from the older benchmark video and it is clearly labeled;
the original video/data were not overwritten. Soudelor, Mangkhut and Meranti
reuse the exact saved original member archives. All mean routes and common-grid
pressure fields are rechecked against their saved video arrays. No inference
or fitting is required for the three archived issues or for observed-only plots.

Before promotion: develop a genuine evolving RMW/outer structure model, fit any
calibration only on declared whole-storm training data, freeze a genuinely unused
storm cohort, validate surface-wind period/height and quadrant definitions, and
report errors plus failed/missing coverage for identical leads through +120 h.
