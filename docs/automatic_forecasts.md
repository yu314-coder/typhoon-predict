# Automatic 1.2 forecast archive

The existing repository runs the released Python checkpoint on **GitHub-hosted
CPU runners**, not on a visitor's browser or the owner's Mac. Every six hours
(00:17, 06:17, 12:17, 18:17 UTC), the workflow first checks all current in-domain
JMA storms, then processes a bounded historical batch. GitHub may delay schedules;
public-repository schedules can be disabled after 60 days of inactivity.

New outputs are **one deterministic member**, never labelled as 50-member means.
The historical target starts in **1970**. Recent playback takes priority:
**4,015 independent six-hour issues from all 135 eligible Western Pacific storms
in 2022–2026** in the current observed snapshot. Each tick initializes a new
+120-hour run, with that issue's centre, intensity, motion and nine causal analyses.
Fung-wong's 39 consecutive ticks are first for playback verification. These
are not shifted copies of an earlier route and are not selected by forecast error.
Older backfill retains the first supported issue of each uncovered storm.
Out-of-domain global storms are not
silently extrapolated. A storm record is not a promise that usable weather exists.

Nine consecutive six-hour weather analyses end no later than issue time. Live
GFS is f000 only, with its final analysis at most twelve hours before the JMA
issue. Live wind and motion are explicitly missing; native detail is masked.
The resulting GFS transfer is experimental. Historical NOAA NCEP reanalyses are
retrospective, not operationally available inputs; fitting-year overlaps are
labelled and these forecasts are not a fresh independent evaluation.

The input-only archive is checksum-pinned in `release_tools/history_inputs.json`.
Weights and source modules must match the release hashes before inference.
Complete finite +6…+120-hour outputs are saved on the `forecast-data` branch.
Old issues are immutable; new outputs are added, not substituted into old dates.
The live job is deduplicated by storm and exact analysis time. Historical errors
are retried after 24 hours while other storms can continue.

Public files beneath that branch's `data/` directory:

- `catalog.json`: available storms, exact issues, member counts and worker status.
- `status.json`: completion counts, live-issue availability, errors and run URL.
- `coverage.json`: 1970 boundary, domain and input-queue classifications.
- `forecasts/{id}.json`: geographic route, central pressure, provenance.
- `fields/{id}.json.gz`: losslessly compressed genuine model basin-pressure
  arrays in hPa. Existing uncompressed files remain readable. The public API
  transparently returns ordinary JSON for either format.

The Weather Lab API merges these results with its existing verified archive.
The GitHub Pages website can use the same public History API. A cache miss or
source outage remains explicit; no future observation is substituted as forecast.

## Playback and scale

The History player defaults to Western Pacific and orders storms by formation
time (first observed record), newest first. Its issue-time slider advances in
six-hour ticks. Each tick requires an exact matching saved initialization; it
pauses on an uncomputed tick rather than showing a previous issue as a new one.
A separate preview-lead selector chooses the pressure field from that issue.
The original one-issue animation remains available.

The full 1970-onward snapshot contains approximately 65,146 eligible six-hour
ticks across 1,919 WP storm records. Only the requested recent subset is queued
at six-hour frequency; the older full-tick expansion is not launched. Runtime
and compressed output size should be measured on the cloud pilot before
estimating total storage and wall-clock completion. The queued combined plan
contains 5,568 issues (4,015 recent ticks and 1,553 older first issues).
Scheduled batches attempt up to 500 issues or 45 minutes, whichever comes first.
Missing recent NOAA reanalysis may delay individual historical ticks; the
source error is retained and retried after 24 hours, never filled with truth.

The read-only hosted integration run
[36554550793](https://github.com/yu314-coder/typhoon-predict/actions/runs/36554550793)
verified two consecutive Fung-wong initializations: six hours apart, distinct
causal input hashes, distinct routes and distinct pressure arrays, with 20
forecast leads each. The second forecast completed about 1.27 seconds after the
first; the two-run pipeline took 6.5 seconds excluding package setup. This tiny
warm-cache sample is a throughput lower bound, not an ETA. At that rate the
recent cohort needs about 1.4 compute-hours and the full archive about 23 hours,
before weather retrieval, initialization, failures, publishing and scheduler
gaps. At 500 issues per six-hour scheduled batch, even ideal recent completion
takes roughly two days; NOAA availability can make it longer.

A measured basin-field sequence used 44,311 bytes with lossless gzip: about
170 MiB for 4,015 issues, or 2.7 GiB for 65,146, excluding routes, catalogues,
Git history and existing outputs. These are single-sample storage estimates,
not fixed quotas or guaranteed total sizes.
