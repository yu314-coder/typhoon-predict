# Automatic 1.2 forecast archive

The existing repository runs the released Python checkpoint on **GitHub-hosted
CPU runners**, not on a visitor's browser or the owner's Mac. Every hour at
minute 17 (UTC and Asia/Taipei), the workflow checks all current in-domain
JMA storms. Lightweight backup checks at minutes 32, 47 and 57 start the same
writer only when the last published live check is at least one hour old.
The gate skips active or queued archive writers; the shared non-cancelling
publication lock remains on the inference job. Scheduled runs use
`historical_limit=0`; historical batches remain explicit or existing continuations.
GitHub may delay or drop schedules; this is not an exact wall-clock guarantee.
Public-repository schedules can be disabled after 60 days of inactivity.

The existing `automatic-forecasts.yml` schedule is the only live inference
scheduler; no second Sites or Mac automation is required. A new, validated JMA
analysis produces a new +120-hour issue and a separate **50-member live ensemble**
with genuine common-grid pressure means. Already completed issue IDs are skipped.
An hourly check does not imply that JMA or the six-hour GFS analysis grid supplies
new input every hour; unavailable sources remain explicit. The Live and History
views follow the latest completed
issue unless the viewer deliberately selects an archive. Updates continue with
the browser and Mac closed. The Live page polls every minute while visible and
again on focus or reconnect; History retains its five-minute poll. `status.json` exposes the last check, requested
3600-second interval, ensemble member count, source errors and cloud run URL.

Historical backfill now continues in **back-to-back serialized batches**: after
successfully publishing a productive batch, the workflow immediately dispatches
the next batch if uncomputed issues are ready. The same concurrency group permits
only one runner at a time. There can still be GitHub queue and environment setup
gaps; this is not a permanently running server. The hourly schedule remains
for live updates and delayed source retries. Continuation stops when all work is
complete, only cooling-down failures remain, a batch makes no successful progress,
or inference/publishing fails. A targeted single-issue retry does not start a chain.
No model, input boundary, membership, or forecast values are changed by this policy.

Historical outputs are **one deterministic member**, never labelled as the
separate 50-member live means. Recovery shares the same publication lock and
checks for overdue live input at an hourly threshold before starting a batch.
The historical target starts in **1970**. Recent playback takes priority:
**4,015 independent six-hour issues from all 135 eligible Western Pacific storms
in 2022–2026** in the current observed snapshot. Each tick initializes a new
+120-hour run, with that issue's centre, intensity, motion and nine causal analyses.
Fung-wong's 39 consecutive ticks are first for playback verification. These
are not shifted copies of an earlier route and are not selected by forecast error.
The completed first phase is preserved. The next phase adds **26,575 six-hour
issues for 1996–2021**; 2022 is already covered and is not rerun. The combined
queue now contains **32,143 issues**, including **30,590 six-hour ticks** and
1,553 preserved older first issues. All 5,568 original issue definitions and
completed forecasts remain unchanged. Older backfill retains the first
supported issue of each uncovered storm.
Out-of-domain global storms are not
silently extrapolated. A storm record is not a promise that usable weather exists.

Nine consecutive six-hour weather analyses end no later than issue time. Live
GFS is f000 only, with its final analysis at most twelve hours before the JMA
issue. Live sustained wind is read from JMA's analysis in knots when available;
missing wind remains missing. Motion remains explicitly missing; native detail
is masked. Initialization uses the analysis valid time, not bulletin publication
time, and rejects future or stale analyses. JMA's future forecast points are
comparison data only, never model inputs.
The resulting GFS transfer is experimental. Historical NOAA NCEP reanalyses are
retrospective, not operationally available inputs; fitting-year overlaps are
labelled and these forecasts are not a fresh independent evaluation.

NOAA **ended NCEP/NCAR Reanalysis 1 at March 17, 2026**
([source notice](https://psl.noaa.gov/data/reanalysis/)). Later historical issues
now use nine exact six-hour **archived GFS f000 analyses** from NOAA's public
archive. This is explicitly labelled an experimental GFS-input transfer, not
the original R1 evaluation or a claim of operational availability at the issue.
The complete history window uses one provider; R1 and GFS are not mixed within
an initialization. No later-valid analysis, forecast step, nearest-date fill,
or future observation can replace a missing analysis. Each new archive forecast
records the nine source URLs, valid times and GRIB-subset SHA-256 hashes.
Only the eight required GRIB messages are fetched with checked byte ranges;
timestamps, levels, units and missing values are checked again after decoding.
New analyses are cached across cloud batches to avoid re-downloading overlapping
history windows.

The input-only archive is checksum-pinned in `release_tools/history_inputs.json`.
Weights and source modules must match the release hashes before inference.
Complete finite +6…+120-hour outputs are saved on the `forecast-data` branch.
Old issues are immutable; new outputs are added, not substituted into old dates.
The live job is deduplicated by storm and exact analysis time. Historical errors
are retried after 24 hours while other storms can continue. A versioned input
reader correction releases old failures for one immediate retry; a new failure
then backs off normally. A manual exact-ID retry also bypasses the cooldown,
without starting an automatic batch chain. Status separately reports unresolved
planned-issue errors; obsolete issue IDs from earlier queues are preserved under
`retired_source_errors`, not presented as unresolved current work. Publication
and chaining require the independent saved-output audit to pass.

Public files beneath that branch's `data/` directory:

- `catalog.json`: available storms, exact issues, member counts and worker status.
- `status.json`: completion counts, live-issue availability, errors and run URL.
- `coverage.json`: 1970 boundary, domain and input-queue classifications.
- `verification.json`: an independent saved-output audit of every completed
  planned issue: release/member identity, observed +0 alignment, causal history,
  21 route points, 20 finite physical pressure grids, exact leads and catalogue
  coverage. `complete` is true only when every pinned planned ID is present.
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
ticks across 1,919 WP storm records. The requested 1996-onward subset is queued
at six-hour frequency; pre-1996 full-tick expansion is not launched. Runtime
and compressed output size should be measured on the cloud pilot before
estimating total storage and wall-clock completion. The queued combined plan
contains 32,143 issues (30,590 ticks and 1,553 older first issues).
Scheduled batches attempt up to 500 issues or 45 minutes, whichever comes first.
An unavailable archived analysis may delay an individual historical tick; the
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
gaps. The former six-hour pause between productive backfill batches is removed;
NOAA availability can still block completion regardless of compute throughput.

A measured basin-field sequence used 44,311 bytes with lossless gzip: about
170 MiB for 4,015 issues, or 2.7 GiB for 65,146, excluding routes, catalogues,
Git history and existing outputs. These are single-sample storage estimates,
not fixed quotas or guaranteed total sizes.
