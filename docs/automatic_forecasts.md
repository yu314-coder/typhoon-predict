# Automatic 1.2 forecast archive

The existing repository runs the released Python checkpoint on **GitHub-hosted
CPU runners**, not on a visitor's browser or the owner's Mac. Every six hours
(00:17, 06:17, 12:17, 18:17 UTC), the workflow first checks all current in-domain
JMA storms, then processes a bounded historical batch. GitHub may delay schedules;
public-repository schedules can be disabled after 60 days of inactivity.

New outputs are **one deterministic member**, never labelled as 50-member means.
The historical target starts in **1970**. Backfill selects the first supported
issue of each previously uncovered Western Pacific storm without looking at
forecast errors or requiring future truth. Out-of-domain global storms are not
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
- `fields/{id}.json`: genuine model basin-pressure arrays in hPa.

The Weather Lab API merges these results with its existing verified archive.
The GitHub Pages website can use the same public History API. A cache miss or
source outage remains explicit; no future observation is substituted as forecast.
