# Experimental wind and circular-radius diagnostics

The frozen 1.2 weights and neural architecture are unchanged. Maximum wind is
the existing learned auxiliary `vmax` output, in knots, not a resolved wind map.
Its averaging period is not harmonized to JMA, and skill has not been established
by the route/pressure benchmark. Do not treat this product as operational advice.

`wind_estimation.py` optionally diagnoses each model-generated pressure member:

1. Sample complete circular rings about that member's predicted centre, no more
   finely than the source/reconstruction grid, out to at most 480 km.
2. Compute the azimuthal-mean radial pressure gradient (hPa converted to Pa).
3. Solve `Vg^2/r + |f| Vg = (1/rho) dp/dr`, with air density 1.15 kg/m3, then
   apply a fixed 0.8 ocean surface-reduction assumption. This is uncalibrated;
   motion, land friction, asymmetry, boundary-layer jets and averaging periods
   are not represented. No future observations or official routes are inputs.
4. Report a resolved gradient-wind peak/RMW and outward 34/50/64-kt crossings.
   These are **circular equivalent radius estimates**, not quadrant radii and
   not isobar radii. Coarse (>50 km) source grids, incomplete rings, unresolved
   peaks and missing crossings produce nulls, not invented zeros or edge radii.
5. Diagnose members individually, then summarize. A mean is supplied only when
   every member has that estimate. Valid counts and p10/p90 describe spread;
   they are not calibrated confidence intervals. Never diagnose a mean map and
   call it a mean of 50 radius predictions.

The 20-km moving core is a learned reconstruction. A resampled 0.25-degree
regional image does not add resolved observations. Even an available estimate
therefore remains experimental. A single gridded basin mean cannot reconstruct
the missing member diagnostics. Legacy forecasts without saved member/core
outputs retain unavailable estimates instead of being rerun or relabelled.

For the saved 50-member regional composite, core origins were not archived.
That release's core spans +/-640 km and centre association is restricted to
300 km about its origin. Consequently this read-only legacy exporter samples
only rings within 300 km of each predicted centre, conservatively inside the
core footprint; a missing crossing beyond that limit remains unavailable.
New forecasts use the actual moving-core coordinates and their valid masks.

Defaults are fixed physical assumptions, not fitted to showcase outcomes. Before
claiming skill, freeze eligible whole-storm held-out cases; score +6 to +120 h
maximum winds and agency-compatible wind radii separately, with coverage,
unresolved cases, land/ocean breakdowns and whole-storm uncertainty. No such
validation or calibration is claimed by this implementation.

References: [NOAA PaHM theory and limitations](https://noaa-ocs-modeling.github.io/PaHM/pahm_manual.pdf)
and [Chavas, Reed and Knaff (2017)](https://doi.org/10.1038/s41467-017-01546-9).
