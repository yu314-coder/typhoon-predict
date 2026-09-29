# Trackformer 1.2 — research candidate

This release packages a verified moving-pressure-core model as **Trackformer 1.2**. It supersedes the withdrawn, unrelated route/scalar 1.2 candidate; the older root-level `trackformer_1_2.py` is not compatible with these field weights. Trackformer 1.1 remains available as a separate release.

The attached archive includes inference-only weights, exact source modules, an input wrapper, a data-contract/provenance manifest, detailed documentation, **270-case direction/route-shape benchmark bars**, paired route examples and recent **Surigae pressure maps with labelled isobars**. The training checkpoint was internally labelled version `1.2.73`, epoch 4; its SHA-256 is `f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0`. The exported inference weights SHA-256 is `db49f36e85a3766defc4c172746897a1f783705d1ce8e6f9dfb8e87ae1d902cb`.

On the 270-case development cohort, 1.1 versus 1.2 mean-of-50 scores are: direction error **56.22° vs 44.82°**, route-shape similarity **0.7273 vs 0.8249**, path similarity **0.4983 vs 0.5727**, and mean track error **902.3 vs 714.4 km**. Direction uses 5,382 common valid steps. The old 1.1 report had stored latitude/longitude under a local-kilometre key; the new comparison repairs that coordinate interpretation and records the correction. Original artifacts remain available. These results are development comparisons, and no matched 1.1 pressure-field bar is claimed.

The Surigae illustration is a single forecast issued 2026-09-27 12 UTC, with 4 hPa isobars at +6/+24/+36 h. It is separate from the 50-member historical benchmark. Reproduction data and metric definitions are included. The model weights are unchanged in this documentation revision.

This is not an operational warning service. Do not use for safety-critical decisions. A genuinely untouched storm-level holdout is still required before generalization claims.
