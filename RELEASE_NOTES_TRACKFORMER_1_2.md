# Trackformer 1.2 — research candidate

This release packages a verified moving-pressure-core model as **Trackformer 1.2**. It supersedes the withdrawn, unrelated route/scalar 1.2 candidate; the older root-level `trackformer_1_2.py` is not compatible with these field weights. Trackformer 1.1 remains available as a separate release.

The attached archive includes inference-only weights, exact source modules, an input wrapper, a data-contract/provenance manifest, detailed documentation, **270-case direction/route-shape benchmark bars**, paired route examples and recent **Surigae pressure maps with labelled isobars**. The training checkpoint was internally labelled version `1.2.73`, epoch 4; its SHA-256 is `f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0`. The exported inference weights SHA-256 is `db49f36e85a3766defc4c172746897a1f783705d1ce8e6f9dfb8e87ae1d902cb`.

On the 270-case development cohort, 1.1 versus 1.2 mean-of-50 scores are: direction error **56.22° vs 44.82°**, route-shape similarity **0.7273 vs 0.8249**, path similarity **0.4983 vs 0.5727**, and mean track error **902.3 vs 714.4 km**. Direction uses 5,382 common valid steps. The old 1.1 report had stored latitude/longitude under a local-kilometre key; the new comparison repairs that coordinate interpretation and records the correction. Original artifacts remain available. These results are development comparisons, and no matched 1.1 pressure-field bar is claimed.

The route gallery now shows six explicitly selected best-performing examples from distinct storms, with the selection rule and all case scores supplied. It is not representative evidence; the aggregate bars retain all 270 cases. A separate larger benchmark has frozen **270 distinct storms / 1,473 daily issues**, with daily scores averaged per storm and storms weighted equally. Its forecasts are being computed; results are pending.

The README preview now links to an interactive PRAPIROON pressure/route showcase with playback, lead selection, hover details and observed-track overlays. It uses actual saved 50-member mean pressure fields and a disclosed selected-best rule. The main paper has been replaced with an architecture-only Trackformer 1.2 description; the 1.1 source is preserved separately.

The Surigae illustration is a single forecast issued 2026-09-27 12 UTC, with 4 hPa isobars at +6/+24/+36 h. It is separate from the 50-member historical benchmark. Reproduction data and metric definitions are included. The model weights are unchanged in this documentation revision.

This is not an operational warning service. Do not use for safety-critical decisions. A genuinely untouched storm-level holdout is still required before generalization claims.
