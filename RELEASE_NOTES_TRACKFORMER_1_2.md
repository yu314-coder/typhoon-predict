# Trackformer 1.2 — research candidate

This release packages a verified moving-pressure-core model as **Trackformer 1.2**. It supersedes the withdrawn, unrelated route/scalar 1.2 candidate; the older root-level `trackformer_1_2.py` is not compatible with these field weights. Trackformer 1.1 remains available as a separate release.

The attached archive includes inference-only weights, the exact source modules, a strict input wrapper, a data-contract/provenance manifest, model documentation, saved TIP comparison bars and a real model-generated Dolphin pressure-map example. The training checkpoint was internally labelled version `1.2.73`, epoch 4; its SHA-256 is `f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0`. The exported inference weights SHA-256 is `db49f36e85a3766defc4c172746897a1f783705d1ce8e6f9dfb8e87ae1d902cb`.

On ten overlapping TIP starts, the 50-input-member mean reduced mean track error from 383.9 km (1.1) to 370.5 km, **but raised central-pressure MAE from 20.1 to 30.7 hPa**. These are development diagnostics, not proof of generalization. Trackformer 1.1 does not output a native MSLP forecast field, so the 1.2 pressure map is an example rather than a paired 1.1 field comparison.

This is not an operational warning service. Do not use for safety-critical decisions. A genuinely untouched storm-level holdout is still required before generalization claims.
