# Evaluation assets

The current **completed three-model comparison** scores Trackformer 1.1, Trackformer 1.2 and Google's **WeatherNext Cyclones Mini `<2024` (software v0.3.0)** on **1,473 daily starts / 270 Western Pacific storms**, through +120 h. Mini's RTX 3070 CUDA run is complete and verified; Trackformer 1.2 is a mean of 50 members and Mini is one member.

- [Announcement chart](released_daily/model_1_2_benchmark.png) and [shared three-model scores](released_daily/released_daily_benchmark.json).
- [Completed DeepMind results](deepmind_daily/benchmark.json), [CUDA completion receipt](deepmind_daily/verification.json) and [independent publication audit](deepmind_daily/publication_audit.json).
- [Daily-case protocol](../docs/daily_storm_benchmark.md), [DeepMind model identity and period breakdowns](../docs/deepmind_daily_benchmark.md) and [Trackformer evaluation definitions](../docs/trackformer_1_2_evaluation.md).
- [Matched pressure, wind and radius evaluation](../docs/intensity_benchmark.md): pressure uses 134 shared daily starts / 40 storms, not all 270 storms. Unsupported scores stay unavailable, never zero.

Scores average valid leads within each day, days within each storm, then storms equally. The historical 1980–1999 group overlaps Mini's fitting years; recent 2024+ results are reported separately. This is development evidence, not a certified untouched holdout or an equal-compute architecture ablation.

For geographically aligned routes, compare forecasts and observations at the **same valid times on the same map**. Centred shape similarity alone cannot establish alignment. Never translate, rotate or rescale a forecast to make it match the observed route.

Reproduce the current three-model announcement chart from saved results with `python release_tools/plot_model_announcement.py`. This renders audited scores without new inference or weather retrieval; NumPy and Matplotlib are required. The paper is in [`../paper/`](../paper/), and saved pressure arrays and preview provenance are under [`release_data/`](release_data/).

## Archived diagnostics and selected examples

Older figures, the [TIP diagnostic](trackformer_1_2_vs_1_1_tip_metrics.json) and [showcase selection](trackformer_1_2_showcase_selection.json) are preserved for provenance, not presented as the current benchmark. See the [showcase archive](../docs/showcase_archive.md) for selected pressure-map films and their limitations.
