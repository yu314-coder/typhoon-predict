# Evaluation assets

This directory holds the current published comparison figures, metrics, selected forecast previews and the numerical inputs under [`release_data/`](release_data/). The architecture paper is separate in [`../paper/`](../paper/).

- [270-case comparison and definitions](../docs/trackformer_1_2_evaluation.md): 270 issue times from 90 storms; the 1.2 output is a 50-member input-perturbation mean.
- [Expanded daily-issue protocol](../docs/daily_storm_benchmark.md): 1,473 daily issues from 270 storms; results are not implied by the existing 270-case charts.
- [TIP diagnostic](trackformer_1_2_vs_1_1_tip_metrics.json): retain pressure regressions as well as improvements.
- [Showcase selection](trackformer_1_2_showcase_selection.json): deliberately selected examples, not typical performance or untouched validation.

For geographically aligned routes, use errors between forecasts and observations at the **same valid times on the same map**. Centred/scale-normalized shape similarity alone cannot establish alignment. Never translate, rotate or rescale a forecast to make it match the observed route.

From the repository root, reproduce the 270-case figures and Surigae isobars using `python release_tools/plot_release_270.py`. The script defaults to this directory and uses its saved plot inputs; it does not run model inference. Rendering requires NumPy and Matplotlib. The [Yagi video builder](../release_tools/build_yagi_video.py) likewise uses saved model fields, not synthetic illustrations. Its [provenance](release_data/yagi_video.json) reports geographic route alignment separately from pressure error.
