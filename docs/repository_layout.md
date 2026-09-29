# Current repository layout

The current source tree is focused on the selected pressure-field **Trackformer 1.2** release. The implementation remains at `models/trackformer_1_2_field/`; module filenames such as `v165_base.py` are required architecture dependencies, not unused legacy files. They and their source hashes have not been changed.

## Cleanup on 29 September 2026

Removed from the current branch:

- The withdrawn route/scalar `trackformer_1_2.py` candidate and its incompatible `examples/predict_trackformer_1_2.py` entry point.
- Root-level 1.1 inference/route/intensity/temporal modules and their old analysis utility, plus `colab_train_v17.ipynb`.
- The old 1.1 paper and superseded 1.2.26/matched100 comparison assets.

Moved current benchmark figures, metrics and plot data from `paper/` into `evaluation/`, updating documentation and reproduction paths. `paper/trackformer.tex` and `paper/trackformer.pdf` stay at their existing locations. Published HTML and MP4 URLs under `docs/` stay unchanged.

The 1.1-versus-1.2 comparison arrays, TIP diagnostic, current paper, weights, model source hashes and release tags are retained. No local research checkpoints, running benchmark, Hugging Face files or historical release assets were deleted by this source-tree cleanup.

## Recovery

The complete pre-cleanup repository remains in [commit 459a3d014aa4794652ceba3ecdffcde0beada2cd](https://github.com/yu314-coder/typhoon-predict/tree/459a3d014aa4794652ceba3ecdffcde0beada2cd). The [Trackformer 1.1 release](https://github.com/yu314-coder/typhoon-predict/releases/tag/trackformer-1.1) remains available. This cleanup does not rewrite Git history.
