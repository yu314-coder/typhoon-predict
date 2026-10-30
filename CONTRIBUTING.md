# Contributing to TrackFormer

Anyone may inspect, fork, modify and retrain the public release under its MIT license. No contributor agreement or account approval is required to download or use the code and weights. Submitting a GitHub pull request requires your own GitHub account; local use does not.

## Reproduce first

1. Follow `training/trackformer_1_2/README.md` and pin the asset revision from `PUBLICATION.md`.
2. Verify dataset/checkpoint SHA-256 values before changing anything. Keep original weights and source artifacts unchanged and write experiments into new output directories.
3. Use `--preflight` for the archive/architecture check and `test_training_cpu.py` for the portable CPU loss/backward check. These reconstruct a writable local mmap, so use your own data copy rather than an active trainer's directory.
4. For pressure export changes, use the existing `release_tools/test_pressure_field_export.py` checks as applicable. There is no requirement to run the full multi-hour training job for a documentation correction.

## Share a change

Open an issue or pull request at https://github.com/yu314-coder/typhoon-predict . Explain the problem, changed behavior, exact model/data revision and relevant checks. Document new dependencies and retain third-party notices. Never commit credentials, private data or unrelated large caches. New project contributions should use the existing MIT license; third-party material must keep its own license and attribution.

For scientific claims, preserve issue-time causal inputs and whole-storm partitions; keep future observations as labels only. Compare on identical cases, grids, times and masks. Report misses and false alarms as well as mean errors, and identify previously reused development data. Do not infer forecast skill from a smoother map or training loss. Label changed architectures and weights as new candidates rather than silently replacing released 1.2.

The MIT license permits broad reuse; scientific review criteria apply to what this repository accepts or claims, not to your permission to fork or experiment independently.
