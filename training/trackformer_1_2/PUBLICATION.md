# Immutable publication and execution checks

- Source publication on GitHub: https://github.com/yu314-coder/typhoon-predict/commit/ac05e1974ac90ad7819bf0483da8e8637d880525
- Source and all nine assets on Hugging Face: https://huggingface.co/euler314/typhoon-predict/commit/2d1464e9a310e6a6e8840ef1c23017fdededd57b
- All asset hashes remain fixed in `assets.json`.

Pin the binary download to the completed publication:

```bash
python training/trackformer_1_2/download_assets.py --revision 2d1464e9a310e6a6e8840ef1c23017fdededd57b
```

The original trainer's CPU preflight passed, including all six input-file hashes, exact split counts, normalization and original warm-start loading (21,452,595 parameters). The portable CPU check passed on one coarse and one native TRAIN example: finite forward/loss/backward, nonzero attention gradient, pressure exactly read from the predicted field, rejection of an added future input, and full TRAIN sampler coverage. Zero optimizer updates were performed. See `EXECUTION_CHECKS.json`. This is an execution check, not a new end-to-end training reproduction or an accuracy claim.

```bash
python training/trackformer_1_2/test_training_cpu.py
```

This test also reconstructs the loader's local basin mmap; use a dedicated writable data copy, never another trainer's data directory. Original GPU-specific tests are archived without rewriting their historical relative-path assumptions in `source/v173/test_model.py`; they were not rerun on GPU for this publication.
