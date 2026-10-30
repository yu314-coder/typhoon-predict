# Open access and reuse

TrackFormer 1.2 is publicly accessible without account approval, a download gate, a paid API or Google Drive permissions. Project code and project model weights/checkpoints are distributed under the [MIT license](LICENSE). You may use, study, modify, retrain, redistribute, sublicense and sell copies, including commercial use, subject to retaining the required copyright and permission notice. No research-only or non-commercial restriction is added by this project. The research/safety cautions describe capability limits; they do not replace or narrow the MIT license.

## What is public

| Component | Public location |
| --- | --- |
| Architecture, inference and physical pressure exporter | `models/trackformer_1_2_field/` on both repositories |
| Original training source, losses, sampler, curriculum and resume logic | `training/trackformer_1_2/source/v173/` |
| Portable launcher, configuration, tests and exact saved training history | `training/trackformer_1_2/` |
| Exact six derived dataset files, masks and original dataset manifest | Hugging Face: `training/trackformer_1_2/data/` |
| Original warm start and selected full training checkpoint with optimizer/RNG state | Hugging Face: `training/trackformer_1_2/checkpoints/` |
| Hashes and anonymous checksum-verifying asset downloader | `training/trackformer_1_2/assets.json` and `download_assets.py` |
| Benchmark protocols, measured results and figure provenance | `evaluation/`, `docs/` and `release_tools/` |

[GitHub source](https://github.com/yu314-coder/typhoon-predict) · [Hugging Face source and binary assets](https://huggingface.co/euler314/typhoon-predict) · [Training instructions](training/trackformer_1_2/README.md) · [Immutable publication links](training/trackformer_1_2/PUBLICATION.md) · [Contributing](CONTRIBUTING.md)

Large arrays and optimizer checkpoints are hosted on Hugging Face and linked from GitHub, rather than duplicated in ordinary Git history. Downloads are standard public HTTPS; the asset downloader uses Python's standard library, verifies exact hashes and requires no token. Training/inference still require the documented open-source runtime dependencies. There is no hosted-service dependency in the published training launcher.

## Upstream data terms

NOAA/IBTrACS and Copernicus/ERA5 data retain their own provenance, attribution and applicable upstream terms. The project's MIT license cannot remove third-party conditions. See [the source notices](training/trackformer_1_2/THIRD_PARTY_NOTICES.md) and [data provenance](docs/trackformer_1_2_training_data.md). The root `license: mit` metadata describes project software and model artifacts, not a blanket relicensing of all weather data.

## What is not claimed

The frozen derived arrays and their exact hashes are public, but the original raw IBTrACS snapshot/download ID was not recorded. A complete original runtime lockfile, original process argv and earlier warm-start training history have not been recovered. The archived recipe and warm-start checkpoint make the released v173 path runnable; bitwise full retraining across devices has not been demonstrated. The report selects epoch 4, while the saved aggregate validation-loss minimum is epoch 2. Previously inspected tests and examples are development evidence, not an untouched holdout.

Public availability is not a claim that every earlier 1.2 experiment or later rainfall experiment is this released model. No unavailable artifact, qualification result or score is invented to claim complete reproducibility. These remaining gaps stay visible so others can audit and improve the release.
