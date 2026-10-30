# TrackFormer 1.2: open training package

This is the actual released pressure-field model: development **1.2.73 (v173), epoch 4**, 21,452,595 parameters. It is separate from older route-only 1.2 experiments and later rainfall models.

## Published contents

- `source/v173/`: original, byte-identical model, baseline architecture, archive loader, objective, curriculum, sampler, trainer, resume logic, design and original Metal tests.
- `source/data_preparation/`: original historical ERA5 download/plan utilities and dataset contracts. The small full-field pilot described there is **not** the six-file training corpus.
- `assets.json`: exact sizes and SHA-256 values for all six original derived archives, the original manifest, warm-start checkpoint and selected full training checkpoint.
- `data/`: those exact public arrays, including original split plan, labels/masks and static geography. Binary files are hosted on Hugging Face at the identical path, not committed into GitHub source history.
- `checkpoints/v168_epoch_003.pt`: the original, mandatory warm start, previously split across Drive files.
- `checkpoints/v173_epoch_004.pt`: the selected full training checkpoint, with model, optimizer, normalization, four-epoch history and Python/NumPy/Torch CPU RNG states.
- `config.json`, `checkpoint-history.json`, `SOURCE_MANIFEST.json`: archived recipe, exact saved loss history and source hashes.
- `train.py`: a path-portability wrapper; it calls the unmodified original trainer with explicit paths and settings. No module import starts training.
- Model export, physical pressure-map rendering and benchmark tools remain in the repository's `release_tools/` and `models/trackformer_1_2_field/`.

Both [GitHub](https://github.com/yu314-coder/typhoon-predict/tree/main/training/trackformer_1_2) and [Hugging Face](https://huggingface.co/euler314/typhoon-predict/tree/main/training/trackformer_1_2) carry the source and instructions. Hugging Face serves the large assets (2,644,261,044 bytes in total); no Drive permission is required.

## Download and verify

From the repository root, after obtaining this source folder:

```bash
python training/trackformer_1_2/download_assets.py
python training/trackformer_1_2/download_assets.py --verify-only
python -m pip install -r training/trackformer_1_2/requirements.txt
```

The downloader uses only Python's standard library. It publishes each local file only after exact size/SHA verification and does not overwrite a differing existing file. Use `--revision <immutable-HF-commit>` from `PUBLICATION.md` to pin the upload; even the default `main` URL must match the fixed hashes. For a writable data/output directory on an external disk, copy the package there or set `--destination`; then pass its `data` and `checkpoints` paths to the launcher.

## Preflight, then explicitly start training

```bash
python training/trackformer_1_2/train.py --device mps --output runs/retrain-1.2 --preflight
python training/trackformer_1_2/train.py --device mps --output runs/retrain-1.2
```

Use `--device cuda` for NVIDIA or `--device cpu` for CPU. This wrapper explicitly sets the archived notebook/design recipe: effective batch 10, microbatch 4, workers 3, four epochs; pass flags to change execution settings deliberately. It uses FP32 even though the historical notebook's compatibility AMP positional flag was 1. The unchanged original entry point is also available:

```bash
python training/trackformer_1_2/source/v173/v173_pressure_train.py 10 3 0 \
  --microbatch 4 --epochs 4 --device mps \
  --manifest training/trackformer_1_2/data/manifest.json \
  --warm-start training/trackformer_1_2/checkpoints/v168_epoch_003.pt \
  --output runs/retrain-1.2
```

The original loader hashes all arrays, reconstructs a writable basin mmap beside the dataset, recomputes train-only normalization and enforces counts **13,949 / 1,041 / 1,195** with native-pressure subsets **800 / 100 / 100**. Allow space for that extra approximately 390 MB mmap and approximately 260 MB checkpoint files. Do not point preflight/training at an active trainer's data/output directories. The trainer uses a per-output lock; the same output resumes its `last.pt`. SIGINT/SIGTERM request a checkpoint at an optimizer boundary. Training horizons are 4/8/12/20 six-hour steps; validation always uses 20 steps. Four-step truncated backpropagation, gradient clipping, AdamW settings and all losses are in the original source.

## Provenance and honest reproduction limits

- Dataset manifest: `04fd0d30620137aad20173dcd59e22da2a6c20fae2f17de593e91a59517459c1`.
- Warm start: `99ed776097e05ad6df03375465d9618741bed63a4ee57724d3ba98e295039d30`.
- Selected full checkpoint: `f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0`.
- Inference-only weights: `db49f36e85a3766defc4c172746897a1f783705d1ce8e6f9dfb8e87ae1d902cb`.

The selected full checkpoint is epoch **4**, used by the report; **it is not the smallest saved aggregate validation loss** (epoch 2 is 0.6939101; epoch 4 is 0.7283264). A fresh run's `best.pt` follows the original minimum-validation criterion and need not be the report's epoch-4 checkpoint. Do not conflate these choices. The supplied warm start makes the actual v173 training path runnable without reconstructing every earlier experimental model. Historical v168 optimization logs and complete original runtime package versions/accelerator RNG states are not archived here. The recipe is preserved from the original notebook/design, not a newly recovered complete original process argv. Bitwise end-to-end retraining across devices has **not** been demonstrated.

The exact derived data are published; the raw IBTrACS release/download ID was not recorded in the frozen manifest. Current upstream downloads may differ and cannot be claimed byte-identical. Fitting uses 2000–2021, validation 2022–2023, original test 2024–2025, with complete storms separated. Later cache records are not fitting years. All future fields/positions are labels only. Repeatedly inspected tests/showcases are development evidence, not a fresh unused holdout. This publication does not rerun training, change released weights or modify active rainfall training.

## Licenses and attribution

Project source and project checkpoints retain the repository's MIT license. Upstream weather/track data retain their own terms and attributions, not the MIT software license; see `THIRD_PARTY_NOTICES.md`. Downloaded full checkpoints use Python pickle containers: load these identified, checksum-verified project artifacts only.
