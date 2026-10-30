# TrackFormer 1.2 training

**[Read / download the complete training manual (106-page PDF)](../../docs/TrackFormer_1.2_Training_and_Reproducibility.pdf).**

The single PDF contains the entire training method and data process, active losses, settings, checkpoint history, reproduction commands, limitations and all 40 text-source/provenance files. Its embedded ZIP preserves their exact bytes. You do not need to open separate Markdown documents to read the method. The original source files remain executable and inspectable here.

Large assets are public at the same [Hugging Face training path](https://huggingface.co/euler314/typhoon-predict/tree/main/training/trackformer_1_2); `assets.json` and `download_assets.py` verify their hashes without tokens or Drive permissions. Code and weights use MIT; upstream data attribution remains in `THIRD_PARTY_NOTICES.md` and the PDF.

Important: the archived notebook/config recipe uses batch 10, while the selected full checkpoint records 4,384 updates, compatible with batch 16 for the published four-epoch sampler. The PDF documents the evidence and provides both commands without claiming recovery of the original argv. See `records/selected-checkpoint-training.json` for the machine-readable inspection. The report selects epoch 4; the saved validation-loss minimum is epoch 2.
