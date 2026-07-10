# Typhoon Predict

Typhoon Predict is an ERA5-conditioned tropical-cyclone track research model. It combines recent storm-track history with a local atmospheric reanalysis patch and produces a probabilistic forecast at multiple lead times.

This repository contains a trained PyTorch checkpoint and a portable inference script. It is a research project, not an operational warning system.

## Model

The model is an ensemble neural network with three parts:

1. A convolutional field encoder processes the ERA5 patch. It uses convolution, GELU activations, batch normalization, strided downsampling, and global average pooling.
2. A bidirectional GRU encodes the recent cyclone track history.
3. A fusion MLP combines the atmospheric and track embeddings. Separate heads predict the output mean and log scale, while a latent projection adds correlated ensemble variation.

The checkpoint uses four historical track steps and predicts seven lead times: 6, 12, 24, 48, 72, 96, and 120 hours. Each output step contains latitude and longitude displacement plus additional track variables used by the training target.

## Training configuration

The included checkpoint was trained with:

- Western Pacific basin data (`WP`)
- ERA5 data beginning in 1979
- 8 degree atmospheric patches at 0.5 degree resolution
- Four historical track steps
- 1,024 maximum training windows
- Batch size 64
- Learning rate `2e-4`
- Weight decay `1e-4`
- Up to 80 epochs with early stopping patience 12
- 50 ensemble members at inference time
- Random seed 42

The checkpoint stores the model weights, training configuration, feature scalers, and ERA5 normalization statistics required for inference.

## Local inference

```bash
cd /Volumes/D/typhoon_predict
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_inference.py --checkpoint best.pt --output forecast.json
```

The script automatically selects Apple MPS on Apple Silicon, CUDA when added to the script for a compatible environment, or CPU otherwise. Use `--device cpu` to force CPU execution.

The current script is a demonstration inference path. It uses a mean-normalized atmospheric input when a matching live ERA5 patch is unavailable. For real research evaluation, replace that input with a correctly time-aligned ERA5 patch using the same variables, grid, normalization statistics, and storm-centered coordinates used during training.

## Minimum system requirements

For local inference with the included checkpoint:

- Operating system: macOS, Linux, or Windows
- Python: 3.10 or newer
- Processor: 64-bit CPU with four cores or more
- Memory: 8 GB RAM minimum; 16 GB recommended
- Storage: 2 GB free space for the repository, virtual environment, checkpoint, and generated forecasts
- GPU: not required; CPU inference is supported. Apple Silicon can use MPS when available.

For faster experimentation, training, or batch inference, use a CUDA-capable NVIDIA GPU with at least 16 GB VRAM. An A100 or H100 is suitable for the full training workflow, but is not required to run the released checkpoint.

## Output and visualization

The inference script writes JSON containing the ensemble mean and percentile bounds. `forecast_world_map.html` is an example Leaflet visualization of one generated result; it is not part of the model architecture and can be replaced with a generic forecast viewer.

## Data pipeline

Training requires:

- Tropical-cyclone best-track fixes for the target basin
- ERA5 atmospheric variables on a regular grid
- Storm-centered patch extraction
- Time-aligned history/target windows
- Train/validation/test splits separated by time to avoid leakage

ERA5 is produced by the Copernicus Climate Change Service implemented by ECMWF. Users must obtain the required track and ERA5 data under their own access and licensing terms.

## Model formats

The supported format is the original PyTorch checkpoint (`best.pt`). GGUF is intended mainly for llama.cpp-compatible language and tensor models; it is not a compatible runtime format for this custom convolutional encoder, GRU, and probabilistic ensemble head. Converting it to GGUF would not make it runnable in llama.cpp.

For production deployment, use PyTorch on CPU, Apple MPS, or CUDA. An ONNX or TorchScript export could be added for a fixed deterministic member, but it must preserve preprocessing, scaler state, output decoding, and ensemble sampling behavior and should be validated against the PyTorch implementation.

## Limitations

- The included checkpoint is a research baseline, not an operational forecast system.
- Forecast quality depends strongly on correct ERA5 inputs and track preprocessing.
- The supplied demonstration run may use a mean-normalized atmospheric proxy when live ERA5 data is absent.
- Probabilistic spread is model-generated uncertainty, not a calibrated warning cone.

## License and safety

Check the licenses for the source datasets and derived products before redistribution. Do not use this repository for evacuation, aviation, maritime, emergency-management, or other safety-critical decisions.
