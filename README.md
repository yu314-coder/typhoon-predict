# Bavi Typhoon Track Research Model

This repository contains a local inference package for a research typhoon-track model trained from an ERA5-conditioned CNN-GRU ensemble checkpoint. It runs on Apple Silicon with PyTorch MPS, on CPU, or on CUDA.

## Important limitation

The included Bavi result is a research proxy, not an operational warning product. The checkpoint was trained on historical ERA5 data, but the matching 2026 ERA5 atmospheric field was not available locally. The supplied run therefore uses the latest official track fixes and a mean-normalized atmospheric input. Do not use it for safety, evacuation, shipping, or aviation decisions.

## Files

- `best.pt`: trained PyTorch checkpoint.
- `run_bavi_mac.py`: portable local inference script.
- `bavi_mac_proxy_forecast.json`: generated forecast output.
- `bavi_mac_world_map.html`: interactive Leaflet world map.

## Run locally on macOS

```bash
cd /Volumes/D/typhoon_predict
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_bavi_mac.py --checkpoint best.pt --output bavi_mac_proxy_forecast.json
```

The script automatically uses Apple MPS when available. Use `--device cpu` to force CPU execution. Open `bavi_mac_world_map.html` through a local web server to view the route.

## Run on Linux/CUDA

Install a CUDA-compatible PyTorch build, then run the same command with `--device auto` or `--device cpu`. The checkpoint is a standard PyTorch state dictionary and does not require Colab.

## Model format and GGUF

GGUF is designed primarily for llama.cpp-style language and compatible tensor models. This checkpoint is a custom convolutional field encoder plus bidirectional GRU and probabilistic ensemble head, so converting it to GGUF would not make it runnable in llama.cpp and would risk changing the forecast behavior.

The supported portable format is the original PyTorch checkpoint (`best.pt`). For deployment, use PyTorch on MPS/CPU/CUDA or add a carefully validated ONNX/TorchScript export for a fixed deterministic member. The scaler objects and normalization arrays stored in `best.pt` must remain with the model.

## Reproducibility

The checkpoint includes the model weights, input/output scalers, field normalization statistics, and training configuration. Results are stochastic because the ensemble samples latent noise. The map displays the ensemble mean and the 10th-to-90th percentile boxes.

## Data and attribution

The current Bavi initialization uses official tropical-cyclone track fixes. ERA5 is a product of the Copernicus Climate Change Service implemented by ECMWF. OpenStreetMap data is used for the map background.

## License

This research package is provided without operational guarantees. Check the source-data and model-training licenses before redistributing derived datasets or deploying the model.
