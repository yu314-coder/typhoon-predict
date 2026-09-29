"""Trackformer 1.2 research inference from a causal, normalized issue packet.

This wrapper deliberately does not fetch or construct weather inputs. The caller
must supply past/issue-time analyses on the exact grids in manifest.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import CoreForecaster


INPUT_SHAPES = {
    "global_history": (9, 8, 25, 33),
    "regional_history": (9, 1, 121, 121),
    "global_static": (4, 25, 33),
    "regional_static": (4, 121, 121),
    "detail_available": (1,),
    "center": (2,),
    "motion": (2,),
    "issue_intensity": (2,),
    "issue_mask": (2,),
}


def load_packet(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as packet:
        if set(packet.files) != set(INPUT_SHAPES) | {"issue_time_ns", "history_time_ns"}:
            raise ValueError("Packet keys must match the documented causal input schema exactly")
        issue = int(packet["issue_time_ns"])
        history = np.asarray(packet["history_time_ns"], dtype=np.int64)
        if history.shape != (9,) or history[-1] != issue or np.any(history > issue):
            raise ValueError("History must contain nine times ending at issue time; future inputs forbidden")
        if not np.array_equal(np.diff(history), np.full(8, 6 * 3600 * 10**9)):
            raise ValueError("History must use consecutive six-hour analyses")
        arrays = {}
        for key, shape in INPUT_SHAPES.items():
            value = np.asarray(packet[key], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key}: expected finite array of shape {shape}, got {value.shape}")
            arrays[key] = value[None]
        return arrays


def forecast(packet: Path, model_dir: Path, device: str = "cpu") -> dict[str, np.ndarray]:
    metadata = json.loads((model_dir / "manifest.json").read_text())
    if metadata["public_version"] != "1.2":
        raise ValueError("Unexpected public model version")
    contract = metadata["data_contract"]
    model = CoreForecaster(contract).to(device).eval()
    state = torch.load(model_dir / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    inputs = {key: torch.from_numpy(value).to(device) for key, value in load_packet(packet).items()}
    with torch.inference_mode():
        running = model.initial(inputs)
        outputs = []
        for _ in range(20):
            running, output = model.step(running)
            outputs.append(output)
    scale = float(contract["normalization"]["std"][0])
    offset = float(contract["normalization"]["mean"][0])
    result = {
        "lead_hours": np.arange(6, 121, 6, dtype=np.int16),
        "track_lat_lon": torch.stack([o["center"][0] for o in outputs]).cpu().numpy(),
        "central_pressure_hpa": torch.stack([o["pressure"][0] for o in outputs]).cpu().numpy(),
        "basin_mslp_hpa": torch.stack([o["global"][0, 0] for o in outputs]).cpu().numpy() * scale + offset,
        "regional_mslp_hpa": torch.stack([o["regional"][0, 0] for o in outputs]).cpu().numpy() * scale + offset,
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite forecast")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, help="Causal normalized .npz issue packet")
    parser.add_argument("output", type=Path, help="Output .npz path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    result = forecast(args.packet, Path(__file__).resolve().parent, args.device)
    np.savez_compressed(args.output, **result)
    print(f"Saved 20 six-hour forecasts to {args.output}")


if __name__ == "__main__":
    main()
