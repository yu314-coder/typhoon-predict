"""Trackformer 1.2 research inference from a causal, normalized issue packet.

This wrapper deliberately does not fetch or construct weather inputs. The caller
must supply past/issue-time analyses on the exact grids in manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
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

PRESSURE_EXPORT_SCHEMA = "trackformer-1.2-pressure-fields-v2"


def export_pressure_fields(outputs, inputs, contract, issue_time_ns):
    """Capture existing forward outputs; never relocate or fit a display vortex."""
    scale = float(contract["normalization"]["std"][0])
    offset = float(contract["normalization"]["mean"][0])
    global_static = inputs["global_static"][0].detach().cpu().numpy()
    regional_static = inputs["regional_static"][0].detach().cpu().numpy()
    leads = np.arange(6, 121, 6, dtype=np.int16)
    if len(outputs) != len(leads):
        raise ValueError("Detailed pressure export requires all twenty forecast leads")
    return {
        "core_mslp_hpa": torch.stack([
            o["core"][0, 0] * scale + offset for o in outputs]).cpu().numpy(),
        "core_latitude_deg": torch.stack([o["core_lat"][0] for o in outputs]).cpu().numpy(),
        "core_longitude_deg": torch.stack([o["core_lon"][0] for o in outputs]).cpu().numpy(),
        "core_valid": torch.stack([o["core_valid"][0, 0] for o in outputs]).cpu().numpy().astype(bool),
        "basin_latitude_deg": global_static[0] * 90,
        "basin_longitude_deg": (global_static[1] + 1) * 180,
        "regional_latitude_deg": regional_static[0] * 90,
        "regional_longitude_deg": (regional_static[1] + 1) * 180,
        "regional_valid": torch.stack([
            o["regional_valid"][0, 0] for o in outputs]).cpu().numpy().astype(bool),
        "track_valid": torch.stack([o["track_valid"][0] for o in outputs]).cpu().numpy().astype(bool),
        "issue_center_lat_lon": inputs["center"][0].detach().cpu().numpy(),
        "issue_time_ns": np.asarray(issue_time_ns, dtype=np.int64),
        "valid_time_ns": issue_time_ns + leads.astype(np.int64) * (3600 * 10**9),
        "member_count": np.asarray(1, dtype=np.int16),
        "core_information_spacing_km": np.asarray(20, dtype=np.float32),
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
    from wind_estimation import diagnose_outputs, summarize_members, KEYS
    metadata = json.loads((model_dir / "manifest.json").read_text())
    if metadata["public_version"] != "1.2":
        raise ValueError("Unexpected public model version")
    contract = metadata["data_contract"]
    model = CoreForecaster(contract).to(device).eval()
    weights = model_dir / "weights.pt"
    with weights.open("rb") as stream:
        weight_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if weight_sha256 != metadata["inference_weights_sha256"]:
        raise ValueError("Weights do not match the released 1.2 manifest")
    state = torch.load(weights, map_location="cpu", weights_only=True)
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
        "maximum_wind_auxiliary_kt": torch.stack([o["vmax"][0] for o in outputs]).cpu().numpy(),
    }
    with np.load(packet, allow_pickle=False) as source:
        issue_time_ns = int(source["issue_time_ns"])
    result.update(export_pressure_fields(outputs, inputs, contract, issue_time_ns))
    diagnostics = [summarize_members(diagnose_outputs(o, contract)) for o in outputs]
    result['maximum_wind_auxiliary_kt_valid'] = np.asarray(
        [d['estimates']['maximum_wind_auxiliary_kt']['mean'] is not None for d in diagnostics], dtype=bool)
    for key in KEYS[1:]:
        values = [d['estimates'][key]['mean'] for d in diagnostics]
        # Numeric zero is storage padding only. Consumers MUST use this mask.
        result[key] = np.asarray([0 if v is None else v for v in values], dtype=np.float32)
        result[key + '_valid'] = np.asarray([v is not None for v in values], dtype=bool)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite forecast")
    result['wind_estimation_json'] = np.asarray(json.dumps(diagnostics, allow_nan=False))
    result['pressure_export_json'] = np.asarray(json.dumps({
        "schema": PRESSURE_EXPORT_SCHEMA,
        "public_version": "1.2",
        "architecture": metadata["architecture"],
        "inference_weights_sha256": weight_sha256,
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "members": 1,
        "units": {"pressure": "hPa", "latitude": "degrees_north", "longitude": "degrees_east"},
        "core_information_spacing_km": 20,
        "core_method": "unchanged learned moving pressure field; physical hPa with original geographic coordinates",
        "native_high_resolution_observations": False,
        "native_detail_history_available": bool(inputs["detail_available"][0, 0].item()),
        "regional_grid": "fixed issue-relative composite; outside moving-core coverage only basin information remains",
        "coverage_policy": "apply core_valid, regional_valid and track_valid; invalid finite storage is not a supported forecast",
        "central_pressure_policy": "existing bilinear moving-core readout at the associated forecast centre; not an independently inserted scalar",
        "ensemble_policy": "one clean member; register physical fields on common geographic coordinates before any ensemble average",
        "forecast_equations_changed": False,
    }, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, help="Causal normalized .npz issue packet")
    parser.add_argument("output", type=Path, help="Output .npz path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--pressure-map", type=Path, help="Optional PNG of basin and actual moving-core pressure (requires Matplotlib)")
    parser.add_argument("--map-lead", type=int, default=120, choices=range(6, 121, 6))
    parser.add_argument("--isobar-interval", type=float, default=4, help="Pressure contour interval in hPa")
    args = parser.parse_args()
    result = forecast(args.packet, Path(__file__).resolve().parent, args.device)
    np.savez_compressed(args.output, **result)
    print(f"Saved 20 six-hour forecasts to {args.output}")
    if args.pressure_map:
        from plot_pressure import render_pressure_map
        render_pressure_map(result, args.pressure_map, args.map_lead, args.isobar_interval)
        print(f"Saved pressure map to {args.pressure_map}")


if __name__ == "__main__":
    main()
