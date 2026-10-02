"""CPU-only export regression using an existing causal batched issue packet.

This checks packaging/field consistency, not forecast skill. It never calls MPS,
changes neural weights, or writes into historical archive/training directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/trackformer_1_2_field"
sys.path.insert(0, str(MODEL))
import predict
from model import CoreForecaster
from baseline_model import base
from plot_pressure import render_pressure_map


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(batched_packet, weights, work):
    work = work.resolve()
    if not work.is_relative_to(Path("/Volumes/D")):
        raise ValueError("Diagnostic artifacts must remain on D")
    work.mkdir(parents=True, exist_ok=True)
    directory = work / "inference"
    directory.mkdir(exist_ok=True)
    for source in MODEL.iterdir():
        if source.is_file() and source.suffix in (".py", ".md", ".json"):
            shutil.copy2(source, directory / source.name)
    shutil.copy2(weights, directory / "weights.pt")
    metadata = json.loads((directory / "manifest.json").read_text())
    for name, expected in metadata["source_module_sha256"].items():
        if sha(directory / name) != expected:
            raise ValueError("Frozen neural module changed: " + name)
    with np.load(batched_packet, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key])[0] for key in predict.INPUT_SHAPES}
        arrays["history_time_ns"] = np.asarray(source["history_time_ns"], dtype=np.int64)
        arrays["issue_time_ns"] = np.asarray(arrays["history_time_ns"][-1], dtype=np.int64)
    packet = work / "causal_issue_packet.npz"
    np.savez_compressed(packet, **arrays)
    captured, origins = [], []

    class CapturingModel(CoreForecaster):
        def step(self, state):
            next_state, out = super().step(state)
            captured.append(out)
            origins.append(next_state["origin"])
            return next_state, out

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with patch.object(predict, "CoreForecaster", CapturingModel):
        result = predict.forecast(packet, directory, "cpu")
    contract = metadata["data_contract"]
    scale, offset = contract["normalization"]["std"][0], contract["normalization"]["mean"][0]
    reference = {
        "track_lat_lon": torch.stack([o["center"][0] for o in captured]).numpy(),
        "central_pressure_hpa": torch.stack([o["pressure"][0] for o in captured]).numpy(),
        "basin_mslp_hpa": torch.stack([o["global"][0,0] for o in captured]).numpy()*scale+offset,
        "regional_mslp_hpa": torch.stack([o["regional"][0,0] for o in captured]).numpy()*scale+offset,
        "maximum_wind_auxiliary_kt": torch.stack([o["vmax"][0] for o in captured]).numpy(),
    }
    for key, expected in reference.items():
        np.testing.assert_array_equal(result[key], expected, err_msg="Original readout changed: " + key)
    alignment = []
    for i, (out, origin) in enumerate(zip(captured, origins)):
        # The original model's own geographic sampler, not a centre/minimum fit.
        grid = CoreForecaster.core_grid(None, out["center"][:,0,None,None], out["center"][:,1,None,None], origin)
        sampled = base.sample_field(torch.from_numpy(result["core_mslp_hpa"][i])[None,None], grid)[0,0,0,0]
        alignment.append(abs(float(sampled) - float(result["central_pressure_hpa"][i])))
        np.testing.assert_array_equal(result["core_latitude_deg"][i], out["core_lat"][0].numpy())
        np.testing.assert_array_equal(result["core_longitude_deg"][i], out["core_lon"][0].numpy())
        np.testing.assert_array_equal(result["core_valid"][i], out["core_valid"][0,0].numpy())
    if max(alignment) > 1e-3:
        raise ValueError("Exported core does not reproduce its original central-pressure readout")
    np.savez_compressed(work / "forecast.npz", **result)
    for lead in (6, 120):
        render_pressure_map(result, work / f"pressure_{lead:03d}h.png", lead, 4)
    receipt = {
        "state": "verified_cpu_export_regression",
        "scientific_benchmark": False,
        "model": "Trackformer 1.2", "members": 1,
        "device": "cpu", "forecast_steps": 20,
        "issue_time_ns": int(result["issue_time_ns"]),
        "core_shape": list(result["core_mslp_hpa"].shape),
        "core_min_hpa": float(result["core_mslp_hpa"][result["core_valid"]].min()),
        "core_max_hpa": float(result["core_mslp_hpa"][result["core_valid"]].max()),
        "central_pressure_alignment_max_error_hpa": max(alignment),
        "original_outputs_bit_identical": list(reference),
        "coordinates_and_masks_match_forward_outputs": True,
        "inference_weights_sha256": sha(weights),
        "source_packet_sha256": sha(batched_packet),
        "forecast_sha256": sha(work / "forecast.npz"),
        "native_detail_history_available": bool(arrays["detail_available"][0]),
        "core_is_learned_reconstruction_not_native_observations": True,
    }
    (work / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batched-packet", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    verify(args.batched_packet, args.weights, args.work)
