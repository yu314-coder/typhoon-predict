#!/usr/bin/env python3
"""Run Trackformer 1.2 on a causal issue packet.

The NPZ may contain the route group:
    field [B,6,25,61], context [B,647], base_position [B,20,2]
and/or the intensity group:
    causal_features [B,1020], anchor_structure [B,20,15]

For optional ocean calibration, also provide ocean_features [B,66] and
ocean_available [B]. All arrays must be constructed from data available at
the forecast issue time or earlier. This script never downloads raw data or
official forecast products. It rejects packet keys that name official forecast
sources so an accidental positive-lead input fails early.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from trackformer_1_2 import MODEL_VERSION, Trackformer12, local_position_to_latlon


FORBIDDEN_SOURCE_TOKENS = (
    "jma", "jtwc", "ecmwf", "gfs", "gefs", "weathernext", "deepmind",
    "weatherlab",
)


def load_causal_packet(path: Path) -> dict[str, np.ndarray]:
    """Load an NPZ packet and reject obvious official-forecast source keys."""

    with np.load(path, allow_pickle=False) as source:
        packet = {key: source[key] for key in source.files}
    forbidden = [
        key for key in packet
        if any(token in key.lower() for token in FORBIDDEN_SOURCE_TOKENS)
    ]
    if forbidden:
        raise ValueError(
            "packet contains forbidden official-forecast source keys: "
            + ", ".join(sorted(forbidden))
        )
    return packet


def scalar_metadata(packet: dict[str, np.ndarray], key: str) -> float | None:
    """Return a scalar issue-location metadata value when one is present."""

    value = packet.get(key)
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{key} must contain exactly one scalar value")
    return float(array.reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="causal issue packet .npz")
    parser.add_argument("--output", required=True, type=Path, help="output forecast .npz")
    parser.add_argument("--model-root", type=Path, help="extracted models/trackformer_1_2 directory; omit to use Hugging Face")
    parser.add_argument("--device", default=None, help="auto, cpu, mps, or cuda")
    parser.add_argument("--issue-latitude", type=float, help="optional latitude for route conversion")
    parser.add_argument("--issue-longitude", type=float, help="optional longitude for route conversion")
    args = parser.parse_args()

    packet = load_causal_packet(args.input)
    model = Trackformer12.from_pretrained(args.model_root, device=None if args.device in (None, "auto") else args.device)
    result = model.predict_issue_packet(packet)
    if not result:
        raise ValueError(
            "issue packet has no complete forecast group; provide route keys "
            "(field, context, base_position) and/or intensity keys "
            "(causal_features, anchor_structure)"
        )
    flat: dict[str, np.ndarray] = {}
    for head, values in result.items():
        for key, value in values.items():
            flat[f"{head}_{key}"] = np.asarray(value)
    issue_latitude = args.issue_latitude if args.issue_latitude is not None else scalar_metadata(packet, "issue_latitude")
    issue_longitude = args.issue_longitude if args.issue_longitude is not None else scalar_metadata(packet, "issue_longitude")
    if "route" in result and issue_latitude is not None and issue_longitude is not None:
        latitude, longitude = local_position_to_latlon(result["route"]["position_100km"], issue_latitude, issue_longitude)
        flat["route_latitude"] = latitude
        flat["route_longitude"] = longitude
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **flat)
    print(f"model: Trackformer {MODEL_VERSION}")
    print(f"device: {model.device}")
    print(f"wrote {args.output}")
    for head, values in result.items():
        shapes = ", ".join(f"{key}={np.asarray(value).shape}" for key, value in values.items())
        print(f"{head}: {shapes}")


if __name__ == "__main__":
    main()
