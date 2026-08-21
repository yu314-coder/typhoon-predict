#!/usr/bin/env python3
"""Run the public Trackformer 1.2 API on a causal issue packet.

The NPZ must contain either the route group:
    field [B,6,25,61], context [B,647], base_position [B,20,2]
or the intensity group:
    causal_features [B,1020], anchor_structure [B,20,15]

For the optional ocean calibration, also provide ocean_features [B,66] and
ocean_available [B].  All arrays must be constructed from data available at
the forecast issue time or earlier.  This script never downloads official
forecast products.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from trackformer_1_2 import Trackformer12, local_position_to_latlon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="causal issue packet .npz")
    parser.add_argument("--output", required=True, type=Path, help="output .npz")
    parser.add_argument("--model-root", type=Path, help="extracted models/trackformer_1_2 directory")
    parser.add_argument("--device", default=None, help="auto, cpu, mps, or cuda")
    parser.add_argument("--issue-latitude", type=float, help="optional latitude for route conversion")
    parser.add_argument("--issue-longitude", type=float, help="optional longitude for route conversion")
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as source:
        packet = {key: source[key] for key in source.files}
    model = Trackformer12.from_pretrained(args.model_root, device=None if args.device in (None, "auto") else args.device)
    result = model.predict_issue_packet(packet)
    flat: dict[str, np.ndarray] = {}
    for head, values in result.items():
        for key, value in values.items():
            flat[f"{head}_{key}"] = np.asarray(value)
    if "route" in result and args.issue_latitude is not None and args.issue_longitude is not None:
        latitude, longitude = local_position_to_latlon(result["route"]["position_100km"], args.issue_latitude, args.issue_longitude)
        flat["route_latitude"] = latitude
        flat["route_longitude"] = longitude
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **flat)
    print(f"wrote {args.output}")
    print("heads:", ", ".join(result))


if __name__ == "__main__":
    main()
