#!/usr/bin/env python3
"""Evaluate the 1.2.12 causal route adapter with the 1.2.11 intensity head."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import v212_causal_motion_route as v212  # noqa: E402
import scripts.test_trackformer_1211_intensity_dolphin as base  # noqa: E402


OUT_ROOT = ROOT / "v212" / "causal_motion_route"
OUT_JSON = OUT_ROOT / "dolphin_20260807_evaluation.json"
OUT_GRAPH = ROOT / "paper" / "dolphin_trackformer_1_2_12_motion_route_graph.png"
OUT_MAP = ROOT / "paper" / "dolphin_trackformer_1_2_12_motion_route_world_map.png"
OUT_HTML = ROOT / "paper" / "dolphin_trackformer_1_2_12_motion_route.html"


def _route_and_environment(history: list[dict], inputs: dict[str, np.ndarray], sst: np.ndarray):
    """Build the base causal route, then apply only the issue-time motion adapter."""

    v127 = base.v127_case.v127
    fields, pressure, latitude, longitude = base.dolphin._load_analysis()
    base_lat = float(history[-1]["lat"])
    base_lon = float(history[-1]["lon"])
    base_displacement, diagnostics = v127.build_weighted_route_fast(
        fields,
        latitude,
        longitude,
        base_lat,
        base_lon,
        pressure,
        available=(1.0, 1.0),
        route_variants=v127.ROUTE_VARIANTS_ACTIVE,
        curvature_variants=v127.CURVATURE_VARIANTS_ACTIVE,
        tendency_scales=v127.TENDENCY_SCALES_ACTIVE,
        history_motion_km_per_6h=base.dolphin._recent_motion(history),
    )
    displacement, motion_diagnostics = v212.apply_motion_anchor(
        base_displacement,
        base.dolphin._recent_motion(history),
    )
    cumulative = np.cumsum(displacement, axis=0)
    lat = base_lat + cumulative[:, 1] / 111.2
    lon = (base_lon + cumulative[:, 0] / (111.2 * np.maximum(np.cos(np.deg2rad(lat)), 0.20))) % 360.0
    terrain_np = np.load(v127.TERRAIN_PATH, allow_pickle=False)
    terrain = {key: terrain_np[key] for key in terrain_np.files}
    land, distance = v127._land_features(lat, lon, terrain)
    environment = v127._environment_features(
        fields[0],
        fields[1],
        pressure[0],
        pressure[1],
        sst,
        True,
        base_lat,
        base_lon,
        lat,
        lon,
        cumulative,
        land,
        distance,
        latitude,
        longitude,
    )
    points = np.concatenate([
        np.asarray([[base_lat, base_lon]], dtype="float32"),
        np.stack([lat, lon], axis=1).astype("float32"),
    ])
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "version": v212.VERSION,
        "route_policy": "Trackformer 1.1 causal analysis route plus validation-selected motion anchor",
        "base_route_displacement_last_km": base_displacement[-1].astype("float32").tolist(),
        "motion_anchor": motion_diagnostics,
        "positions": points.tolist(),
        "environment_feature_names": [str(value) for value in v127.ENV_FEATURE_NAMES],
    })
    return points, environment, diagnostics


def main() -> None:
    # Reuse the already quality-gated 1.2.11 intensity inference while
    # replacing only its causal route/environment builder.
    base.v127_case._route_and_environment = _route_and_environment
    base.OUT_ROOT = OUT_ROOT
    base.OUT_JSON = OUT_JSON
    base.OUT_GRAPH = OUT_GRAPH
    base.OUT_MAP = OUT_MAP
    base.OUT_HTML = OUT_HTML
    base.MODEL_NAME = "Trackformer 1.2.12"
    base.MODEL_VERSION = "1.2.12"
    base.ROUTE_LABEL = "Trackformer 1.2.12 causal motion-anchored route"
    base.INFERENCE_NOTICE = (
        "Trackformer 1.2.12 starts with the current/past analysis-field route and applies a "
        "validation-selected, speed-capped, decaying anchor from observed issue-time motion. "
        "The route remains curved by the causal analysis fields. JMA/JTWC paths are loaded only "
        "as comparison overlays and were never used as model inputs or calibration targets. "
        "The intensity head remains the frozen 1.2.11 head evaluated on the updated route."
    )
    payload = base._load_case()
    payload["version"] = "Trackformer1.2.12-causal-motion-route-with-1.2.11-intensity-Dolphin"
    payload["model"] = "Trackformer 1.2.12 causal motion-anchored route + Trackformer 1.2.11 intensity"
    payload["scores"]["trackformer_1_2_12_route"] = payload["scores"].pop("trackformer_1_2_11")
    payload["route_policy"] = {
        "manifest": str((OUT_ROOT / "trackformer_1_2_12_route_policy.json").relative_to(ROOT)),
        "base_route": "Trackformer 1.1 current/t-12/t-24 analysis ensemble",
        "adapter": v212.VERSION,
        "official_forecasts_used_as_inputs": False,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    base._draw_graph(payload)
    base._draw_map(payload)
    base._write_html(payload)
    print(json.dumps({
        "json": str(OUT_JSON),
        "graph": str(OUT_GRAPH),
        "map": str(OUT_MAP),
        "html": str(OUT_HTML),
        "scores": payload["scores"],
        "official_forecasts_used_as_inputs": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
