#!/usr/bin/env python3
"""Fail fast if official JMA/JTWC forecast points enter the v37 route.

This is a small release guard, not a claim that static text analysis replaces
review. It checks the production forecast module's ordering and the saved
forecast provenance fields used by the current release artifact.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "v37_protected_forecast.py"
FORECAST = ROOT / "track_build" / "dolphin_v37_current_ibtracs_jma_forecast.json"
COMPARISON = ROOT / "track_build" / "dolphin_v37_current_ibtracs_jma_official_comparison.json"


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    route_start = source.index("def load_gfs_vortex_multilevel_ensemble")
    route_end = source.index("def load_route_models")
    route_code = source[route_start:route_end]
    forbidden_route_tokens = ("parse_current_jma", "parse_current_jtwc", "jma_points", "jtwc_points")
    leaked = [token for token in forbidden_route_tokens if token in route_code]
    if leaked:
        raise AssertionError(f"official forecast token entered route loader: {leaked}")

    run_code = source[source.index("def run_forecast"):]
    route_pos = run_code.index("route_loader()")
    official_pos = run_code.index("parse_current_jma()")
    if route_pos >= official_pos:
        raise AssertionError("official forecast parsing must occur after route generation")

    payload = json.loads(FORECAST.read_text(encoding="utf-8"))
    backbone = f"{payload.get('route_backbone', '')} {payload.get('data_policy', '')}".lower()
    if "jtwc" in backbone or "official forecast" in backbone:
        raise AssertionError("saved route provenance claims an official forecast input")
    if not payload.get("forecast"):
        raise AssertionError("saved v37 forecast is empty")

    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    policy = comparison.get("input_policy", "")
    if "comparison-only" not in policy.lower():
        raise AssertionError("comparison artifact does not mark official forecasts comparison-only")
    print("v37 source policy OK: official JMA/JTWC forecast points are comparison-only")


if __name__ == "__main__":
    main()
