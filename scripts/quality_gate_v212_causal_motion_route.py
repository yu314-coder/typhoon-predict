#!/usr/bin/env python3
"""Evaluate the causal 1.2.12 route adapter on storm-held-out splits."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import v212_causal_motion_route as v212  # noqa: E402


TRACK_PATH = ROOT / "track_build" / "track_windows_v13.npz"
ROUTE_PATH = ROOT / "track_build" / "v127_v11_full_route_environment.npz"
OUT_PATH = ROOT / "v212" / "causal_motion_route" / "trackformer_1_2_12_route_policy.json"


def _first_storm_year(storm_id: np.ndarray, year: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}
    for storm, value in zip(storm_id.astype(str), year.astype(int)):
        result[str(storm)] = min(result.get(str(storm), 9999), int(value))
    return result


def _metrics(route: np.ndarray, target: np.ndarray) -> dict:
    error = np.linalg.norm(
        np.cumsum(route, axis=1) - np.cumsum(target, axis=1), axis=2
    )
    return {
        "windows": int(len(route)),
        "mean_track_error_km": round(float(error.mean()), 4),
        "track_error_120h_km": round(float(error[:, -1].mean()), 4),
        "endpoint_rms_km": round(float(np.sqrt(np.mean(error[:, -1] ** 2))), 4),
        "track_error_by_lead_km": [round(float(value), 4) for value in error.mean(axis=0)],
    }


def main() -> None:
    with np.load(TRACK_PATH, allow_pickle=False) as track, np.load(ROUTE_PATH, allow_pickle=False) as route:
        route_indices = route["indices"].astype("int64")
        base_route = route["displacement_km"].astype("float32")
        target = track["target"][route_indices, :, :2].astype("float32")
        raw_motion = (
            track["track"][route_indices, -1, 2:4].astype("float32")
            * track["track_std"][None, 2:4]
            + track["track_mean"][None, 2:4]
        )
        first_year = _first_storm_year(track["storm_id"], track["year"])
        storm_ids = track["storm_id"][route_indices].astype(str)

    split_positions = {
        "train": np.asarray([i for i, storm in enumerate(storm_ids) if first_year[storm] <= 2015], dtype="int64"),
        "val": np.asarray([i for i, storm in enumerate(storm_ids) if 2015 < first_year[storm] <= 2019], dtype="int64"),
        "test": np.asarray([i for i, storm in enumerate(storm_ids) if first_year[storm] >= 2020], dtype="int64"),
    }
    corrected = np.stack([
        v212.apply_motion_anchor(
            base_route[index], raw_motion[index],
            zonal_max_weight=v212.DEFAULT_ZONAL_MAX_WEIGHT,
            meridional_max_weight=v212.DEFAULT_MERIDIONAL_MAX_WEIGHT,
            zonal_tau_hours=v212.DEFAULT_ZONAL_TAU_HOURS,
            meridional_tau_hours=v212.DEFAULT_MERIDIONAL_TAU_HOURS,
            motion_cap_km_per_6h=v212.DEFAULT_MOTION_CAP_KM_PER_6H,
        )[0]
        for index in range(len(base_route))
    ]).astype("float32")

    metrics = {}
    for name, positions in split_positions.items():
        metrics[name] = {
            "baseline_v1_1": _metrics(base_route[positions], target[positions]),
            "candidate_v1_2_12": _metrics(corrected[positions], target[positions]),
        }

    payload = {
        "version": v212.VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection": {
            "fit_split": "2016-2019 validation storms",
            "evaluation_split": "untouched 2020+ test storms",
            "policy": "fixed capped exponentially decaying recent-motion anchor selected without official forecast paths",
            "zonal_max_weight": v212.DEFAULT_ZONAL_MAX_WEIGHT,
            "meridional_max_weight": v212.DEFAULT_MERIDIONAL_MAX_WEIGHT,
            "zonal_tau_hours": v212.DEFAULT_ZONAL_TAU_HOURS,
            "meridional_tau_hours": v212.DEFAULT_MERIDIONAL_TAU_HOURS,
            "motion_cap_km_per_6h": v212.DEFAULT_MOTION_CAP_KM_PER_6H,
        },
        "input_policy": {
            "current_and_past_analysis_fields": True,
            "issue_time_recent_motion": True,
            "future_observed_rows_as_inputs": False,
            "official_forecasts_as_inputs": False,
            "jma_or_jtwc_paths_used_for_calibration": False,
        },
        "metrics": metrics,
        "promotion_rule": "candidate must improve both validation and untouched test mean and 120-hour route error; official forecasts remain comparison-only overlays",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
