#!/usr/bin/env python3
"""Causal mean route using multiple analysis-level steering views.

Every member reads only valid-time analysis winds.  The level/radius views are
fixed before inference and are averaged after each route is integrated.  This
keeps disagreement visible while allowing the inner low-level steering flow
to contribute separately from the broad deep-layer mean.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


ROUTE_VARIANTS = (
    {
        "name": "850_hpa_inner_3_5deg",
        "level_weights": (1.0, 0.0, 0.0),
        "ring_min_deg": 3.0,
        "ring_max_deg": 5.0,
        "weight": 0.65,
    },
    {
        "name": "deep_layer_mean_3_8deg",
        "level_weights": (0.269, 0.500, 0.231),
        "ring_min_deg": 3.0,
        "ring_max_deg": 8.0,
        "weight": 0.20,
    },
    {
        "name": "850_hpa_outer_4_7deg",
        "level_weights": (1.0, 0.0, 0.0),
        "ring_min_deg": 4.0,
        "ring_max_deg": 7.0,
        "weight": 0.10,
    },
    {
        "name": "500_hpa_inner_3_5deg",
        "level_weights": (0.0, 1.0, 0.0),
        "ring_min_deg": 3.0,
        "ring_max_deg": 5.0,
        "weight": 0.05,
    },
)
CURVATURE_VARIANTS = (0.0, 0.20, 0.40)
SNAPSHOT_WEIGHTS = (0.50, 0.30, 0.20)


def build_level_analysis_mean_route(
    current_levels: np.ndarray,
    history_levels: np.ndarray,
    available: np.ndarray,
    base_latitude: float,
    base_longitude: float,
    snapshots: Sequence[dict] | None = None,
    route_variants: Sequence[dict] = ROUTE_VARIANTS,
    curvature_variants: Sequence[float] = CURVATURE_VARIANTS,
    snapshot_weights: Sequence[float] = SNAPSHOT_WEIGHTS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return east/north displacement members and normalized weights.

    ``current_levels`` has shape ``(3, 2, 17, 17)`` and history has shape
    ``(6, 2, 17, 17)`` in the order 850/500/200 hPa, u/v.  No target track
    row or positive forecast lead is consumed.
    """

    current_levels = np.asarray(current_levels, dtype="float32")
    history_levels = np.asarray(history_levels, dtype="float32")
    available = np.asarray(available, dtype="float64").reshape(-1)
    if current_levels.shape != (3, 2, 17, 17) or history_levels.shape != (6, 2, 17, 17):
        raise ValueError(
            f"expected current (3,2,17,17) and history (6,2,17,17), got "
            f"{current_levels.shape} and {history_levels.shape}"
        )
    if len(route_variants) == 0:
        raise ValueError("route_variants must not be empty")
    snapshot_base_weights = np.asarray(tuple(snapshot_weights), dtype="float64")
    if snapshot_base_weights.shape != (3,) or np.any(snapshot_base_weights < 0.0):
        raise ValueError("snapshot_weights must contain three non-negative values")
    curvature_values = np.asarray(tuple(curvature_variants), dtype="float64")
    if curvature_values.size == 0 or np.any(~np.isfinite(curvature_values)):
        raise ValueError("curvature_variants must contain finite values")
    curvature_values = np.clip(curvature_values, 0.0, 0.60)
    variant_weights = np.asarray([float(item["weight"]) for item in route_variants], dtype="float64")
    if np.any(~np.isfinite(variant_weights)) or np.any(variant_weights < 0.0) or variant_weights.sum() <= 0.0:
        raise ValueError("route variant weights must be finite and positive")
    variant_weights /= variant_weights.sum()

    valid_snapshot = np.ones(3, dtype="float64")
    if available.size >= 1:
        valid_snapshot[1] = float(available[0] > 0.5)
    if available.size >= 2:
        valid_snapshot[2] = float(available[1] > 0.5)
    weighted_snapshots = snapshot_base_weights * valid_snapshot
    if weighted_snapshots.sum() <= 0.0:
        raise RuntimeError("level analysis route has no valid analysis snapshot")
    weighted_snapshots /= weighted_snapshots.sum()

    fields = [
        current_levels,
        history_levels[:3],
        history_levels[3:6],
    ]
    grid_limit = 20.0
    grid_step = 2.5
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False, dtype="float32")
    earth_radius_m = 6_371_000.0

    def bilinear(field: np.ndarray, north: np.ndarray, east: np.ndarray) -> np.ndarray:
        row = np.clip((north + grid_limit) / grid_step, 0.0, 16.0)
        column = np.clip((east + grid_limit) / grid_step, 0.0, 16.0)
        row0 = np.floor(row).astype("int64")
        column0 = np.floor(column).astype("int64")
        row1 = np.minimum(row0 + 1, 16)
        column1 = np.minimum(column0 + 1, 16)
        row_fraction = row - row0
        column_fraction = column - column0
        return np.stack(
            [
                field[channel, row0, column0] * (1.0 - row_fraction) * (1.0 - column_fraction)
                + field[channel, row1, column0] * row_fraction * (1.0 - column_fraction)
                + field[channel, row0, column1] * (1.0 - row_fraction) * column_fraction
                + field[channel, row1, column1] * row_fraction * column_fraction
                for channel in range(2)
            ],
            axis=1,
        )

    snapshot_rows = [dict(item) for item in (snapshots or ({}, {}, {}))]
    while len(snapshot_rows) < 3:
        snapshot_rows.append({})
    route_states: list[np.ndarray] = []
    route_weights: list[float] = []
    member_rows: list[dict] = []
    for snapshot_index, snapshot in enumerate(fields):
        if not valid_snapshot[snapshot_index]:
            continue
        snapshot_rows[snapshot_index] = {
            **snapshot_rows[snapshot_index],
            "snapshot_index": snapshot_index,
            "ensemble_snapshot_weight": round(float(weighted_snapshots[snapshot_index]), 8),
        }
        for variant_index, variant in enumerate(route_variants):
            level_weights = np.asarray(variant["level_weights"], dtype="float32")
            if level_weights.shape != (3,) or level_weights.sum() <= 0.0:
                raise ValueError(f"invalid level weights for {variant.get('name', variant_index)}")
            level_weights /= level_weights.sum()
            wind_field = np.tensordot(level_weights, snapshot, axes=(0, 0)).astype("float32")
            radii = np.arange(
                float(variant["ring_min_deg"]),
                float(variant["ring_max_deg"]) + 0.001,
                0.5,
                dtype="float32",
            )
            ring_north = (radii[:, None] * np.sin(angles)[None, :]).reshape(-1)
            ring_east = (radii[:, None] * np.cos(angles)[None, :]).reshape(-1)

            def flow_from(north: float = 0.0, east: float = 0.0) -> tuple[float, float]:
                samples = bilinear(wind_field, north + ring_north, east + ring_east)
                return float(samples[:, 0].mean()), float(samples[:, 1].mean())

            base_u, base_v = flow_from()
            base_step = np.asarray(
                [(0.76 * base_u - 2.03) * 21.6, (0.78 * base_v + 0.40) * 21.6],
                dtype="float32",
            )
            for curvature_fraction in curvature_values:
                latitude = float(base_latitude)
                longitude = float(base_longitude)
                steps = np.zeros((20, 2), dtype="float32")
                waypoints = []
                clamped_samples = 0
                for lead in range(20):
                    relative_north = latitude - float(base_latitude)
                    relative_east = ((longitude - float(base_longitude) + 180.0) % 360.0) - 180.0
                    sample_north = relative_north + ring_north
                    sample_east = relative_east + ring_east
                    clamped_samples += int(
                        np.count_nonzero(
                            (sample_north < -grid_limit)
                            | (sample_north > grid_limit)
                            | (sample_east < -grid_limit)
                            | (sample_east > grid_limit)
                        )
                    )
                    local_u, local_v = flow_from(relative_north, relative_east)
                    local_step = np.asarray(
                        [(0.76 * local_u - 2.03) * 21.6, (0.78 * local_v + 0.40) * 21.6],
                        dtype="float32",
                    )
                    step = (1.0 - float(curvature_fraction)) * base_step + float(curvature_fraction) * local_step
                    steps[lead] = step
                    latitude += float(step[1]) / 111.2
                    longitude += float(step[0]) / (111.2 * max(math.cos(math.radians(latitude)), 0.15))
                    longitude %= 360.0
                    if lead in (0, 3, 7, 11, 15, 19):
                        waypoints.append(
                            {
                                "lead_hours": (lead + 1) * 6,
                                "latitude": round(latitude, 4),
                                "longitude": round(longitude, 4),
                                "u_mean_mps": round(local_u, 4),
                                "v_mean_mps": round(local_v, 4),
                                "east_km_per_6h": round(float(step[0]), 3),
                                "north_km_per_6h": round(float(step[1]), 3),
                            }
                        )
                member_index = len(route_states)
                route_states.append(steps)
                route_weights.append(
                    float(weighted_snapshots[snapshot_index])
                    * float(variant_weights[variant_index])
                    / float(curvature_values.size)
                )
                member_rows.append(
                    {
                        "member_index": member_index,
                        "snapshot_index": snapshot_index,
                        "variant_index": variant_index,
                        "variant": str(variant["name"]),
                        "level_weights": level_weights.tolist(),
                        "ring_degrees": [float(variant["ring_min_deg"]), float(variant["ring_max_deg"])],
                        "curvature_fraction": round(float(curvature_fraction), 4),
                        "variant_weight": round(float(variant_weights[variant_index]), 8),
                        "boundary_clamped_ring_samples": clamped_samples,
                        "sampled_waypoints": waypoints,
                    }
                )

    states = np.stack(route_states).astype("float32")
    weights = np.asarray(route_weights, dtype="float64")
    weights /= weights.sum()
    metadata = {
        "available": True,
        "policy": "strict analysis-only multi-level/radius ensemble mean",
        "route_input_policy": (
            "Only current, t-12 h, and t-24 h analysis wind fields at or before the issue time were used. "
            "No forecast lead field, official forecast track, future observed row, or Tip data was used."
        ),
        "analysis_field_snapshots": snapshot_rows,
        "route_weights": weights.tolist(),
        "snapshot_weights": weighted_snapshots.tolist(),
        "route_variants": [
            {
                "name": str(item["name"]),
                "level_weights": list(item["level_weights"]),
                "ring_degrees": [float(item["ring_min_deg"]), float(item["ring_max_deg"])],
                "weight": float(variant_weights[index]),
            }
            for index, item in enumerate(route_variants)
        ],
        "curvature_variants": [round(float(value), 4) for value in curvature_values],
        "ensemble_mean_method": (
            f"weighted mean after integrating {len(member_rows)} members: "
            f"{int(np.count_nonzero(valid_snapshot))} causal snapshots x "
            f"{len(route_variants)} level/radius views x {curvature_values.size} curvature variants"
        ),
        "member_count": len(member_rows),
        "grid_spacing_degrees": 2.5,
        "curvature_method": "bilinear analysis-wind sampling around each evolving predicted center",
        "motion_slopes": [0.76, 0.78],
        "beta_intercept_mps": [-2.03, 0.40],
        "forecast_files_opened": [],
    }
    return states, weights, metadata
