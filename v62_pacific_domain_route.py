#!/usr/bin/env python3
"""Causal western-Pacific state and steering route.

This module first extrapolates a bounded regional atmospheric state from the
current, t-12, and t-24 analysis fields.  It then integrates the track using
steering views that reach across the western Pacific rather than only a small
storm-centered ring.  The pressure map and low-center diagnostics are derived
from the same analysis-only state.

No positive-lead weather field, official forecast track, or future observation
is accepted as an input.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from v61_big_system_route import (
    CURVATURE_VARIANTS,
    SNAPSHOT_WEIGHTS,
    TENDENCY_SCALES,
    build_route,
)


VERSION = "v62-causal-western-pacific-state-route"
PACIFIC_LON_RANGE = (100.0, 190.0)
PACIFIC_LAT_RANGE = (0.0, 60.0)
LEAD_HOURS = tuple(range(0, 121, 6))

# The outer views make Japan, the East China Sea, Taiwan, the Philippines,
# the subtropical ridge, and the western Pacific trough visible together.
PACIFIC_ROUTE_VARIANTS = (
    {"name": "inner_850", "level_weights": (1.0, 0.0, 0.0), "ring_degrees": (3.0, 6.0), "pressure_fraction": 0.05, "weight": 0.10},
    {"name": "deep_inner", "level_weights": (0.269, 0.500, 0.231), "ring_degrees": (4.0, 10.0), "pressure_fraction": 0.10, "weight": 0.14},
    {"name": "broad_850_ridge", "level_weights": (1.0, 0.0, 0.0), "ring_degrees": (8.0, 20.0), "pressure_fraction": 0.15, "weight": 0.18},
    {"name": "pacific_850_environment", "level_weights": (1.0, 0.0, 0.0), "ring_degrees": (12.0, 32.0), "pressure_fraction": 0.18, "weight": 0.18},
    {"name": "pacific_deep_environment", "level_weights": (0.269, 0.500, 0.231), "ring_degrees": (10.0, 30.0), "pressure_fraction": 0.22, "weight": 0.20},
    {"name": "broad_500_trough", "level_weights": (0.0, 1.0, 0.0), "ring_degrees": (12.0, 34.0), "pressure_fraction": 0.16, "weight": 0.11},
    {"name": "outer_200_jet", "level_weights": (0.0, 0.0, 1.0), "ring_degrees": (15.0, 35.0), "pressure_fraction": 0.08, "weight": 0.09},
)


def _clip_delta(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype="float32"), copy=True)
    for channel in range(values.shape[0]):
        scale = float(np.nanpercentile(np.abs(values[channel]), 98.0))
        if scale > 0.0 and math.isfinite(scale):
            values[channel] = np.clip(values[channel], -2.0 * scale, 2.0 * scale)
    return values


def build_pacific_route(
    fields: np.ndarray,
    pressure: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    base_latitude: float,
    base_longitude: float,
    history_motion_km_per_6h: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a broad-domain causal route from analysis-only inputs."""

    members, weights, metadata = build_route(
        fields,
        latitude,
        longitude,
        base_latitude,
        base_longitude,
        pressure,
        available=(1.0, 1.0),
        route_variants=PACIFIC_ROUTE_VARIANTS,
        curvature_variants=CURVATURE_VARIANTS,
        snapshot_weights=SNAPSHOT_WEIGHTS,
        tendency_scales=TENDENCY_SCALES,
        history_motion_km_per_6h=history_motion_km_per_6h,
    )
    metadata = {
        **metadata,
        "version": VERSION,
        "domain": {
            "longitude_east": list(PACIFIC_LON_RANGE),
            "latitude_north": list(PACIFIC_LAT_RANGE),
            "steering_ring_max_degrees": 35.0,
        },
        "large_system_policy": "The route samples the complete analysis grid through broad 850/500/200-hPa and SLP-gradient views; nearby lows are represented by the same causal pressure field.",
    }
    return members, weights, metadata


def forecast_pacific_state(
    fields: np.ndarray,
    pressure: np.ndarray,
    lead_hours: Sequence[int] = LEAD_HOURS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return causal whole-domain pressure and multilevel states.

    The forecast state is a bounded extrapolation of analysis tendency.  It is
    deliberately not a claimed NWP forecast: no future weather product is
    read, and no official forecast field is substituted.
    """

    fields = np.asarray(fields, dtype="float32")
    pressure = np.asarray(pressure, dtype="float32")
    if fields.shape[0] != 3 or pressure.shape[0] != 3:
        raise ValueError(f"expected three causal snapshots, got {fields.shape} and {pressure.shape}")
    recent_fields = fields[0] - fields[1]
    older_fields = fields[1] - fields[2]
    field_tendency = _clip_delta(0.6 * recent_fields + 0.4 * older_fields)
    recent_pressure = pressure[0] - pressure[1]
    older_pressure = pressure[1] - pressure[2]
    pressure_tendency = _clip_delta(0.6 * recent_pressure[None, ...] + 0.4 * older_pressure[None, ...])[0]
    scale = float(np.dot(np.asarray(TENDENCY_SCALES), np.asarray((0.15, 0.45, 0.40))))
    state_fields = []
    state_pressure = []
    for hours in lead_hours:
        progress = min(2.0, 0.5 * max(float(hours), 0.0) / 12.0)
        state_fields.append(fields[0] + scale * progress * field_tendency)
        state_pressure.append(pressure[0] + scale * progress * pressure_tendency)
    return np.stack(state_fields).astype("float32"), np.stack(state_pressure).astype("float32"), {
        "version": VERSION,
        "lead_hours": [int(value) for value in lead_hours],
        "analysis_tendency_scale": scale,
        "tendency_method": "0.6 * (current - t-12) + 0.4 * (t-12 - t-24), clipped per channel at the 98th percentile",
        "input_policy": "current, t-12, and t-24 analysis fields only; no positive-lead or official forecast field",
        "domain": {
            "longitude_east": list(PACIFIC_LON_RANGE),
            "latitude_north": list(PACIFIC_LAT_RANGE),
        },
    }


def _distance_degrees(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    delta_lon = ((lon_a - lon_b + 180.0) % 360.0) - 180.0
    return math.hypot(lat_a - lat_b, delta_lon * math.cos(math.radians(0.5 * (lat_a + lat_b))))


def detect_pressure_systems(
    pressure: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    storm_latitude: float,
    storm_longitude: float,
    maximum: int = 8,
) -> list[dict]:
    """Find candidate closed lows from an analysis-only SLP field.

    These are weather-field vortices, not labels imported from a typhoon
    warning center.  They are used for diagnostics and route context.
    """

    try:
        from scipy.ndimage import maximum_filter, minimum_filter
    except ImportError:
        return []
    latitude = np.asarray(latitude, dtype="float32")
    longitude = np.asarray(longitude, dtype="float32")
    pressure = np.asarray(pressure, dtype="float32")
    lat_mask = (latitude >= PACIFIC_LAT_RANGE[0]) & (latitude <= PACIFIC_LAT_RANGE[1])
    lon_mask = (longitude >= PACIFIC_LON_RANGE[0]) & (longitude <= PACIFIC_LON_RANGE[1])
    if not lat_mask.any() or not lon_mask.any():
        return []
    local = pressure[np.ix_(lat_mask, lon_mask)]
    minimum = minimum_filter(local, size=13, mode="nearest")
    surrounding_maximum = maximum_filter(local, size=41, mode="nearest")
    threshold = float(np.nanpercentile(local, 18.0))
    candidates = np.argwhere((local <= minimum + 0.05) & (local <= threshold) & ((surrounding_maximum - local) >= 2.0))
    rows = []
    lat_values = latitude[lat_mask]
    lon_values = longitude[lon_mask]
    for row, column in candidates:
        lat = float(lat_values[row])
        lon = float(lon_values[column])
        if _distance_degrees(lat, lon, storm_latitude, storm_longitude) < 8.0:
            continue
        rows.append({
            "latitude": round(lat, 3),
            "longitude": round(lon, 3),
            "pressure_hpa": round(float(local[row, column]), 2),
            "local_low_prominence_hpa": round(float(surrounding_maximum[row, column] - local[row, column]), 2),
            "kind": "analysis low/vortex candidate",
        })
    rows.sort(key=lambda item: (-item["local_low_prominence_hpa"], item["pressure_hpa"]))
    selected = []
    for row in rows:
        if any(_distance_degrees(row["latitude"], row["longitude"], item["latitude"], item["longitude"]) < 4.0 for item in selected):
            continue
        selected.append(row)
        if len(selected) >= maximum:
            break
    return selected


__all__ = [
    "VERSION",
    "PACIFIC_LON_RANGE",
    "PACIFIC_LAT_RANGE",
    "LEAD_HOURS",
    "PACIFIC_ROUTE_VARIANTS",
    "build_pacific_route",
    "forecast_pacific_state",
    "detect_pressure_systems",
]
