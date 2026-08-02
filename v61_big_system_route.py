#!/usr/bin/env python3
"""Causal large-system steering ensemble for out-of-archive cases.

v61 is a physical route candidate, not a consensus of official forecasts.
Each member reads only analysis snapshots at or before the issue time.  It
samples multiple pressure levels over inner and synoptic-scale rings around
the evolving center, and a small SLP-gradient component can represent the
large-scale pressure field.  The final route is the weighted mean of the
integrated members so curvature and disagreement remain visible.

Input field layout:
    fields: (snapshots, 7, latitude, longitude)
    channels: hgt500, u850, v850, u500, v500, u200, v200
    snapshots: current, t-12, t-24 (earlier snapshots may be zeroed)
    pressure: (snapshots, latitude, longitude) SLP in hPa, optional

No forecast lead field, official forecast track, or future observation is
accepted by this module.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


VERSION = "v61-causal-dynamic-big-system-steering-ensemble"
LEADS = 20
LEVEL_WEIGHTS = np.asarray([0.269, 0.500, 0.231], dtype="float32")
MOTION_SLOPES = np.asarray([0.76, 0.78], dtype="float32")
MOTION_INTERCEPTS = np.asarray([-2.03, 0.40], dtype="float32")
TENDENCY_SCALES = (0.0, 0.65, 1.25)
TENDENCY_SCALE_WEIGHTS = (0.15, 0.45, 0.40)


ROUTE_VARIANTS = (
    {
        "name": "inner_850",
        "level_weights": (1.0, 0.0, 0.0),
        "ring_degrees": (3.0, 5.0),
        "pressure_fraction": 0.00,
        "weight": 0.30,
    },
    {
        "name": "deep_layer_inner",
        "level_weights": (0.269, 0.500, 0.231),
        "ring_degrees": (3.0, 8.0),
        "pressure_fraction": 0.04,
        "weight": 0.22,
    },
    {
        "name": "broad_850_ridge",
        "level_weights": (1.0, 0.0, 0.0),
        "ring_degrees": (5.0, 11.0),
        "pressure_fraction": 0.05,
        "weight": 0.16,
    },
    {
        "name": "broad_500_trough",
        "level_weights": (0.0, 1.0, 0.0),
        "ring_degrees": (5.0, 12.0),
        "pressure_fraction": 0.04,
        "weight": 0.12,
    },
    {
        "name": "synoptic_deep",
        "level_weights": (0.269, 0.500, 0.231),
        "ring_degrees": (7.0, 16.0),
        "pressure_fraction": 0.10,
        "weight": 0.12,
    },
    {
        "name": "outer_200_jet",
        "level_weights": (0.0, 0.0, 1.0),
        "ring_degrees": (8.0, 18.0),
        "pressure_fraction": 0.02,
        "weight": 0.08,
    },
)
CURVATURE_VARIANTS = (0.0, 0.20, 0.40)
SNAPSHOT_WEIGHTS = (0.50, 0.30, 0.20)


def _sorted_axes(latitude: np.ndarray, longitude: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitude = np.asarray(latitude, dtype="float32").reshape(-1)
    longitude = np.asarray(longitude, dtype="float32").reshape(-1)
    field = np.asarray(field, dtype="float32")
    lat_order = np.argsort(latitude)
    lon_order = np.argsort(longitude)
    return latitude[lat_order], longitude[lon_order], field[..., lat_order, :][..., :, lon_order]


def _longitude_queries(values: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    low, high = float(longitude[0]), float(longitude[-1])
    if low >= 0.0 and high > 180.0:
        return np.mod(values, 360.0)
    if low < 0.0 and high <= 180.0:
        return ((values + 180.0) % 360.0) - 180.0
    return np.clip(values, low, high)


def _bilinear(field: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, query_lat: np.ndarray, query_lon: np.ndarray) -> np.ndarray:
    """Bilinear sample a 2-D field or a 2-channel field on a regular grid."""

    lat, lon, values = _sorted_axes(latitude, longitude, field)
    query_lat = np.clip(np.asarray(query_lat, dtype="float64"), float(lat[0]), float(lat[-1]))
    query_lon = _longitude_queries(query_lon, lon)
    row = np.interp(query_lat, lat, np.arange(len(lat), dtype="float64"))
    column = np.interp(query_lon, lon, np.arange(len(lon), dtype="float64"))
    row0 = np.floor(row).astype("int64")
    col0 = np.floor(column).astype("int64")
    row1 = np.minimum(row0 + 1, len(lat) - 1)
    col1 = np.minimum(col0 + 1, len(lon) - 1)
    rf = row - row0
    cf = column - col0
    if values.ndim == 2:
        return (
            values[row0, col0] * (1.0 - rf) * (1.0 - cf)
            + values[row1, col0] * rf * (1.0 - cf)
            + values[row0, col1] * (1.0 - rf) * cf
            + values[row1, col1] * rf * cf
        ).astype("float32")
    return np.stack([
        _bilinear(values[channel], lat, lon, query_lat, query_lon)
        for channel in range(values.shape[0])
    ], axis=1).astype("float32")


def _geostrophic_field(pressure: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Approximate geostrophic wind from causal SLP gradients."""

    pressure = np.asarray(pressure, dtype="float32")
    lat, lon, sorted_pressure = _sorted_axes(latitude, longitude, pressure)
    earth_radius = 6_371_000.0
    omega = 7.2921159e-5
    dy = np.gradient(np.deg2rad(lat) * earth_radius)
    dx = np.deg2rad(float(np.median(np.diff(lon)))) * earth_radius * np.cos(np.deg2rad(lat))
    dp_dy = np.gradient(sorted_pressure.astype("float64"), axis=0) * 100.0 / dy[:, None]
    dp_dx = np.gradient(sorted_pressure.astype("float64"), axis=1) * 100.0 / dx[:, None]
    coriolis = 2.0 * omega * np.sin(np.deg2rad(lat))
    coriolis = np.where(np.abs(coriolis) < 2.0e-5, np.sign(coriolis) * 2.0e-5, coriolis)
    coriolis = np.where(coriolis == 0.0, 2.0e-5, coriolis)
    rho = 1.15
    u = -dp_dy / (rho * coriolis[:, None])
    v = dp_dx / (rho * coriolis[:, None])
    return np.stack([u, v]).astype("float32")


def _ring(ring_degrees: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    radii = np.arange(float(ring_degrees[0]), float(ring_degrees[1]) + 0.01, 1.0, dtype="float32")
    angles = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False, dtype="float32")
    north = (radii[:, None] * np.sin(angles)[None, :]).reshape(-1)
    east = (radii[:, None] * np.cos(angles)[None, :]).reshape(-1)
    return north, east


def _step(flow: tuple[float, float]) -> np.ndarray:
    u, v = flow
    return np.asarray([
        (float(MOTION_SLOPES[0]) * u + float(MOTION_INTERCEPTS[0])) * 21.6,
        (float(MOTION_SLOPES[1]) * v + float(MOTION_INTERCEPTS[1])) * 21.6,
    ], dtype="float32")


def _clip_tendency(values: np.ndarray) -> np.ndarray:
    """Limit noisy analysis differences without using any future frame."""

    values = np.nan_to_num(np.asarray(values, dtype="float32"), copy=True)
    if values.ndim < 3:
        return values
    for channel in range(values.shape[0]):
        scale = float(np.nanpercentile(np.abs(values[channel]), 98.0))
        if scale > 0.0 and math.isfinite(scale):
            values[channel] = np.clip(values[channel], -2.0 * scale, 2.0 * scale)
    return values


def build_route(
    fields: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    base_latitude: float,
    base_longitude: float,
    pressure: np.ndarray | None = None,
    available: Sequence[float] = (1.0, 1.0),
    route_variants: Sequence[dict] = ROUTE_VARIANTS,
    curvature_variants: Sequence[float] = CURVATURE_VARIANTS,
    snapshot_weights: Sequence[float] = SNAPSHOT_WEIGHTS,
    tendency_scales: Sequence[float] = TENDENCY_SCALES,
    history_motion_km_per_6h: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return dynamic member displacements, weights, and causal diagnostics.

    The current analysis is advanced with a clipped tendency estimated only
    from the two preceding analysis frames.  The fields are never replaced by
    a positive-lead forecast.  Every member still starts at the same issue
    position, but its broad steering environment can evolve with lead.
    """

    fields = np.asarray(fields, dtype="float32")
    if fields.ndim != 4 or fields.shape[0] != 3 or fields.shape[1] != 7:
        raise ValueError(f"expected fields (3,7,H,W), got {fields.shape}")
    latitude = np.asarray(latitude, dtype="float32")
    longitude = np.asarray(longitude, dtype="float32")
    if pressure is None:
        pressure = np.zeros((3, len(latitude), len(longitude)), dtype="float32")
        pressure_available = np.zeros(3, dtype=bool)
    else:
        pressure = np.asarray(pressure, dtype="float32")
        if pressure.shape != (3, len(latitude), len(longitude)):
            raise ValueError(f"expected pressure (3,H,W), got {pressure.shape}")
        pressure_available = np.isfinite(pressure).all(axis=(1, 2))
    valid_snapshot = np.ones(3, dtype=bool)
    avail = np.asarray(available, dtype="float64").reshape(-1)
    if avail.size >= 1:
        valid_snapshot[1] = bool(avail[0] > 0.5)
    if avail.size >= 2:
        valid_snapshot[2] = bool(avail[1] > 0.5)
    valid_snapshot &= np.isfinite(fields).all(axis=(1, 2, 3))
    if not valid_snapshot[0]:
        raise RuntimeError("current causal analysis snapshot is unavailable")
    snap_weights = np.asarray(snapshot_weights, dtype="float64") * valid_snapshot
    snap_weights /= snap_weights.sum()
    curve_values = np.asarray(tuple(curvature_variants), dtype="float64")
    curve_values = np.clip(curve_values[np.isfinite(curve_values)], 0.0, 0.60)
    if not len(curve_values):
        raise ValueError("curvature_variants is empty")
    tendency_values = np.asarray(tuple(tendency_scales), dtype="float64")
    tendency_weights = np.asarray(tuple(TENDENCY_SCALE_WEIGHTS), dtype="float64")
    if tendency_values.shape != tendency_weights.shape or np.any(~np.isfinite(tendency_values)):
        raise ValueError("tendency_scales must match the fixed tendency weight table")
    tendency_values = np.maximum(tendency_values, 0.0)
    tendency_weights = np.maximum(tendency_weights, 0.0)
    tendency_weights /= tendency_weights.sum()
    variant_weights = np.asarray([float(item["weight"]) for item in route_variants], dtype="float64")
    variant_weights = np.maximum(variant_weights, 0.0)
    variant_weights /= variant_weights.sum()
    geo_fields = np.zeros((3, 2, len(latitude), len(longitude)), dtype="float32")
    for index in range(3):
        if pressure_available[index]:
            geo_fields[index] = _geostrophic_field(pressure[index], latitude, longitude)

    current_fields = fields[0]
    recent_delta = _clip_tendency(fields[0] - fields[1]) if valid_snapshot[1] else np.zeros_like(current_fields)
    older_delta = _clip_tendency(fields[1] - fields[2]) if valid_snapshot[2] else recent_delta.copy()
    tendency_views = (
        recent_delta,
        _clip_tendency(0.5 * (recent_delta + older_delta)),
        older_delta,
    )
    current_geo = geo_fields[0]
    recent_geo_delta = _clip_tendency(geo_fields[0] - geo_fields[1]) if pressure_available[1] else np.zeros_like(current_geo)
    older_geo_delta = _clip_tendency(geo_fields[1] - geo_fields[2]) if pressure_available[2] else recent_geo_delta.copy()
    geo_tendency_views = (
        recent_geo_delta,
        _clip_tendency(0.5 * (recent_geo_delta + older_geo_delta)),
        older_geo_delta,
    )
    history_step = None
    if history_motion_km_per_6h is not None:
        candidate = np.asarray(history_motion_km_per_6h, dtype="float32").reshape(-1)
        if candidate.size == 2 and np.isfinite(candidate).all():
            history_step = candidate

    members: list[np.ndarray] = []
    weights: list[float] = []
    member_rows: list[dict] = []
    current_wind_levels = np.stack(
        [current_fields[1:3], current_fields[3:5], current_fields[5:7]],
        axis=0,
    )
    for snapshot_index in range(3):
        if snap_weights[snapshot_index] <= 0.0:
            continue
        for variant_index, variant in enumerate(route_variants):
            level_weights = np.asarray(variant["level_weights"], dtype="float32")
            level_weights /= level_weights.sum()
            ring_north, ring_east = _ring(tuple(variant["ring_degrees"]))
            wind_field = np.tensordot(level_weights, current_wind_levels, axes=(0, 0)).astype("float32")
            wind_tendency = np.tensordot(
                level_weights,
                np.stack(
                    [
                        tendency_views[snapshot_index][1:3],
                        tendency_views[snapshot_index][3:5],
                        tendency_views[snapshot_index][5:7],
                    ],
                    axis=0,
                ),
                axes=(0, 0),
            ).astype("float32")
            pressure_field = current_geo
            geo_tendency = geo_tendency_views[snapshot_index]
            pressure_fraction = float(variant.get("pressure_fraction", 0.0)) if pressure_available[0] else 0.0

            def flow_from(field: np.ndarray, lat: float, lon: float) -> tuple[float, float]:
                sample_lat = float(lat) + ring_north
                sample_lon = float(lon) + ring_east / max(math.cos(math.radians(float(lat))), 0.20)
                samples = _bilinear(field, latitude, longitude, sample_lat, sample_lon)
                return float(np.nanmean(samples[:, 0])), float(np.nanmean(samples[:, 1]))

            for tendency_index, tendency_scale in enumerate(tendency_values):
                for curvature in curve_values:
                    lat = float(base_latitude)
                    lon = float(base_longitude)
                    previous_step = _step(flow_from(wind_field, lat, lon))
                    steps = np.zeros((LEADS, 2), dtype="float32")
                    waypoints = []
                    for lead in range(LEADS):
                        # A bounded extrapolation of the observed analysis
                        # tendency lets the steering regime change with lead.
                        progress = min(2.0, 0.5 * ((lead + 1) * 6.0 / 12.0))
                        dynamic_wind = wind_field + float(tendency_scale) * progress * wind_tendency
                        dynamic_geo = pressure_field + float(tendency_scale) * progress * geo_tendency
                        local_u, local_v = flow_from(dynamic_wind, lat, lon)
                        local_geo_u, local_geo_v = flow_from(dynamic_geo, lat, lon) if pressure_fraction else (0.0, 0.0)
                        local_flow = (
                            (1.0 - pressure_fraction) * local_u + pressure_fraction * local_geo_u,
                            (1.0 - pressure_fraction) * local_v + pressure_fraction * local_geo_v,
                        )
                        weather_step = _step(local_flow)
                        if history_step is not None:
                            history_weight = 0.20 * math.exp(-((lead + 1) * 6.0) / 36.0)
                            weather_step = (1.0 - history_weight) * weather_step + history_weight * history_step
                        inertia = min(0.18, 0.04 + 0.08 * float(curvature)) if lead else 0.0
                        step = (1.0 - inertia) * weather_step + inertia * previous_step
                        previous_step = step
                        steps[lead] = step
                        lat += float(step[1]) / 111.2
                        lon += float(step[0]) / (111.2 * max(math.cos(math.radians(lat)), 0.20))
                        lon %= 360.0
                        if lead in (0, 3, 7, 11, 15, 19):
                            waypoints.append({
                                "lead_hours": (lead + 1) * 6,
                                "latitude": round(lat, 4),
                                "longitude": round(lon, 4),
                                "u_mean_mps": round(local_flow[0], 4),
                                "v_mean_mps": round(local_flow[1], 4),
                                "tendency_progress": round(progress, 4),
                            })
                    members.append(steps)
                    weights.append(
                        float(snap_weights[snapshot_index])
                        * float(variant_weights[variant_index])
                        * float(tendency_weights[tendency_index])
                        / float(len(curve_values))
                    )
                    member_rows.append({
                        "member_index": len(members) - 1,
                        "snapshot_index": snapshot_index,
                        "variant": str(variant["name"]),
                        "ring_degrees": list(variant["ring_degrees"]),
                        "level_weights": level_weights.tolist(),
                        "pressure_fraction": pressure_fraction,
                        "curvature_fraction": round(float(curvature), 4),
                        "tendency_scale": round(float(tendency_scale), 4),
                        "sampled_waypoints": waypoints,
                    })
    member_weights = np.asarray(weights, dtype="float64")
    member_weights /= member_weights.sum()
    return np.stack(members).astype("float32"), member_weights, {
        "version": VERSION,
        "policy": "causal dynamic multi-level large-system analysis ensemble",
        "input_policy": "current, t-12, and t-24 analysis fields only; the lead evolution is a bounded extrapolation of their observed tendency; no future analysis, forecast product, official track, or future observed row",
        "channels": ["hgt500", "u850", "v850", "u500", "v500", "u200", "v200"],
        "snapshot_weights": snap_weights.tolist(),
        "member_weights": member_weights.tolist(),
        "member_count": len(member_rows),
        "route_variants": [dict(item) for item in route_variants],
        "curvature_variants": [float(value) for value in curve_values],
        "tendency_scales": [float(value) for value in tendency_values],
        "tendency_weights": tendency_weights.tolist(),
        "tendency_method": "bounded current-minus-past analysis tendency, capped at two 12-hour differences",
        "history_motion_km_per_6h": None if history_step is None else history_step.tolist(),
        "member_rows": member_rows,
        "large_systems": [
            "Pacific subtropical ridge represented by broad 850/500-hPa flow and 500-hPa height field",
            "midlatitude trough and jet influence represented by broad 500/200-hPa rings",
            "pressure-gradient steering represented by causal SLP geostrophic component when supplied",
        ],
    }


def integrate_from_issue(member_displacements: np.ndarray, base_latitude: float, base_longitude: float) -> list[list[dict]]:
    """Convert member km steps to serializable geographic paths."""

    paths = []
    for member in np.asarray(member_displacements, dtype="float32"):
        lat, lon = float(base_latitude), float(base_longitude)
        points = []
        for lead, step in enumerate(member, start=1):
            lon += float(step[0]) / (111.2 * max(math.cos(math.radians(lat)), 0.20))
            lat += float(step[1]) / 111.2
            lon %= 360.0
            points.append({
                "lead_hours": lead * 6,
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
            })
        paths.append(points)
    return paths


def weighted_route(member_displacements: np.ndarray, member_weights: np.ndarray) -> np.ndarray:
    return np.tensordot(np.asarray(member_weights, dtype="float32"), np.asarray(member_displacements, dtype="float32"), axes=(0, 0)).astype("float32")


__all__ = [
    "VERSION",
    "ROUTE_VARIANTS",
    "CURVATURE_VARIANTS",
    "SNAPSHOT_WEIGHTS",
    "build_route",
    "integrate_from_issue",
    "weighted_route",
]
