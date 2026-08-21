"""Public Trackformer 1.2 inference API.

The release checkpoints are trained on causal issue-time inputs.  This module
does not download or accept JMA/JTWC/ECMWF/GFS/GEFS forecast products.  It
expects the same preprocessed arrays used by the released checkpoints:

* route: a six-channel [SLP, 500-hPa height] analysis field for three times,
  a 647-dimensional causal context vector, and a 20-step base route in 100-km
  local displacement units;
* intensity: the 1,020-dimensional causal feature vector and the frozen
  20-lead physical anchor structure;
* ocean structure calibration: the 66-dimensional three-time ocean summary,
  the same causal feature vector, and the physical anchor structure.

The explicit contracts are intentional.  They prevent a convenience wrapper
from silently substituting a forecast field for an issue-time analysis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import maximum_filter, minimum_filter


LEADS = 20
LEAD_HOURS = tuple(range(6, 121, 6))
STRUCTURE_DIM = 15
ROUTE_CONTEXT_DIM = 647
INTENSITY_FEATURE_DIM = 1020
INTENSITY_INPUT_DIM = 1320
OCEAN_FEATURE_DIM = 66
FIELD_HEIGHT = 25
FIELD_WIDTH = 61
ROUTE_SYNOPTIC_DIM = 270
ROUTE_SYSTEM_DIM = 46
ROUTE_INTERACTION_DIM = 16
ROUTE_GEOGRAPHY_DIM = 224
ROUTE_KINEMATIC_DIM = 11
TRACK_WINDOW_LENGTH = 9
TRACK_FEATURE_DIM = 54


def _select_device(value: str | torch.device | None) -> torch.device:
    if value is not None:
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch(value: np.ndarray, trailing: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype="float32")
    if array.shape == trailing:
        array = array[None, ...]
    if array.ndim != len(trailing) + 1 or array.shape[1:] != trailing:
        raise ValueError(f"{name} must have shape [batch, {', '.join(map(str, trailing))}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _same_batch(*arrays: np.ndarray) -> None:
    sizes = {int(value.shape[0]) for value in arrays}
    if len(sizes) != 1:
        raise ValueError(f"all inputs must have the same batch size, got {sorted(sizes)}")


def prepare_route_field(
    slp: np.ndarray,
    hgt500: np.ndarray,
    slp_valid: np.ndarray | None = None,
    hgt500_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Encode physical three-time SLP/H500 analysis grids for the route head.

    ``slp`` and ``hgt500`` are physical arrays with shape ``[B, 3, H, W]``.
    The channel order is ``SLP_t0, H500_t0, SLP_t1, H500_t1, SLP_t2,
    H500_t2``.  Missing-frame masks are 0/1 arrays with shape ``[B, 3]`` or
    ``[B, 3, 1, 1]``.  The route checkpoint was trained with SLP in hPa and
    H500 in geopotential metres.
    """

    slp = _batch(slp, (3, slp.shape[-2], slp.shape[-1]) if np.asarray(slp).ndim == 4 else (3, FIELD_HEIGHT, FIELD_WIDTH), "slp")
    hgt500 = _batch(hgt500, (3, slp.shape[-2], slp.shape[-1]), "hgt500")
    _same_batch(slp, hgt500)
    if slp.shape[1:] != hgt500.shape[1:]:
        raise ValueError(f"slp and hgt500 grids must match, got {slp.shape} and {hgt500.shape}")
    slp_valid = np.ones((slp.shape[0], 3), dtype="float32") if slp_valid is None else np.asarray(slp_valid, dtype="float32").reshape(slp.shape[0], 3)
    hgt500_valid = np.ones((slp.shape[0], 3), dtype="float32") if hgt500_valid is None else np.asarray(hgt500_valid, dtype="float32").reshape(slp.shape[0], 3)
    field = np.empty((slp.shape[0], 6, slp.shape[2], slp.shape[3]), dtype="float32")
    for frame in range(3):
        field[:, 2 * frame] = ((slp[:, frame] - 1010.0) / 20.0) * slp_valid[:, frame, None, None]
        field[:, 2 * frame + 1] = ((hgt500[:, frame] - 5500.0) / 300.0) * hgt500_valid[:, frame, None, None]
    return field


def build_base_position(base_route_100km: np.ndarray) -> np.ndarray:
    """Convert per-lead local route increments to cumulative positions.

    The released route checkpoint was trained around a frozen incumbent route.
    Pass those causal per-lead increments here when reproducing that exact
    route pipeline; the function does not infer a route from future labels.
    """

    route = _batch(base_route_100km, (LEADS, 2), "base_route_100km")
    return np.cumsum(route, axis=1).astype("float32")


def build_kinematic_base_position(
    current_motion_100km: np.ndarray,
    previous_motion_100km: np.ndarray | None = None,
) -> np.ndarray:
    """Build a causal persistence base route when no incumbent route is available.

    ``current_motion_100km`` is the observed six-hour local displacement.  The
    optional previous displacement is accepted for auditability, but the
    fallback deliberately persists the current motion rather than pretending
    to reproduce the private incumbent route used during training.
    """

    current = _batch(current_motion_100km, (2,), "current_motion_100km")
    if previous_motion_100km is not None:
        previous = _batch(previous_motion_100km, (2,), "previous_motion_100km")
        _same_batch(current, previous)
    increments = np.repeat(current[:, None, :], LEADS, axis=1)
    return build_base_position(increments)


def normalize_route_context(
    context: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    """Apply the train-only normalization used by the route checkpoint."""

    value = _batch(context, (ROUTE_CONTEXT_DIM,), "context")
    mean = np.asarray(feature_mean, dtype="float32").reshape(-1)
    std = np.maximum(np.asarray(feature_std, dtype="float32").reshape(-1), 1.0e-4)
    if mean.shape != (ROUTE_CONTEXT_DIM,) or std.shape != (ROUTE_CONTEXT_DIM,):
        raise ValueError("route context statistics must each have shape [647]")
    return ((value - mean[None, :]) / std[None, :]).astype("float32")


def _nearest_static(
    static_map: np.ndarray,
    latitude_grid: np.ndarray,
    longitude_grid: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    fields = np.asarray(static_map, dtype="float32")
    lat_grid = np.asarray(latitude_grid, dtype="float32").reshape(-1)
    lon_grid = np.asarray(longitude_grid, dtype="float32").reshape(-1)
    lat = np.asarray(latitude, dtype="float32").reshape(-1)
    lon = np.mod(np.asarray(longitude, dtype="float32").reshape(-1), 360.0)
    iy = np.clip(np.searchsorted(lat_grid, lat), 1, len(lat_grid) - 1)
    iy = np.where(np.abs(lat_grid[iy - 1] - lat) <= np.abs(lat_grid[iy] - lat), iy - 1, iy)
    ix = np.clip(np.searchsorted(lon_grid, lon), 1, len(lon_grid) - 1)
    ix = np.where(np.abs(lon_grid[ix - 1] - lon) <= np.abs(lon_grid[ix] - lon), ix - 1, ix)
    return fields[:, iy, ix].T.astype("float32")


def build_route_geography_features(
    route_position_100km: np.ndarray,
    issue_latitude: np.ndarray | float,
    issue_longitude: np.ndarray | float,
    latitude_grid: np.ndarray,
    longitude_grid: np.ndarray,
    static_map: np.ndarray,
    feature_names: Iterable[str] | None = None,
) -> np.ndarray:
    """Construct the exact 224-feature static route geography block.

    ``static_map`` is ``[C,H,W]``.  When all 28 release channels are passed,
    ``feature_names`` must name them; an eight-channel map may be passed in
    the trained order listed in ``ROUTE_GEOGRAPHY_CHANNELS``.
    """

    position = _batch(route_position_100km, (LEADS, 2), "route_position_100km")
    lat0 = np.asarray(issue_latitude, dtype="float32").reshape(-1)
    lon0 = np.asarray(issue_longitude, dtype="float32").reshape(-1)
    if len(lat0) == 1 and len(position) > 1:
        lat0 = np.repeat(lat0, len(position))
    if len(lon0) == 1 and len(position) > 1:
        lon0 = np.repeat(lon0, len(position))
    _same_batch(position, lat0[:, None], lon0[:, None])
    names = list(feature_names) if feature_names is not None else None
    selected_names = (
        "land_fraction", "coast_proximity", "land_fraction_75km",
        "land_fraction_150km", "land_fraction_300km", "land_buffer_25km",
        "land_buffer_50km", "land_local_std_150km",
    )
    static = np.asarray(static_map, dtype="float32")
    if static.ndim != 3:
        raise ValueError("static_map must have shape [channels, latitude, longitude]")
    if names is None:
        if static.shape[0] != len(selected_names):
            raise ValueError("feature_names is required when static_map is not the eight-channel trained subset")
        selected = static
    else:
        indices = []
        for name in selected_names:
            if name not in names:
                raise ValueError(f"static_map is missing required channel {name!r}")
            indices.append(names.index(name))
        selected = static[np.asarray(indices, dtype="int64")]
    route_lat = lat0[:, None] + position[..., 1] * 100.0 / 111.0
    route_lon = lon0[:, None] + position[..., 0] * 100.0 / (
        111.0 * np.maximum(np.cos(np.deg2rad(route_lat)), 0.2)
    )
    route = _nearest_static(
        selected, latitude_grid, longitude_grid,
        route_lat.reshape(-1), route_lon.reshape(-1),
    ).reshape(len(position), LEADS, len(selected_names))
    delta = np.diff(route, axis=1)
    summary = np.concatenate([
        route[:, 0], route[:, -1], route.mean(axis=1), route.max(axis=1),
        route.min(axis=1), route[:, -1] - route[:, 0],
        delta.mean(axis=1), delta[:, -1],
    ], axis=1)
    output = np.concatenate([route.reshape(len(position), -1), summary], axis=1)
    if output.shape[1] != ROUTE_GEOGRAPHY_DIM:
        raise RuntimeError(f"route geography builder produced {output.shape[1]} features, expected 224")
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def build_route_kinematic_features(
    current_motion_100km: np.ndarray,
    previous_motion_100km: np.ndarray,
    issue_latitude: np.ndarray | float,
    issue_longitude: np.ndarray | float,
) -> np.ndarray:
    """Construct the exact 11-feature issue-time motion block."""

    current = _batch(current_motion_100km, (2,), "current_motion_100km")
    previous = _batch(previous_motion_100km, (2,), "previous_motion_100km")
    lat = np.asarray(issue_latitude, dtype="float32").reshape(-1)
    lon = np.asarray(issue_longitude, dtype="float32").reshape(-1)
    if len(lat) == 1 and len(current) > 1:
        lat = np.repeat(lat, len(current))
    if len(lon) == 1 and len(current) > 1:
        lon = np.repeat(lon, len(current))
    _same_batch(current, previous, lat[:, None], lon[:, None])
    speed = np.linalg.norm(current, axis=1)
    turn = np.arctan2(
        current[:, 0] * previous[:, 1] - current[:, 1] * previous[:, 0],
        (current * previous).sum(axis=1) + 1.0e-4,
    )
    return np.stack([
        lat / 30.0, np.mod(lon, 360.0) / 180.0,
        np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
        current[:, 0] / 30.0, current[:, 1] / 30.0, speed / 30.0,
        np.sin(turn), np.cos(turn),
        (current[:, 0] - previous[:, 0]) / 30.0,
        (current[:, 1] - previous[:, 1]) / 30.0,
    ], axis=1).astype("float32")


def build_route_context(
    synoptic_features: np.ndarray,
    system_features: np.ndarray,
    interaction_features: np.ndarray,
    geography_features: np.ndarray,
    kinematic_features: np.ndarray,
    base_route_100km: np.ndarray,
    base_position_100km: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate the exact causal 647-feature route context.

    All components are issue-time features.  ``base_route_100km`` is the
    causal incumbent-route input; it is not an official forecast product.
    """

    synoptic = _batch(synoptic_features, (ROUTE_SYNOPTIC_DIM,), "synoptic_features")
    system = _batch(system_features, (ROUTE_SYSTEM_DIM,), "system_features")
    interaction = _batch(interaction_features, (ROUTE_INTERACTION_DIM,), "interaction_features")
    geography = _batch(geography_features, (ROUTE_GEOGRAPHY_DIM,), "geography_features")
    kinematic = _batch(kinematic_features, (ROUTE_KINEMATIC_DIM,), "kinematic_features")
    route = _batch(base_route_100km, (LEADS, 2), "base_route_100km")
    position = build_base_position(route) if base_position_100km is None else _batch(base_position_100km, (LEADS, 2), "base_position_100km")
    _same_batch(synoptic, system, interaction, geography, kinematic, route, position)
    context = np.concatenate([
        synoptic, system, interaction, geography, kinematic,
        route.reshape(len(route), -1), position.reshape(len(position), -1),
    ], axis=1)
    if context.shape[1] != ROUTE_CONTEXT_DIM:
        raise RuntimeError(f"route context builder produced {context.shape[1]} features, expected 647")
    return np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _grid_distance(
    base_latitude: np.ndarray,
    base_longitude: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(latitude, dtype="float32")[None, :, None]
    lon = np.asarray(longitude, dtype="float32")[None, None, :]
    center_lat = np.asarray(base_latitude, dtype="float32")[:, None, None]
    center_lon = np.asarray(base_longitude, dtype="float32")[:, None, None]
    dy = np.broadcast_to((lat - center_lat) * 111.2, (len(center_lat), len(latitude), len(longitude)))
    delta_lon = (lon - center_lon + 180.0) % 360.0 - 180.0
    dx = delta_lon * 111.2 * np.cos(np.deg2rad((lat + center_lat) * 0.5))
    return dx.astype("float32"), dy.astype("float32"), np.hypot(dx, dy).astype("float32")


def _synoptic_extrema(
    field: np.ndarray,
    base_latitude: np.ndarray,
    base_longitude: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    kind: str,
) -> np.ndarray:
    dx, dy, distance = _grid_distance(base_latitude, base_longitude, latitude, longitude)
    if kind == "low":
        local = minimum_filter(field, size=(1, 5, 5), mode="nearest")
        background = maximum_filter(field, size=(1, 9, 9), mode="nearest")
        strength = background - field
        candidate = (
            (np.abs(field - local) <= 0.26) & (strength >= 0.75)
            & (distance >= 450.0) & (distance <= 2200.0)
            & (np.asarray(latitude)[None, :, None] >= 0.0)
            & (np.asarray(latitude)[None, :, None] <= 60.0)
        )
        value_scale, strength_scale = 1000.0, 10.0
    elif kind == "ridge":
        local = maximum_filter(field, size=(1, 7, 7), mode="nearest")
        background = minimum_filter(field, size=(1, 11, 11), mode="nearest")
        strength = field - background
        candidate = (
            (np.abs(field - local) <= 10.0) & (strength >= 18.0)
            & (distance >= 450.0) & (distance <= 2600.0)
            & (np.asarray(latitude)[None, :, None] >= 10.0)
            & (np.asarray(latitude)[None, :, None] <= 45.0)
            & (np.asarray(longitude)[None, None, :] >= 115.0)
            & (np.asarray(longitude)[None, None, :] <= 205.0)
        )
        value_scale, strength_scale = 1000.0, 1000.0
    else:
        raise ValueError("kind must be 'low' or 'ridge'")
    score = np.where(candidate, strength, -np.inf).reshape(len(field), -1)
    top_index = np.argpartition(score, -3, axis=1)[:, -3:]
    top_score = np.take_along_axis(score, top_index, axis=1)
    order = np.argsort(-top_score, axis=1)
    top_index = np.take_along_axis(top_index, order, axis=1)
    top_score = np.take_along_axis(top_score, order, axis=1)
    row = np.arange(len(field))[:, None]
    iy, ix = np.divmod(top_index, field.shape[-1])
    picked_dx = dx[row, iy, ix]
    picked_dy = dy[row, iy, ix]
    picked_distance = distance[row, iy, ix]
    picked_value = field[row, iy, ix]
    valid = np.isfinite(top_score).astype("float32")
    angle = np.arctan2(picked_dy, picked_dx)
    counts = np.stack([
        np.sum(candidate & (distance <= 750.0), axis=(1, 2)),
        np.sum(candidate & (distance <= 1250.0), axis=(1, 2)),
        np.sum(candidate & (distance <= 1750.0), axis=(1, 2)),
    ], axis=1).astype("float32")
    values = np.stack([
        valid, picked_dx / 1000.0, picked_dy / 1000.0,
        picked_distance / 1000.0,
        np.nan_to_num(top_score, nan=0.0, posinf=0.0, neginf=0.0) / strength_scale,
        picked_value / value_scale,
        np.sin(angle) * valid, np.cos(angle) * valid,
    ], axis=-1)
    return np.concatenate([values.reshape(len(field), -1), counts], axis=1).astype("float32")


def build_synoptic_features(
    slp: np.ndarray,
    hgt500: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    issue_latitude: np.ndarray | float,
    issue_longitude: np.ndarray | float,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Build the exact 270-feature causal low/ridge catalog.

    The three frames must be ordered ``t0, t-6h, t-12h``.  The function uses
    only the supplied analysis fields and never reads a positive-lead product.
    """

    slp = np.asarray(slp, dtype="float32")
    hgt500 = np.asarray(hgt500, dtype="float32")
    if slp.ndim == 3:
        slp = slp[None, ...]
    if hgt500.ndim == 3:
        hgt500 = hgt500[None, ...]
    if slp.ndim != 4 or hgt500.shape != slp.shape or slp.shape[1] != 3:
        raise ValueError("slp and hgt500 must both have shape [B, 3, H, W]")
    lat = np.asarray(latitude, dtype="float32").reshape(-1)
    lon = np.asarray(longitude, dtype="float32").reshape(-1)
    if slp.shape[-2:] != (len(lat), len(lon)):
        raise ValueError("analysis field shape does not match latitude/longitude grids")
    base_lat = np.asarray(issue_latitude, dtype="float32").reshape(-1)
    base_lon = np.asarray(issue_longitude, dtype="float32").reshape(-1)
    if len(base_lat) == 1 and len(slp) > 1:
        base_lat = np.repeat(base_lat, len(slp))
    if len(base_lon) == 1 and len(slp) > 1:
        base_lon = np.repeat(base_lon, len(slp))
    _same_batch(slp, hgt500, base_lat[:, None], base_lon[:, None])
    frame_valid = np.ones((len(slp), 3), dtype="float32") if valid is None else np.asarray(valid, dtype="float32").reshape(len(slp), 3)
    parts: list[np.ndarray] = []
    for values, kind in ((slp, "low"), (hgt500, "ridge")):
        frames = []
        for frame in range(3):
            item = _synoptic_extrema(values[:, frame], base_lat, base_lon, lat, lon, kind)
            frames.append(item * frame_valid[:, frame, None])
        stacked = np.stack(frames, axis=1)
        parts.extend([
            stacked.reshape(len(stacked), -1),
            stacked[:, 0] - stacked[:, 1],
            stacked[:, 1] - stacked[:, 2],
        ])
    output = np.concatenate(parts, axis=1)
    if output.shape[1] != ROUTE_SYNOPTIC_DIM:
        raise RuntimeError(f"synoptic builder produced {output.shape[1]} features, expected 270")
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _system_frame_summary(field: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    ridge = (latitude[:, None] >= 15.0) & (latitude[:, None] <= 35.0)
    ridge = ridge & (longitude[None, :] >= 130.0) & (longitude[None, :] <= 180.0)
    broad = (latitude[:, None] >= 5.0) & (latitude[:, None] <= 40.0)
    broad = broad & (longitude[None, :] >= 115.0) & (longitude[None, :] <= 180.0)
    hgt = field[0]
    ridge_values = hgt[ridge]
    broad_values = hgt[broad]
    if ridge_values.size:
        baseline = float(np.percentile(ridge_values, 60.0))
        weights = np.maximum(ridge_values - baseline, 0.0) + 1.0e-3
        lat_grid, lon_grid = np.broadcast_arrays(latitude[:, None], longitude[None, :])
        ridge_lat = float(np.average(lat_grid[ridge], weights=weights))
        ridge_lon = float(np.average(lon_grid[ridge], weights=weights))
    else:
        ridge_lat, ridge_lon = 25.0, 155.0
    grad_lat, grad_lon = np.gradient(hgt)
    mean = lambda values, mask: float(values[mask].mean()) if np.any(mask) else 0.0
    std = lambda values, mask: float(values[mask].std()) if np.any(mask) else 0.0
    return np.asarray([
        mean(hgt, ridge), std(hgt, ridge), float(hgt[ridge].max()) if ridge_values.size else 0.0,
        float(hgt[ridge].min()) if ridge_values.size else 0.0, ridge_lat, ridge_lon,
        mean(grad_lat, ridge), mean(grad_lon, ridge),
        mean(field[1], broad), mean(field[2], broad), mean(field[3], broad),
        mean(field[4], broad), mean(field[5], broad), mean(field[6], broad),
        mean(field[3], ridge), mean(field[4], ridge), mean(field[1], ridge),
        mean(field[2], ridge), mean(field[5], ridge), mean(field[6], ridge),
        mean(hgt, broad), std(hgt, broad), float(hgt[broad].max()) if broad_values.size else 0.0,
        float(hgt[broad].min()) if broad_values.size else 0.0,
    ], dtype="float32")


def _normalize_context_block(value: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    output = np.asarray(value, dtype="float32")
    if mean is None and std is None:
        return output
    if mean is None or std is None:
        raise ValueError("context mean and std must be supplied together")
    mean_value = np.asarray(mean, dtype="float32").reshape(1, -1)
    std_value = np.maximum(np.asarray(std, dtype="float32").reshape(1, -1), 1.0e-5)
    if output.shape[1] != mean_value.shape[1]:
        raise ValueError("context statistics do not match the feature block")
    return np.clip((output - mean_value) / std_value, -5.0, 5.0).astype("float32")


def build_route_system_features(
    analysis_field: np.ndarray,
    previous_analysis_field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    issue_latitude: np.ndarray | float,
    issue_longitude: np.ndarray | float,
    analysis_age_hours: np.ndarray | float = 0.0,
    valid: np.ndarray | float = 1.0,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> np.ndarray:
    """Build the 46-feature causal Pacific-High/system context block."""

    current = np.asarray(analysis_field, dtype="float32")
    previous = np.asarray(previous_analysis_field, dtype="float32")
    if current.ndim == 3:
        current = current[None, ...]
    if previous.ndim == 3:
        previous = previous[None, ...]
    if current.ndim != 4 or current.shape[1] != 7 or previous.shape != current.shape:
        raise ValueError("analysis fields must both have shape [B, 7, H, W]")
    lat_grid = np.asarray(latitude, dtype="float32").reshape(-1)
    lon_grid = np.asarray(longitude, dtype="float32").reshape(-1)
    if current.shape[-2:] != (len(lat_grid), len(lon_grid)):
        raise ValueError("analysis field shape does not match latitude/longitude grids")
    lat = np.asarray(issue_latitude, dtype="float32").reshape(-1)
    lon = np.asarray(issue_longitude, dtype="float32").reshape(-1)
    if len(lat) == 1 and len(current) > 1:
        lat = np.repeat(lat, len(current))
    if len(lon) == 1 and len(current) > 1:
        lon = np.repeat(lon, len(current))
    _same_batch(current, previous, lat[:, None], lon[:, None])
    ages = np.asarray(analysis_age_hours, dtype="float32").reshape(-1)
    flags = np.asarray(valid, dtype="float32").reshape(-1)
    if len(ages) == 1 and len(current) > 1:
        ages = np.repeat(ages, len(current))
    if len(flags) == 1 and len(current) > 1:
        flags = np.repeat(flags, len(current))
    summaries = np.stack([_system_frame_summary(item, lat_grid, lon_grid) for item in current], axis=0)
    previous_summary = np.stack([_system_frame_summary(item, lat_grid, lon_grid) for item in previous], axis=0)
    delta_columns = np.asarray([0, 2, 4, 5, 10, 11, 12, 13], dtype="int64")
    deltas = summaries[:, delta_columns] - previous_summary[:, delta_columns]
    local = []
    for index, item in enumerate(current):
        iy = int(np.argmin(np.abs(lat_grid - lat[index])))
        ix = int(np.argmin(np.abs(lon_grid - lon[index])))
        ridge_lat, ridge_lon = summaries[index, 4], summaries[index, 5]
        grad_lat, grad_lon = np.gradient(item[0])
        local.append(np.asarray([
            *item[:, iy, ix],
            (ridge_lat - lat[index]) * 111.2,
            ((ridge_lon - lon[index] + 180.0) % 360.0 - 180.0) * 111.2 * np.cos(np.deg2rad(lat[index])),
            item[0, iy, ix] - summaries[index, 0], grad_lat[iy, ix], grad_lon[iy, ix],
        ], dtype="float32"))
    raw = np.concatenate([summaries, deltas, np.stack(local), flags[:, None], np.clip(ages, -1.0, 3.0)[:, None]], axis=1)
    if raw.shape[1] != ROUTE_SYSTEM_DIM:
        raise RuntimeError(f"system builder produced {raw.shape[1]} features, expected 46")
    raw[flags <= 0.0] = 0.0
    return _normalize_context_block(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), feature_mean, feature_std)


def build_nearby_interaction_features(
    nearby_latitude: np.ndarray,
    nearby_longitude: np.ndarray,
    nearby_vmax: np.ndarray,
    nearby_age_hours: np.ndarray,
    issue_latitude: np.ndarray | float,
    issue_longitude: np.ndarray | float,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> np.ndarray:
    """Build the 16-feature causal interaction block from other observed storms.

    The nearby arrays are ``[B,N]`` and must exclude the target storm.  Points
    older than six hours or later than the issue time are ignored.
    """

    lat_values = np.asarray(nearby_latitude, dtype="float32")
    lon_values = np.asarray(nearby_longitude, dtype="float32")
    wind_values = np.asarray(nearby_vmax, dtype="float32")
    age_values = np.asarray(nearby_age_hours, dtype="float32")
    if lat_values.ndim == 1:
        lat_values = lat_values[None, :]
    if lon_values.ndim == 1:
        lon_values = lon_values[None, :]
    if wind_values.ndim == 1:
        wind_values = wind_values[None, :]
    if age_values.ndim == 1:
        age_values = age_values[None, :]
    if not (lat_values.shape == lon_values.shape == wind_values.shape == age_values.shape):
        raise ValueError("nearby arrays must have matching shape [B, N]")
    lat = np.asarray(issue_latitude, dtype="float32").reshape(-1)
    lon = np.asarray(issue_longitude, dtype="float32").reshape(-1)
    if len(lat) == 1 and len(lat_values) > 1:
        lat = np.repeat(lat, len(lat_values))
    if len(lon) == 1 and len(lon_values) > 1:
        lon = np.repeat(lon, len(lon_values))
    result = np.zeros((len(lat_values), ROUTE_INTERACTION_DIM), dtype="float32")
    for index in range(len(result)):
        active = np.isfinite(lat_values[index]) & np.isfinite(lon_values[index]) & np.isfinite(wind_values[index])
        active &= np.isfinite(age_values[index]) & (age_values[index] >= 0.0) & (age_values[index] <= 6.0)
        if not np.any(active):
            continue
        other_lat = lat_values[index, active].astype("float64")
        other_lon = lon_values[index, active].astype("float64")
        dx = ((other_lon - lon[index] + 180.0) % 360.0 - 180.0) * 111.2 * np.cos(np.deg2rad(lat[index]))
        dy = (other_lat - lat[index]) * 111.2
        distance = np.hypot(dx, dy)
        wind = np.clip(wind_values[index, active], 0.0, 180.0)
        result[index, :4] = [np.count_nonzero(distance <= 200.0), np.count_nonzero(distance <= 500.0), np.count_nonzero(distance <= 1000.0), np.count_nonzero(distance <= 1500.0)]
        nearest = int(np.argmin(distance))
        result[index, 4:8] = [distance[nearest], dx[nearest], dy[nearest], wind[nearest]]
        result[index, 8] = float(wind.max())
        weights = (wind + 10.0) * np.exp(-distance / 600.0)
        total = float(weights.sum())
        if total > 0.0:
            result[index, 9:12] = [(weights * wind).sum() / total, (weights * dx).sum() / total, (weights * dy).sum() / total]
            result[index, 12:14] = [(wind * dx / (distance + 150.0)).sum(), (wind * dy / (distance + 150.0)).sum()]
        result[index, 14] = float(age_values[index, active].min())
        result[index, 15] = float(active.sum())
    return _normalize_context_block(result, feature_mean, feature_std)


def build_intensity_context_features(
    basin_features: np.ndarray,
    basin_valid: np.ndarray,
    basin_age_hours: np.ndarray,
    system_features: np.ndarray,
    interaction_features: np.ndarray,
    system_valid: np.ndarray,
    system_age_hours: np.ndarray,
    global_features: np.ndarray,
    global_valid: np.ndarray,
    global_age_hours: np.ndarray,
) -> np.ndarray:
    """Concatenate the exact 183-feature causal intensity context block."""

    parts = [
        _batch(basin_features, (42,), "basin_features"),
        np.asarray(basin_valid, dtype="float32").reshape(-1, 1),
        np.clip(np.asarray(basin_age_hours, dtype="float32").reshape(-1, 1), -1.0, 6.0) / 6.0,
        _batch(system_features, (46,), "system_features"),
        _batch(interaction_features, (16,), "interaction_features"),
        np.asarray(system_valid, dtype="float32").reshape(-1, 1),
        np.clip(np.asarray(system_age_hours, dtype="float32").reshape(-1, 1), -1.0, 6.0) / 6.0,
        _batch(global_features, (73,), "global_features"),
        np.asarray(global_valid, dtype="float32").reshape(-1, 1),
        np.clip(np.asarray(global_age_hours, dtype="float32").reshape(-1, 1), -1.0, 6.0) / 6.0,
    ]
    _same_batch(*parts)
    context = np.concatenate(parts, axis=1)
    if context.shape[1] != 183:
        raise RuntimeError(f"intensity context builder produced {context.shape[1]} features, expected 183")
    return np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def local_position_to_latlon(
    position_100km: np.ndarray,
    issue_latitude: np.ndarray | float,
    issue_longitude: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert model local displacement output to latitude/longitude degrees."""

    position = _batch(position_100km, (LEADS, 2), "position_100km")
    lat0 = np.asarray(issue_latitude, dtype="float32").reshape(-1)
    lon0 = np.asarray(issue_longitude, dtype="float32").reshape(-1)
    if len(lat0) == 1 and len(position) > 1:
        lat0 = np.repeat(lat0, len(position))
    if len(lon0) == 1 and len(position) > 1:
        lon0 = np.repeat(lon0, len(position))
    if len(lat0) != len(position) or len(lon0) != len(position):
        raise ValueError("issue latitude/longitude must have one value per batch item")
    latitude = lat0[:, None] + position[..., 1] * 100.0 / 111.0
    longitude = lon0[:, None] + position[..., 0] * 100.0 / (
        111.0 * np.maximum(np.cos(np.deg2rad(latitude)), 0.2)
    )
    return latitude.astype("float32"), np.mod(longitude, 360.0).astype("float32")


def build_patch_summary(patch: np.ndarray, available: np.ndarray | float = 1.0) -> np.ndarray:
    """Build the causal 8-statistics-plus-availability patch summary."""

    value = np.asarray(patch, dtype="float32")
    if value.ndim == 3:
        value = value[:, None]
    if value.ndim != 4:
        raise ValueError("patch must have shape [B, channels, height, width]")
    count, channels, height, width = value.shape
    cy, cx = height // 2, width // 2
    inner = value[..., max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3]
    broad = value[..., max(0, cy - 4):cy + 5, max(0, cx - 4):cx + 5]
    ring = value[..., max(0, cy - 6):min(height, cy + 7), max(0, cx - 6):min(width, cx + 7)]
    north = value[..., max(0, cy - 4):max(1, cy - 1), max(0, cx - 1):cx + 2].mean(axis=(-2, -1))
    south = value[..., cy + 2:min(height, cy + 5), max(0, cx - 1):cx + 2].mean(axis=(-2, -1))
    west = value[..., max(0, cy - 1):cy + 2, max(0, cx - 4):max(1, cx - 1)].mean(axis=(-2, -1))
    east = value[..., max(0, cy - 1):cy + 2, cx + 2:min(width, cx + 5)].mean(axis=(-2, -1))
    available_value = np.asarray(available, dtype="float32")
    if available_value.ndim == 0:
        available_value = np.full((count, 1), float(available_value), dtype="float32")
    else:
        available_value = available_value.reshape(count, -1)
    features = np.concatenate([
        value[..., cy, cx].reshape(count, -1),
        inner.mean(axis=(-2, -1)).reshape(count, -1),
        inner.std(axis=(-2, -1)).reshape(count, -1),
        broad.mean(axis=(-2, -1)).reshape(count, -1),
        ring.mean(axis=(-2, -1)).reshape(count, -1),
        ring.std(axis=(-2, -1)).reshape(count, -1),
        (north - south).reshape(count, -1),
        (east - west).reshape(count, -1),
        available_value,
    ], axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def build_terrain_samples(
    issue_latitude: np.ndarray | float,
    issue_longitude: np.ndarray | float,
    latitude_grid: np.ndarray,
    longitude_grid: np.ndarray,
    static_map: np.ndarray,
) -> np.ndarray:
    """Sample the six causal land/sea points used by the structure head."""

    lat = np.asarray(issue_latitude, dtype="float32").reshape(-1)
    lon = np.asarray(issue_longitude, dtype="float32").reshape(-1)
    if len(lat) == 1 and len(lon) > 1:
        lat = np.repeat(lat, len(lon))
    if len(lon) == 1 and len(lat) > 1:
        lon = np.repeat(lon, len(lat))
    if len(lat) != len(lon):
        raise ValueError("issue latitude/longitude must have one value per batch item")
    fields = np.asarray(static_map, dtype="float32")
    if fields.ndim != 3:
        raise ValueError("static_map must have shape [channels, latitude, longitude]")
    grid_lat = np.asarray(latitude_grid, dtype="float32").reshape(-1)
    grid_lon = np.asarray(longitude_grid, dtype="float32").reshape(-1)
    offsets_lat = np.asarray([0.0, 150.0 / 111.0, -150.0 / 111.0, 0.0, 0.0, 500.0 / 111.0], dtype="float32")
    cos_lat = np.maximum(np.cos(np.deg2rad(lat)), 0.2)
    offsets_lon = np.asarray([0.0, 0.0, 0.0, 150.0, -150.0, 0.0], dtype="float32")[None, :] / (111.0 * cos_lat[:, None])
    sample_lat = np.clip(lat[:, None] + offsets_lat[None, :], grid_lat[0], grid_lat[-1])
    sample_lon = np.mod(lon[:, None] + offsets_lon, 360.0)
    lat_index = np.clip(np.rint((sample_lat - grid_lat[0]) / (grid_lat[1] - grid_lat[0])).astype("int64"), 0, len(grid_lat) - 1)
    lon_index = np.clip(np.rint((sample_lon - grid_lon[0]) / (grid_lon[1] - grid_lon[0])).astype("int64"), 0, len(grid_lon) - 1)
    result = fields[:, lat_index, lon_index].transpose(1, 2, 0)
    return np.nan_to_num(result.reshape(len(lat), -1), nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def build_ocean_features(ocean_summary: np.ndarray) -> np.ndarray:
    """Flatten current, -12 h, and -24 h 22-value ocean summaries to 66."""

    summary = np.asarray(ocean_summary, dtype="float32")
    if summary.ndim == 2 and summary.shape == (3, 22):
        summary = summary[None, ...]
    if summary.ndim != 3 or summary.shape[1:] != (3, 22):
        raise ValueError("ocean_summary must have shape [B, 3, 22]")
    return np.nan_to_num(summary.reshape(len(summary), OCEAN_FEATURE_DIM), nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _season_features(issue_time_ns: np.ndarray) -> np.ndarray:
    value = np.asarray(issue_time_ns, dtype="int64").reshape(-1)
    dates = value.astype("datetime64[ns]").astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype("datetime64[D]")
    day = (dates - years).astype("timedelta64[D]").astype("float32")
    hour = ((value // (3_600 * 1_000_000_000)) % 24).astype("float32")
    phase = 2.0 * np.pi * (day + hour / 24.0) / 365.25
    return np.stack([
        np.sin(phase), np.cos(phase), np.sin(2.0 * phase), np.cos(2.0 * phase),
    ], axis=1).astype("float32")


def build_causal_structure_features(
    track_window: np.ndarray,
    analysis_patches: np.ndarray,
    analysis_valid: np.ndarray,
    sst_patches: np.ndarray,
    sst_valid: np.ndarray,
    terrain_features: np.ndarray,
    context_features: np.ndarray,
    issue_latitude: np.ndarray | float,
    issue_longitude: np.ndarray | float,
    issue_time_ns: np.ndarray,
    track_mean: np.ndarray | None = None,
    track_std: np.ndarray | None = None,
    history_available: np.ndarray | None = None,
    track_normalized: bool = True,
) -> np.ndarray:
    """Build the exact 1,020-dimensional causal structure feature vector.

    ``track_window`` is the normalized ``9 x 54`` archive representation by
    default.  Supply ``track_mean`` and ``track_std`` to reconstruct the raw
    physical rows needed for the current-state block; set ``track_normalized``
    to ``False`` when the passed rows are physical and should be normalized.
    Analysis patches
    must already be in the released DLM4 representation (decoded ``q / 31.75``)
    and ordered ``t0, t-12 h, t-24 h``.  No future observations are accepted.
    """

    track = np.asarray(track_window, dtype="float32")
    if track.ndim == 2 and track.shape == (TRACK_WINDOW_LENGTH, TRACK_FEATURE_DIM):
        track = track[None, ...]
    if track.ndim != 3 or track.shape[1:] != (TRACK_WINDOW_LENGTH, TRACK_FEATURE_DIM):
        raise ValueError("track_window must have shape [B, 9, 54]")
    if track_mean is not None or track_std is not None:
        if track_mean is None or track_std is None:
            raise ValueError("track_mean and track_std must be supplied together")
        mean = np.asarray(track_mean, dtype="float32").reshape(1, 1, TRACK_FEATURE_DIM)
        std = np.asarray(track_std, dtype="float32").reshape(1, 1, TRACK_FEATURE_DIM)
        if track_normalized:
            raw_track = track * std + mean
        else:
            raw_track = track.copy()
            track = (track - mean) / np.maximum(std, 1.0e-6)
    else:
        raw_track = None
    analysis = np.asarray(analysis_patches, dtype="float32")
    if analysis.ndim == 4 and analysis.shape[:2] == (3, 4):
        analysis = analysis[None, ...]
    if analysis.ndim != 5 or analysis.shape[1:3] != (3, 4) or analysis.shape[-2:] != (17, 17):
        raise ValueError("analysis_patches must have shape [B, 3, 4, 17, 17]")
    valid = np.asarray(analysis_valid, dtype="float32").reshape(len(track), 3)
    history = valid[:, 1:] if history_available is None else np.asarray(history_available, dtype="float32").reshape(len(track), 2)
    field_features = build_patch_summary(analysis.reshape(-1, 4, 17, 17), 1.0).reshape(len(track), 3, -1).reshape(len(track), -1)
    field_availability = np.concatenate([valid[:, :1], valid[:, 1:], history], axis=1)
    sst = np.asarray(sst_patches, dtype="float32")
    if sst.ndim == 4 and sst.shape[:2] == (3, 1):
        sst = sst[None, ...]
    if sst.ndim != 5 or sst.shape[1:3] != (3, 1) or sst.shape[-2:] != (17, 17):
        raise ValueError("sst_patches must have shape [B, 3, 1, 17, 17]")
    sst_ok = np.asarray(sst_valid, dtype="float32").reshape(len(track), 3)
    sst_features = build_patch_summary(sst.reshape(-1, 1, 17, 17), sst_ok.reshape(-1, 1)).reshape(len(track), 3, -1).reshape(len(track), -1)
    terrain = _batch(terrain_features, (168,), "terrain_features")
    context = _batch(context_features, (183,), "context_features")
    lat = np.asarray(issue_latitude, dtype="float32").reshape(-1)
    lon = np.asarray(issue_longitude, dtype="float32").reshape(-1)
    time_value = np.asarray(issue_time_ns, dtype="int64").reshape(-1)
    if len(lat) == 1 and len(track) > 1:
        lat = np.repeat(lat, len(track))
    if len(lon) == 1 and len(track) > 1:
        lon = np.repeat(lon, len(track))
    if len(time_value) == 1 and len(track) > 1:
        time_value = np.repeat(time_value, len(track))
    _same_batch(track, analysis, sst, terrain, context, lat[:, None], lon[:, None], time_value[:, None])
    if raw_track is None:
        raise ValueError("track_mean and track_std are required to reconstruct physical current-state features")
    raw_last = raw_track[:, -1]
    raw_prev = raw_track[:, -2]
    current_state = np.concatenate([
        raw_last[:, 4:20] / np.asarray([80.0, 1000.0, 300.0] + [300.0] * 13, dtype="float32"),
        (raw_last[:, 4:20] - raw_prev[:, 4:20]) / np.asarray([40.0, 20.0, 100.0] + [100.0] * 13, dtype="float32"),
        raw_last[:, 28:40],
    ], axis=1).astype("float32")
    location = np.stack([lat / 90.0, np.mod(lon, 360.0) / 180.0, np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat))], axis=1).astype("float32")
    output = np.concatenate([
        track.reshape(len(track), -1), current_state, field_features, field_availability,
        sst_features, terrain, context, location, _season_features(time_value),
    ], axis=1)
    if output.shape[1] != INTENSITY_FEATURE_DIM:
        raise RuntimeError(f"causal structure builder produced {output.shape[1]} features, expected 1020")
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def build_causal_anchor_structure(observed_structure: np.ndarray) -> np.ndarray:
    """Build a causal persistence/trend anchor when no incumbent is supplied.

    This is a documented fallback, not the private frozen Trackformer 1.2.26
    anchor used to train the released residual head.  Supply the real anchor
    to ``predict_intensity`` when reproducing the trained residual pipeline.
    ``observed_structure`` is physical ``[B, 3, 15]`` history ordered oldest to
    newest, with columns ``vmax, pressure, RMW, R34/R50/R64``.
    """

    history = np.asarray(observed_structure, dtype="float32")
    if history.ndim == 2 and history.shape == (3, STRUCTURE_DIM):
        history = history[None, ...]
    if history.ndim == 2 and history.shape == (STRUCTURE_DIM,):
        history = history[None, None, ...]
    if history.ndim != 3 or history.shape[-1] != STRUCTURE_DIM or history.shape[1] not in (1, 2, 3):
        raise ValueError("observed_structure must have shape [B, 1|2|3, 15]")
    latest = history[:, -1]
    if history.shape[1] >= 2:
        delta = latest - history[:, -2]
    else:
        delta = np.zeros_like(latest)
    lead = np.arange(1, LEADS + 1, dtype="float32")[None, :, None]
    trend = (1.0 - np.exp(-lead / 6.0)) * delta[:, None, :]
    output = latest[:, None, :] + trend
    return _project_structure(output).astype("float32")


def _structure_transform(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype="float32")
    result = np.zeros_like(value, dtype="float32")
    result[..., 0] = (value[..., 0] - 50.0) / 35.0
    result[..., 1] = (value[..., 1] - 1000.0) / 20.0
    result[..., 2:] = np.log1p(np.maximum(value[..., 2:], 0.0) / 50.0)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _structure_inverse(value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value, dtype="float32")
    result[..., 0] = value[..., 0] * 35.0 + 50.0
    result[..., 1] = value[..., 1] * 20.0 + 1000.0
    result[..., 2:] = np.expm1(np.maximum(value[..., 2:], 0.0)) * 50.0
    return result


def _project_structure(value: np.ndarray) -> np.ndarray:
    output = value.astype("float32", copy=True)
    output[..., 0] = np.clip(output[..., 0], 0.0, 250.0)
    output[..., 1] = np.clip(output[..., 1], 850.0, 1060.0)
    output[..., 2:] = np.clip(output[..., 2:], 0.0, 1000.0)
    output[..., 7:11] = np.minimum(output[..., 7:11], output[..., 3:7])
    output[..., 11:15] = np.minimum(output[..., 11:15], output[..., 7:11])
    return output


class _WholePacificRouteModel(nn.Module):
    def __init__(self, config: dict[str, Any], context_dim: int) -> None:
        super().__init__()
        width = int(config["width"])
        field_width = int(config["field_width"])
        context_width = int(config["context_width"])
        dropout = float(config["dropout"])
        layers = int(config["layers"])
        heads = int(config["heads"])
        self.max_correction = float(config["max_correction_100km"])
        self.field = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, field_width, 3, padding=1), nn.GroupNorm(8, field_width), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.field_projection = nn.Linear(field_width, field_width)
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, context_width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(context_width, width), nn.GELU(),
        )
        self.fusion = nn.Sequential(nn.Linear(field_width + width, width), nn.LayerNorm(width), nn.GELU())
        self.lead = nn.Parameter(torch.randn(1, LEADS, width) * 0.02)
        start = max(0, min(LEADS, int(config.get("lead_start", 0))))
        ramp = torch.zeros(LEADS, dtype=torch.float32)
        if start < LEADS:
            ramp[start:] = torch.linspace(0.0, 1.0, LEADS - start) ** float(config.get("lead_ramp_power", 1.0))
        self.register_buffer("correction_ramp", ramp.view(1, LEADS, 1))
        position = torch.arange(LEADS, dtype=torch.float32).view(LEADS, 1)
        divisor = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
        encoding = torch.zeros(LEADS, width)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        block = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(block, layers)
        self.route_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))
        self.land_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, field: torch.Tensor, context: torch.Tensor, base_position: torch.Tensor):
        field_embedding = self.field_projection(self.field(field).flatten(1))
        context_embedding = self.context(context)
        fused = self.fusion(torch.cat([field_embedding, context_embedding], dim=1))
        tokens = self.temporal(fused[:, None, :] + self.lead + self.position_encoding)
        correction = torch.tanh(self.route_head(tokens)) * self.max_correction * self.correction_ramp
        position = base_position + correction
        land_probability = torch.sigmoid(self.land_head(tokens)).squeeze(-1)
        return position, land_probability, correction


class _ResidualCurveModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        width = int(config["width"])
        input_dim = INTENSITY_INPUT_DIM
        dropout = float(config["dropout"])
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width), nn.GELU())
        self.lead = nn.Parameter(torch.randn(1, LEADS, width) * 0.02)
        position = torch.arange(LEADS, dtype=torch.float32).view(LEADS, 1)
        divisor = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
        encoding = torch.zeros(LEADS, width)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("position", encoding.unsqueeze(0))
        block = nn.TransformerEncoderLayer(width, int(config["heads"]), width * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(block, int(config["layers"]))
        self.output = nn.Linear(width, STRUCTURE_DIM)
        self.register_buffer("residual_limit", torch.tensor([1.20, 1.20, 0.90] + [1.05] * 12, dtype=torch.float32).view(1, 1, -1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = self.input(features)
        tokens = self.temporal(context[:, None, :] + self.lead + self.position)
        return torch.tanh(self.output(tokens)) * self.residual_limit


class Trackformer12:
    """Load and run the released Trackformer 1.2 causal members."""

    def __init__(self, model_root: str | Path, device: str | torch.device | None = None):
        self.model_root = self._resolve_root(Path(model_root))
        self.device = _select_device(device)
        manifest_path = self.model_root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        stats_path = self.model_root / "trackformer_1_2_feature_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(f"missing feature statistics: {stats_path}")
        with np.load(stats_path, allow_pickle=False) as stats:
            self.feature_mean = stats["feature_mean"].astype("float32")
            self.feature_std = np.maximum(stats["feature_std"].astype("float32"), 1.0e-4)
        route_stats_path = self.model_root / "trackformer_1_2_route_context_stats.npz"
        self.route_context_mean: np.ndarray | None = None
        self.route_context_std: np.ndarray | None = None
        if route_stats_path.exists():
            with np.load(route_stats_path, allow_pickle=False) as route_stats:
                self.route_context_mean = route_stats["feature_mean"].astype("float32")
                self.route_context_std = np.maximum(route_stats["feature_std"].astype("float32"), 1.0e-4)
        track_stats_path = self.model_root / "trackformer_1_2_track_stats.npz"
        self.track_mean: np.ndarray | None = None
        self.track_std: np.ndarray | None = None
        if track_stats_path.exists():
            with np.load(track_stats_path, allow_pickle=False) as track_stats:
                self.track_mean = track_stats["track_mean"].astype("float32")
                self.track_std = np.maximum(track_stats["track_std"].astype("float32"), 1.0e-6)
        system_stats_path = self.model_root / "trackformer_1_2_system_context_stats.npz"
        self.system_context_stats: dict[str, np.ndarray] | None = None
        if system_stats_path.exists():
            with np.load(system_stats_path, allow_pickle=False) as system_stats:
                self.system_context_stats = {
                    key: system_stats[key].astype("float32")
                    for key in ("system_mean", "system_std", "interaction_mean", "interaction_std")
                }
        self.route_members = self._load_route_members()
        self.intensity_members = self._load_intensity_members()
        ocean_path = self.model_root / "trackformer_1_2_ocean_structure.joblib"
        self.ocean_structure = None
        if ocean_path.exists():
            try:
                import joblib
                self.ocean_structure = joblib.load(ocean_path)
            except ImportError as exc:
                raise ImportError("joblib is required for ocean structure calibration") from exc

    @staticmethod
    def _resolve_root(path: Path) -> Path:
        direct = path / "manifest.json"
        nested = path / "models" / "trackformer_1_2" / "manifest.json"
        if direct.exists():
            return path
        if nested.exists():
            return nested.parent
        raise FileNotFoundError(f"could not find Trackformer 1.2 manifest below {path}")

    @classmethod
    def from_pretrained(cls, model_root: str | Path | None = None, repo_id: str = "euler314/typhoon-predict", device: str | torch.device | None = None) -> "Trackformer12":
        """Load from an extracted release bundle or the Hugging Face model repo."""
        if model_root is None:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError("install huggingface_hub or pass model_root to from_pretrained") from exc
            model_root = snapshot_download(repo_id=repo_id, allow_patterns=["models/trackformer_1_2/*", "README.md"])
        return cls(model_root, device=device)

    def _load_route_members(self) -> list[_WholePacificRouteModel]:
        members = []
        for seed in (0, 1):
            payload = torch.load(self.model_root / f"trackformer_1_2_route_seed{seed}.pt", map_location="cpu", weights_only=False)
            model = _WholePacificRouteModel(payload["config"], int(payload["context_dim"]))
            missing, unexpected = model.load_state_dict(payload["model"], strict=False)
            # The published route checkpoints predate serialization of the
            # non-learned correction ramp.  Their config has no lead-ramp
            # fields, so preserve the checkpoint behavior with an all-one
            # ramp rather than silently applying a new schedule.
            if missing == ["correction_ramp"] and not unexpected and "lead_start" not in payload["config"]:
                model.correction_ramp.fill_(1.0)
            elif missing or unexpected:
                raise RuntimeError(f"incompatible route checkpoint {seed}: missing={missing}, unexpected={unexpected}")
            members.append(model.to(self.device).eval())
        return members

    def _load_intensity_members(self) -> list[_ResidualCurveModel]:
        members = []
        for seed in (0, 1):
            payload = torch.load(self.model_root / f"trackformer_1_2_intensity_seed{seed}.pt", map_location="cpu", weights_only=False)
            model = _ResidualCurveModel(payload["config"])
            model.load_state_dict(payload["model"], strict=True)
            members.append(model.to(self.device).eval())
        return members

    @torch.no_grad()
    def predict_route(
        self,
        field: np.ndarray,
        context: np.ndarray,
        base_position: np.ndarray,
        batch_size: int = 64,
        context_is_normalized: bool = False,
    ) -> dict[str, np.ndarray]:
        """Predict the 20-lead route from issue-time causal route tensors.

        ``field`` is normalized by :func:`prepare_route_field` and has shape
        ``[B, 6, H, W]``; ``context`` is raw ``[B, 647]`` unless
        ``context_is_normalized=True``; and ``base_position`` is ``[B, 20, 2]``
        in cumulative 100-km local displacement units.
        """
        field = _batch(field, (6, np.asarray(field).shape[-2], np.asarray(field).shape[-1]), "field")
        context = _batch(context, (ROUTE_CONTEXT_DIM,), "context")
        base_position = _batch(base_position, (LEADS, 2), "base_position")
        if field.shape[-2:] != (FIELD_HEIGHT, FIELD_WIDTH):
            raise ValueError(f"field must be the released [25, 61] Pacific grid, got {field.shape[-2:]}")
        _same_batch(field, context, base_position)
        if not context_is_normalized:
            if self.route_context_mean is None or self.route_context_std is None:
                raise FileNotFoundError("route context normalization stats are not present in this model bundle")
            context = normalize_route_context(context, self.route_context_mean, self.route_context_std)
        positions: list[np.ndarray] = []
        land: list[np.ndarray] = []
        for start in range(0, len(field), batch_size):
            stop = min(start + batch_size, len(field))
            inputs = (torch.from_numpy(field[start:stop]).to(self.device), torch.from_numpy(context[start:stop]).to(self.device), torch.from_numpy(base_position[start:stop]).to(self.device))
            members = [model(*inputs) for model in self.route_members]
            positions.append(torch.stack([item[0] for item in members]).mean(0).cpu().numpy())
            land.append(torch.stack([item[1] for item in members]).mean(0).cpu().numpy())
        return {"position_100km": np.concatenate(positions).astype("float32"), "land_probability": np.concatenate(land).astype("float32"), "lead_hours": np.asarray(LEAD_HOURS, dtype="int32")}

    @torch.no_grad()
    def predict_intensity(self, causal_features: np.ndarray, anchor_structure: np.ndarray, batch_size: int = 256) -> dict[str, np.ndarray]:
        """Predict wind, pressure, RMW, and quadrant radii.

        ``causal_features`` is the exact pre-residual 1,020-dimensional vector
        built from current/past observations and analysis summaries.  The
        ``anchor_structure`` is a physical ``[B, 20, 15]`` frozen causal
        Trackformer 1.2.26 curve ordered as ``vmax, pressure, RMW, R34[4],
        R50[4], R64[4]``.
        """
        causal_features = _batch(causal_features, (INTENSITY_FEATURE_DIM,), "causal_features")
        anchor = _batch(anchor_structure, (LEADS, STRUCTURE_DIM), "anchor_structure")
        _same_batch(causal_features, anchor)
        anchor_transformed = _structure_transform(anchor)
        features = np.concatenate([causal_features, anchor_transformed.reshape(len(anchor), -1)], axis=1)
        if features.shape[1] != len(self.feature_mean):
            raise ValueError(f"feature vector plus anchor must have {len(self.feature_mean)} columns, got {features.shape[1]}")
        normalized = ((features - self.feature_mean) / self.feature_std).astype("float32")
        members: list[np.ndarray] = []
        for start in range(0, len(normalized), batch_size):
            stop = min(start + batch_size, len(normalized))
            tensor = torch.from_numpy(normalized[start:stop]).to(self.device)
            delta = torch.stack([model(tensor) for model in self.intensity_members]).mean(0).cpu().numpy()
            members.append(_project_structure(_structure_inverse(anchor_transformed[start:stop] + delta)))
        return {"structure": np.concatenate(members).astype("float32"), "lead_hours": np.asarray(LEAD_HOURS, dtype="int32")}

    def predict_ocean_structure(self, causal_features: np.ndarray, ocean_features: np.ndarray, anchor_structure: np.ndarray, ocean_available: np.ndarray | float = 1.0) -> dict[str, np.ndarray]:
        """Apply the released causal OHC/D26/D20 calibration to the anchor.

        This is a separately validated calibration head trained around the
        frozen 1.2.26 structure anchor.  It is intentionally returned as its
        own forecast rather than silently stacking it on the neural residual
        head, which would be an unvalidated model change.
        """
        if self.ocean_structure is None:
            raise FileNotFoundError("trackformer_1_2_ocean_structure.joblib is not present")
        causal_features = _batch(causal_features, (INTENSITY_FEATURE_DIM,), "causal_features")
        ocean_features = _batch(ocean_features, (OCEAN_FEATURE_DIM,), "ocean_features")
        anchor = _batch(anchor_structure, (LEADS, STRUCTURE_DIM), "anchor_structure")
        _same_batch(causal_features, ocean_features, anchor)
        available = np.asarray(ocean_available, dtype="float32").reshape(-1)
        if len(available) == 1:
            available = np.repeat(available, len(anchor))
        if len(available) != len(anchor):
            raise ValueError("ocean_available must have one value per batch item")
        raw = np.concatenate([ocean_features, _structure_transform(anchor[:, -1]), causal_features], axis=1)
        fitted = self.ocean_structure
        normalized = ((raw - fitted["feature_mean"]) / np.maximum(fitted["feature_std"], 1.0e-4)).astype("float32")
        output = anchor.copy()
        groups = {"vmax": np.asarray([0]), "pressure": np.asarray([1]), "r34": np.arange(3, 7), "r50": np.arange(7, 11), "r64": np.arange(11, 15)}
        for group, columns in groups.items():
            meta = fitted["selection"][group]
            corrections = np.zeros((len(anchor), LEADS), dtype="float32")
            for lead, item in enumerate(fitted["models"][group]):
                selected = np.asarray(item["features"], dtype="int64")
                corrections[:, lead] = item["model"].predict(normalized[:, selected]).astype("float32")
            fraction = np.linspace(0.0, 1.0, LEADS, dtype="float32")
            ramp = np.zeros(LEADS, dtype="float32")
            start = int(meta["start"])
            ramp[start:] = fraction[start:] ** float(meta["power"])
            trusted = available[:, None] * ramp[None, :]
            if group in ("vmax", "pressure"):
                output[..., columns] += float(meta["alpha"]) * trusted[..., None] * corrections[..., None]
            else:
                log_value = np.log1p(np.maximum(output[..., columns], 0.0) / 50.0)
                log_value += float(meta["alpha"]) * trusted[..., None] * corrections[..., None]
                output[..., columns] = np.expm1(np.maximum(log_value, 0.0)) * 50.0
            output = _project_structure(output)
        return {"structure": output.astype("float32"), "lead_hours": np.asarray(LEAD_HOURS, dtype="int32"), "ocean_available": available.astype("float32")}

    def predict_issue_packet(self, packet: dict[str, np.ndarray]) -> dict[str, Any]:
        """Run every head present in an issue packet.

        Required route keys are ``field``, ``context``, and ``base_position``.
        Required intensity keys are ``causal_features`` and
        ``anchor_structure``.  Ocean calibration additionally accepts
        ``ocean_features`` and ``ocean_available``.
        """
        result: dict[str, Any] = {}
        if {"field", "context", "base_position"}.issubset(packet):
            result["route"] = self.predict_route(
                packet["field"], packet["context"], packet["base_position"],
                context_is_normalized=bool(packet.get("context_is_normalized", False)),
            )
        if {"causal_features", "anchor_structure"}.issubset(packet):
            result["intensity"] = self.predict_intensity(packet["causal_features"], packet["anchor_structure"])
            if "ocean_features" in packet:
                result["ocean_structure"] = self.predict_ocean_structure(packet["causal_features"], packet["ocean_features"], packet["anchor_structure"], packet.get("ocean_available", 1.0))
        if not result:
            raise ValueError("packet has no complete route or intensity input group")
        return result


__all__ = [
    "FIELD_HEIGHT", "FIELD_WIDTH", "INTENSITY_FEATURE_DIM", "INTENSITY_INPUT_DIM", "LEADS", "LEAD_HOURS",
    "OCEAN_FEATURE_DIM", "ROUTE_CONTEXT_DIM", "STRUCTURE_DIM", "Trackformer12",
    "build_base_position", "build_causal_anchor_structure", "build_causal_structure_features",
    "build_intensity_context_features", "build_kinematic_base_position", "build_ocean_features",
    "build_nearby_interaction_features", "build_patch_summary", "build_route_context",
    "build_route_geography_features", "build_route_kinematic_features",
    "build_route_system_features", "build_synoptic_features", "build_terrain_samples",
    "local_position_to_latlon", "normalize_route_context", "prepare_route_field",
]
