#!/usr/bin/env python3
"""Causal v62 intensity and wind-structure inference.

The v62 route is responsible for position and the western-Pacific pressure
state.  This module supplies the previously missing storm-structure outputs
without changing that route: maximum sustained wind, central pressure, radius
of maximum wind, and the four-quadrant R34/R50/R64 radii.

The weights are the validated v37G spatial structure ensemble.  They are
loaded only for inference, and are conditioned on the same nine-step observed
track window plus the current v23-compatible four-channel analysis patch.
No positive-lead atmospheric field or official agency forecast is consumed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


LEADS = 20
TARGET_SCALE = np.asarray([100.0, 100.0, 35.0, 20.0, 50.0] + [50.0] * 12, dtype="float32")
STRUCTURE_SCALE = TARGET_SCALE[2:]
THERMO_ENV_COLS = (
    [4, 5, 6, 7]
    + list(range(8, 20))
    + list(range(24, 40))
    + [44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
)


def sinusoidal(length: int, width: int) -> torch.Tensor:
    position = torch.arange(length).unsqueeze(1).float()
    divisor = torch.exp(torch.arange(0, width, 2).float() * (-np.log(10000.0) / width))
    result = torch.zeros(length, width)
    result[:, 0::2] = torch.sin(position * divisor)
    result[:, 1::2] = torch.cos(position * divisor)
    return result


class StructureSpatialV37G(nn.Module):
    """Inference copy of the architecture used by the frozen v37G weights."""

    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.track_proj = nn.Linear(len(THERMO_ENV_COLS), width)
        self.register_buffer("track_time", sinusoidal(9, width).unsqueeze(0))
        track_layer = nn.TransformerEncoderLayer(
            width,
            heads,
            width * 4,
            0.12,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.track_encoder = nn.TransformerEncoder(track_layer, layers)
        self.field_encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.field_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.field_norm = nn.LayerNorm(width)
        self.field_pos = nn.Parameter(torch.randn(1, 16, width) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            width,
            heads,
            width * 4,
            0.12,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.query = nn.Parameter(torch.randn(1, LEADS, width) * 0.02)
        self.register_buffer("lead_time", sinusoidal(LEADS, width).unsqueeze(0))
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.state = nn.Linear(width, 15)
        self.log_scale = nn.Linear(width, 15)

    def forward(
        self,
        track: torch.Tensor,
        field: torch.Tensor,
        current: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        track_tokens = self.track_encoder(self.track_proj(track[:, :, THERMO_ENV_COLS]) + self.track_time)
        field_tokens = self.field_pool(self.field_encoder(field)).flatten(2).transpose(1, 2)
        field_tokens = self.field_norm(field_tokens + self.field_pos)
        memory = torch.cat([track_tokens, field_tokens], dim=1)
        query = (self.query + self.lead_time).expand(track.shape[0], -1, -1)
        hidden = self.decoder(query, memory)
        state = self.state(hidden)
        state = state.clone()
        state[:, :, :2] = state[:, :, :2] + (current * available)[:, None, :]
        return state, self.log_scale(hidden)


def _device(requested: str | None) -> torch.device:
    value = requested or os.environ.get("V62_INTENSITY_DEVICE")
    if value:
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _calibrated_wind(states: np.ndarray, current_wind: float, calibration: dict) -> np.ndarray:
    alphas = calibration.get("wind_blend_alpha")
    if not alphas:
        return states[:, :, 0]
    predicted = states[:, :, 0]
    result = np.empty_like(predicted)
    for lead, alpha in enumerate(alphas[:LEADS]):
        result[:, lead] = float(alpha) * predicted[:, lead] + (1.0 - float(alpha)) * current_wind
    return np.clip(result, 0.0, 190.0)


def _calibrated_pressure(
    states: np.ndarray,
    current_wind: float,
    current_pressure: float,
    previous_wind: float,
    previous_pressure: float,
    calibrated_wind: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    joint = calibration.get("pressure_joint_calibrations")
    if not joint:
        return np.clip(states[:, :, 1], 850.0, 1025.0)
    predicted_pressure = states[:, :, 1]
    result = np.empty_like(predicted_pressure)
    for lead, item in enumerate(joint[:LEADS]):
        features = np.column_stack([
            np.full(len(predicted_pressure), 1.0, dtype="float32"),
            np.full(len(predicted_pressure), current_pressure, dtype="float32"),
            np.full(len(predicted_pressure), current_wind, dtype="float32"),
            calibrated_wind[:, lead],
            predicted_pressure[:, lead],
            predicted_pressure[:, lead] - current_pressure,
            calibrated_wind[:, lead] - current_wind,
            np.full(len(predicted_pressure), current_pressure - previous_pressure, dtype="float32"),
            np.full(len(predicted_pressure), current_wind - previous_wind, dtype="float32"),
        ])
        mean = np.asarray(item["mean"], dtype="float32")
        scale = np.maximum(np.asarray(item["scale"], dtype="float32"), 1e-6)
        normalized = (features - mean) / scale
        normalized[:, 0] = 1.0
        result[:, lead] = normalized @ np.asarray(item["beta"], dtype="float32")
    return np.clip(result, 850.0, 1025.0)


def _sanitize(states: np.ndarray) -> np.ndarray:
    """Apply output bounds and preserve the R34 >= R50 >= R64 ordering."""

    result = np.asarray(states, dtype="float32").copy()
    result[:, :, 0] = np.clip(result[:, :, 0], 0.0, 190.0)
    result[:, :, 1] = np.clip(result[:, :, 1], 850.0, 1025.0)
    result[:, :, 2] = np.clip(result[:, :, 2], 0.0, 300.0)
    radii = np.clip(result[:, :, 3:15], 0.0, 1000.0)
    for offset in range(4):
        radii[:, :, 4 + offset] = np.minimum(radii[:, :, 4 + offset], radii[:, :, offset])
        radii[:, :, 8 + offset] = np.minimum(radii[:, :, 8 + offset], radii[:, :, 4 + offset])
    result[:, :, 3:15] = radii
    return result


def _row(mean: np.ndarray, spread: np.ndarray, lead: int) -> dict:
    radii = mean[lead, 3:15]
    radius_spread = spread[lead, 3:15]
    return {
        "vmax_kt": round(float(mean[lead, 0]), 2),
        "vmax_spread_kt": round(float(spread[lead, 0]), 2),
        "central_pressure_hpa": round(float(mean[lead, 1]), 2),
        "pressure_spread_hpa": round(float(spread[lead, 1]), 2),
        "rmw_km": round(float(mean[lead, 2]), 2),
        "rmw_spread_km": round(float(spread[lead, 2]), 2),
        "wind_radii_km": np.round(radii, 2).tolist(),
        "wind_radii_spread_km": np.round(radius_spread, 2).tolist(),
        # Keep the map renderer's generic pressure field alias as well.
        "pressure_hpa": round(float(mean[lead, 1]), 2),
    }


def _map_grid_feature(
    pressure: np.ndarray,
    fields: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    query_latitude: float,
    query_longitude: float,
) -> dict:
    """Extract a storm-relative signal from one forecast pressure map.

    The map can be much coarser than the route grid, especially for the Tip
    reanalysis.  We therefore use a nearby pressure minimum, a broad annulus
    environment, and quadrant-wise anomaly extents rather than pretending that
    a single grid cell is an exact storm center.
    """

    pressure = np.asarray(pressure, dtype="float32")
    fields = np.asarray(fields, dtype="float32")
    latitude = np.asarray(latitude, dtype="float32").reshape(-1)
    longitude = np.asarray(longitude, dtype="float32").reshape(-1)
    lat_order = np.argsort(latitude)
    lon_order = np.argsort(longitude)
    latitude = latitude[lat_order]
    longitude = longitude[lon_order]
    pressure = pressure[np.ix_(lat_order, lon_order)]
    fields = fields[:, lat_order, :][:, :, lon_order]
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    delta_lon = ((lon_grid - float(query_longitude) + 180.0) % 360.0) - 180.0
    delta_lat = lat_grid - float(query_latitude)
    distance_deg = np.hypot(delta_lat, delta_lon * np.cos(np.deg2rad(float(query_latitude))))
    query_row, query_column = np.unravel_index(int(np.nanargmin(distance_deg)), distance_deg.shape)
    query_pressure = float(pressure[query_row, query_column])
    query_wind850 = float(np.hypot(fields[1, query_row, query_column], fields[2, query_row, query_column]) * 1.94384) if fields.shape[0] >= 3 else float("nan")
    # A minimum farther than this is likely a separate synoptic system rather
    # than the cyclone represented by the route point.  Keep the local query
    # as a fallback instead of allowing a distant low to control intensity.
    nearby = distance_deg <= 6.0
    finite = np.isfinite(pressure)
    candidate = np.where(nearby & finite, pressure, np.inf)
    if not np.isfinite(candidate).any():
        row, column = query_row, query_column
    else:
        row, column = np.unravel_index(int(np.argmin(candidate)), candidate.shape)
    center_latitude = float(latitude[row])
    center_longitude = float(longitude[column])
    center_pressure = float(pressure[row, column])
    center_delta_lon = ((lon_grid - center_longitude + 180.0) % 360.0) - 180.0
    center_delta_lat = lat_grid - center_latitude
    center_distance_deg = np.hypot(
        center_delta_lat,
        center_delta_lon * np.cos(np.deg2rad(center_latitude)),
    )
    annulus = (
        (center_distance_deg >= 2.5)
        & (center_distance_deg <= 6.0)
        & np.isfinite(pressure)
    )
    environment = float(np.nanmedian(pressure[annulus])) if annulus.any() else center_pressure + 15.0
    anomaly = np.maximum(environment - pressure, 0.0)
    # Four degrees is a conservative upper bound for a pressure-derived
    # tropical wind footprint on this coarse map.  Without it, a weak broad
    # gradient would be misreported as an 800-km R34.
    core = (center_distance_deg <= 4.0) & np.isfinite(anomaly)
    peak_anomaly = float(np.nanmax(anomaly[core])) if core.any() else 0.0
    radii: list[float] = []
    # These are pressure-anomaly fractions used as a stable proxy for the
    # R34/R50/R64 shape.  The learned radius output remains the absolute
    # baseline; only its map-observed expansion/contraction is applied.
    for fraction in (0.25, 0.50, 0.70):
        for quadrant in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)):
            north_sign, east_sign = quadrant
            quadrant_mask = core & (center_delta_lat * north_sign >= 0.0) & (center_delta_lon * east_sign >= 0.0)
            threshold = peak_anomaly * fraction
            valid = quadrant_mask & (anomaly >= threshold) if peak_anomaly > 0.0 else np.zeros_like(core)
            if valid.any():
                radii.append(float(np.nanmax(center_distance_deg[valid]) * 111.2))
            else:
                radii.append(float("nan"))
    wind850 = float(np.hypot(fields[1, row, column], fields[2, row, column]) * 1.94384) if fields.shape[0] >= 3 else float("nan")
    center_offset_km = float(distance_deg[row, column] * 111.2)
    center_trusted = bool(center_offset_km <= 333.6 and (environment - center_pressure) >= 4.0)
    offset_degrees = center_offset_km / 111.2
    center_confidence = 1.0 if center_trusted else float(np.clip(1.0 - 0.75 * (offset_degrees - 3.0) / 3.0, 0.25, 0.75))
    return {
        "map_min_pressure_hpa": round(center_pressure, 2),
        "map_query_pressure_hpa": round(query_pressure, 2) if np.isfinite(query_pressure) else None,
        "map_environment_pressure_hpa": round(environment, 2),
        "map_pressure_deficit_hpa": round(max(0.0, environment - center_pressure), 2),
        "map_center_latitude": round(center_latitude, 3),
        "map_center_longitude": round(center_longitude % 360.0, 3),
        "map_center_offset_km": round(center_offset_km, 2),
        "map_center_trusted": center_trusted,
        "map_center_confidence": round(center_confidence, 3),
        "map_wind850_kt": round(wind850, 2) if np.isfinite(wind850) else None,
        "map_query_wind850_kt": round(query_wind850, 2) if np.isfinite(query_wind850) else None,
        "map_pressure_radii_km": [round(value, 2) if np.isfinite(value) else None for value in radii],
    }


def couple_forecast_to_pressure_map(
    structure_rows: list[dict],
    pressure_states: np.ndarray,
    field_states: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    base_latitude: float,
    base_longitude: float,
    forecast_points: list[dict],
    current_wind: float,
    current_pressure: float,
) -> tuple[list[dict], dict]:
    """Use the v62 pressure-map trajectory to correct structure forecasts.

    The correction is anchored to the observed current wind and pressure, so
    coarse reanalysis cannot replace a known storm intensity.  Future changes
    in the map minimum drive bounded pressure/wind changes, and changes in the
    map's quadrant anomaly extent rescale the learned radii.  This is the same
    forecast-state family used by the track route, not a future official field.
    """

    pressure_states = np.asarray(pressure_states, dtype="float32")
    field_states = np.asarray(field_states, dtype="float32")
    if pressure_states.ndim != 3 or field_states.ndim != 4 or field_states.shape[1] < 3:
        raise ValueError(f"invalid v62 map state shapes: {pressure_states.shape}, {field_states.shape}")
    if len(structure_rows) != len(forecast_points) or len(pressure_states) < len(structure_rows) + 1:
        raise ValueError("map states, route points, and structure rows have incompatible lengths")
    if not np.isfinite(current_wind) or not np.isfinite(current_pressure):
        return structure_rows, {
            "enabled": False,
            "reason": "current observed wind and pressure are unavailable",
            "official_forecasts_used": False,
        }

    features = [_map_grid_feature(
        pressure_states[index],
        field_states[index],
        latitude,
        longitude,
        base_latitude if index == 0 else float(forecast_points[index - 1]["lat"]),
        base_longitude if index == 0 else float(forecast_points[index - 1]["lon"]),
    ) for index in range(len(structure_rows) + 1)]
    # Use a trusted nearby minimum when available.  Otherwise sample the map
    # at the route point; this keeps another low/typhoon from hijacking the
    # structure forecast merely because it is the strongest minimum nearby.
    effective_minimum = np.asarray([
        float(item["map_query_pressure_hpa"])
        + float(item["map_center_confidence"]) * (
            float(item["map_min_pressure_hpa"]) - float(item["map_query_pressure_hpa"])
        )
        for item in features
    ], dtype="float32")
    effective_deficit = np.asarray([
        float(item["map_pressure_deficit_hpa"]) * float(item["map_center_confidence"])
        for item in features
    ], dtype="float32")
    effective_wind850 = np.asarray([
        float(item["map_query_wind850_kt"])
        + float(item["map_center_confidence"]) * (
            float(item["map_wind850_kt"]) - float(item["map_query_wind850_kt"])
        )
        for item in features
    ], dtype="float32")
    # A short causal smoother prevents a one-cell minimum handoff from making
    # a six-hour intensity jump while retaining the map's lead-time trend.
    smooth_minimum = effective_minimum.copy()
    smooth_deficit = effective_deficit.copy()
    smooth_wind850 = effective_wind850.copy()
    for index in range(1, len(smooth_minimum)):
        smooth_minimum[index] = 0.65 * smooth_minimum[index - 1] + 0.35 * effective_minimum[index]
        smooth_deficit[index] = 0.65 * smooth_deficit[index - 1] + 0.35 * effective_deficit[index]
        if np.isfinite(effective_wind850[index]) and np.isfinite(smooth_wind850[index - 1]):
            smooth_wind850[index] = 0.65 * smooth_wind850[index - 1] + 0.35 * effective_wind850[index]
    reference_radii = np.asarray([
        float(value) if value is not None else np.nan
        for value in features[0]["map_pressure_radii_km"]
    ], dtype="float32")
    corrected: list[dict] = []
    map_weight = 0.35
    for index, base in enumerate(structure_rows, start=1):
        row = dict(base)
        row_weight = map_weight * (0.55 + 0.45 * float(features[index]["map_center_confidence"]))
        map_pressure = float(current_pressure + smooth_minimum[index] - smooth_minimum[0])
        pressure_signal = float(np.clip(
            0.55 * (smooth_minimum[0] - smooth_minimum[index])
            + 0.45 * (smooth_deficit[index] - smooth_deficit[0]),
            -30.0,
            30.0,
        ))
        wind_signal = pressure_signal
        if np.isfinite(smooth_wind850[index]) and np.isfinite(smooth_wind850[0]):
            wind_signal += float(np.clip(0.18 * (smooth_wind850[index] - smooth_wind850[0]), -8.0, 8.0))
        map_wind = float(current_wind + 0.70 * wind_signal)
        row["central_pressure_hpa"] = round(float(np.clip(
            (1.0 - row_weight) * float(base["central_pressure_hpa"]) + row_weight * map_pressure,
            850.0,
            1025.0,
        )), 2)
        row["pressure_hpa"] = row["central_pressure_hpa"]
        row["vmax_kt"] = round(float(np.clip(
            (1.0 - row_weight) * float(base["vmax_kt"]) + row_weight * map_wind,
            0.0,
            190.0,
        )), 2)
        row["pressure_spread_hpa"] = round(float(np.hypot(
            float(base.get("pressure_spread_hpa", 0.0)),
            row_weight * abs(map_pressure - float(base["central_pressure_hpa"])),
        )), 2)
        row["vmax_spread_kt"] = round(float(np.hypot(
            float(base.get("vmax_spread_kt", 0.0)),
            row_weight * abs(map_wind - float(base["vmax_kt"])),
        )), 2)
        map_radii = np.asarray([
            float(value) if value is not None else np.nan
            for value in features[index]["map_pressure_radii_km"]
        ], dtype="float32")
        base_radii = np.asarray(base["wind_radii_km"], dtype="float32")
        if not features[0]["map_center_trusted"] or not features[index]["map_center_trusted"]:
            map_radii[:] = np.nan
        if np.isfinite(reference_radii).any() and np.isfinite(map_radii).any():
            valid = np.isfinite(reference_radii) & np.isfinite(map_radii) & (reference_radii >= 20.0)
            ratios = np.ones(12, dtype="float32")
            ratios[valid] = np.clip(map_radii[valid] / reference_radii[valid], 0.60, 1.50)
            if valid.any():
                ratios[~valid] = float(np.clip(np.nanmedian(ratios[valid]), 0.60, 1.50))
            radius_factor = 1.0 + 0.25 * (ratios - 1.0)
            adjusted_radii = base_radii * radius_factor
        else:
            adjusted_radii = base_radii
        adjusted_radii = np.clip(adjusted_radii, 0.0, 1000.0)
        for quadrant in range(4):
            adjusted_radii[4 + quadrant] = min(adjusted_radii[4 + quadrant], adjusted_radii[quadrant])
            adjusted_radii[8 + quadrant] = min(adjusted_radii[8 + quadrant], adjusted_radii[4 + quadrant])
        row["wind_radii_km"] = np.round(adjusted_radii, 2).tolist()
        map_rmw_ratio = 1.0
        if features[0]["map_center_trusted"] and features[index]["map_center_trusted"]:
            ref_r64 = float(np.nanmedian(reference_radii[8:])) if np.isfinite(reference_radii[8:]).any() else np.nan
            future_r64 = float(np.nanmedian(map_radii[8:])) if np.isfinite(map_radii[8:]).any() else np.nan
            if np.isfinite(ref_r64) and np.isfinite(future_r64) and ref_r64 >= 20.0:
                map_rmw_ratio = float(np.clip(future_r64 / ref_r64, 0.75, 1.25))
                row["rmw_km"] = round(float(np.clip(
                    float(base["rmw_km"]) * (1.0 + 0.15 * (map_rmw_ratio - 1.0)),
                    0.0,
                    300.0,
                )), 2)
        row["pressure_map_features"] = features[index]
        corrected.append(row)
    metadata = {
        "enabled": True,
        "method": "v62 forecast pressure-map query plus confidence-weighted local-minimum/anomaly-radius coupling",
        "map_weight": map_weight,
        "minimum_confidence": "local minimum contribution decays with route-to-minimum offset; route-point map pressure remains the fallback",
        "wind_pressure_anchor": "observed current wind/pressure; map drives only future changes",
        "radius_method": "learned radius baseline rescaled by trusted local quadrant pressure-anomaly extent; max four-degree footprint",
        "map_features": features,
        "official_forecasts_used": False,
        "positive_lead_weather_product_used": False,
    }
    return corrected, metadata


class V62IntensityEnsemble:
    """Load the frozen v37G experts and emit calibrated v62 structure rows."""

    def __init__(
        self,
        checkpoint_root: Path,
        calibration_path: Path | None = None,
        device: str | None = None,
    ):
        self.checkpoint_root = Path(checkpoint_root)
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.device = _device(device)
        paths = sorted(self.checkpoint_root.glob("structure_spatial_v37g_seed*.pt"))
        if len(paths) < 3:
            raise FileNotFoundError(
                f"expected the three validated v37G intensity checkpoints in {self.checkpoint_root}; found {len(paths)}"
            )
        self.models: list[StructureSpatialV37G] = []
        for path in paths[:3]:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            config = payload["config"]
            model = StructureSpatialV37G(config["width"], config["layers"], config["heads"])
            model.load_state_dict(payload["model"])
            self.models.append(model.to(self.device).eval())
        self.calibration = {}
        if self.calibration_path and self.calibration_path.exists():
            self.calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))

    @torch.no_grad()
    def predict(
        self,
        track: np.ndarray,
        field: np.ndarray,
        current_wind: float,
        current_pressure: float,
        previous_wind: float,
        previous_pressure: float,
    ) -> tuple[list[dict], dict]:
        track = np.asarray(track, dtype="float32")
        field = np.asarray(field, dtype="float32")
        if track.shape != (9, 54):
            raise ValueError(f"v62 intensity track must have shape (9, 54), got {track.shape}")
        if field.shape != (4, 17, 17):
            raise ValueError(f"v62 intensity field must have shape (4, 17, 17), got {field.shape}")
        if not np.isfinite(track).all() or not np.isfinite(field).all():
            raise ValueError("v62 intensity inputs contain non-finite values")
        current_available = float(np.isfinite(current_wind) and np.isfinite(current_pressure))
        current_values = np.nan_to_num(
            np.asarray([current_wind, current_pressure], dtype="float32") / TARGET_SCALE[2:4],
            nan=0.0,
        )
        available = np.asarray([current_available, current_available], dtype="float32")
        track_tensor = torch.from_numpy(track[None]).to(self.device)
        field_tensor = torch.from_numpy(field[None]).to(self.device)
        current_tensor = torch.from_numpy(current_values[None]).to(self.device)
        available_tensor = torch.from_numpy(available[None]).to(self.device)
        states = np.stack([
            model(track_tensor, field_tensor, current_tensor, available_tensor)[0][0].detach().cpu().numpy()
            for model in self.models
        ]).astype("float32")
        states *= STRUCTURE_SCALE[None, None, :]
        if current_available:
            calibrated_wind = _calibrated_wind(states, float(current_wind), self.calibration)
            states[:, :, 1] = _calibrated_pressure(
                states,
                float(current_wind),
                float(current_pressure),
                float(previous_wind) if np.isfinite(previous_wind) else float(current_wind),
                float(previous_pressure) if np.isfinite(previous_pressure) else float(current_pressure),
                calibrated_wind,
                self.calibration,
            )
            states[:, :, 0] = calibrated_wind
        states = _sanitize(states)
        mean = states.mean(axis=0)
        spread = states.std(axis=0)
        rows = [_row(mean, spread, lead) for lead in range(LEADS)]
        metadata = {
            "model": "frozen v37G spatial structure ensemble used as the v62 intensity head",
            "checkpoint_count": len(self.models),
            "calibration": str(self.calibration_path) if self.calibration_path and self.calibration_path.exists() else None,
            "field_contract": "4x17x17 v23-compatible analysis patch, q/31.75 then clipped to [-4,4]",
            "outputs": ["vmax_kt", "central_pressure_hpa", "rmw_km", "wind_radii_km"],
            "ensemble_spread": "standard deviation across the three frozen spatial experts",
            "official_forecasts_used": False,
            "positive_lead_weather_used": False,
            "device": str(self.device),
        }
        return rows, metadata
