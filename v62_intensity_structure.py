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
