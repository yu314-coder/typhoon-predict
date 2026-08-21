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


LEADS = 20
LEAD_HOURS = tuple(range(6, 121, 6))
STRUCTURE_DIM = 15
ROUTE_CONTEXT_DIM = 647
INTENSITY_FEATURE_DIM = 1020
INTENSITY_INPUT_DIM = 1320
OCEAN_FEATURE_DIM = 66
FIELD_HEIGHT = 25
FIELD_WIDTH = 61


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
    def predict_route(self, field: np.ndarray, context: np.ndarray, base_position: np.ndarray, batch_size: int = 64) -> dict[str, np.ndarray]:
        """Predict the 20-lead route from issue-time causal route tensors.

        ``field`` is normalized by :func:`prepare_route_field` and has shape
        ``[B, 6, H, W]``; ``context`` is ``[B, 647]``; and ``base_position``
        is ``[B, 20, 2]`` in cumulative 100-km local displacement units.
        """
        field = _batch(field, (6, np.asarray(field).shape[-2], np.asarray(field).shape[-1]), "field")
        context = _batch(context, (ROUTE_CONTEXT_DIM,), "context")
        base_position = _batch(base_position, (LEADS, 2), "base_position")
        if field.shape[-2:] != (FIELD_HEIGHT, FIELD_WIDTH):
            raise ValueError(f"field must be the released [25, 61] Pacific grid, got {field.shape[-2:]}")
        _same_batch(field, context, base_position)
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
            result["route"] = self.predict_route(packet["field"], packet["context"], packet["base_position"])
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
    "local_position_to_latlon", "prepare_route_field",
]
