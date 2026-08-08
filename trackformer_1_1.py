"""Public Trackformer1.1 loading and inference API."""

from __future__ import annotations

from pathlib import Path

from trackformer_1_1_intensity import Trackformer11IntensityEnsemble
from trackformer_1_1_route import (
    LEAD_HOURS,
    build_pacific_route,
    detect_pressure_systems,
    forecast_pacific_state,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PACKAGE_ROOT / "models" / "trackformer_1_1"
CALIBRATION_PATH = MODEL_ROOT / "trackformer_1_1_calibration.json"


def load_intensity(device: str | None = None) -> Trackformer11IntensityEnsemble:
    """Load the frozen Trackformer1.1 intensity and structure experts."""

    return Trackformer11IntensityEnsemble(MODEL_ROOT, CALIBRATION_PATH, device=device)


__all__ = [
    "CALIBRATION_PATH",
    "LEAD_HOURS",
    "MODEL_ROOT",
    "build_pacific_route",
    "detect_pressure_systems",
    "forecast_pacific_state",
    "load_intensity",
]
