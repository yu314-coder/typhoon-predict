#!/usr/bin/env python3
"""Causal motion-anchored route adapter for Trackformer 1.2.12.

The base route is still generated from current, t-12, and t-24 analysis
fields.  This adapter only blends each base displacement with the recent
observed six-hour motion available at issue time.  The motion is capped for
outlier resistance and decays gradually so the analysis-field curvature is
retained at longer leads.

The coefficients are a fixed validation-storm policy selected from the
prepared archive.  JMA/JTWC forecast paths are never read by this module.
"""

from __future__ import annotations

import math

import numpy as np


VERSION = "Trackformer1.2.12-causal-motion-anchored-route"
DEFAULT_ZONAL_MAX_WEIGHT = 0.60
DEFAULT_MERIDIONAL_MAX_WEIGHT = 0.84
DEFAULT_ZONAL_TAU_HOURS = 120.0
DEFAULT_MERIDIONAL_TAU_HOURS = 240.0
DEFAULT_MOTION_CAP_KM_PER_6H = 120.0


def motion_weights(
    n_leads: int,
    lead_interval_hours: float = 6.0,
    zonal_max_weight: float = DEFAULT_ZONAL_MAX_WEIGHT,
    meridional_max_weight: float = DEFAULT_MERIDIONAL_MAX_WEIGHT,
    zonal_tau_hours: float = DEFAULT_ZONAL_TAU_HOURS,
    meridional_tau_hours: float = DEFAULT_MERIDIONAL_TAU_HOURS,
) -> np.ndarray:
    """Return causal zonal and meridional weights for each positive lead."""

    if n_leads <= 0:
        return np.zeros(0, dtype="float32")
    values = (zonal_max_weight, meridional_max_weight, zonal_tau_hours, meridional_tau_hours)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("motion policy coefficients must be finite")
    if (
        zonal_tau_hours <= 0.0 or meridional_tau_hours <= 0.0
        or not 0.0 <= zonal_max_weight <= 1.0
        or not 0.0 <= meridional_max_weight <= 1.0
    ):
        raise ValueError("expected motion weights in [0, 1] and positive decay times")
    lead = np.arange(1, n_leads + 1, dtype="float64") * float(lead_interval_hours)
    return np.stack([
        float(zonal_max_weight) * np.exp(-lead / float(zonal_tau_hours)),
        float(meridional_max_weight) * np.exp(-lead / float(meridional_tau_hours)),
    ], axis=1).astype("float32")


def apply_motion_anchor(
    displacement_km: np.ndarray,
    history_motion_km_per_6h: tuple[float, float] | np.ndarray | None,
    *,
    zonal_max_weight: float = DEFAULT_ZONAL_MAX_WEIGHT,
    meridional_max_weight: float = DEFAULT_MERIDIONAL_MAX_WEIGHT,
    zonal_tau_hours: float = DEFAULT_ZONAL_TAU_HOURS,
    meridional_tau_hours: float = DEFAULT_MERIDIONAL_TAU_HOURS,
    motion_cap_km_per_6h: float = DEFAULT_MOTION_CAP_KM_PER_6H,
) -> tuple[np.ndarray, dict]:
    """Blend a causal recent-motion prior into a route displacement sequence.

    ``displacement_km`` is ``[lead, east, north]`` in km per six-hour step.
    No future route or official forecast data is accepted.  The returned
    diagnostics make the policy and the exact issue-time motion auditable.
    """

    base = np.asarray(displacement_km, dtype="float32")
    if base.ndim != 2 or base.shape[1] != 2:
        raise ValueError(f"expected displacement shape (leads, 2), got {base.shape}")
    weights = motion_weights(
        len(base),
        zonal_max_weight=zonal_max_weight,
        meridional_max_weight=meridional_max_weight,
        zonal_tau_hours=zonal_tau_hours,
        meridional_tau_hours=meridional_tau_hours,
    )
    candidate = None
    if history_motion_km_per_6h is not None:
        values = np.asarray(history_motion_km_per_6h, dtype="float32").reshape(-1)
        if values.size == 2 and np.isfinite(values).all():
            candidate = values
    if candidate is None:
        return base.copy(), {
            "version": VERSION,
            "applied": False,
            "reason": "issue-time recent motion unavailable",
            "weights": weights.tolist(),
        }

    raw_speed = float(np.linalg.norm(candidate))
    if not math.isfinite(motion_cap_km_per_6h) or motion_cap_km_per_6h <= 0.0:
        raise ValueError("motion_cap_km_per_6h must be positive")
    scale = min(1.0, float(motion_cap_km_per_6h) / max(raw_speed, 1.0e-6))
    capped = candidate * np.float32(scale)
    corrected = (1.0 - weights) * base + weights * capped[None, :]
    return corrected.astype("float32"), {
        "version": VERSION,
        "applied": True,
        "input_policy": "issue-time and past observed motion only; no official forecast path",
        "zonal_max_weight": float(zonal_max_weight),
        "meridional_max_weight": float(meridional_max_weight),
        "zonal_tau_hours": float(zonal_tau_hours),
        "meridional_tau_hours": float(meridional_tau_hours),
        "motion_cap_km_per_6h": float(motion_cap_km_per_6h),
        "history_motion_km_per_6h": candidate.astype("float32").tolist(),
        "capped_motion_km_per_6h": capped.astype("float32").tolist(),
        "raw_motion_speed_km_per_6h": raw_speed,
        "capped_motion_speed_km_per_6h": float(np.linalg.norm(capped)),
        "weights": weights.tolist(),
        "base_displacement_last_km": base[-1].astype("float32").tolist(),
        "corrected_displacement_last_km": corrected[-1].astype("float32").tolist(),
    }


__all__ = [
    "VERSION",
    "DEFAULT_ZONAL_MAX_WEIGHT",
    "DEFAULT_MERIDIONAL_MAX_WEIGHT",
    "DEFAULT_ZONAL_TAU_HOURS",
    "DEFAULT_MERIDIONAL_TAU_HOURS",
    "DEFAULT_MOTION_CAP_KM_PER_6H",
    "motion_weights",
    "apply_motion_anchor",
]
