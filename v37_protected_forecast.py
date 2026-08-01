#!/usr/bin/env python3
"""Run the local v37 Dolphin forecast and render an offline comparison map.

Route: an offline ensemble of the public deterministic GFS and 31 GEFS member
850-hPa vortex tracks. The route is derived from relative-vorticity maxima in
the public forecast fields, then averaged at prediction level.

Structure: prediction-level ensemble of the locally trained v37C structure
branch for wind, pressure, RMW, and wind radii. No Colab or API key is used.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "hf-space"))
sys.path.insert(0, str(ROOT / "v37"))
sys.path.insert(0, str(ROOT))

import trackformer_v23 as v23_module  # noqa: E402
from trackformer_v23 import TARGET_SCALE as V23_TARGET_SCALE  # noqa: E402
from trackformer_v23 import build_v23  # noqa: E402
from v37_protected_train import (  # noqa: E402
    DEVICE,
    StructureV37,
    TARGET_SCALE,
    StructureEnsemble,
)
from paper.draw_dolphin_v36_world_map import (  # noqa: E402
    OBSERVED as HISTORIC_OBSERVED,
    annotate_point,
    category,
    draw_world,
    intensity_color,
    load_coastlines,
    lon360,
    plot_track,
    safe_json,
)
from scripts.build_v36_ibtracs_jma_official_comparison import (  # noqa: E402
    parse_jma as parse_current_jma,
    parse_jtwc as parse_current_jtwc,
)
from scripts.predict_ibtracs_jma_only import (  # noqa: E402
    build_track_window,
    distance_to_land,
    load_land_points,
)


# The released v23 module keeps its annulus mask as a module-level tensor. Move
# that inference-only constant to the selected Mac device before calling it.
v23_module.ANN = v23_module.ANN.to(DEVICE)


CURRENT_FIELD_PATH = ROOT / "track_build" / "dolphin_dlm4_current.npz"
REAL_FIELD_PATH = CURRENT_FIELD_PATH if CURRENT_FIELD_PATH.exists() else ROOT / "track_build" / "dolphin_dlm4_real.npz"
STRUCTURE_FIELD_PATH = ROOT / "track_build" / "dolphin_v37e_field_current.npz"
FIELD_SCALE_PATH = ROOT / "track_build" / "dlm4_int8.npz"
INTENSITY_CALIBRATION_PATH = ROOT / "v37" / "structure_spatial" / "v37g_intensity_calibration.json"
NORM_PATH = (
    ROOT / "v37" / "v23_norm_stats.npz"
    if (ROOT / "v37" / "v23_norm_stats.npz").exists()
    else ROOT / "hf-space" / "v23_norm_stats.npz"
)
GFS_FORECAST_PATH = ROOT / "track_build" / "dolphin_gfs_forecast_current.npz"
GFS_RAW_ROOT = ROOT / "v37" / "current_gfs_forecast"
GEFS_FORECAST_PATH = ROOT / "track_build" / "dolphin_gefs_ensemble_current.npz"
V36_PATH = ROOT / "track_build" / "dolphin_v36_forecast.json"
MODEL_VERSION = os.environ.get("V37_MODEL_VERSION", "v37D")
ARTIFACT_TAG = os.environ.get("V37_ARTIFACT_TAG", MODEL_VERSION.lower())
OUTPUT_JSON = ROOT / "track_build" / f"dolphin_{ARTIFACT_TAG}_forecast.json"
OUTPUT_HTML = ROOT / "paper" / f"dolphin_{ARTIFACT_TAG}_world_map.html"
OUTPUT_PNG = ROOT / "paper" / f"dolphin_{ARTIFACT_TAG}_world_map.png"
OUTPUT_CHART = ROOT / "paper" / f"dolphin_{ARTIFACT_TAG}_intensity.png"
SYNOPTIC_CHART = ROOT / "paper" / f"dolphin_{ARTIFACT_TAG}_gfs_synoptic.png"
ROUTE_POLICY = os.environ.get(
    "V37_ROUTE_POLICY",
    "gfs_vortex_multilevel_ensemble"
    if MODEL_VERSION in {"v37E", "v37G", "v37I"}
    else "gfs_vortex_ensemble",
)

# Fitted on the v37G validation split only.  Columns are [bias, current
# pressure, current wind, predicted wind, predicted wind-current wind].
V37G_PRESSURE_CALIBRATION = np.asarray([
    [1000.50738285, 0.02460214, -0.28285116, -0.40560690, -0.12275596],
    [1016.48099380, 0.00912400, -0.22241769, -0.48416475, -0.26174734],
    [1020.67204532, 0.00479206, -0.22822725, -0.48884163, -0.26061459],
    [1022.16127175, 0.00307313, -0.23933334, -0.48408614, -0.24475306],
    [1022.43088666, 0.00256861, -0.24670074, -0.47988360, -0.23318302],
    [1020.78661195, 0.00384297, -0.25140424, -0.47479007, -0.22338108],
    [1018.94386261, 0.00516678, -0.25607299, -0.46638211, -0.21026107],
    [1017.61039103, 0.00563250, -0.25770410, -0.45482876, -0.19708188],
    [1016.56513587, 0.00579834, -0.25750026, -0.44598846, -0.18849875],
    [1014.94106138, 0.00661454, -0.25355923, -0.43889053, -0.18507679],
    [1013.18266190, 0.00764134, -0.24802907, -0.43303550, -0.18488951],
    [1011.11954158, 0.00890055, -0.24345248, -0.42434574, -0.18052015],
    [1009.47435587, 0.00953451, -0.23880529, -0.41216439, -0.17323989],
    [1008.78391696, 0.00924911, -0.23472763, -0.39824354, -0.16388785],
    [1009.28070956, 0.00779342, -0.23044502, -0.38586971, -0.15638540],
    [1008.36732102, 0.00801385, -0.22553302, -0.37584795, -0.15015629],
    [1007.24155945, 0.00865660, -0.21984638, -0.36813814, -0.14789540],
    [1005.41738324, 0.00999198, -0.21353241, -0.36228363, -0.14726690],
    [1003.55385746, 0.01140877, -0.20662796, -0.35769043, -0.15262705],
    [1001.77095583, 0.01272495, -0.19723051, -0.35648195, -0.15970671],
], dtype="float32")


def apply_v37g_pressure_calibration(
    structure_states: np.ndarray,
    current_wind: float,
    current_pressure: float,
    calibrated_wind: np.ndarray | None = None,
) -> np.ndarray:
    """Apply validation-fitted pressure consistency coefficients per lead."""

    if INTENSITY_CALIBRATION_PATH.exists():
        payload = json.loads(INTENSITY_CALIBRATION_PATH.read_text(encoding="utf-8"))
        joint_calibrations = payload.get("pressure_joint_calibrations")
        if joint_calibrations:
            previous_wind = float(OBSERVED[-2]["vmax_kt"])
            previous_pressure = float(OBSERVED[-2]["pressure_hpa"])
            predicted_wind = (
                np.asarray(calibrated_wind, dtype="float32")
                if calibrated_wind is not None
                else structure_states[:, :, 0]
            )
            predicted_pressure = structure_states[:, :, 1]
            calibrated = np.empty_like(predicted_pressure)
            for lead, calibration in enumerate(joint_calibrations):
                features = np.column_stack([
                    np.full(len(predicted_pressure), 1.0, dtype="float32"),
                    np.full(len(predicted_pressure), current_pressure, dtype="float32"),
                    np.full(len(predicted_pressure), current_wind, dtype="float32"),
                    predicted_wind[:, lead],
                    predicted_pressure[:, lead],
                    predicted_pressure[:, lead] - current_pressure,
                    predicted_wind[:, lead] - current_wind,
                    np.full(len(predicted_pressure), current_pressure - previous_pressure, dtype="float32"),
                    np.full(len(predicted_pressure), current_wind - previous_wind, dtype="float32"),
                ])
                mean = np.asarray(calibration["mean"], dtype="float32")
                scale = np.asarray(calibration["scale"], dtype="float32")
                normalized = (features - mean) / scale
                normalized[:, 0] = 1.0
                calibrated[:, lead] = normalized @ np.asarray(calibration["beta"], dtype="float32")
            return np.clip(calibrated, 850.0, 1025.0)
        previous_wind = float(OBSERVED[-2]["vmax_kt"])
        previous_pressure = float(OBSERVED[-2]["pressure_hpa"])
        predicted_wind = structure_states[:, :, 0]
        calibrated = np.empty_like(structure_states[:, :, 1])
        for lead, calibration in enumerate(payload["calibrations"]):
            features = np.column_stack([
                np.full(len(predicted_wind), 1.0, dtype="float32"),
                np.full(len(predicted_wind), current_pressure, dtype="float32"),
                np.full(len(predicted_wind), current_wind, dtype="float32"),
                predicted_wind[:, lead],
                predicted_wind[:, lead] - current_wind,
                np.full(len(predicted_wind), current_pressure - previous_pressure, dtype="float32"),
                np.full(len(predicted_wind), current_wind - previous_wind, dtype="float32"),
            ])
            mean = np.asarray(calibration["mean"], dtype="float32")
            scale = np.asarray(calibration["scale"], dtype="float32")
            normalized = (features - mean) / scale
            normalized[:, 0] = 1.0
            calibrated[:, lead] = normalized @ np.asarray(calibration["beta"], dtype="float32")
        return np.clip(calibrated, 850.0, 1025.0)

    # Preserve the earlier v37G artifact as a fallback if the calibration
    # manifest has not yet been generated in a fresh checkout.
    predicted_wind = structure_states[:, :, 0]
    design = np.stack([
        np.ones_like(predicted_wind),
        np.full_like(predicted_wind, current_pressure),
        np.full_like(predicted_wind, current_wind),
        predicted_wind,
        predicted_wind - current_wind,
    ], axis=-1)
    return np.clip(np.einsum("nlf,lf->nl", design, V37G_PRESSURE_CALIBRATION), 850.0, 1025.0)


def apply_v37g_wind_calibration(
    structure_states: np.ndarray,
    current_wind: float,
    current_pressure: float,
) -> np.ndarray:
    """Apply the validation-fitted v37G wind anchor when it is available."""

    if not INTENSITY_CALIBRATION_PATH.exists():
        return structure_states[:, :, 0]
    payload = json.loads(INTENSITY_CALIBRATION_PATH.read_text(encoding="utf-8"))
    alphas = payload.get("wind_blend_alpha")
    if not alphas:
        return structure_states[:, :, 0]
    predicted_wind = structure_states[:, :, 0]
    calibrated = np.empty_like(predicted_wind)
    for lead, alpha in enumerate(alphas):
        calibrated[:, lead] = float(alpha) * predicted_wind[:, lead] + (1.0 - float(alpha)) * current_wind
    return np.clip(calibrated, 0.0, 190.0)


def load_current_observed() -> list[dict]:
    """Build a synchronized 6-hour input history ending at the current JMA analysis."""

    forecast_path = ROOT / "data" / "jma" / "TC2615_forecast.json"
    parts = json.loads(forecast_path.read_text(encoding="utf-8"))
    analysis = next(
        item for item in parts
        if isinstance(item.get("part"), dict) and item["part"].get("en") == "Analysis"
    )
    track = analysis["track"]["typhoon"]
    valid_time = dt.datetime.fromisoformat(analysis["validtime"]["UTC"].replace("Z", "+00:00"))
    # The current JMA analysis path is 3-hourly.  Use the last nine points at
    # six-hour spacing so the input lines up with the cached GFS cycle keys.
    start = max(0, len(track) - 18)
    selected = list(range(start, len(track), 2))
    if len(selected) != 9:
        raise RuntimeError(f"Expected nine 6-hour JMA analysis points, found {len(selected)}")

    historic = {point["time"]: point for point in HISTORIC_OBSERVED}
    fallback_intensity = {
        "2026-07-30T12:00Z": (139.0, 915.0),
        "2026-07-30T18:00Z": (135.0, 920.0),
        "2026-07-31T00:00Z": (130.0, 925.0),
        "2026-07-31T06:00Z": (130.0, 927.0),
        "2026-07-31T12:00Z": (130.0, 927.0),
    }
    records = []
    for index in selected:
        point_time = valid_time - dt.timedelta(hours=3 * (len(track) - 1 - index))
        time_text = point_time.isoformat().replace("+00:00", "Z")
        if time_text in historic:
            intensity = historic[time_text]
            vmax = float(intensity["vmax_kt"])
            pressure = intensity["pressure_hpa"]
            pressure = float(pressure) if pressure is not None else np.nan
        else:
            vmax, pressure = fallback_intensity.get(time_text, (130.0, 927.0))
        records.append({
            "time": time_text,
            "lat": float(track[index][0]),
            "lon": float(track[index][1]),
            "vmax_kt": vmax,
            "pressure_hpa": pressure,
        })
    return records


OBSERVED = load_current_observed()


def make_records() -> list[dict]:
    records = []
    land_lat, land_lon = load_land_points()
    for point in OBSERVED:
        lat = float(point["lat"])
        lon = float(point["lon"])
        row = {
            "time_utc": point["time"],
            "lat": lat,
            "lon": lon,
            "vmax_kt": float(point["vmax_kt"]),
            "pressure_hpa": float(point["pressure_hpa"]) if point["pressure_hpa"] is not None else np.nan,
            "dist2land_km": distance_to_land(lat, lon % 360.0, land_lat, land_lon),
            "rmw_nm": np.nan,
            "roci_nm": np.nan,
        }
        for name in (
            "r34_ne_nm", "r34_se_nm", "r34_sw_nm", "r34_nw_nm",
            "r50_ne_nm", "r50_se_nm", "r50_sw_nm", "r50_nw_nm",
            "r64_ne_nm", "r64_se_nm", "r64_sw_nm", "r64_nw_nm",
        ):
            row[name] = np.nan
        records.append(row)
    return records


def field_key(time_text: str) -> str:
    return time_text.replace("-", "").replace("T", "_")[:11]


def make_steering_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The full public-GFS field is opt-in.  v37E's flow-only expert uses the
    # legacy two-channel cache so its training and inference distributions
    # remain identical; the full-field experiment can be enabled explicitly.
    use_full_field = os.environ.get(
        "V37_USE_FULL_STRUCTURE_FIELD", "1" if MODEL_VERSION in {"v37G", "v37I"} else "0"
    ) == "1"
    if use_full_field and STRUCTURE_FIELD_PATH.exists():
        archive = np.load(STRUCTURE_FIELD_PATH, allow_pickle=True)
        current = np.asarray(archive["normalized"], dtype="float32")
        if current.shape != (4, 17, 17):
            raise RuntimeError(
                f"unexpected v37E structure field shape: {current.shape}; expected (4, 17, 17)"
            )
        history = np.repeat(current[None], 2, axis=0).reshape(8, 17, 17)
        return current, history, np.zeros((2,), dtype="float32")

    real = np.load(REAL_FIELD_PATH)
    scales = np.load(FIELD_SCALE_PATH)["scale"][2:4].astype("float32")
    keys = [field_key(point["time"]) for point in OBSERVED]
    current_index = len(OBSERVED) - 1

    def patch(index: int) -> np.ndarray:
        value = np.zeros((4, 17, 17), dtype="float32")
        key = keys[index]
        if key in real:
            raw = np.asarray(real[key], dtype="float32")
            value[2:4] = np.clip(raw / scales[:, None, None], -4.0, 4.0)
        return value

    current = patch(current_index)
    history = np.zeros((8, 17, 17), dtype="float32")
    available = np.zeros((2,), dtype="float32")
    for slot, back in enumerate((2, 4)):
        index = current_index - back
        if index >= 0 and keys[index] in real:
            history[slot * 4:(slot + 1) * 4] = patch(index)
            available[slot] = 1.0
        else:
            history[slot * 4:(slot + 1) * 4] = current
    return current, history, available


def load_gfs_guided_steps() -> tuple[np.ndarray | None, dict]:
    """Integrate public GFS deep-layer flow into a deterministic route guide.

    The v23 route network is retained for member deviations, but its learned
    mean is not allowed to override the future GFS steering direction.  The
    coefficients are the documented motion-vs-steering regression used when
    the deep-layer target was extracted.
    """

    if not GFS_FORECAST_PATH.exists():
        return None, {"available": False, "reason": "forecast cache missing"}
    cache = np.load(GFS_FORECAST_PATH, allow_pickle=True)
    grid_lat = np.asarray(cache["latitude"], dtype="float32")
    grid_lon = np.asarray(cache["longitude"], dtype="float32")
    fields_u = np.asarray(cache["u"], dtype="float32")
    fields_v = np.asarray(cache["v"], dtype="float32")
    leads = np.asarray(cache["leads"], dtype="int16")
    if fields_u.shape[0] != 20 or fields_v.shape != fields_u.shape:
        raise RuntimeError(f"unexpected GFS forecast cache shape: {fields_u.shape} / {fields_v.shape}")

    lat_step = float(np.median(np.diff(grid_lat)))
    lon_step = float(np.median(np.diff(grid_lon)))
    lat_offsets = np.arange(-8.0, 8.0001, 0.25, dtype="float32")
    lon_offsets = np.arange(-8.0, 8.0001, 0.25, dtype="float32")
    latitude = float(OBSERVED[-1]["lat"])
    longitude = float(OBSERVED[-1]["lon"])
    steps = np.zeros((20, 2), dtype="float32")
    flow_rows = []
    positions = []

    for index in range(20):
        lat_mesh = latitude + lat_offsets[:, None]
        lon_mesh = longitude + lon_offsets[None, :]
        distance = np.hypot(
            lat_mesh - latitude,
            (lon_mesh - longitude) * math.cos(math.radians(latitude)),
        )
        annulus = (distance >= 3.0) & (distance <= 8.0)
        rows = np.clip(np.rint((lat_mesh[:, 0] - grid_lat[0]) / lat_step).astype(int), 0, len(grid_lat) - 1)
        cols = np.mod(np.rint((lon_mesh[0] % 360.0 - grid_lon[0]) / lon_step).astype(int), len(grid_lon))
        u_mean = float(fields_u[index][rows[:, None], cols[None, :]][annulus].mean())
        v_mean = float(fields_v[index][rows[:, None], cols[None, :]][annulus].mean())
        # 0.76/0.78 are the observed-motion slopes; the small beta-drift
        # intercept keeps the guide from becoming a raw-wind extrapolator.
        east_km = (0.76 * u_mean - 2.03) * 21.6
        north_km = (0.78 * v_mean + 0.40) * 21.6
        steps[index] = [east_km, north_km]
        latitude += north_km / 111.2
        longitude += east_km / (111.2 * max(math.cos(math.radians(latitude)), 0.15))
        longitude %= 360.0
        flow_rows.append({"lead_hours": int(leads[index]), "u_mean_mps": round(u_mean, 4), "v_mean_mps": round(v_mean, 4)})
        positions.append({"lead_hours": int(leads[index]), "latitude": round(latitude, 4), "longitude": round(longitude, 4)})

    metadata = {
        "available": True,
        "cache": str(GFS_FORECAST_PATH),
        "cycle": "2026-07-31T12:00:00Z",
        "levels_hpa": [850, 500, 200],
        "weights": [0.269, 0.500, 0.231],
        "motion_slopes": [0.76, 0.78],
        "beta_intercept_mps": [-2.03, 0.40],
        "flow_annulus_degrees": [3.0, 8.0],
        "flow": flow_rows,
        "guide_positions": positions,
    }
    return steps, metadata


def _relative_vorticity(u: np.ndarray, v: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    """Return 1e-5 s^-1 relative vorticity on a regular lat/lon grid."""

    radius_m = 6_371_000.0
    latitude_m = np.deg2rad(grid_lat.astype("float64")) * radius_m
    dy = np.gradient(latitude_m)
    longitude_step = float(np.median(np.diff(grid_lon)))
    dx = np.deg2rad(longitude_step) * radius_m * np.cos(np.deg2rad(grid_lat))
    dvdx = np.gradient(v.astype("float64"), axis=-1) / dx[:, None]
    dudy = np.gradient(u.astype("float64"), axis=-2) / dy[:, None]
    return ((dvdx - dudy) * 1.0e5).astype("float32")


def _advance_vortex(
    latitude: float,
    longitude: float,
    u850: np.ndarray,
    v850: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    radius_degrees: float = 4.0,
    smooth_sigma: float = 1.0,
) -> tuple[float, float, float]:
    """Find the next low-level cyclonic center near the previous center."""

    from scipy.ndimage import gaussian_filter

    vorticity = _relative_vorticity(u850, v850, grid_lat, grid_lon)
    if smooth_sigma > 0:
        vorticity = gaussian_filter(vorticity, smooth_sigma)
    row_mask = np.abs(grid_lat - latitude) <= radius_degrees
    lon_distance = ((grid_lon - longitude + 180.0) % 360.0) - 180.0
    column_mask = np.abs(lon_distance) <= radius_degrees
    rows = np.flatnonzero(row_mask)
    columns = np.flatnonzero(column_mask)
    if not len(rows) or not len(columns):
        return latitude, longitude, float("nan")
    local = vorticity[np.ix_(rows, columns)]
    if not np.isfinite(local).any():
        return latitude, longitude, float("nan")
    peak = np.unravel_index(np.nanargmax(local), local.shape)
    next_latitude = float(grid_lat[rows[peak[0]]])
    next_longitude = float(grid_lon[columns[peak[1]]]) % 360.0
    if distance_km(latitude, longitude, next_latitude, next_longitude) > 500.0:
        return latitude, longitude, float(local[peak])
    return next_latitude, next_longitude, float(local[peak])


def _track_vortex_frames(
    u_frames: np.ndarray,
    v_frames: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitudes = np.zeros(len(u_frames), dtype="float64")
    longitudes = np.zeros(len(u_frames), dtype="float64")
    scores = np.zeros(len(u_frames), dtype="float64")
    latitude = float(OBSERVED[-1]["lat"])
    longitude = float(OBSERVED[-1]["lon"])
    for index, (u850, v850) in enumerate(zip(u_frames, v_frames)):
        latitude, longitude, score = _advance_vortex(
            latitude, longitude, u850, v850, grid_lat, grid_lon
        )
        latitudes[index] = latitude
        longitudes[index] = longitude
        scores[index] = score
    return latitudes, longitudes, scores


def load_gfs_vortex_ensemble() -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a deterministic-GFS plus GEFS member vortex-track ensemble."""

    import xarray as xr

    deterministic_path = sorted(GFS_RAW_ROOT.glob("gfs_*.grib2"))
    deterministic_lat = deterministic_lon = None
    deterministic_u = []
    deterministic_v = []
    for path in deterministic_path:
        with xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
                "indexpath": "",
            },
        ) as dataset:
            if deterministic_lat is None:
                deterministic_lat = np.asarray(dataset["latitude"].values, dtype="float32")
                deterministic_lon = np.asarray(dataset["longitude"].values, dtype="float32")
            deterministic_u.append(np.asarray(dataset["u"].isel(isobaricInhPa=0).values, dtype="float32"))
            deterministic_v.append(np.asarray(dataset["v"].isel(isobaricInhPa=0).values, dtype="float32"))
    if len(deterministic_u) == 20:
        det_lat, det_lon, det_score = _track_vortex_frames(
            np.stack(deterministic_u), np.stack(deterministic_v), deterministic_lat, deterministic_lon
        )
    else:
        det_lat = det_lon = det_score = None

    ge = np.load(GEFS_FORECAST_PATH, allow_pickle=True)
    if "u_levels" not in ge or "v_levels" not in ge:
        raise RuntimeError("GEFS cache lacks per-level winds; rebuild the ensemble cache")
    ge_u = np.asarray(ge["u_levels"][:, :, 0], dtype="float32")
    ge_v = np.asarray(ge["v_levels"][:, :, 0], dtype="float32")
    ge_lat = np.asarray(ge["latitude"], dtype="float32")
    ge_lon = np.asarray(ge["longitude"], dtype="float32")
    member_latitudes = []
    member_longitudes = []
    member_scores = []
    for index in range(ge_u.shape[0]):
        latitude, longitude, score = _track_vortex_frames(
            ge_u[index], ge_v[index], ge_lat, ge_lon
        )
        member_latitudes.append(latitude)
        member_longitudes.append(longitude)
        member_scores.append(score)
    if det_lat is not None:
        member_latitudes.insert(0, det_lat)
        member_longitudes.insert(0, det_lon)
        member_scores.insert(0, det_score)
    if not member_latitudes:
        raise RuntimeError("No public GFS/GEFS vortex tracks were available")
    metadata = {
        "available": True,
        "policy": "50% deterministic GFS vortex track plus 50% 31-member GEFS vortex-track mean",
        "deterministic_member": bool(det_lat is not None),
        "member_count": len(member_latitudes),
        "gefs_cache": str(GEFS_FORECAST_PATH),
        "gfs_raw_root": str(GFS_RAW_ROOT),
        "cycle": "2026-07-31T12:00:00Z",
        "level_hpa": 850,
        "search_radius_degrees": 4.0,
        "smoothing_sigma_gridpoints": 1.0,
        "control_weight": 0.5,
        "gefs_member_mean_weight": 0.5,
        "member_peak_vorticity_1e5_s-1": np.asarray(member_scores).mean(axis=0).round(4).tolist(),
    }
    return np.stack(member_latitudes), np.stack(member_longitudes), metadata


def load_gfs_vortex_multilevel_ensemble() -> tuple[np.ndarray, np.ndarray, dict]:
    """Use a lower-level control route and a multi-level GEFS route ensemble.

    The 850-hPa vortex is the center anchor.  A small 500/200-hPa contribution
    lets the ensemble account for the evolving deep-layer circulation without
    allowing an upper trough to replace the storm center in the deterministic
    control member.
    """

    base_lats, base_lons, base_metadata = load_gfs_vortex_ensemble()
    ge = np.load(GEFS_FORECAST_PATH, allow_pickle=True)
    if "u_levels" not in ge or "v_levels" not in ge:
        raise RuntimeError("GEFS cache lacks per-level winds; rebuild the ensemble cache")
    level_weights = np.asarray([0.70, 0.25, 0.05], dtype="float64")
    ge_u = np.asarray(ge["u_levels"], dtype="float32")
    ge_v = np.asarray(ge["v_levels"], dtype="float32")
    ge_lat = np.asarray(ge["latitude"], dtype="float32")
    ge_lon = np.asarray(ge["longitude"], dtype="float32")
    member_latitudes = [base_lats[0]]
    member_longitudes = [base_lons[0]]
    for member in range(ge_u.shape[0]):
        level_latitudes = []
        level_longitudes = []
        for level in range(ge_u.shape[2]):
            latitude, longitude, _ = _track_vortex_frames(
                ge_u[member, :, level],
                ge_v[member, :, level],
                ge_lat,
                ge_lon,
            )
            level_latitudes.append(latitude)
            level_longitudes.append(longitude)
        member_latitudes.append(np.average(np.stack(level_latitudes), axis=0, weights=level_weights))
        member_longitudes.append(np.average(np.stack(level_longitudes), axis=0, weights=level_weights))

    metadata = dict(base_metadata)
    metadata.update({
        "policy": "50% deterministic GFS 850-hPa vortex track plus 50% GEFS multi-level vortex-track mean",
        "level_hpa": [850, 500, 200],
        "level_weights_gefs": level_weights.tolist(),
        "control_level_hpa": 850,
        "control_weight": 0.5,
        "gefs_member_mean_weight": 0.5,
    })
    return np.stack(member_latitudes), np.stack(member_longitudes), metadata


def load_gefs_mean_multilevel_route() -> tuple[np.ndarray, np.ndarray, dict]:
    """Track the GEFS ensemble-mean fields at 850/500/200 hPa.

    The historical no-key hindcast showed that averaging the public GEFS
    fields before locating the vortex is more stable than giving the
    deterministic GFS control member half of the route weight.  The three
    level tracks are retained as weighted route members so their vertical
    disagreement remains visible in the route spread.
    """

    ge = np.load(GEFS_FORECAST_PATH, allow_pickle=True)
    if "u_levels" not in ge or "v_levels" not in ge:
        raise RuntimeError("GEFS cache lacks per-level winds; rebuild the ensemble cache")
    ge_u = np.asarray(ge["u_levels"], dtype="float32")
    ge_v = np.asarray(ge["v_levels"], dtype="float32")
    ge_lat = np.asarray(ge["latitude"], dtype="float32")
    ge_lon = np.asarray(ge["longitude"], dtype="float32")
    if ge_u.ndim != 5 or ge_v.shape != ge_u.shape or ge_u.shape[2] != 3:
        raise RuntimeError(f"unexpected GEFS per-level cache shape: {ge_u.shape} / {ge_v.shape}")
    level_weights = np.asarray([0.70, 0.25, 0.05], dtype="float64")
    level_latitudes = []
    level_longitudes = []
    level_scores = []
    for level in range(ge_u.shape[2]):
        latitude, longitude, score = _track_vortex_frames(
            ge_u[:, :, level].mean(axis=0),
            ge_v[:, :, level].mean(axis=0),
            ge_lat,
            ge_lon,
        )
        level_latitudes.append(latitude)
        level_longitudes.append(longitude)
        level_scores.append(score)
    metadata = {
        "available": True,
        "policy": "GEFS ensemble-mean fields tracked independently at 850/500/200 hPa",
        "gefs_cache": str(GEFS_FORECAST_PATH),
        "cycle": "2026-07-31T12:00:00Z",
        "level_hpa": [850, 500, 200],
        "level_weights": level_weights.tolist(),
        "member_count": 3,
        "ensemble_members_averaged_before_tracking": int(ge_u.shape[0]),
        "level_peak_vorticity_1e5_s-1": np.stack(level_scores).mean(axis=0).round(4).tolist(),
    }
    return np.stack(level_latitudes), np.stack(level_longitudes), metadata


def load_adaptive_gfs_gefs_route() -> tuple[np.ndarray, np.ndarray, dict]:
    """Select between the existing GFS/GEFS route and the GEFS-mean route.

    Historical hindcasts show that the deterministic GFS control can lock on
    to a different circulation.  Agreement with the independent GEFS field
    mean is therefore used as a forecast-time confidence signal; no future
    observations or official forecast positions enter this gate.
    """

    gfs_lats, gfs_lons, gfs_metadata = load_gfs_vortex_multilevel_ensemble()
    gefs_lats, gefs_lons, gefs_metadata = load_gefs_mean_multilevel_route()
    gfs_weights = np.full(gfs_lats.shape[0], 0.5 / max(gfs_lats.shape[0] - 1, 1), dtype="float64")
    gfs_weights[0] = 0.5
    gefs_weights = np.asarray([0.70, 0.25, 0.05], dtype="float64")
    gfs_mean_lat = np.average(gfs_lats, axis=0, weights=gfs_weights)
    gfs_mean_lon = np.average(gfs_lons, axis=0, weights=gfs_weights)
    gefs_mean_lat = np.average(gefs_lats, axis=0, weights=gefs_weights)
    gefs_mean_lon = np.average(gefs_lons, axis=0, weights=gefs_weights)
    threshold_km = float(os.environ.get("V37_ROUTE_AGREEMENT_KM", "300"))
    disagreement_km = distance_km(
        float(gfs_mean_lat[-1]),
        float(gfs_mean_lon[-1]),
        float(gefs_mean_lat[-1]),
        float(gefs_mean_lon[-1]),
    )
    use_gfs = disagreement_km <= threshold_km
    chosen_lats, chosen_lons = (gfs_lats, gfs_lons) if use_gfs else (gefs_lats, gefs_lons)
    chosen_weights = gfs_weights if use_gfs else gefs_weights
    metadata = {
        "available": True,
        "policy": "adaptive GFS/GEFS route: retain GFS/GEFS member route when it agrees with GEFS field mean, otherwise use GEFS field mean",
        "chosen_route": "gfs_vortex_multilevel_ensemble" if use_gfs else "gefs_mean_multilevel",
        "agreement_threshold_km": threshold_km,
        "final_route_disagreement_km": round(disagreement_km, 3),
        "gfs_route": gfs_metadata,
        "gefs_mean_route": gefs_metadata,
        "route_weights": chosen_weights.tolist(),
        "member_count": int(chosen_lats.shape[0]),
    }
    return chosen_lats, chosen_lons, metadata


def load_route_models() -> list[torch.nn.Module]:
    models = []
    for checkpoint_path in sorted((ROOT / "models" / "v23").glob("v23_seed*.pt")):
        model = build_v23().to(DEVICE).eval()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        models.append(model)
    if not models:
        raise FileNotFoundError("No local v23 checkpoints found")
    return models


def load_structure_models() -> list[torch.nn.Module]:
    paths = sorted((ROOT / "v37" / "protected" / "checkpoints").glob("structure_v37_seed*.pt"))
    if not paths:
        raise FileNotFoundError("No v37C structure checkpoints found")
    models = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload["config"]
        model = StructureV37(config["width"], config["layers"], config["heads"]).to(DEVICE).eval()
        model.load_state_dict(payload["model"])
        models.append(model)
    return models


def load_structure_fusion_models() -> list[torch.nn.Module]:
    from v37_structure_fusion_train import StructureFusionV37D

    if MODEL_VERSION == "v37E":
        checkpoint_root = ROOT / "v37" / "structure_flowonly" / "checkpoints"
        pattern = "structure_fusion_v37e_seed*.pt"
    else:
        checkpoint_root = ROOT / "v37" / "structure_fusion" / "checkpoints"
        pattern = "structure_fusion_v37d_seed*.pt"
    paths = sorted(checkpoint_root.glob(pattern))
    models = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload["config"]
        model = StructureFusionV37D(config["width"], config["layers"], config["heads"]).to(DEVICE).eval()
        model.load_state_dict(payload["model"])
        models.append(model)
    return models


def load_structure_spatial_models() -> list[torch.nn.Module]:
    from v37_structure_spatial_train import StructureSpatialV37G

    checkpoint_root = ROOT / "v37" / "structure_spatial" / "checkpoints"
    tag = MODEL_VERSION.lower()
    paths = sorted(checkpoint_root.glob(f"structure_spatial_{tag}_seed*.pt"))
    # v37I is trained as a separate candidate. Keep a deliberate fallback to
    # the validated three-member v37G ensemble while the candidate is
    # incomplete. A single interrupted seed must not silently become the
    # production intensity ensemble.
    if len(paths) < 3 and MODEL_VERSION == "v37I":
        paths = sorted(checkpoint_root.glob("structure_spatial_v37g_seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"No {MODEL_VERSION} spatial structure checkpoints found")
    models = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload["config"]
        model = StructureSpatialV37G(config["width"], config["layers"], config["heads"]).to(DEVICE).eval()
        model.load_state_dict(payload["model"])
        models.append(model)
    return models


def integrate(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    member_lats = np.zeros((len(states), 20), dtype="float64")
    member_lons = np.zeros((len(states), 20), dtype="float64")
    for member, sequence in enumerate(states):
        latitude = float(OBSERVED[-1]["lat"])
        longitude = float(OBSERVED[-1]["lon"])
        for lead, state in enumerate(sequence):
            east, north = float(state[0]), float(state[1])
            latitude += north / 111.2
            longitude += east / (111.2 * max(math.cos(math.radians(latitude)), 0.15))
            member_lats[member, lead] = latitude
            member_lons[member, lead] = longitude % 360.0
    return member_lats, member_lons


def track_spread(
    member_lats: np.ndarray,
    member_lons: np.ndarray,
    lead: int,
    weights: np.ndarray | None = None,
) -> float:
    if weights is None:
        weights = np.full(member_lats.shape[0], 1.0 / member_lats.shape[0], dtype="float64")
    lat = float(np.average(member_lats[:, lead], weights=weights))
    lon = float(np.average(member_lons[:, lead], weights=weights))
    distances = np.hypot(
        member_lats[:, lead] - lat,
        ((member_lons[:, lead] - lon + 180.0) % 360.0 - 180.0) * math.cos(math.radians(lat)),
    )
    return float(np.average(distances, weights=weights) * 111.2)


def distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    dlat = math.radians(lat_b - lat_a)
    dlon = math.radians(((lon_b - lon_a + 180.0) % 360.0) - 180.0)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(math.radians(lat_a)) * math.cos(math.radians(lat_b)) * math.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(a)))


def official_lead(point: dict) -> int:
    if isinstance(point.get("lead_hours"), int):
        return int(point["lead_hours"])
    marker = point["time"]
    start = marker.rfind("(+")
    if start < 0:
        return 0
    return int(marker[start + 2:marker.index("h)", start)])


def comparison_errors(forecast: list[dict], official: dict) -> dict:
    issue_time = dt.datetime.fromisoformat(OBSERVED[-1]["time"].replace("Z", "+00:00"))
    model_times = [
        dt.datetime.fromisoformat(item["valid_time_utc"].replace("Z", "+00:00"))
        for item in forecast
    ]

    def interpolate(target_time: dt.datetime) -> tuple[dict, float] | None:
        if target_time < model_times[0] or target_time > model_times[-1]:
            return None
        for index, model_time in enumerate(model_times):
            if target_time == model_time:
                point = dict(forecast[index])
                return point, float(point["lead_hours"])
            if target_time < model_time:
                before = forecast[index - 1]
                after = forecast[index]
                before_time = model_times[index - 1]
                fraction = (target_time - before_time).total_seconds() / (
                    model_time - before_time
                ).total_seconds()
                lon_delta = ((after["longitude"] - before["longitude"] + 180.0) % 360.0) - 180.0
                point = dict(before)
                point["latitude"] = before["latitude"] + fraction * (after["latitude"] - before["latitude"])
                point["longitude"] = (before["longitude"] + fraction * lon_delta) % 360.0
                for key in ("vmax_kt", "central_pressure_hpa"):
                    point[key] = before[key] + fraction * (after[key] - before[key])
                point["lead_hours"] = before["lead_hours"] + fraction * (
                    after["lead_hours"] - before["lead_hours"]
                )
                return point, float(point["lead_hours"])
        return None

    result = {}
    for name in ("jtwc", "jma"):
        rows = []
        for point in official[name]:
            lead = official_lead(point)
            target_time = dt.datetime.fromisoformat(point["time"].replace("Z", "+00:00"))
            interpolated = interpolate(target_time)
            if interpolated is None:
                continue
            predicted, forecast_lead = interpolated
            rows.append({
                "lead_hours": lead,
                "forecast_lead_hours": round(forecast_lead, 2),
                "official_valid_time_utc": point["time"],
                "track_error_km": round(distance_km(predicted["latitude"], predicted["longitude"], point["lat"], point["lon"]), 2),
                "predicted_vmax_kt": predicted["vmax_kt"],
                "official_vmax_kt": point["vmax_kt"],
                "predicted_lat": predicted["latitude"],
                "predicted_lon": predicted["longitude"],
                "official_lat": point["lat"],
                "official_lon": point["lon"],
            })
        result[name] = rows
    return result


def run_forecast() -> dict:
    records = make_records()
    stats = np.load(NORM_PATH)
    normalized, velocity_pair, _ = build_track_window(
        records,
        stats["tmean"].astype("float32"),
        stats["tstd"].astype("float32"),
        *load_land_points(),
    )
    current, history, available = make_steering_inputs()
    structure_models = [] if MODEL_VERSION in {"v37G", "v37I"} else load_structure_models()
    fusion_models = [] if MODEL_VERSION in {"v37G", "v37I"} else load_structure_fusion_models()
    spatial_models = load_structure_spatial_models() if MODEL_VERSION in {"v37G", "v37I"} else []
    route_args = [
        torch.from_numpy(normalized[None]).to(DEVICE),
        torch.from_numpy(velocity_pair[None]).to(DEVICE),
        torch.from_numpy(current[None]).to(DEVICE),
        torch.from_numpy(history[None]).to(DEVICE),
        torch.from_numpy(available[None]).to(DEVICE),
    ]
    track_tensor = torch.from_numpy(normalized[None]).to(DEVICE)
    with torch.no_grad():
        if spatial_models:
            fusion_field = torch.from_numpy(current[None]).to(DEVICE)
            current_values = np.asarray(
                [OBSERVED[-1]["vmax_kt"], OBSERVED[-1]["pressure_hpa"]], dtype="float32"
            )
            current_available = np.isfinite(current_values).astype("float32")
            current_values = np.nan_to_num(current_values / TARGET_SCALE[2:4], nan=0.0)
            current_tensor = torch.from_numpy(current_values[None]).to(DEVICE)
            available_tensor = torch.from_numpy(current_available[None]).to(DEVICE)
            structure_states = torch.stack(
                [model(track_tensor, fusion_field, current_tensor, available_tensor)[0][0] for model in spatial_models]
            ).cpu().numpy()
            structure_states = structure_states * TARGET_SCALE[2:][None, None, :]
            if np.all(current_available > 0.0):
                current_wind = float(current_values[0] * TARGET_SCALE[2])
                current_pressure = float(current_values[1] * TARGET_SCALE[3])
                calibrated_wind = apply_v37g_wind_calibration(
                    structure_states,
                    current_wind,
                    current_pressure,
                )
                structure_states[:, :, 1] = apply_v37g_pressure_calibration(
                    structure_states,
                    current_wind,
                    current_pressure,
                    calibrated_wind=calibrated_wind,
                )
                # Apply the independent wind anchor first, then use that
                # forecast-time wind together with raw pressure in the joint
                # pressure calibration.
                structure_states[:, :, 0] = calibrated_wind
        else:
            base_states = torch.stack([model(track_tensor)[0][0] for model in structure_models]).cpu().numpy()
            base_states = base_states * TARGET_SCALE[2:][None, None, :]
        if not spatial_models and fusion_models:
            fusion_field = torch.from_numpy(current[None]).to(DEVICE)
            fusion_states = torch.stack(
                [model(track_tensor, fusion_field)[0][0] for model in fusion_models]
            ).cpu().numpy()
            fusion_states = fusion_states * TARGET_SCALE[2:][None, None, :]
            # Validation-selected, variable-specific expert weights: the field
            # branch contributes most to wind, moderately to pressure/radii,
            # and little to RMW where it did not generalize consistently.
            blend = np.asarray(
                [0.90, 0.50, 0.20] + [0.40] * 4 + [0.50] * 8,
                dtype="float32",
            )
            if len(base_states) != len(fusion_states):
                raise RuntimeError("v37D structure experts have different seed counts")
            structure_states = base_states + blend[None, None, :] * (fusion_states - base_states)
        elif not spatial_models:
            structure_states = base_states

    if ROUTE_POLICY in {"gfs_vortex_ensemble", "gfs_vortex_multilevel_ensemble", "gefs_mean_multilevel", "gfs_gefs_adaptive"}:
        route_loader = {
            "gfs_vortex_ensemble": load_gfs_vortex_ensemble,
            "gfs_vortex_multilevel_ensemble": load_gfs_vortex_multilevel_ensemble,
            "gefs_mean_multilevel": load_gefs_mean_multilevel_route,
            "gfs_gefs_adaptive": load_adaptive_gfs_gefs_route,
        }[ROUTE_POLICY]
        member_lats, member_lons, gfs_metadata = route_loader()
        route_member_count = int(member_lats.shape[0])
        if "route_weights" in gfs_metadata:
            route_weights = np.asarray(gfs_metadata["route_weights"], dtype="float64")
        elif ROUTE_POLICY == "gefs_mean_multilevel":
            route_weights = np.asarray([0.70, 0.25, 0.05], dtype="float64")
        else:
            route_weights = np.full(route_member_count, 0.5 / max(route_member_count - 1, 1), dtype="float64")
            route_weights[0] = 0.5
        route_backbone = gfs_metadata["policy"]
    else:
        route_models = load_route_models()
        with torch.no_grad():
            route_states = torch.stack([model(*route_args)[0][0] for model in route_models]).cpu().numpy()
            route_states = route_states * V23_TARGET_SCALE.numpy()[None, None, :]
        gfs_steps, gfs_metadata = load_gfs_guided_steps()
        if gfs_steps is not None:
            learned_mean = route_states[..., :2].mean(axis=0, keepdims=True)
            route_states[..., :2] = gfs_steps[None] + 0.25 * (route_states[..., :2] - learned_mean)
        member_lats, member_lons = integrate(route_states)
        route_member_count = len(route_models)
        route_weights = np.full(route_member_count, 1.0 / route_member_count, dtype="float64")
        route_backbone = "Public GFS deep-layer physical route guide plus frozen TrackFormer v23 residual ensemble"

    mean_lats = np.average(member_lats, axis=0, weights=route_weights)
    mean_lons = np.average(member_lons, axis=0, weights=route_weights)
    structure_mean = structure_states.mean(axis=0)
    structure_spread = structure_states.std(axis=0)
    issue_time = dt.datetime.fromisoformat(OBSERVED[-1]["time"].replace("Z", "+00:00"))
    forecast = []
    for index, lead in enumerate(range(6, 121, 6)):
        state = structure_mean[index]
        valid = issue_time + dt.timedelta(hours=lead)
        forecast.append({
            "lead_hours": lead,
            "valid_time_utc": valid.isoformat().replace("+00:00", "Z"),
            "latitude": round(float(mean_lats[index]), 4),
            "longitude": round(float(mean_lons[index]), 4),
            "track_spread_km": round(track_spread(member_lats, member_lons, index, route_weights), 2),
            "vmax_kt": round(float(np.clip(state[0], 0.0, 190.0)), 2),
            "vmax_spread_kt": round(float(structure_spread[index, 0]), 2),
            "central_pressure_hpa": round(float(np.clip(state[1], 850.0, 1025.0)), 2),
            "pressure_spread_hpa": round(float(structure_spread[index, 1]), 2),
            "rmw_km": round(float(np.clip(state[2], 0.0, 300.0)), 2),
            "wind_radii_km": np.clip(state[3:15], 0.0, 1000.0).round(2).tolist(),
        })

    _, jma_points, _ = parse_current_jma()
    _, jtwc_points = parse_current_jtwc()
    official = {"jma": jma_points, "jtwc": jtwc_points}
    v36 = json.loads(V36_PATH.read_text(encoding="utf-8"))
    use_full_structure_field = (
        os.environ.get("V37_USE_FULL_STRUCTURE_FIELD", "1" if MODEL_VERSION in {"v37G", "v37I"} else "0") == "1"
        and STRUCTURE_FIELD_PATH.exists()
    )
    inference_field_path = STRUCTURE_FIELD_PATH if use_full_structure_field else REAL_FIELD_PATH
    payload = {
        "model": MODEL_VERSION + " GFS/GEFS vortex-route structure ensemble",
        "issue_time_utc": OBSERVED[-1]["time"],
        "members": {
            "route": route_member_count,
            "structure_base": len(structure_models),
            "structure_fusion": len(fusion_models),
            "structure_spatial": len(spatial_models),
        },
        "route_member_weights": route_weights.round(8).tolist(),
        "route_backbone": route_backbone,
        "structure_branch": (
            f"Prediction-level ensemble of {len(spatial_models)} {MODEL_VERSION} spatial field experts with current wind/pressure residual anchor"
            if spatial_models
            else f"Prediction-level blend of three v37C track-only StructureV37 experts and three {MODEL_VERSION} field-conditioned experts"
        ),
        "field_normalization": (
            "Training q/31.75; inference uses public-GFS SLP anomaly/tendency and weighted 850/500/200 u/v divided by the stored dlm4 scales"
            if use_full_structure_field
            else "Training q/31.75; inference uses the legacy public-GFS DLM u/v cache in channels 2/3"
        ),
        "pressure_calibration": (
            "v37G validation-only pressure and wind consistency layers using current pressure/wind, recent tendencies, and predicted wind"
            if MODEL_VERSION in {"v37G", "v37I"}
            else "none"
        ),
        "intensity_calibration_manifest": (
            str(INTENSITY_CALIBRATION_PATH)
            if MODEL_VERSION in {"v37G", "v37I"} and INTENSITY_CALIBRATION_PATH.exists()
            else None
        ),
        "data_policy": "Local IBTrACS/JMA-derived track history plus cached public NOAA GFS fields; no CDS/API key",
        "field_cache": str(inference_field_path),
        "gfs_route_guide": gfs_metadata,
        "forecast": forecast,
        "route_members": [
            {
                "member_index": int(index),
                "latitude": member_lats[index].round(4).tolist(),
                "longitude": member_lons[index].round(4).tolist(),
            }
            for index in range(member_lats.shape[0])
        ],
        "observed": OBSERVED,
        "v36_reference": v36,
        "official": official,
        "comparison_errors": comparison_errors(forecast, official),
        "sources": {
            "ibtracs": "https://www.ncei.noaa.gov/products/international-best-track-archive",
            "gfs_nomads": "https://nomads.ncep.noaa.gov/info.php?page=opendap_grib_migration",
            "jma": "https://www.jma.go.jp/bosai/map.html#3/30/140/&elem=root&typhoon=all&lang=en&contents=typhoon",
            "jtwc": "https://www.metoc.navy.mil/jtwc/products/wp1226web.txt",
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def create_chart(payload: dict) -> None:
    forecast = payload["forecast"]
    x = [point["lead_hours"] for point in forecast]
    wind = np.asarray([point["vmax_kt"] for point in forecast])
    wind_spread = np.asarray([point["vmax_spread_kt"] for point in forecast])
    pressure = np.asarray([point["central_pressure_hpa"] for point in forecast])
    pressure_spread = np.asarray([point["pressure_spread_hpa"] for point in forecast])
    figure, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(x, wind, color="#b91c1c", linewidth=2.6, label=f"{MODEL_VERSION} mean")
    axes[0].fill_between(x, wind - wind_spread, wind + wind_spread, color="#b91c1c", alpha=0.13, label="structure spread")
    axes[0].set_ylabel("Maximum wind (kt)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(x, pressure, color="#1d4ed8", linewidth=2.6, label=f"{MODEL_VERSION} mean")
    axes[1].fill_between(x, pressure - pressure_spread, pressure + pressure_spread, color="#1d4ed8", alpha=0.13, label="structure spread")
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Central pressure (hPa)")
    axes[1].set_xlabel("Forecast lead (hours)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    figure.suptitle(f"Dolphin {MODEL_VERSION} GFS/GEFS vortex-route intensity forecast")
    figure.savefig(OUTPUT_CHART, dpi=180, facecolor="white")
    plt.close(figure)


def create_png(payload: dict) -> None:
    coastlines = load_coastlines()
    forecast = payload["forecast"]
    v36 = payload["v36_reference"]["forecast"]
    jtwc = payload["official"]["jtwc"]
    jma = payload["official"]["jma"]
    figure, (global_ax, detail_ax) = plt.subplots(1, 2, figsize=(19, 9.5), gridspec_kw={"width_ratios": [1.65, 1.0]}, constrained_layout=True)
    for axis in (global_ax, detail_ax):
        draw_world(axis, coastlines, detail=axis is detail_ax)
        axis.plot([lon360(point["lon"]) for point in OBSERVED], [point["lat"] for point in OBSERVED], color="#111827", linewidth=2.4, zorder=5)
        axis.scatter([lon360(point["lon"]) for point in OBSERVED], [point["lat"] for point in OBSERVED], color="#111827", s=13, zorder=6)
        for member in payload.get("route_members", []):
            axis.plot(
                [lon360(value) for value in member["longitude"]],
                member["latitude"],
                color="#b91c1c",
                linewidth=0.65,
                alpha=0.13,
                zorder=2,
            )
        plot_track(axis, forecast, color="#b91c1c", linestyle="-", linewidth=2.8)
        plot_track(axis, v36, color="#64748b", linestyle="--", linewidth=1.7)
        plot_track(axis, jtwc, color="#c2410c", linestyle="--", linewidth=2.0)
        plot_track(axis, jma, color="#047857", linestyle="-.", linewidth=2.0)
        for point in forecast[::2]:
            radius_lat = max(float(point["track_spread_km"]) / 111.2, 0.2)
            radius_lon = radius_lat / max(0.15, abs(math.cos(math.radians(point["latitude"]))))
            axis.add_patch(Ellipse((lon360(point["longitude"]), point["latitude"]), 2 * radius_lon, 2 * radius_lat, facecolor="#b91c1c", edgecolor="#b91c1c", alpha=0.06, linewidth=0.7, zorder=2))
        axis.scatter([lon360(point["longitude"]) for point in forecast], [point["latitude"] for point in forecast], s=36, c=[intensity_color(point["vmax_kt"]) for point in forecast], edgecolors="white", linewidths=0.75, zorder=7)
        axis.scatter([lon360(point["lon"]) for point in jtwc], [point["lat"] for point in jtwc], s=28, facecolors="white", edgecolors="#c2410c", linewidths=1.4, zorder=7)
        axis.scatter([lon360(point["lon"]) for point in jma], [point["lat"] for point in jma], s=28, facecolors="white", edgecolors="#047857", linewidths=1.4, zorder=7)
    for point in forecast:
        if point["lead_hours"] % 24 == 0:
            annotate_point(detail_ax, lon360(point["longitude"]), point["latitude"], f"+{point['lead_hours']}h\n{point['vmax_kt']:.0f} kt\n{point['central_pressure_hpa']:.0f} hPa", "#b91c1c")
    global_ax.set_title("Global / Pacific-centered overview", loc="left", fontsize=13, fontweight="bold")
    detail_ax.set_title(f"Western Pacific detail: {MODEL_VERSION} intensity markers", loc="left", fontsize=13, fontweight="bold")
    figure.suptitle(f"Dolphin: {MODEL_VERSION} GFS/GEFS vortex-route ensemble versus v36, JTWC, and JMA\nissue {payload['issue_time_utc']}", fontsize=16, fontweight="bold")
    detail_ax.legend(handles=[
        Line2D([0], [0], color="#111827", linewidth=2.4, label="Observed"),
        Line2D([0], [0], color="#b91c1c", linewidth=2.8, label=f"{MODEL_VERSION} route + intensity"),
        Line2D([0], [0], color="#64748b", linewidth=1.7, linestyle="--", label="v36 reference"),
        Line2D([0], [0], color="#c2410c", linewidth=2.0, linestyle="--", label="JTWC official"),
        Line2D([0], [0], color="#047857", linewidth=2.0, linestyle="-.", label="JMA official"),
        Patch(facecolor="#b91c1c", alpha=0.08, edgecolor="#b91c1c", label=f"{MODEL_VERSION} route spread"),
    ], loc="lower left", fontsize=8.1, framealpha=0.96)
    figure.text(0.5, 0.005, f"{MODEL_VERSION} route mean uses the public GFS/GEFS vortex ensemble; point colors are the structure blend. JTWC 1-min, JMA 10-min.", ha="center", fontsize=9, color="#334155")
    figure.savefig(OUTPUT_PNG, dpi=190, facecolor="white")
    plt.close(figure)


def leaflet_point(point: dict, model: str) -> dict:
    if model == "v37":
        return {
            "model": f"{MODEL_VERSION} GFS/GEFS vortex-route ensemble",
            "lead_hours": point["lead_hours"],
            "time": point["valid_time_utc"],
            "lat": point["latitude"],
            "lon": point["longitude"],
            "vmax_kt": point["vmax_kt"],
            "pressure_hpa": point["central_pressure_hpa"],
            "vmax_spread_kt": point["vmax_spread_kt"],
            "pressure_spread_hpa": point["pressure_spread_hpa"],
            "track_spread_km": point["track_spread_km"],
            "wind_convention": "1-min model estimate",
        }
    if model == "v36":
        return {
            "model": "v36A reference",
            "lead_hours": point["lead_hours"],
            "time": point["valid_time_utc"],
            "lat": point["latitude"],
            "lon": point["longitude"],
            "vmax_kt": point["vmax_kt"],
            "pressure_hpa": point["central_pressure_hpa"],
            "track_spread_km": point["track_spread_km"],
            "wind_convention": "1-min model estimate",
        }
    result = dict(point)
    result["model"] = "JTWC official" if model == "jtwc" else "JMA/RSMC Tokyo official"
    result["wind_convention"] = "1-min sustained" if model == "jtwc" else "10-min sustained"
    return result


def create_html(payload: dict) -> None:
    v37_points = [leaflet_point(point, "v37") for point in payload["forecast"]]
    v36_points = [leaflet_point(point, "v36") for point in payload["v36_reference"]["forecast"]]
    jtwc_points = [leaflet_point(point, "jtwc") for point in payload["official"]["jtwc"]]
    jma_points = [leaflet_point(point, "jma") for point in payload["official"]["jma"]]
    observed = [{
        "model": "Observed",
        "time": point["time"],
        "lat": point["lat"],
        "lon": point["lon"],
        "vmax_kt": point["vmax_kt"],
        "pressure_hpa": point["pressure_hpa"],
        "wind_convention": "1-min input estimate",
    } for point in payload["observed"]]
    route_members = [
        {"latitude": member["latitude"], "longitude": member["longitude"]}
        for member in payload.get("route_members", [])
    ]
    member_counts = payload.get("members", {})
    route_count = int(member_counts.get("route", len(route_members)))
    structure_parts = [f"{route_count} route members"]
    for key, label in (
        ("structure_base", "base experts"),
        ("structure_fusion", "fusion experts"),
        ("structure_spatial", "spatial field experts"),
    ):
        count = int(member_counts.get(key, 0))
        if count:
            structure_parts.append(f"{count} {label}")
    structure_counts = " + ".join(structure_parts)
    data = {"v37": v37_points, "v36": v36_points, "jtwc": jtwc_points, "jma": jma_points, "observed": observed, "route_members": route_members}
    data_json = safe_json(data).replace("</", "<\\/")
    rows = []
    for point in v37_points:
        rows.append(
            "<tr>"
            f"<td>+{point['lead_hours']} h</td><td>{point['time']}</td>"
            f"<td>{point['lat']:.2f}</td><td>{point['lon']:.2f}E</td>"
            f"<td><b>{point['vmax_kt']:.0f} kt</b><small> +/- {point['vmax_spread_kt']:.1f}</small></td>"
            f"<td><b>{point['pressure_hpa']:.0f} hPa</b><small> +/- {point['pressure_spread_hpa']:.1f}</small></td>"
            f"<td>{point['track_spread_km']:.0f} km</td></tr>"
        )
    comparison = payload["comparison_errors"]
    comparison_rows = []
    for model_name in ("jtwc", "jma"):
        for row in comparison[model_name]:
            comparison_rows.append(
                f"<tr><td>{html.escape(model_name.upper())}</td><td>+{row['lead_hours']} h</td>"
                f"<td>+{row['forecast_lead_hours']:.0f} h</td><td>{row['track_error_km']:.0f} km</td>"
                f"<td>{row['predicted_vmax_kt']:.0f} kt</td>"
                f"<td>{row['official_vmax_kt']:.0f} kt</td></tr>"
            )
    final = v37_points[-1]
    html_text = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dolphin __MODEL_VERSION__ GFS/GEFS vortex-route world map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#5b6875;--line:#d4dde5;--surface:#fff;--bg:#eef3f6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:24px}h1{margin:0 0 6px;font-size:28px}h2{margin:0 0 10px;font-size:17px}.sub{color:var(--muted);max-width:1100px;margin:0 0 16px}.notice{background:#fff8e6;border-left:4px solid #d97706;padding:11px 14px;margin:12px 0 16px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0}.card{background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:12px}.card b{display:block;font-size:18px;margin-top:4px}.muted{color:var(--muted)}.mapwrap{background:var(--surface);border:1px solid var(--line);border-radius:7px;overflow:hidden}#map{height:650px}.legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 18px;padding:13px 15px;border-top:1px solid var(--line)}.legend span{display:flex;align-items:center;gap:8px}.swatch{width:27px;height:3px;display:inline-block;flex:none}.section{margin-top:16px;background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:15px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{color:var(--muted);font-size:12px}small{display:block;color:var(--muted);font-size:11px}.source{color:var(--muted);font-size:12px}a{color:#0f5f9f}.intensity-label{background:transparent;border:0;color:#17202a;font-weight:700;text-shadow:0 1px 2px white,0 -1px 2px white,1px 0 2px white,-1px 0 2px white}
@media(max-width:900px){main{padding:12px}h1{font-size:22px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}#map{height:540px}.legend{grid-template-columns:1fr}table{font-size:12px}}
</style></head><body><main>
<h1>Dolphin: __MODEL_VERSION__ GFS/GEFS vortex-route forecast</h1>
<p class="sub">__ROUTE_DESCRIPTION__ The point colors and labels come from the field-conditioned structure-expert blend. The map includes the saved v36 reference and current JTWC/JMA tracks.</p>
<div class="notice"><b>Research output, not an operational warning.</b> The route uses local IBTrACS/JMA-derived history and cached public NOAA GFS/GEFS fields; official forecast positions are comparison-only. __MODEL_VERSION__ averages tracks after integration, not raw winds. Official wind conventions differ: JTWC is 1-minute sustained and JMA is 10-minute sustained.</div>
<div class="cards"><div class="card"><span class="muted">__MODEL_VERSION__ issue</span><b>__ISSUE__</b><span class="muted">__STRUCTURE_COUNTS__</span></div>
<div class="card"><span class="muted">__MODEL_VERSION__ at +120 h</span><b>__FINAL_POS__</b><span class="muted">__FINAL_WIND__ kt / __FINAL_PRESSURE__ hPa</span></div>
<div class="card"><span class="muted">+120 h route spread</span><b>__FINAL_SPREAD__ km</b><span class="muted">mean distance from __ROUTE_COUNT__ route members</span></div>
<div class="card"><span class="muted">Route guide</span><b>__ROUTE_GUIDE_TITLE__</b><span class="muted">__ROUTE_GUIDE_DETAIL__</span></div></div>
<div class="mapwrap"><div id="map" aria-label="Dolphin __MODEL_VERSION__ world map comparing __MODEL_VERSION__, v36, JTWC, and JMA"></div><div class="legend">
<span><i class="swatch" style="background:#111827"></i>Observed track</span><span><i class="swatch" style="background:#b91c1c"></i>__MODEL_VERSION__ route and intensity markers</span>
<span><i class="swatch" style="background:#64748b;border-top:2px dashed #64748b;height:0"></i>v36 reference</span><span><i class="swatch" style="background:#c2410c;border-top:2px dashed #c2410c;height:0"></i>JTWC official</span>
<span><i class="swatch" style="background:#047857;border-top:2px dashed #047857;height:0"></i>JMA/RSMC Tokyo official</span><span>Point popup: wind, central pressure, spread, and valid time</span>
</div></div>
<section class="section"><h2>__MODEL_VERSION__ forecast values used on the map</h2><table><thead><tr><th>Lead</th><th>Valid UTC</th><th>Lat</th><th>Lon</th><th>Wind</th><th>Pressure</th><th>Track spread</th></tr></thead><tbody>__ROWS__</tbody></table></section>
<section class="section"><h2>Track error against current local official points</h2><table><thead><tr><th>Source</th><th>Official lead</th><th>Model lead at same valid UTC</th><th>Position error</th><th>__MODEL_VERSION__ wind</th><th>Official wind</th></tr></thead><tbody>__COMPARISON_ROWS__</tbody></table><p class="source">Errors are computed at the official point's valid UTC time by linear interpolation between model forecast points. The official products use different issue times and wind conventions.</p></section>
<section class="section"><h2>Intensity chart</h2><img src="dolphin_v37d_intensity.png" alt="__MODEL_VERSION__ wind and central pressure forecast" style="width:100%;max-width:1100px"></section>
<section class="section"><h2>Pressure-map diagnostic</h2><img src="dolphin_v37d_pressure_maps.png" alt="__MODEL_VERSION__ storm-relative pressure maps" style="width:100%;max-width:1200px"><p class="source">Parametric reconstruction from the predicted center pressure, maximum wind, RMW, and quadrant wind radii; this is not a learned spatial pressure field.</p></section>
__SYNOPTIC_SECTION__
<section class="section source"><h2>Data and method</h2><p>__METHOD_DESCRIPTION__</p><p>Sources: <a href="https://www.ncei.noaa.gov/products/international-best-track-archive">NOAA IBTrACS</a>, <a href="https://nomads.ncep.noaa.gov/info.php?page=opendap_grib_migration">NOAA NOMADS GFS/GEFS access</a>, <a href="https://www.jma.go.jp/bosai/map.html#3/30/140/&elem=root&typhoon=all&lang=en&contents=typhoon">JMA typhoon information</a>, and <a href="https://www.metoc.navy.mil/jtwc/products/wp1226web.txt">JTWC product</a>. Official lines are current local captures; their issue times differ from the model issue, so these rows are a valid-time-aligned overlay rather than a formal same-issue skill score.</p></section>
</main><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script><script>
const DATA=__DATA__;const map=L.map('map',{worldCopyJump:true}).setView([20,170],3);L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:7,attribution:'&copy; OpenStreetMap contributors'}).addTo(map);
const windColor=kt=>kt>=137?'#7c3aed':kt>=113?'#b91c1c':kt>=96?'#dc2626':kt>=83?'#ea580c':kt>=64?'#ca8a04':kt>=34?'#0891b2':'#64748b';
const popup=p=>{const pressure=p.pressure_hpa==null?'':`<b>Central pressure:</b> ${p.pressure_hpa.toFixed(0)} hPa<br>`;const lead=p.lead_hours==null?'':`<b>Lead:</b> +${p.lead_hours} h<br>`;const spread=p.track_spread_km==null?'':`<b>Track spread:</b> ${p.track_spread_km.toFixed(0)} km<br>`;return `<b>${p.model}</b><br>${lead}<b>Valid:</b> ${p.time}<br><b>Position:</b> ${p.lat.toFixed(2)}N, ${p.lon.toFixed(2)}E<br><b>Wind:</b> ${p.vmax_kt.toFixed(0)} kt (${p.wind_convention})<br>${pressure}${spread}`};
const line=(items,color,options={})=>L.polyline(items.map(p=>[p.lat,p.lon]),Object.assign({color,weight:3,opacity:.9},options)).addTo(map);
line(DATA.observed,'#111827',{weight:3});DATA.observed.forEach(p=>L.circleMarker([p.lat,p.lon],{radius:3,color:'#111827',fillColor:'#111827',fillOpacity:.9}).bindPopup(popup(p)).addTo(map));
(DATA.route_members||[]).forEach(member=>L.polyline(member.latitude.map((lat,index)=>[lat,member.longitude[index]]),{color:'#b91c1c',weight:1,opacity:.14}).addTo(map));
line(DATA.v37,'#b91c1c',{weight:4});DATA.v37.forEach((p,i)=>{if(i%2===0)L.circle([p.lat,p.lon],{radius:(p.track_spread_km||1)*1000,color:'#b91c1c',weight:1,opacity:.25,fillOpacity:.035}).addTo(map);const m=L.circleMarker([p.lat,p.lon],{radius:6,color:'#fff',weight:1.5,fillColor:windColor(p.vmax_kt),fillOpacity:.98}).bindPopup(popup(p)).addTo(map);if(p.lead_hours%24===0)m.bindTooltip(`+${p.lead_hours}h · ${p.vmax_kt.toFixed(0)} kt · ${p.pressure_hpa.toFixed(0)} hPa`,{permanent:true,direction:'right',className:'intensity-label',offset:[7,0]})});
line(DATA.v36,'#64748b',{weight:2,dashArray:'7 7'});DATA.v36.forEach(p=>L.circleMarker([p.lat,p.lon],{radius:4,color:'#64748b',fillColor:'#fff',fillOpacity:1,weight:1.5}).bindPopup(popup(p)).addTo(map));
line(DATA.jtwc,'#c2410c',{weight:3,dashArray:'9 7'});DATA.jtwc.forEach(p=>L.circleMarker([p.lat,p.lon],{radius:5,color:'#c2410c',fillColor:'#fff',fillOpacity:1,weight:2}).bindPopup(popup(p)).addTo(map));
line(DATA.jma,'#047857',{weight:3,dashArray:'3 7'});DATA.jma.forEach(p=>L.circleMarker([p.lat,p.lon],{radius:5,color:'#047857',fillColor:'#fff',fillOpacity:1,weight:2}).bindPopup(popup(p)).addTo(map));
map.fitBounds(DATA.observed.concat(DATA.v37,DATA.v36,DATA.jtwc,DATA.jma).map(p=>[p.lat,p.lon]),{padding:[24,24]});
</script></body></html>"""
    html_text = html_text.replace("__DATA__", data_json)
    html_text = html_text.replace("__ISSUE__", html.escape(payload["issue_time_utc"]))
    html_text = html_text.replace("__FINAL_POS__", f"{final['lat']:.2f}N, {final['lon']:.2f}E")
    html_text = html_text.replace("__FINAL_WIND__", f"{final['vmax_kt']:.0f}")
    html_text = html_text.replace("__FINAL_PRESSURE__", f"{final['pressure_hpa']:.0f}")
    html_text = html_text.replace("__FINAL_SPREAD__", f"{final['track_spread_km']:.0f}")
    html_text = html_text.replace("__STRUCTURE_COUNTS__", html.escape(structure_counts))
    html_text = html_text.replace("__ROUTE_COUNT__", str(route_count))
    route_guide_title = {
        "gefs_mean_multilevel": "GEFS mean vortex",
        "gfs_gefs_adaptive": "Adaptive GFS/GEFS",
    }.get(ROUTE_POLICY, "GFS + GEFS vortex")
    route_guide_detail = {
        "gefs_mean_multilevel": "weighted 850/500/200 hPa field means",
        "gfs_gefs_adaptive": "agreement-gated physical route",
    }.get(ROUTE_POLICY, "850 hPa relative-vorticity maxima")
    html_text = html_text.replace("__ROUTE_GUIDE_TITLE__", route_guide_title)
    html_text = html_text.replace("__ROUTE_GUIDE_DETAIL__", route_guide_detail)
    html_text = html_text.replace("__ROWS__", "".join(rows))
    html_text = html_text.replace("__COMPARISON_ROWS__", "".join(comparison_rows))
    route_description = {
        "gfs_vortex_multilevel_ensemble": (
            "The red route is the mean after integrating one deterministic GFS 850-hPa vortex track and 31 GEFS multi-level (850/500/200 hPa) vortex tracks."
        ),
        "gefs_mean_multilevel": (
            "The red route is the weighted mean of GEFS ensemble-mean vortex tracks at 850/500/200 hPa; the three level tracks remain visible as spread members."
        ),
        "gfs_gefs_adaptive": (
            "The red route uses the GFS/GEFS member route when its +120 h consensus agrees with the independent GEFS field-mean route; otherwise it switches to the GEFS field-mean route."
        ),
    }.get(
        ROUTE_POLICY,
        "The red route is the mean after integrating one deterministic GFS and 31 GEFS 850-hPa vortex tracks.",
    )
    method_description = (
        f"Route: {payload['route_backbone']}. Structure: {payload['structure_branch']}. "
        f"Field normalization: {payload['field_normalization']}. "
        f"Pressure layer: {payload['pressure_calibration']}. "
        "The structure branch cannot change the route. Pressure values are central-pressure predictions; "
        "the pressure panels below are a parametric diagnostic, not a learned full pressure-grid forecast."
    )
    html_text = html_text.replace("__METHOD_DESCRIPTION__", html.escape(method_description))
    synoptic_section = ""
    if SYNOPTIC_CHART.exists():
        synoptic_section = (
            '<section class="section"><h2>NOAA GFS synoptic background</h2>'
            f'<img src="{html.escape(SYNOPTIC_CHART.name)}" '
            f'alt="{html.escape(MODEL_VERSION)} NOAA GFS pressure and wind background" '
            'style="width:100%;max-width:1450px">'
            '<p class="source">This chart uses the no-key public NOAA GFS '
            'mean-sea-level pressure field and weighted 850/500/200 hPa winds. '
            f'{html.escape(MODEL_VERSION)} supplies the center, intensity, and route overlay; '
            'the field background is not a learned full-grid pressure forecast. '
            'The dashed gray line is the v36A reference.</p></section>'
        )
    html_text = html_text.replace("__SYNOPTIC_SECTION__", synoptic_section)
    html_text = html_text.replace("__MODEL_VERSION__", html.escape(MODEL_VERSION))
    html_text = html_text.replace("__ROUTE_DESCRIPTION__", route_description)
    html_text = html_text.replace("v37D", html.escape(MODEL_VERSION))
    html_text = html_text.replace("dolphin_v37d_", f"dolphin_{ARTIFACT_TAG}_")
    OUTPUT_HTML.write_text(html_text, encoding="utf-8")


def main() -> None:
    payload = run_forecast()
    create_chart(payload)
    create_png(payload)
    create_html(payload)
    print(json.dumps({
        "forecast_json": str(OUTPUT_JSON),
        "world_map_png": str(OUTPUT_PNG),
        "world_map_html": str(OUTPUT_HTML),
        "intensity_chart": str(OUTPUT_CHART),
        "final": payload["forecast"][-1],
        "comparison_errors": payload["comparison_errors"],
    }, indent=2))


if __name__ == "__main__":
    main()
