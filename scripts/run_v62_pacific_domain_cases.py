#!/usr/bin/env python3
"""Run strict-causal western-Pacific pressure-state forecasts for two cases."""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import draw_dolphin_tip_case_maps as base_maps  # noqa: E402
from scripts.run_v61_big_system_cases import (  # noqa: E402
    DOLPHIN_FILES,
    DOLPHIN_LEVELS,
    DOLPHIN_SOURCE,
    TIP_FILES,
    TIP_LEVELS,
    TIP_SOURCE,
    _decode_analysis,
    _local_route,
    _recent_motion,
    _route_points,
    _score,
)
from scripts.predict_ibtracs_jma_only import build_track_window, load_land_points  # noqa: E402
from v62_intensity_structure import (  # noqa: E402
    V62IntensityEnsemble,
    couple_forecast_to_pressure_map,
)
from v62_pacific_domain_route import (  # noqa: E402
    CAUSAL_ONLY,
    LEAD_HOURS,
    PACIFIC_LAT_RANGE,
    PACIFIC_LON_RANGE,
    build_pacific_route,
    detect_pressure_systems,
    forecast_pacific_state,
)
from v61_big_system_route import integrate_from_issue, weighted_route  # noqa: E402


PACIFIC_WEIGHT = 0.25
LOCAL_WEIGHT = 1.0 - PACIFIC_WEIGHT
DOLPHIN_JSON = ROOT / "track_build" / "dolphin_v62_pacific_domain_case_map.json"
TIP_JSON = ROOT / "track_build" / "tip_v62_pacific_domain_case_map.json"
DOLPHIN_HTML = ROOT / "paper" / "dolphin_v62_pacific_domain_world_map.html"
TIP_HTML = ROOT / "paper" / "tip_v62_pacific_domain_world_map.html"
DOLPHIN_PNG = ROOT / "paper" / "dolphin_v62_pacific_domain_world_map.png"
TIP_PNG = ROOT / "paper" / "tip_v62_pacific_domain_world_map.png"
DOLPHIN_PRESSURE_PNG = ROOT / "paper" / "dolphin_v62_pacific_pressure_forecast.png"
TIP_PRESSURE_PNG = ROOT / "paper" / "tip_v62_pacific_pressure_forecast.png"
DOLPHIN_INTENSITY_PNG = ROOT / "paper" / "dolphin_v62_intensity_structure.png"
TIP_INTENSITY_PNG = ROOT / "paper" / "tip_v62_intensity_structure.png"
V23_STATS_PATH = ROOT / "v37" / "v23_norm_stats.npz"
FIELD_SCALE_PATH = ROOT / "track_build" / "dlm4_int8.npz"
INTENSITY_CHECKPOINT_ROOT = ROOT / "v37" / "structure_spatial" / "checkpoints"
INTENSITY_CALIBRATION_PATH = ROOT / "v37" / "structure_spatial" / "v37g_intensity_calibration.json"
TIP_INTENSITY_SOURCE = ROOT / "track_build" / "tip_v37_cfsr_causal_19791012.json"

_INTENSITY_MODEL: V62IntensityEnsemble | None = None


def _as_utc(value: object) -> dt.datetime:
    """Normalize an ISO or NumPy timestamp for the causal input guard."""

    if isinstance(value, dt.datetime):
        result = value
    elif isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("analysis file contains NaT")
        seconds = value.astype("datetime64[s]").astype(np.int64)
        result = dt.datetime.fromtimestamp(int(seconds), tz=dt.timezone.utc)
    else:
        result = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=dt.timezone.utc)
    return result.astimezone(dt.timezone.utc)


def _single_coordinate(dataset: xr.Dataset, name: str) -> object:
    if name not in dataset.coords:
        raise RuntimeError(f"causal input guard: GRIB dataset has no {name} coordinate")
    values = np.asarray(dataset[name].values).reshape(-1)
    if len(values) != 1:
        raise RuntimeError(f"causal input guard: GRIB dataset has multiple {name} values")
    return values[0]


def _validate_analysis_file(path: Path, issue_time: dt.datetime) -> dict:
    """Reject forecast-step or post-issue weather files before decoding them."""

    valid_times = []
    for level_type in ("meanSea", "isobaricInhPa"):
        dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": level_type},
                "indexpath": "",
            },
        )
        try:
            valid_time = _as_utc(_single_coordinate(dataset, "valid_time"))
            if "step" in dataset.coords:
                step = np.asarray(dataset["step"].values).reshape(-1).astype("timedelta64[ns]")
                if np.any(step != np.timedelta64(0, "ns")):
                    raise RuntimeError(
                        f"causal input guard rejected forecast step in {path}: {step.tolist()}"
                    )
            if valid_time > issue_time:
                raise RuntimeError(
                    f"causal input guard rejected future weather file {path}: "
                    f"valid_time={valid_time.isoformat()} > issue={issue_time.isoformat()}"
                )
            valid_times.append(valid_time)
        finally:
            dataset.close()
    if len(set(valid_times)) != 1:
        raise RuntimeError(f"causal input guard found inconsistent valid times in {path}: {valid_times}")
    return {
        "path": str(path.relative_to(ROOT)),
        "valid_time_utc": valid_times[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "step_hours": 0,
        "accepted_as": "analysis_or_reanalysis",
    }


def _validate_history(rows: list[dict], time_key: str, issue_time: dt.datetime, label: str) -> dict:
    times = []
    for index, row in enumerate(rows):
        if time_key not in row:
            raise RuntimeError(f"causal input guard: {label}[{index}] has no {time_key}")
        value = _as_utc(row[time_key])
        if value > issue_time:
            raise RuntimeError(
                f"causal input guard rejected future {label}[{index}]: "
                f"time={value.isoformat()} > issue={issue_time.isoformat()}"
            )
        times.append(value)
    if not times:
        raise RuntimeError(f"causal input guard: {label} is empty")
    return {
        "label": label,
        "time_key": time_key,
        "row_count": len(times),
        "max_time_utc": max(times).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _causal_input_guard(
    issue_time: dt.datetime,
    weather_files: tuple[Path, ...],
    histories: list[tuple[str, list[dict], str]],
) -> dict:
    if not CAUSAL_ONLY:
        raise RuntimeError("v62 causal-only configuration was disabled")
    weather = [_validate_analysis_file(path, issue_time) for path in weather_files]
    history = [_validate_history(rows, time_key, issue_time, label) for label, rows, time_key in histories]
    return {
        "enabled": True,
        "policy": "analysis/reanalysis only; every weather valid_time <= issue_time and every GRIB step = 0",
        "issue_time_utc": issue_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weather_files": weather,
        "histories": history,
        "positive_lead_weather_rejected": True,
        "future_history_rows_rejected": True,
        "official_forecasts_used_for_inference": False,
    }


def _intensity_model() -> V62IntensityEnsemble:
    global _INTENSITY_MODEL
    if _INTENSITY_MODEL is None:
        _INTENSITY_MODEL = V62IntensityEnsemble(INTENSITY_CHECKPOINT_ROOT, INTENSITY_CALIBRATION_PATH)
    return _INTENSITY_MODEL


def _intensity_record(row: dict, time_key: str) -> dict:
    """Make the v37G track schema explicit without inventing missing labels."""

    result = dict(row)
    result["time_utc"] = str(row.get("time_utc", row.get(time_key, "")))
    for key in (
        "vmax_kt", "pressure_hpa", "rmw_nm", "roci_nm", "dist2land_km",
        "r34_ne_nm", "r34_se_nm", "r34_sw_nm", "r34_nw_nm",
        "r50_ne_nm", "r50_se_nm", "r50_sw_nm", "r50_nw_nm",
        "r64_ne_nm", "r64_se_nm", "r64_sw_nm", "r64_nw_nm",
    ):
        result.setdefault(key, float("nan"))
    return result


def _intensity_track(rows: list[dict], time_key: str) -> tuple[np.ndarray, list[dict]]:
    if not V23_STATS_PATH.exists():
        raise FileNotFoundError(V23_STATS_PATH)
    stats = np.load(V23_STATS_PATH)
    records = [_intensity_record(row, time_key) for row in rows]
    normalized, _, _ = build_track_window(
        records,
        stats["tmean"].astype("float32"),
        stats["tstd"].astype("float32"),
        *load_land_points(),
    )
    return normalized, records


def _predict_intensity(rows: list[dict], time_key: str, field: np.ndarray) -> tuple[list[dict], dict]:
    track, records = _intensity_track(rows, time_key)
    current = records[-1]
    previous = records[-2] if len(records) > 1 else current
    forecast, metadata = _intensity_model().predict(
        track,
        field,
        float(current["vmax_kt"]),
        float(current["pressure_hpa"]),
        float(previous["vmax_kt"]),
        float(previous["pressure_hpa"]),
    )
    metadata["calibration"] = str(INTENSITY_CALIBRATION_PATH.relative_to(ROOT))
    return forecast, metadata


def _tip_intensity_field(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    """Rebuild the v23 4-channel field using only Tip data through the issue."""

    slp = np.asarray(archive["slp_hpa"], dtype="float32")
    u = np.asarray(archive["u_dlm"], dtype="float32")
    v = np.asarray(archive["v_dlm"], dtype="float32")
    scale = np.asarray(np.load(FIELD_SCALE_PATH)["scale"], dtype="float32")
    if len(slp) < 9:
        raise ValueError("Tip causal archive must contain the nine issue-history frames")

    def frame(index: int) -> np.ndarray:
        tendency_index = max(0, index - 4)
        raw = np.stack([
            slp[index] - float(np.nanmean(slp[index])),
            slp[index] - slp[tendency_index],
            u[index],
            v[index],
        ]).astype("float32")
        return np.clip(raw / scale[:, None, None], -4.0, 4.0)

    return frame(8)


def _draw_intensity_chart(path: Path, forecast: list[dict], title: str) -> None:
    leads = np.asarray([row["lead_hours"] for row in forecast], dtype="float32")
    wind = np.asarray([row["vmax_kt"] for row in forecast], dtype="float32")
    wind_spread = np.asarray([row.get("vmax_spread_kt", 0.0) for row in forecast], dtype="float32")
    pressure = np.asarray([row["central_pressure_hpa"] for row in forecast], dtype="float32")
    pressure_spread = np.asarray([row.get("pressure_spread_hpa", 0.0) for row in forecast], dtype="float32")
    rmw = np.asarray([row["rmw_km"] for row in forecast], dtype="float32")
    rmw_spread = np.asarray([row.get("rmw_spread_km", 0.0) for row in forecast], dtype="float32")
    radii = np.asarray([row["wind_radii_km"] for row in forecast], dtype="float32")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    axes[0].plot(leads, wind, color="#b91c1c", linewidth=2.4, label="vmax")
    axes[0].fill_between(leads, wind - wind_spread, wind + wind_spread, color="#b91c1c", alpha=0.15, label="ensemble spread")
    axes[0].set_ylabel("Maximum wind (kt)")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.35)
    axes[0].legend(loc="best")
    axes[1].plot(leads, pressure, color="#1d4ed8", linewidth=2.4, label="central pressure")
    axes[1].fill_between(leads, pressure - pressure_spread, pressure + pressure_spread, color="#1d4ed8", alpha=0.15, label="ensemble spread")
    axes[1].set_ylabel("Central pressure (hPa)")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.35)
    axes[1].legend(loc="best")
    labels = ((0, "R34 mean", "#be185d"), (4, "R50 mean", "#7c3aed"), (8, "R64 mean", "#0e7490"))
    for offset, label, color in labels:
        values = radii[:, offset:offset + 4].mean(axis=1)
        axes[2].plot(leads, values, color=color, linewidth=2.0, label=label)
    axes[2].plot(leads, rmw, color="#111827", linewidth=1.8, linestyle="--", label="RMW")
    axes[2].fill_between(leads, rmw - rmw_spread, rmw + rmw_spread, color="#111827", alpha=0.10)
    axes[2].set_xlabel("Forecast lead (hours)")
    axes[2].set_ylabel("Radius (km)")
    axes[2].set_ylim(bottom=0)
    axes[2].grid(alpha=0.35)
    axes[2].legend(loc="best", ncol=2)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _append_intensity_section(page: str, image_name: str, metadata: dict) -> str:
    policy = html.escape(
        "The v62 intensity head predicts maximum wind, central pressure, RMW, and directional R34/R50/R64 radii from the current/past track window and current analysis patch only. "
        f"{metadata['checkpoint_count']} frozen v37G spatial experts provide the mean and spread."
    )
    section = f'<section><h2>Predicted intensity and wind structure</h2><p>{policy}</p><img src="{html.escape(image_name)}" alt="Causal v62 intensity and wind-structure forecast"></section>'
    return page.replace("</main>", section + "</main>")


def _member_points(members: np.ndarray, base_latitude: float, base_longitude: float, issue: dt.datetime) -> list[list[dict]]:
    paths = []
    for path in integrate_from_issue(members, base_latitude, base_longitude):
        paths.append([
            {
                "lead_hours": int(point["lead_hours"]),
                "valid_time_utc": (issue + dt.timedelta(hours=int(point["lead_hours"]))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latitude": float(point["latitude"]),
                "longitude": float(point["longitude"]),
                "lat": float(point["latitude"]),
                "lon": float(point["longitude"]),
            }
            for point in path
        ])
    return paths


def _system_series(
    pressure_states: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    forecast: list[dict],
    base_latitude: float,
    base_longitude: float,
) -> list[dict]:
    result = []
    for index, hours in enumerate(LEAD_HOURS):
        if index == 0:
            lat, lon = base_latitude, base_longitude
        else:
            point = forecast[index - 1]
            lat, lon = float(point["latitude"]), float(point["longitude"])
        result.append({
            "lead_hours": int(hours),
            "systems": detect_pressure_systems(pressure_states[index], latitude, longitude, lat, lon),
        })
    return result


def _draw_pressure_forecast(
    path: Path,
    fields: np.ndarray,
    pressure: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    observed: list[dict],
    forecast: list[dict],
    systems: list[dict],
    title: str,
    comparison_routes: dict[str, tuple[str, list[dict]]] | None = None,
) -> None:
    lat_mask = (latitude >= PACIFIC_LAT_RANGE[0]) & (latitude <= PACIFIC_LAT_RANGE[1])
    lon_mask = (longitude >= PACIFIC_LON_RANGE[0]) & (longitude <= PACIFIC_LON_RANGE[1])
    lat = latitude[lat_mask]
    lon = longitude[lon_mask]
    if not lat_mask.any() or not lon_mask.any():
        raise RuntimeError("Pacific plotting domain is outside the analysis grid")
    slp = pressure[:, lat_mask][:, :, lon_mask]
    hgt = fields[:, 0][:, lat_mask][:, :, lon_mask]
    u850 = fields[:, 1][:, lat_mask][:, :, lon_mask]
    v850 = fields[:, 2][:, lat_mask][:, :, lon_mask]
    u500 = fields[:, 3][:, lat_mask][:, :, lon_mask]
    v500 = fields[:, 4][:, lat_mask][:, :, lon_mask]
    u200 = fields[:, 5][:, lat_mask][:, :, lon_mask]
    v200 = fields[:, 6][:, lat_mask][:, :, lon_mask]
    wind850 = np.hypot(u850, v850) * 1.94384
    wind200 = np.hypot(u200, v200) * 1.94384

    def regular_levels(values: np.ndarray, step: float, lower: float, upper: float) -> np.ndarray:
        start = math.floor(float(np.nanpercentile(values, lower)) / step) * step
        stop = math.ceil(float(np.nanpercentile(values, upper)) / step) * step
        if stop <= start:
            stop = start + step
        return np.arange(start, stop + step, step)

    # The NCDR-style layering is intentionally analysis-oriented: dense SLP
    # contours, 500-hPa height/wind, and low-level wind-speed structure all
    # come from the same causal state shown in the route diagnostics.
    slp_levels = regular_levels(slp, 2.0, 1.0, 99.0)
    slp_major_levels = slp_levels[::2]
    height_levels = regular_levels(hgt, 60.0, 2.0, 98.0)
    height_major_levels = height_levels[::2]
    wind850_stop = max(40.0, math.ceil(float(np.nanpercentile(wind850, 99.0)) / 10.0) * 10.0)
    wind850_levels = np.arange(20.0, wind850_stop + 10.0, 10.0)
    wind850_major_levels = wind850_levels[::2]
    jet_start = max(40.0, math.floor(float(np.nanpercentile(wind200, 65.0)) / 10.0) * 10.0)
    jet_stop = max(jet_start + 20.0, math.ceil(float(np.nanpercentile(wind200, 99.0)) / 10.0) * 10.0)
    jet_levels = np.arange(jet_start, jet_stop + 10.0, 10.0)
    display_indices = [0, 4, 8, 12, 16, 20]
    coastlines = base_maps.load_coastlines()
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    contour_handle = None
    wind_handle = None
    regional_labels = (
        ("China", 110.5, 31.5),
        ("Japan", 139.5, 36.8),
        ("Taiwan", 121.0, 23.6),
        ("Philippines", 123.0, 12.0),
    )
    lead_labels = (24, 48, 72, 96, 120)
    lon_step = float(np.nanmedian(np.diff(lon))) if len(lon) > 1 else 0.25
    barb_stride = max(1, int(round(3.0 / max(lon_step, 0.01))))
    for axis, index in zip(axes.flat, display_indices):
        # Light wind-speed shading reveals the low-level circulation without
        # replacing the pressure field as the primary background.
        wind_handle = axis.contourf(
            lon,
            lat,
            wind850[index],
            levels=wind850_levels,
            cmap="RdPu",
            alpha=0.18,
            extend="max",
        )
        contour_handle = axis.contourf(
            lon,
            lat,
            slp[index],
            levels=slp_levels,
            cmap="Spectral_r",
            alpha=0.78,
            extend="both",
        )
        minor_slp = axis.contour(lon, lat, slp[index], levels=slp_levels, colors="#17202a", linewidths=0.32, alpha=0.62)
        major_slp = axis.contour(lon, lat, slp[index], levels=slp_major_levels, colors="#111827", linewidths=0.75, alpha=0.86)
        axis.clabel(major_slp, inline=True, fontsize=6.3, fmt="%1.0f", inline_spacing=2)
        height_minor = axis.contour(lon, lat, hgt[index], levels=height_levels, colors="#7e22ce", linewidths=0.42, alpha=0.58)
        height_major = axis.contour(lon, lat, hgt[index], levels=height_major_levels, colors="#581c87", linewidths=0.82, alpha=0.82)
        axis.clabel(height_major, inline=True, fontsize=6.2, fmt="%1.0f", inline_spacing=2)
        low_wind = axis.contour(lon, lat, wind850[index], levels=wind850_major_levels, colors="#be185d", linewidths=0.5, alpha=0.72)
        axis.clabel(low_wind, inline=True, fontsize=5.4, fmt="%1.0f", inline_spacing=2)
        if len(jet_levels):
            jet = axis.contour(lon, lat, wind200[index], levels=jet_levels, colors="#0e7490", linewidths=0.52, linestyles="--", alpha=0.7)
            if len(jet_levels) > 1:
                axis.clabel(jet, inline=True, fontsize=5.2, fmt="%1.0f", inline_spacing=2)

        # 500-hPa barbs are the primary flow depiction, matching the NCDR
        # chart convention; values are converted from m/s to knots.
        axis.barbs(
            lon[::barb_stride],
            lat[::barb_stride],
            u500[index][::barb_stride, ::barb_stride] * 1.94384,
            v500[index][::barb_stride, ::barb_stride] * 1.94384,
            length=4.4,
            linewidth=0.28,
            color="#1d4ed8",
            alpha=0.68,
            barb_increments={"half": 5, "full": 10, "flag": 50},
        )
        base_maps.draw_coastlines(axis, coastlines)
        observed_lons = [float(row["lon"]) for row in observed]
        observed_lats = [float(row["lat"]) for row in observed]
        axis.plot(observed_lons, observed_lats, color="#111827", linewidth=1.5, label="observed input")
        if index:
            route = forecast[:index]
            route_lons = [observed_lons[-1]] + [float(row["lon"]) for row in route]
            route_lats = [observed_lats[-1]] + [float(row["lat"]) for row in route]
            axis.plot(route_lons, route_lats, color="#b91c1c", linewidth=2.4, label="v62 route", zorder=7)
            axis.scatter(route_lons[-1], route_lats[-1], color="#b91c1c", edgecolor="white", linewidth=0.65, s=30, zorder=8)
            if comparison_routes:
                for label, (color, route_points) in comparison_routes.items():
                    route = route_points[:index]
                    route_lons = [observed_lons[-1]] + [float(row["lon"]) for row in route]
                    route_lats = [observed_lats[-1]] + [float(row["lat"]) for row in route]
                    axis.plot(route_lons, route_lats, color=color, linewidth=1.8, linestyle="--", label=label, zorder=6)
                    axis.scatter(route_lons[-1], route_lats[-1], color=color, edgecolor="white", linewidth=0.55, s=22, zorder=7)
            for lead in lead_labels:
                if lead > LEAD_HOURS[index]:
                    continue
                point = forecast[lead // 6 - 1]
                axis.scatter(float(point["lon"]), float(point["lat"]), color="#b91c1c", edgecolor="white", linewidth=0.55, s=19, zorder=8)
                axis.annotate(
                    f"+{lead}h",
                    (float(point["lon"]), float(point["lat"])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=5.8,
                    color="#991b1b",
                    fontweight="bold",
                    zorder=9,
                )
        else:
            axis.scatter(observed_lons[-1], observed_lats[-1], color="#b91c1c", s=22, zorder=6, label="issue position")
        for system in systems[index]["systems"]:
            axis.scatter(system["longitude"], system["latitude"], marker="x", color="#0f766e", s=35, linewidth=1.2, zorder=6)
            axis.annotate(
                f"L {system['pressure_hpa']:.0f}",
                (system["longitude"], system["latitude"]),
                xytext=(4, -8),
                textcoords="offset points",
                fontsize=5.3,
                color="#115e59",
                zorder=7,
            )
        for label, label_lon, label_lat in regional_labels:
            axis.text(label_lon, label_lat, label, fontsize=6.2, color="#334155", alpha=0.85, ha="center", zorder=6)
        axis.set_xlim(*PACIFIC_LON_RANGE)
        axis.set_ylim(*PACIFIC_LAT_RANGE)
        axis.set_xticks(range(100, 191, 10))
        axis.set_yticks(range(0, 61, 10))
        axis.set_xlabel("Longitude (E)", fontsize=7)
        axis.set_ylabel("Latitude (N)", fontsize=7)
        axis.tick_params(labelsize=6.5)
        axis.grid(color="#94a3b8", linewidth=0.28, alpha=0.42)
        valid_time = observed[-1].get("time", "") if index == 0 else forecast[index - 1].get("valid_time_utc", "")
        axis.set_title(f"Causal state  +{LEAD_HOURS[index]} h\n{valid_time.replace('T', ' ').replace('Z', ' UTC')}", fontsize=9.2, fontweight="bold")
        axis.text(0.01, 0.985, "MSLP / 500-hPa HGT / 500-hPa WIND", transform=axis.transAxes, va="top", fontsize=5.5, color="#1e293b", bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5})
        if index == 0:
            handles = [
                Line2D([0], [0], color="#111827", linewidth=1.5, label="observed input"),
                Line2D([0], [0], color="#b91c1c", linewidth=2.4, label="v62 route"),
                Line2D([0], [0], color="#7e22ce", linewidth=0.9, label="500-hPa height"),
                Line2D([0], [0], color="#0e7490", linewidth=0.8, linestyle="--", label="200-hPa jet speed"),
            ]
            if comparison_routes:
                handles[2:2] = [
                    Line2D([0], [0], color=color, linewidth=1.8, linestyle="--", label=label)
                    for label, (color, _) in comparison_routes.items()
                ]
            axis.legend(handles=handles, loc="lower left", fontsize=6.5, framealpha=0.86)
    fig.colorbar(contour_handle, ax=axes, pad=0.018, shrink=0.88, label="Mean sea-level pressure (hPa)")
    fig.colorbar(wind_handle, ax=axes, orientation="horizontal", pad=0.045, fraction=0.035, label="850-hPa wind speed (kt; translucent shading and magenta contours)")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    fig.text(0.5, 0.004, "Causal state from current/t-12/t-24 analysis tendency only. Black: MSLP; purple: 500-hPa height; blue barbs: 500-hPa wind; cyan dashed: 200-hPa jet; teal x: analysis low candidates.", ha="center", fontsize=8.2, color="#334155")
    fig.savefig(path, dpi=210, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _append_pressure_section(page: str, image_name: str, policy: str) -> str:
    section = f'<section><h2>Whole western-Pacific causal pressure forecast</h2><p>{html.escape(policy)}</p><img src="{html.escape(image_name)}" alt="Whole western-Pacific causal pressure forecast and route"></section>'
    return page.replace("</main>", section + "</main>")


def _dolphin() -> dict:
    source = json.loads(DOLPHIN_SOURCE.read_text(encoding="utf-8"))
    records = source["current_track_history"]
    issue = dt.datetime.fromisoformat(source["issue_time_utc"].replace("Z", "+00:00"))
    causal_guard = _causal_input_guard(
        issue,
        DOLPHIN_FILES,
        [("Dolphin current_track_history", records, "time")],
    )
    fields, pressure, latitude, longitude = _decode_analysis(DOLPHIN_FILES)
    levels = np.load(DOLPHIN_LEVELS, allow_pickle=True)
    base_latitude, base_longitude = float(records[-1]["lat"]), float(records[-1]["lon"])
    members, weights, diagnostics = build_pacific_route(fields, pressure, latitude, longitude, base_latitude, base_longitude, _recent_motion(records, "time"))
    pacific_route = weighted_route(members, weights)
    local_route, _, local_diagnostics = _local_route(levels, base_latitude, base_longitude)
    route = LOCAL_WEIGHT * local_route + PACIFIC_WEIGHT * pacific_route
    ensemble = LOCAL_WEIGHT * local_route[None, :, :] + PACIFIC_WEIGHT * members
    observed = [{"time": row["time"], "lat": float(row["lat"]), "lon": float(row["lon"])} for row in records]
    state_fields, state_pressure, state_diagnostics = forecast_pacific_state(fields, pressure)
    base_forecast = _route_points(route, base_latitude, base_longitude, issue)
    intensity, intensity_metadata = _predict_intensity(records, "time", np.asarray(levels["current"], dtype="float32"))
    intensity, map_metadata = couple_forecast_to_pressure_map(
        intensity,
        state_pressure,
        state_fields,
        latitude,
        longitude,
        base_latitude,
        base_longitude,
        base_forecast,
        float(records[-1]["vmax_kt"]),
        float(records[-1]["pressure_hpa"]),
    )
    intensity_metadata["pressure_map_coupling"] = map_metadata
    forecast = _route_points(route, base_latitude, base_longitude, issue, intensity=intensity)
    systems = _system_series(state_pressure, latitude, longitude, forecast, base_latitude, base_longitude)
    payload = {
        "storm": "Dolphin",
        "issue_time_utc": source["issue_time_utc"],
        "model": "v62 whole western-Pacific causal state route + intensity/structure head (25% Pacific domain + 75% local)",
        "observed": observed,
        "forecast": forecast,
        "ensemble_forecasts": _member_points(ensemble, base_latitude, base_longitude, issue),
        "official": {"jtwc": [base_maps.point(row) for row in source["jtwc_official"]], "jma": [base_maps.point(row) for row in source["jma_official"]]},
        "route_diagnostics": {"pacific_weight": PACIFIC_WEIGHT, "local_weight": LOCAL_WEIGHT, "pacific": diagnostics, "local": local_diagnostics, "pressure_state": state_diagnostics, "pressure_systems": systems, "analysis_files": [str(path.relative_to(ROOT)) for path in DOLPHIN_FILES], "causal_input_guard": causal_guard},
        "intensity_model": intensity_metadata,
        "input_policy": "Strict causal-only mode: the runner rejects any GRIB forecast step or weather valid_time after the issue. Only current/past GFS analysis files and the observed track history are used; the whole 100-190E, 0-60N state is extrapolated from current/t-12/t-24 analysis tendency. Official JMA/JTWC tracks are overlays only.",
        "causal_input_guard": causal_guard,
        "future_rows_used_for_inference": 0,
        "official_forecasts_used_for_inference": False,
        "forecast_products_used": [],
        "forecast_intensity_source": "v62_intensity_structure.py using the frozen v37G spatial structure ensemble, coupled to the causal v62 pressure-map state; current/past analysis only",
        "other_typhoon_input": "No storm-center forecast or official typhoon track was used. Nearby lows/vortex candidates come from the causal SLP field.",
    }
    _draw_pressure_forecast(DOLPHIN_PRESSURE_PNG, state_fields, state_pressure, latitude, longitude, observed, forecast, systems, "Dolphin v62 whole western-Pacific causal pressure state")
    _draw_intensity_chart(DOLPHIN_INTENSITY_PNG, forecast, "Dolphin v62 causal intensity and wind structure")
    base_maps.DOLPHIN_JSON, base_maps.DOLPHIN_HTML, base_maps.DOLPHIN_PNG = DOLPHIN_JSON, DOLPHIN_HTML, DOLPHIN_PNG
    base_maps.draw_dolphin_png(payload, base_maps.load_coastlines())
    DOLPHIN_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    page = base_maps.html_shell("Dolphin v62 whole western-Pacific world map", "dolphin-map", payload, "The red route is v62 output. The Pacific pressure map is a causal analysis-tendency extrapolation, not an imported forecast product.", "<p><b>Input:</b> Current and past GFS analysis only. Broad rings cover the western Pacific from China/Japan through Taiwan, the Philippines, and the open Pacific.</p>", DOLPHIN_PNG, "dolphin")
    page = _append_pressure_section(page, DOLPHIN_PRESSURE_PNG.name, payload["input_policy"])
    DOLPHIN_HTML.write_text(_append_intensity_section(page, DOLPHIN_INTENSITY_PNG.name, intensity_metadata), encoding="utf-8")
    return payload


def _tip() -> dict:
    source = json.loads(TIP_SOURCE.read_text(encoding="utf-8"))
    case = source["cases"][0]
    intensity_source = json.loads(TIP_INTENSITY_SOURCE.read_text(encoding="utf-8"))
    intensity_rows = intensity_source["input_history"]
    issue = dt.datetime.fromisoformat(case["issue_time_utc"].replace("Z", "+00:00"))
    causal_guard = _causal_input_guard(
        issue,
        TIP_FILES,
        [
            ("Tip observed_before_issue", case["observed_before_issue"], "time_utc"),
            ("Tip intensity input_history", intensity_rows, "time_utc"),
        ],
    )
    fields, pressure, latitude, longitude = _decode_analysis(TIP_FILES)
    levels = np.load(TIP_LEVELS, allow_pickle=True)
    issue_row = case["observed_before_issue"][-1]
    base_latitude, base_longitude = float(issue_row["lat"]), float(issue_row["lon"])
    members, weights, diagnostics = build_pacific_route(fields, pressure, latitude, longitude, base_latitude, base_longitude, _recent_motion(case["observed_before_issue"], "time_utc"))
    pacific_route = weighted_route(members, weights)
    local_route, _, local_diagnostics = _local_route(levels, base_latitude, base_longitude)
    route = LOCAL_WEIGHT * local_route + PACIFIC_WEIGHT * pacific_route
    ensemble = LOCAL_WEIGHT * local_route[None, :, :] + PACIFIC_WEIGHT * members
    intensity_field = _tip_intensity_field(np.load(ROOT / "v37_cfsr" / "tip_1979_causal" / "cfsr_tip_19791012.npz", allow_pickle=True))
    observed = [{"time": row["time_utc"], "lat": float(row["lat"]), "lon": float(row["lon"])} for row in case["observed_before_issue"]]
    state_fields, state_pressure, state_diagnostics = forecast_pacific_state(fields, pressure)
    base_forecast = _route_points(route, base_latitude, base_longitude, issue)
    intensity, intensity_metadata = _predict_intensity(intensity_rows, "time_utc", intensity_field)
    intensity, map_metadata = couple_forecast_to_pressure_map(
        intensity,
        state_pressure,
        state_fields,
        latitude,
        longitude,
        base_latitude,
        base_longitude,
        base_forecast,
        float(intensity_rows[-1]["vmax_kt"]),
        float(intensity_rows[-1]["pressure_hpa"]),
    )
    intensity_metadata["pressure_map_coupling"] = map_metadata
    forecast = _route_points(route, base_latitude, base_longitude, issue, intensity=intensity)
    systems = _system_series(state_pressure, latitude, longitude, forecast, base_latitude, base_longitude)
    # Read verification truth only after all inference has completed.
    truth = [{"time_utc": row["time_utc"], "lat": float(row["lat"]), "lon": float(row["lon"])} for row in case["truth_after_issue"]]
    tip_case = {"issue_time_utc": case["issue_time_utc"], "observed_before_issue": observed, "forecast": forecast, "ensemble_forecasts": _member_points(ensemble, base_latitude, base_longitude, issue), "truth_after_issue": [{"time": row["time_utc"], "lat": row["lat"], "lon": row["lon"]} for row in truth], "score": _score(forecast, truth, issue_row), "persistence_score": case["persistence_score"], "intensity_model": intensity_metadata, "route_diagnostics": {"pacific_weight": PACIFIC_WEIGHT, "local_weight": LOCAL_WEIGHT, "pacific": diagnostics, "local": local_diagnostics, "pressure_state": state_diagnostics, "pressure_systems": systems, "analysis_files": [str(path.relative_to(ROOT)) for path in TIP_FILES], "causal_input_guard": causal_guard}}
    payload = {"storm": "Tip", "model": "v62 whole western-Pacific causal state route + intensity/structure head (25% Pacific domain + 75% local)", "cases": [tip_case], "intensity_model": intensity_metadata, "input_policy": "Strict causal-only mode: the runner rejects any GRIB forecast step or weather valid_time after the issue. Only current/past CFSR analysis/reanalysis files and pre-issue track history are used; the whole 100-190E, 0-60N state is extrapolated from current/t-12/t-24 analysis tendency. Later IBTrACS rows are read only after inference for verification.", "causal_input_guard": causal_guard, "future_rows_used_for_inference": 0, "official_jma_jtwc_forecasts_used": False, "forecast_products_used": [], "forecast_intensity_source": "v62_intensity_structure.py using the frozen v37G spatial structure ensemble, coupled to the causal v62 pressure-map state; current/past analysis only", "other_typhoon_input": "No storm-center forecast or official typhoon track was used. Nearby lows/vortex candidates come from the causal SLP field.", "tip_used_for_training_or_calibration": False}
    _draw_pressure_forecast(TIP_PRESSURE_PNG, state_fields, state_pressure, latitude, longitude, observed, forecast, systems, "Typhoon Tip v62 whole western-Pacific causal pressure state")
    _draw_intensity_chart(TIP_INTENSITY_PNG, forecast, "Typhoon Tip v62 causal intensity and wind structure")
    base_maps.TIP_JSON, base_maps.TIP_HTML, base_maps.TIP_PNG = TIP_JSON, TIP_HTML, TIP_PNG
    base_maps.draw_tip_png(payload, base_maps.load_coastlines())
    TIP_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    page = base_maps.html_shell("Typhoon Tip v62 whole western-Pacific world map", "tip-map", payload, "The red route is v62 output. Later IBTrACS rows are verification truth only; no future weather or official forecast was passed to the route.", "<p><b>Input:</b> Current and past CFSR analysis only. The route uses a Pacific-wide pressure/flow state covering China, Japan, Taiwan, the Philippines, and the western Pacific.</p>", TIP_PNG, "tip")
    page = _append_pressure_section(page, TIP_PRESSURE_PNG.name, payload["input_policy"])
    TIP_HTML.write_text(_append_intensity_section(page, TIP_INTENSITY_PNG.name, intensity_metadata), encoding="utf-8")
    return payload


def main() -> None:
    dolphin = _dolphin()
    tip = _tip()
    print(json.dumps({"dolphin": {"html": str(DOLPHIN_HTML), "pressure_map": str(DOLPHIN_PRESSURE_PNG), "intensity_chart": str(DOLPHIN_INTENSITY_PNG), "final": dolphin["forecast"][-1]}, "tip": {"html": str(TIP_HTML), "pressure_map": str(TIP_PRESSURE_PNG), "intensity_chart": str(TIP_INTENSITY_PNG), "score": tip["cases"][0]["score"], "final": tip["cases"][0]["forecast"][-1]}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
