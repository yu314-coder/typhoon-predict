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
from v62_pacific_domain_route import (  # noqa: E402
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
    fields, pressure, latitude, longitude = _decode_analysis(DOLPHIN_FILES)
    levels = np.load(DOLPHIN_LEVELS, allow_pickle=True)
    base_latitude, base_longitude = float(records[-1]["lat"]), float(records[-1]["lon"])
    members, weights, diagnostics = build_pacific_route(fields, pressure, latitude, longitude, base_latitude, base_longitude, _recent_motion(records, "time"))
    pacific_route = weighted_route(members, weights)
    local_route, _, local_diagnostics = _local_route(levels, base_latitude, base_longitude)
    route = LOCAL_WEIGHT * local_route + PACIFIC_WEIGHT * pacific_route
    ensemble = LOCAL_WEIGHT * local_route[None, :, :] + PACIFIC_WEIGHT * members
    forecast = _route_points(route, base_latitude, base_longitude, issue)
    observed = [{"time": row["time"], "lat": float(row["lat"]), "lon": float(row["lon"])} for row in records]
    state_fields, state_pressure, state_diagnostics = forecast_pacific_state(fields, pressure)
    systems = _system_series(state_pressure, latitude, longitude, forecast, base_latitude, base_longitude)
    payload = {
        "storm": "Dolphin",
        "issue_time_utc": source["issue_time_utc"],
        "model": "v62 whole western-Pacific causal state route (25% Pacific domain + 75% local)",
        "observed": observed,
        "forecast": forecast,
        "ensemble_forecasts": _member_points(ensemble, base_latitude, base_longitude, issue),
        "official": {"jtwc": [base_maps.point(row) for row in source["jtwc_official"]], "jma": [base_maps.point(row) for row in source["jma_official"]]},
        "route_diagnostics": {"pacific_weight": PACIFIC_WEIGHT, "local_weight": LOCAL_WEIGHT, "pacific": diagnostics, "local": local_diagnostics, "pressure_state": state_diagnostics, "pressure_systems": systems, "analysis_files": [str(path.relative_to(ROOT)) for path in DOLPHIN_FILES]},
        "input_policy": "Only GFS f000 analysis files at or before the 2026-07-31 15Z issue were opened. The whole 100-190E, 0-60N state is extrapolated from current/t-12/t-24 analysis tendency; official JMA/JTWC tracks are overlays only.",
        "future_rows_used_for_inference": 0,
        "official_forecasts_used_for_inference": False,
        "forecast_products_used": [],
        "forecast_intensity_source": None,
        "other_typhoon_input": "No storm-center forecast or official typhoon track was used. Nearby lows/vortex candidates come from the causal SLP field.",
    }
    _draw_pressure_forecast(DOLPHIN_PRESSURE_PNG, state_fields, state_pressure, latitude, longitude, observed, forecast, systems, "Dolphin v62 whole western-Pacific causal pressure state")
    base_maps.DOLPHIN_JSON, base_maps.DOLPHIN_HTML, base_maps.DOLPHIN_PNG = DOLPHIN_JSON, DOLPHIN_HTML, DOLPHIN_PNG
    base_maps.draw_dolphin_png(payload, base_maps.load_coastlines())
    DOLPHIN_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    page = base_maps.html_shell("Dolphin v62 whole western-Pacific world map", "dolphin-map", payload, "The red route is v62 output. The Pacific pressure map is a causal analysis-tendency extrapolation, not an imported forecast product.", "<p><b>Input:</b> Current and past GFS analysis only. Broad rings cover the western Pacific from China/Japan through Taiwan, the Philippines, and the open Pacific.</p>", DOLPHIN_PNG, "dolphin")
    DOLPHIN_HTML.write_text(_append_pressure_section(page, DOLPHIN_PRESSURE_PNG.name, payload["input_policy"]), encoding="utf-8")
    return payload


def _tip() -> dict:
    source = json.loads(TIP_SOURCE.read_text(encoding="utf-8"))
    case = source["cases"][0]
    issue = dt.datetime.fromisoformat(case["issue_time_utc"].replace("Z", "+00:00"))
    fields, pressure, latitude, longitude = _decode_analysis(TIP_FILES)
    levels = np.load(TIP_LEVELS, allow_pickle=True)
    issue_row = case["observed_before_issue"][-1]
    base_latitude, base_longitude = float(issue_row["lat"]), float(issue_row["lon"])
    members, weights, diagnostics = build_pacific_route(fields, pressure, latitude, longitude, base_latitude, base_longitude, _recent_motion(case["observed_before_issue"], "time_utc"))
    pacific_route = weighted_route(members, weights)
    local_route, _, local_diagnostics = _local_route(levels, base_latitude, base_longitude)
    route = LOCAL_WEIGHT * local_route + PACIFIC_WEIGHT * pacific_route
    ensemble = LOCAL_WEIGHT * local_route[None, :, :] + PACIFIC_WEIGHT * members
    forecast = _route_points(route, base_latitude, base_longitude, issue)
    observed = [{"time": row["time_utc"], "lat": float(row["lat"]), "lon": float(row["lon"])} for row in case["observed_before_issue"]]
    truth = [{"time_utc": row["time_utc"], "lat": float(row["lat"]), "lon": float(row["lon"])} for row in case["truth_after_issue"]]
    state_fields, state_pressure, state_diagnostics = forecast_pacific_state(fields, pressure)
    systems = _system_series(state_pressure, latitude, longitude, forecast, base_latitude, base_longitude)
    tip_case = {"issue_time_utc": case["issue_time_utc"], "observed_before_issue": observed, "forecast": forecast, "ensemble_forecasts": _member_points(ensemble, base_latitude, base_longitude, issue), "truth_after_issue": [{"time": row["time_utc"], "lat": row["lat"], "lon": row["lon"]} for row in truth], "score": _score(forecast, truth, issue_row), "persistence_score": case["persistence_score"], "route_diagnostics": {"pacific_weight": PACIFIC_WEIGHT, "local_weight": LOCAL_WEIGHT, "pacific": diagnostics, "local": local_diagnostics, "pressure_state": state_diagnostics, "pressure_systems": systems, "analysis_files": [str(path.relative_to(ROOT)) for path in TIP_FILES]}}
    payload = {"storm": "Tip", "model": "v62 whole western-Pacific causal state route (25% Pacific domain + 75% local)", "cases": [tip_case], "input_policy": "Only CFSR analysis/reanalysis files at or before the 1979-10-12 00Z issue were opened. The whole 100-190E, 0-60N state is extrapolated from current/t-12/t-24 analysis tendency; later IBTrACS rows are verification truth only.", "future_rows_used_for_inference": 0, "official_jma_jtwc_forecasts_used": False, "forecast_products_used": [], "forecast_intensity_source": None, "other_typhoon_input": "No storm-center forecast or official typhoon track was used. Nearby lows/vortex candidates come from the causal SLP field.", "tip_used_for_training_or_calibration": False}
    _draw_pressure_forecast(TIP_PRESSURE_PNG, state_fields, state_pressure, latitude, longitude, observed, forecast, systems, "Typhoon Tip v62 whole western-Pacific causal pressure state")
    base_maps.TIP_JSON, base_maps.TIP_HTML, base_maps.TIP_PNG = TIP_JSON, TIP_HTML, TIP_PNG
    base_maps.draw_tip_png(payload, base_maps.load_coastlines())
    TIP_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    page = base_maps.html_shell("Typhoon Tip v62 whole western-Pacific world map", "tip-map", payload, "The red route is v62 output. Later IBTrACS rows are verification truth only; no future weather or official forecast was passed to the route.", "<p><b>Input:</b> Current and past CFSR analysis only. The route uses a Pacific-wide pressure/flow state covering China, Japan, Taiwan, the Philippines, and the western Pacific.</p>", TIP_PNG, "tip")
    TIP_HTML.write_text(_append_pressure_section(page, TIP_PRESSURE_PNG.name, payload["input_policy"]), encoding="utf-8")
    return payload


def main() -> None:
    dolphin = _dolphin()
    tip = _tip()
    print(json.dumps({"dolphin": {"html": str(DOLPHIN_HTML), "pressure_map": str(DOLPHIN_PRESSURE_PNG), "final": dolphin["forecast"][-1]}, "tip": {"html": str(TIP_HTML), "pressure_map": str(TIP_PRESSURE_PNG), "score": tip["cases"][0]["score"]}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
