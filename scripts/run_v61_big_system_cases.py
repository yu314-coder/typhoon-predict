#!/usr/bin/env python3
"""Run and render v61 causal large-system forecasts for Dolphin and Tip.

The route calculation opens only analysis/reanalysis frames at or before each
issue time.  Official tracks are loaded after inference for comparison maps;
they are never passed to v61.  Tip's later IBTrACS rows are scoring truth only.
"""

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

from analysis_level_mean_route import build_level_analysis_mean_route  # noqa: E402
from scripts import draw_dolphin_tip_case_maps as base_maps  # noqa: E402
from v61_big_system_route import (  # noqa: E402
    ROUTE_VARIANTS,
    build_route,
    integrate_from_issue,
    weighted_route,
)


BLEND_LOCAL_WEIGHT = 0.75
DOLPHIN_SOURCE = ROOT / "track_build" / "dolphin_v37p_v37n_same_issue_current.json"
DOLPHIN_LEVELS = ROOT / "track_build" / "dolphin_analysis_causal.npz"
DOLPHIN_FIELD_ROOT = ROOT / "v37" / "current_gfs_structure"
DOLPHIN_FILES = (
    DOLPHIN_FIELD_ROOT / "gfs_20260731_12_full.grib2",
    DOLPHIN_FIELD_ROOT / "gfs_20260730_12_full.grib2",
    DOLPHIN_FIELD_ROOT / "gfs_20260729_12_full.grib2",
)
DOLPHIN_JSON = ROOT / "track_build" / "dolphin_v61_big_system_case_map.json"
DOLPHIN_HTML = ROOT / "paper" / "dolphin_v61_big_system_world_map.html"
DOLPHIN_PNG = ROOT / "paper" / "dolphin_v61_big_system_world_map.png"
DOLPHIN_SYNOPTIC = ROOT / "paper" / "dolphin_v61_big_system_synoptic.png"

TIP_SOURCE = ROOT / "track_build" / "tip_v37N_multi_timing_1979.json"
TIP_LEVELS = ROOT / "v37_cfsr" / "tip_1979_causal" / "cfsr_tip_19791012_levels.npz"
TIP_FIELD_ROOT = ROOT / "v37_cfsr" / "tip_1979_causal" / "raw"
TIP_FILES = (
    TIP_FIELD_ROOT / "pgbhnl.gdas.1979101200.grb2",
    TIP_FIELD_ROOT / "pgbhnl.gdas.1979101100.grb2",
    TIP_FIELD_ROOT / "pgbhnl.gdas.1979101000.grb2",
)
TIP_JSON = ROOT / "track_build" / "tip_v61_big_system_case_map.json"
TIP_HTML = ROOT / "paper" / "tip_v61_big_system_world_map.html"
TIP_PNG = ROOT / "paper" / "tip_v61_big_system_world_map.png"
TIP_SYNOPTIC = ROOT / "paper" / "tip_v61_big_system_synoptic.png"


def _read_grib(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode hgt500, pressure-level winds, and MSLP from one analysis GRIB."""

    def read(short_name: str, type_of_level: str, levels: tuple[float, ...] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": type_of_level, "shortName": short_name},
                "indexpath": "",
            },
        )
        try:
            dataset = dataset.sortby("latitude")
            data = dataset[short_name]
            if levels is not None:
                data = data.sel(isobaricInhPa=list(levels))
            return (
                np.asarray(data.load().values, dtype="float32"),
                np.asarray(dataset.latitude.values, dtype="float32"),
                np.asarray(dataset.longitude.values, dtype="float32"),
            )
        finally:
            dataset.close()

    u, latitude, longitude = read("u", "isobaricInhPa", (850.0, 500.0, 200.0))
    v, _, _ = read("v", "isobaricInhPa", (850.0, 500.0, 200.0))
    height, _, _ = read("gh", "isobaricInhPa", (500.0,))
    pressure, _, _ = read("prmsl", "meanSea")
    fields = np.stack(
        [height[0], u[0], v[0], u[1], v[1], u[2], v[2]],
        axis=0,
    ).astype("float32")
    return fields, pressure.astype("float32") / 100.0, latitude, longitude, u.astype("float32")


def _decode_analysis(files: tuple[Path, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    pressures = []
    latitude = longitude = None
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        field, pressure, frame_latitude, frame_longitude, _ = _read_grib(path)
        if latitude is None:
            latitude, longitude = frame_latitude, frame_longitude
        elif not (np.array_equal(latitude, frame_latitude) and np.array_equal(longitude, frame_longitude)):
            raise RuntimeError(f"analysis grid changed in {path}")
        frames.append(field)
        pressures.append(pressure)
    return np.stack(frames).astype("float32"), np.stack(pressures).astype("float32"), latitude, longitude


def _local_route(level_archive: np.lib.npyio.NpzFile, base_latitude: float, base_longitude: float) -> tuple[np.ndarray, np.ndarray, dict]:
    current = np.asarray(level_archive["current_levels"], dtype="float32")
    history = np.asarray(level_archive["history_levels"], dtype="float32")
    available = np.asarray(level_archive["available"], dtype="float32")
    states, weights, metadata = build_level_analysis_mean_route(
        current,
        history,
        available,
        base_latitude,
        base_longitude,
    )
    return weighted_route(states, weights), states, metadata


def _route_points(route: np.ndarray, base_latitude: float, base_longitude: float, issue_time: dt.datetime, intensity: list[dict] | None = None) -> list[dict]:
    latitude, longitude = float(base_latitude), float(base_longitude)
    points = []
    for index, displacement in enumerate(np.asarray(route, dtype="float32")):
        latitude += float(displacement[1]) / 111.2
        longitude += float(displacement[0]) / (111.2 * max(math.cos(math.radians(latitude)), 0.20))
        point = {
            "lead_hours": (index + 1) * 6,
            "valid_time_utc": (issue_time + dt.timedelta(hours=(index + 1) * 6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "latitude": round(latitude, 4),
            "longitude": round(longitude % 360.0, 4),
            # Keep both schemas: the synoptic renderer uses latitude/
            # longitude, while the shared world-map renderer uses lat/lon.
            "lat": round(latitude, 4),
            "lon": round(longitude % 360.0, 4),
        }
        if intensity and index < len(intensity):
            for key in (
                "vmax_kt", "vmax_spread_kt", "central_pressure_hpa", "pressure_spread_hpa",
                "pressure_hpa", "rmw_km", "rmw_spread_km", "wind_radii_km", "wind_radii_spread_km",
            ):
                if key in intensity[index]:
                    point[key] = intensity[index][key]
        points.append(point)
    return points


def _ensemble_points(members: np.ndarray, base_latitude: float, base_longitude: float, issue_time: dt.datetime) -> list[list[dict]]:
    paths = []
    for path in integrate_from_issue(members, base_latitude, base_longitude):
        paths.append([
            {
                "lead_hours": int(item["lead_hours"]),
                "valid_time_utc": (issue_time + dt.timedelta(hours=int(item["lead_hours"]))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "lat": float(item["latitude"]),
                "lon": float(item["longitude"]),
            }
            for item in path
        ])
    return paths


def _recent_motion(records: list[dict], time_key: str) -> tuple[float, float] | None:
    """Estimate current storm motion from observations at or before issue."""

    if len(records) < 2:
        return None
    rows = records[-5:]
    samples = []
    for previous, current in zip(rows, rows[1:]):
        previous_time = dt.datetime.fromisoformat(str(previous[time_key]).replace("Z", "+00:00"))
        current_time = dt.datetime.fromisoformat(str(current[time_key]).replace("Z", "+00:00"))
        hours = (current_time - previous_time).total_seconds() / 3600.0
        if hours <= 0.0:
            continue
        delta_lon = ((float(current["lon"]) - float(previous["lon"]) + 180.0) % 360.0) - 180.0
        mean_latitude = math.radians(0.5 * (float(current["lat"]) + float(previous["lat"])))
        east_km = delta_lon * 111.2 * math.cos(mean_latitude)
        north_km = (float(current["lat"]) - float(previous["lat"])) * 111.2
        samples.append((east_km * 6.0 / hours, north_km * 6.0 / hours))
    if not samples:
        return None
    weights = np.arange(1.0, len(samples) + 1.0, dtype="float64")
    values = np.asarray(samples, dtype="float64")
    result = (values * weights[:, None]).sum(axis=0) / weights.sum()
    return float(result[0]), float(result[1])


def _score(forecast: list[dict], truth: list[dict], issue: dict) -> dict:
    def distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
        radius = 6371.0
        lat_a, lat_b = math.radians(lat_a), math.radians(lat_b)
        delta_lat = lat_b - lat_a
        delta_lon = math.radians(lon_b - lon_a)
        value = math.sin(delta_lat / 2.0) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
        return 2.0 * radius * math.asin(math.sqrt(max(0.0, min(1.0, value))))

    truth_by_time = {row.get("time_utc"): row for row in truth}
    rows = []
    for point in forecast:
        actual = truth_by_time.get(point["valid_time_utc"])
        if actual is None:
            continue
        rows.append({
            "lead_hours": point["lead_hours"],
            "track_error_km": round(float(distance_km(point["latitude"], point["longitude"], actual["lat"], actual["lon"])), 3),
            "forecast_latitude": point["latitude"],
            "forecast_longitude": point["longitude"],
            "truth_latitude": round(float(actual["lat"]), 4),
            "truth_longitude": round(float(actual["lon"]), 4),
        })
    errors = np.asarray([row["track_error_km"] for row in rows], dtype="float64")
    persistence = np.asarray([
        distance_km(issue["lat"], issue["lon"], row["lat"], row["lon"])
        for row in truth
    ], dtype="float64")
    return {
        "matched_leads": int(len(rows)),
        "track_mae_km": round(float(errors.mean()), 3),
        "track_error_120h_km": next(row["track_error_km"] for row in rows if row["lead_hours"] == 120),
        "persistence_mae_km": round(float(persistence.mean()), 3),
        "persistence_error_120h_km": round(float(persistence[-1]), 3),
        "by_lead": rows,
    }


def _draw_synoptic(path: Path, fields: np.ndarray, pressure: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, observed: list[dict], forecast: list[dict], baseline: list[dict], title: str) -> None:
    lon_min, lon_max = 105.0, 190.0
    lat_min, lat_max = 0.0, 55.0
    lat_mask = (latitude >= lat_min) & (latitude <= lat_max)
    lon_mask = (longitude >= lon_min) & (longitude <= lon_max)
    if not lat_mask.any() or not lon_mask.any():
        raise RuntimeError("synoptic plot domain is outside analysis grid")
    lat = latitude[lat_mask]
    lon = longitude[lon_mask]
    hgt = fields[0, 0][np.ix_(lat_mask, lon_mask)]
    u = fields[0, 1][np.ix_(lat_mask, lon_mask)]
    v = fields[0, 2][np.ix_(lat_mask, lon_mask)]
    slp = pressure[0][np.ix_(lat_mask, lon_mask)]
    fig, axis = plt.subplots(figsize=(14.5, 8.8), constrained_layout=True)
    slp_levels = np.arange(math.floor(float(np.nanpercentile(slp, 2.0)) / 4.0) * 4.0, math.ceil(float(np.nanpercentile(slp, 98.0)) / 4.0) * 4.0 + 4.0, 4.0)
    filled = axis.contourf(lon, lat, slp, levels=slp_levels, cmap="Spectral_r", alpha=0.82, extend="both")
    contours = axis.contour(lon, lat, slp, levels=slp_levels, colors="#17202a", linewidths=0.5)
    axis.clabel(contours, inline=True, fontsize=7, fmt="%1.0f")
    height_contours = axis.contour(lon, lat, hgt, levels=np.arange(math.floor(float(np.nanpercentile(hgt, 5.0)) / 40.0) * 40.0, math.ceil(float(np.nanpercentile(hgt, 95.0)) / 40.0) * 40.0 + 40.0, 40.0), colors="#7e22ce", linewidths=0.75, alpha=0.70)
    axis.clabel(height_contours, inline=True, fontsize=7, fmt="%1.0f")
    stride = max(1, len(lon) // 35)
    axis.barbs(lon[::stride], lat[::stride], u[::stride, ::stride] * 1.94384, v[::stride, ::stride] * 1.94384, length=5, linewidth=0.35, color="#1d4ed8", alpha=0.72)
    if observed:
        axis.plot([p["lon"] for p in observed], [p["lat"] for p in observed], color="#111827", linewidth=2.1, label="observed input")
    if baseline:
        axis.plot([p["longitude"] for p in baseline], [p["latitude"] for p in baseline], color="#64748b", linestyle="--", linewidth=1.8, label="local analysis baseline")
    axis.plot([forecast[0].get("longitude", forecast[0].get("lon"))] + [p["longitude"] for p in forecast], [forecast[0].get("latitude", forecast[0].get("lat"))] + [p["latitude"] for p in forecast], color="#b91c1c", linewidth=3.0, label="v61 dynamic causal route")
    axis.set_xlim(lon_min, lon_max)
    axis.set_ylim(lat_min, lat_max)
    axis.set_xlabel("Longitude (E)")
    axis.set_ylabel("Latitude (N)")
    axis.grid(color="#94a3b8", linewidth=0.35, alpha=0.5)
    axis.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(filled, ax=axis, pad=0.02, label="Mean sea-level pressure (hPa)")
    axis.legend(loc="lower left", fontsize=8)
    fig.text(0.5, 0.012, "Causal analysis at issue time: MSLP contours, 500-hPa height contours, and 850-hPa wind barbs. No positive-lead field is used.", ha="center", fontsize=8.5, color="#334155")
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def _append_synoptic_section(page: str, image_name: str, policy: str) -> str:
    section = f'<section><h2>Large-system analysis used by v61</h2><p>{html.escape(policy)}</p><img src="{html.escape(image_name)}" alt="Causal synoptic analysis background and v61 route"></section>'
    return page.replace("</main>", section + "</main>")


def _dolphin() -> dict:
    source = json.loads(DOLPHIN_SOURCE.read_text(encoding="utf-8"))
    records = source["current_track_history"]
    issue = dt.datetime.fromisoformat(source["issue_time_utc"].replace("Z", "+00:00"))
    fields, pressure, latitude, longitude = _decode_analysis(DOLPHIN_FILES)
    level_archive = np.load(DOLPHIN_LEVELS, allow_pickle=True)
    base_latitude = float(records[-1]["lat"])
    base_longitude = float(records[-1]["lon"])
    global_members, global_weights, global_metadata = build_route(
        fields,
        latitude,
        longitude,
        base_latitude,
        base_longitude,
        pressure,
        available=(1.0, 1.0),
        history_motion_km_per_6h=_recent_motion(records, "time"),
    )
    global_route = weighted_route(global_members, global_weights)
    local_route, local_members, local_metadata = _local_route(level_archive, base_latitude, base_longitude)
    route = BLEND_LOCAL_WEIGHT * local_route + (1.0 - BLEND_LOCAL_WEIGHT) * global_route
    ensemble_members = BLEND_LOCAL_WEIGHT * local_route[None, :, :] + (1.0 - BLEND_LOCAL_WEIGHT) * global_members
    # Position is the only forecast product in this case adapter.  Do not
    # attach intensity/radius values from an older route or official product.
    forecast = _route_points(route, base_latitude, base_longitude, issue)
    baseline = _route_points(local_route, base_latitude, base_longitude, issue)
    observed = [{"time": p["time"], "lat": float(p["lat"]), "lon": float(p["lon"])} for p in records]
    payload = {
        "storm": "Dolphin",
        "issue_time_utc": source["issue_time_utc"],
        "model": "v61 dynamic causal big-system route (75% local multi-level + 25% evolving global synoptic)",
        "observed": observed,
        "forecast": forecast,
        "ensemble_forecasts": _ensemble_points(ensemble_members, base_latitude, base_longitude, issue),
        "official": {
            "jtwc": [base_maps.point(p) for p in source["jtwc_official"]],
            "jma": [base_maps.point(p) for p in source["jma_official"]],
        },
        "baseline_route": baseline,
        "route_diagnostics": {
            "blend_local_weight": BLEND_LOCAL_WEIGHT,
            "global_member_count": len(global_members),
            "global_route_metadata": global_metadata,
            "local_route_metadata": local_metadata,
            "analysis_files": [str(path.relative_to(ROOT)) for path in DOLPHIN_FILES],
        },
        "input_policy": "Only GFS f000 analysis files at 12Z on 2026-07-29, 2026-07-30, and 2026-07-31 were opened. The route evolves the current field only with the observed past-analysis tendency and recent observed motion. Official JMA/JTWC tracks are comparison overlays only; no forecast-derived intensity or radius values are attached.",
        "future_rows_used_for_inference": 0,
        "official_forecasts_used_for_inference": False,
        "forecast_products_used": [],
        "forecast_intensity_source": None,
        "gate_note": "No quality gate or official forecast was used. v61 is a causal dynamic physical large-system route.",
    }
    base_maps.DOLPHIN_JSON = DOLPHIN_JSON
    base_maps.DOLPHIN_HTML = DOLPHIN_HTML
    base_maps.DOLPHIN_PNG = DOLPHIN_PNG
    coastlines = base_maps.load_coastlines()
    base_maps.draw_dolphin_png(payload, coastlines)
    _draw_synoptic(DOLPHIN_SYNOPTIC, fields, pressure, latitude, longitude, observed, forecast, baseline, "Dolphin v61 causal big-system analysis")
    DOLPHIN_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    page = base_maps.html_shell(
        "Dolphin v61 causal big-system world map",
        "dolphin-map",
        payload,
        "The red route is v61 dynamic output. It uses only current/past analysis fields and observed short-term motion; official JMA/JTWC lines are comparison overlays and were not inputs.",
        "<p><b>Model:</b> v61 dynamic causal large-system route. The route blends a local multi-level analysis ensemble with an evolving global synoptic ensemble driven by past-analysis tendency, broad 850/500/200-hPa flow, and causal SLP-gradient steering.</p>",
        DOLPHIN_PNG,
        "dolphin",
    )
    DOLPHIN_HTML.write_text(_append_synoptic_section(page, DOLPHIN_SYNOPTIC.name, payload["input_policy"]), encoding="utf-8")
    return payload


def _tip() -> dict:
    source = json.loads(TIP_SOURCE.read_text(encoding="utf-8"))
    case = source["cases"][0]
    issue = dt.datetime.fromisoformat(case["issue_time_utc"].replace("Z", "+00:00"))
    fields, pressure, latitude, longitude = _decode_analysis(TIP_FILES)
    level_archive = np.load(TIP_LEVELS, allow_pickle=True)
    issue_row = case["observed_before_issue"][-1]
    base_latitude = float(issue_row["lat"])
    base_longitude = float(issue_row["lon"])
    global_members, global_weights, global_metadata = build_route(
        fields,
        latitude,
        longitude,
        base_latitude,
        base_longitude,
        pressure,
        available=(1.0, 1.0),
        history_motion_km_per_6h=_recent_motion(case["observed_before_issue"], "time_utc"),
    )
    global_route = weighted_route(global_members, global_weights)
    local_route, local_members, local_metadata = _local_route(level_archive, base_latitude, base_longitude)
    route = BLEND_LOCAL_WEIGHT * local_route + (1.0 - BLEND_LOCAL_WEIGHT) * global_route
    ensemble_members = BLEND_LOCAL_WEIGHT * local_route[None, :, :] + (1.0 - BLEND_LOCAL_WEIGHT) * global_members
    # Score only the position route.  Later truth rows and prior model routes
    # must not be used to populate forecast wind, pressure, or radius fields.
    forecast = _route_points(route, base_latitude, base_longitude, issue)
    baseline = _route_points(local_route, base_latitude, base_longitude, issue)
    observed = [{"time": p["time_utc"], "lat": float(p["lat"]), "lon": float(p["lon"])} for p in case["observed_before_issue"]]
    truth = [{"time_utc": p["time_utc"], "lat": float(p["lat"]), "lon": float(p["lon"])} for p in case["truth_after_issue"]]
    score = _score(forecast, [{"time_utc": p["time_utc"], "lat": p["lat"], "lon": p["lon"]} for p in truth], issue_row)
    tip_case = {
        "issue_time_utc": case["issue_time_utc"],
        "observed_before_issue": observed,
        "forecast": forecast,
        "ensemble_forecasts": _ensemble_points(ensemble_members, base_latitude, base_longitude, issue),
        "truth_after_issue": [{"time": p["time_utc"], "lat": p["lat"], "lon": p["lon"]} for p in truth],
        "score": score,
        "persistence_score": case["persistence_score"],
        "baseline_route": baseline,
        "route_diagnostics": {
            "blend_local_weight": BLEND_LOCAL_WEIGHT,
            "global_member_count": len(global_members),
            "global_route_metadata": global_metadata,
            "local_route_metadata": local_metadata,
            "analysis_files": [str(path.relative_to(ROOT)) for path in TIP_FILES],
        },
    }
    payload = {
        "storm": "Tip",
        "model": "v61 dynamic causal big-system route (75% local multi-level + 25% evolving global synoptic)",
        "cases": [tip_case],
        "input_policy": "Only CFSR analysis/reanalysis files through the 1979-10-12 00Z issue were opened. The route evolves the current field only with the observed past-analysis tendency and recent observed motion. Later IBTrACS rows are verification truth only; no official forecast product or prior model forecast was used, and no forecast-derived intensity or radius values are attached.",
        "future_rows_used_for_inference": 0,
        "official_jma_jtwc_forecasts_used": False,
        "forecast_products_used": [],
        "tip_used_for_training_or_calibration": False,
        "forecast_intensity_source": None,
        "gate_note": "No quality gate or official forecast was used. v61 is a causal dynamic physical large-system route.",
    }
    base_maps.TIP_JSON = TIP_JSON
    base_maps.TIP_HTML = TIP_HTML
    base_maps.TIP_PNG = TIP_PNG
    coastlines = base_maps.load_coastlines()
    base_maps.draw_tip_png(payload, coastlines)
    _draw_synoptic(TIP_SYNOPTIC, fields, pressure, latitude, longitude, observed, forecast, baseline, "Typhoon Tip v61 causal big-system analysis")
    TIP_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    page = base_maps.html_shell(
        "Typhoon Tip v61 causal big-system world map",
        "tip-map",
        payload,
        "The colored route is v61 dynamic output. Later IBTrACS rows are shown only as verification truth; no future weather or official forecast was passed to the model.",
        base_maps.tip_summary(payload),
        TIP_PNG,
        "tip",
    )
    TIP_HTML.write_text(_append_synoptic_section(page, TIP_SYNOPTIC.name, payload["input_policy"]), encoding="utf-8")
    return payload


def main() -> None:
    dolphin = _dolphin()
    tip = _tip()
    print(json.dumps({
        "dolphin": {"html": str(DOLPHIN_HTML), "synoptic": str(DOLPHIN_SYNOPTIC), "final": dolphin["forecast"][-1]},
        "tip": {"html": str(TIP_HTML), "synoptic": str(TIP_SYNOPTIC), "score": tip["cases"][0]["score"]},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
