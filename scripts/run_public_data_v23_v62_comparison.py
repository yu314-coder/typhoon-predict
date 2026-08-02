#!/usr/bin/env python3
"""Compare v23 and v62 on the same public, causal analysis inputs.

The comparison deliberately separates three routes:

* v23: the released TrackFormer checkpoint, using the same 17x17 analysis
  patch and track history used by the current case;
* v62 Pacific-only: the broad 100-190E, 0-60N route using the full public
  analysis grid;
* v62 full: the current v62 local-plus-Pacific blend.

For Dolphin the analysis source is NOAA NOMADS GFS f000 analysis.  For the
historical Tip replay it is NOAA CFSR analysis.  No positive-lead weather
field, official forecast track, or post-issue observation is passed to any
route.  v23 does not generate a full pressure grid, so the pressure panels
show v62's causal pressure-state extrapolation with all three route lines.
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
import numpy as np
import torch
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hf-space"))
sys.path.insert(0, str(ROOT))

import trackformer_v23 as v23_module  # noqa: E402
from trackformer_v23 import TARGET_SCALE, build_v23  # noqa: E402
from scripts.predict_ibtracs_jma_only import (  # noqa: E402
    build_track_window,
    load_land_points,
)
from scripts.run_v61_big_system_cases import (  # noqa: E402
    DOLPHIN_FILES,
    DOLPHIN_LEVELS,
    _decode_analysis,
    _local_route,
    _recent_motion,
    _route_points,
    TIP_FILES,
    TIP_LEVELS,
)
from scripts.run_v62_pacific_domain_cases import (  # noqa: E402
    DOLPHIN_JSON as DOLPHIN_V62_JSON,
    DOLPHIN_PRESSURE_PNG as DOLPHIN_V62_PRESSURE,
    TIP_JSON as TIP_V62_JSON,
    TIP_PRESSURE_PNG as TIP_V62_PRESSURE,
    _draw_pressure_forecast,
    _system_series,
)
from v61_big_system_route import weighted_route  # noqa: E402
from v62_pacific_domain_route import (  # noqa: E402
    PACIFIC_LAT_RANGE,
    PACIFIC_LON_RANGE,
    build_pacific_route,
    forecast_pacific_state,
)


CHECKPOINT_ROOT = ROOT / "models" / "v23"
STATS_PATH = ROOT / "v37" / "v23_norm_stats.npz"
LAND_LAT, LAND_LON = load_land_points()
LOCAL_WEIGHT = 0.75
PACIFIC_WEIGHT = 0.25
REQUIRED_TRACK_FIELDS = (
    "vmax_kt",
    "pressure_hpa",
    "rmw_nm",
    "roci_nm",
    "r34_ne_nm",
    "r34_se_nm",
    "r34_sw_nm",
    "r34_nw_nm",
    "r50_ne_nm",
    "r50_se_nm",
    "r50_sw_nm",
    "r50_nw_nm",
    "r64_ne_nm",
    "r64_se_nm",
    "r64_sw_nm",
    "r64_nw_nm",
    "dist2land_km",
)


def _canonical_track_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        canonical = dict(row)
        canonical["time_utc"] = canonical.get("time_utc", canonical.get("time"))
        if not canonical["time_utc"]:
            raise ValueError(f"track row has no time: {row!r}")
        for key in REQUIRED_TRACK_FIELDS:
            canonical.setdefault(key, float("nan"))
        result.append(canonical)
    return result


def _load_v23_models() -> list[torch.nn.Module]:
    # v23's module-level annulus tensor is moved by other local forecast
    # modules on import. Keep this reproducible comparison on CPU.
    v23_module.ANN = v23_module.ANN.cpu()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    models = []
    for checkpoint in sorted(CHECKPOINT_ROOT.glob("v23_seed*.pt")):
        model = build_v23().cpu().eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(state)
        models.append(model)
    if not models:
        raise FileNotFoundError(f"no v23 checkpoints under {CHECKPOINT_ROOT}")
    return models


def _route_points_from_v23(displacement: np.ndarray, rows: list[dict]) -> list[dict]:
    issue_time = dt.datetime.fromisoformat(rows[-1]["time_utc"].replace("Z", "+00:00"))
    base_latitude = float(rows[-1]["lat"])
    base_longitude = float(rows[-1]["lon"]) % 360.0
    points = []
    for index, (east_km, north_km) in enumerate(np.cumsum(displacement, axis=0), start=1):
        latitude = base_latitude + float(north_km) / 111.2
        longitude = base_longitude + float(east_km) / (111.2 * max(math.cos(math.radians(latitude)), 0.20))
        longitude %= 360.0
        lead = index * 6
        points.append(
            {
                "lead_hours": lead,
                "valid_time_utc": (issue_time + dt.timedelta(hours=lead)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "lat": round(latitude, 4),
                "lon": round(longitude, 4),
            }
        )
    return points


def _run_v23(rows: list[dict], current: np.ndarray, history: np.ndarray, available: np.ndarray, models: list[torch.nn.Module]) -> dict:
    stats = np.load(STATS_PATH, allow_pickle=True)
    normalized, velocity_pair, _ = build_track_window(
        _canonical_track_rows(rows),
        stats["tmean"].astype("float32"),
        stats["tstd"].astype("float32"),
        LAND_LAT,
        LAND_LON,
    )
    track = torch.from_numpy(normalized[None]).cpu()
    vpair = torch.from_numpy(velocity_pair[None]).cpu()
    current_tensor = torch.from_numpy(np.asarray(current, dtype="float32")[None]).cpu()
    history_tensor = torch.from_numpy(np.asarray(history, dtype="float32")[None]).cpu()
    available_tensor = torch.from_numpy(np.asarray(available, dtype="float32")[None]).cpu()
    outputs = []
    with torch.no_grad():
        for model in models:
            output = model(track, vpair, current_tensor, history_tensor, available_tensor)[0]
            outputs.append((output[0, :, :2] * TARGET_SCALE[:2]).numpy())
    displacement = np.mean(np.stack(outputs, axis=0), axis=0).astype("float32")
    return {
        "label": "v23 same-source 17x17 patch",
        "model": "Released TrackFormer v23 10-seed ensemble",
        "forecast": _route_points_from_v23(displacement, _canonical_track_rows(rows)),
        "seed_count": len(models),
        "input_shape": {"current": list(np.asarray(current).shape), "history": list(np.asarray(history).shape)},
        "input_policy": "Same current/past analysis fields as v62, reduced to v23's four-channel 17x17 patch.",
    }


def _tip_v23_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = ROOT / "v37_cfsr" / "tip_1979_causal" / "cfsr_tip_19791012.npz"
    scale_path = ROOT / "track_build" / "dlm4_int8.npz"
    cache = np.load(cache_path, allow_pickle=True)
    slp = np.asarray(cache["slp_hpa"], dtype="float32")
    u_dlm = np.asarray(cache["u_dlm"], dtype="float32")
    v_dlm = np.asarray(cache["v_dlm"], dtype="float32")
    scale = np.load(scale_path, allow_pickle=True)["scale"].astype("float32")

    def frame(index: int) -> np.ndarray:
        raw = np.stack(
            [slp[index] - float(slp[index].mean()), slp[index] - slp[index - 4], u_dlm[index], v_dlm[index]]
        ).astype("float32")
        return np.clip(raw / scale[:, None, None], -4.0, 4.0).astype("float32")

    current = frame(8)
    history = np.concatenate([frame(6)[None], frame(4)[None]], axis=0).reshape(8, 17, 17)
    return current, history, np.ones((2,), dtype="float32")


def _v62_routes(
    files: tuple[Path, ...],
    levels_path: Path,
    rows: list[dict],
    time_key: str,
    issue_time: dt.datetime,
) -> dict:
    fields, pressure, latitude, longitude = _decode_analysis(files)
    base_latitude, base_longitude = float(rows[-1]["lat"]), float(rows[-1]["lon"])
    members, weights, diagnostics = build_pacific_route(
        fields,
        pressure,
        latitude,
        longitude,
        base_latitude,
        base_longitude,
        _recent_motion(rows, time_key),
    )
    pacific_route = weighted_route(members, weights)
    levels = np.load(levels_path, allow_pickle=True)
    local_route, _, local_diagnostics = _local_route(levels, base_latitude, base_longitude)
    full_route = LOCAL_WEIGHT * local_route + PACIFIC_WEIGHT * pacific_route
    pacific_forecast = _route_points(pacific_route, base_latitude, base_longitude, issue_time)
    full_forecast = _route_points(full_route, base_latitude, base_longitude, issue_time)
    state_fields, state_pressure, state_diagnostics = forecast_pacific_state(fields, pressure)
    observed = [
        {"time": row[time_key], "lat": float(row["lat"]), "lon": float(row["lon"])}
        for row in rows
    ]
    systems = _system_series(
        state_pressure,
        latitude,
        longitude,
        full_forecast,
        base_latitude,
        base_longitude,
    )
    return {
        "fields": fields,
        "pressure": pressure,
        "latitude": latitude,
        "longitude": longitude,
        "state_fields": state_fields,
        "state_pressure": state_pressure,
        "systems": systems,
        "observed": observed,
        "pacific": {
            "label": "v62 Pacific-domain only",
            "model": "v62 broad 100-190E, 0-60N causal route",
            "forecast": pacific_forecast,
            "route_diagnostics": diagnostics,
        },
        "full": {
            "label": "v62 full local + Pacific",
            "model": "Latest v62 75% local multi-level + 25% Pacific-domain route",
            "forecast": full_forecast,
            "route_diagnostics": {
                "local_weight": LOCAL_WEIGHT,
                "pacific_weight": PACIFIC_WEIGHT,
                "local": local_diagnostics,
                "pacific": diagnostics,
                "pressure_state": state_diagnostics,
            },
        },
    }


def _distance_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    mean_lat = math.radians((a_lat + b_lat) * 0.5)
    east = ((a_lon - b_lon + 180.0) % 360.0) - 180.0
    return math.hypot(east * 111.2 * math.cos(mean_lat), (a_lat - b_lat) * 111.2)


def _score(route: list[dict], truth: list[dict], issue: dict) -> dict | None:
    truth_by_time = {row["time_utc"]: row for row in truth}
    rows = []
    for point in route:
        actual = truth_by_time.get(point["valid_time_utc"])
        if actual is None:
            continue
        rows.append(
            {
                "lead_hours": point["lead_hours"],
                "track_error_km": round(
                    _distance_km(point["latitude"], point["longitude"], float(actual["lat"]), float(actual["lon"])),
                    3,
                ),
                "truth_latitude": float(actual["lat"]),
                "truth_longitude": float(actual["lon"]),
            }
        )
    if not rows:
        return None
    errors = np.asarray([row["track_error_km"] for row in rows], dtype="float64")
    persistence = [
        _distance_km(float(issue["lat"]), float(issue["lon"]), float(row["lat"]), float(row["lon"]))
        for row in truth
        if row["time_utc"] in truth_by_time
    ]
    return {
        "matched_leads": len(rows),
        "track_mae_km": round(float(errors.mean()), 3),
        "track_error_120h_km": next((row["track_error_km"] for row in rows if row["lead_hours"] == 120), None),
        "persistence_mae_km": round(float(np.mean(persistence)), 3) if persistence else None,
        "by_lead": rows,
    }


def _draw_coastlines(axis: plt.Axes) -> None:
    path = ROOT / "paper" / "ne_110m_admin_0_countries.geojson"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates", [])
        if geometry.get("type") == "Polygon":
            rings = [ring for ring in coordinates]
        elif geometry.get("type") == "MultiPolygon":
            rings = [ring for polygon in coordinates for ring in polygon]
        else:
            rings = []
        for ring in rings:
            if len(ring) < 2:
                continue
            values = np.asarray(ring, dtype="float32")
            axis.fill(values[:, 0], values[:, 1], color="#eef1f4", zorder=0)
            axis.plot(values[:, 0], values[:, 1], color="#94a3b8", linewidth=0.25, zorder=1)


def _draw_world_map(path: Path, payload: dict) -> None:
    fig, axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    _draw_coastlines(axis)
    observed = payload["observed"]
    axis.plot(
        [point["lon"] for point in observed],
        [point["lat"] for point in observed],
        color="#111827",
        linewidth=2.4,
        marker="o",
        markersize=3.4,
        label="observed input",
    )
    if payload["truth"]:
        truth = payload["truth"]
        axis.plot(
            [observed[-1]["lon"]] + [point["lon"] for point in truth],
            [observed[-1]["lat"]] + [point["lat"] for point in truth],
            color="#047857",
            linestyle="--",
            linewidth=2.0,
            label="post-issue truth (verification only)",
        )
    routes = payload["routes"]
    styles = {
        "v23": ("#2563eb", "--", "v23 same-source patch"),
        "v62_pacific": ("#f59e0b", ":", "v62 Pacific-domain only"),
        "v62_full": ("#b91c1c", "-", "v62 full local + Pacific"),
    }
    for key, (color, linestyle, label) in styles.items():
        route = routes[key]["forecast"]
        axis.plot(
            [observed[-1]["lon"]] + [point["lon"] for point in route],
            [observed[-1]["lat"]] + [point["lat"] for point in route],
            color=color,
            linestyle=linestyle,
            linewidth=2.8 if key == "v62_full" else 2.0,
            marker="o",
            markersize=2.5,
            label=label,
            zorder=5,
        )
        for point in route[3::4]:
            axis.annotate(
                f"+{point['lead_hours']}h",
                (point["lon"], point["lat"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
                color=color,
                zorder=6,
            )
    axis.set_xlim(*PACIFIC_LON_RANGE)
    axis.set_ylim(*PACIFIC_LAT_RANGE)
    axis.set_xticks(range(100, 191, 10))
    axis.set_yticks(range(0, 61, 10))
    axis.set_xlabel("Longitude (E)")
    axis.set_ylabel("Latitude (N)")
    axis.grid(color="#94a3b8", linewidth=0.35, alpha=0.45)
    axis.legend(loc="lower left", fontsize=8, framealpha=0.9)
    axis.set_title(payload["title"], fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.006,
        "All routes use only current/past public analysis data. Official forecast products are not model inputs.",
        ha="center",
        fontsize=8.5,
        color="#334155",
    )
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def _html_page(path: Path, payload: dict, pressure_image: Path) -> None:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    rows = []
    for key, label in (("v23", "v23 same-source patch"), ("v62_pacific", "v62 Pacific-domain only"), ("v62_full", "v62 full local + Pacific")):
        score = payload["scores"].get(key)
        if score is None:
            text = "No post-issue truth available at this live issue time"
        else:
            text = f"MAE {score['track_mae_km']:.1f} km; +120 h {score['track_error_120h_km']:.1f} km"
        final = payload["routes"][key]["forecast"][-1]
        rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{final['latitude']:.2f}N, {final['longitude']:.2f}E</td><td>{html.escape(text)}</td></tr>"
        )
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(payload['title'])}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
<style>
:root {{ color-scheme:light; --ink:#172033; --muted:#64748b; --line:#d7dde5; }}
body {{ margin:0; font-family:system-ui,-apple-system,sans-serif; color:var(--ink); background:#f8fafc; }}
main {{ max-width:1300px; margin:0 auto; padding:18px; }}
h1 {{ margin:0 0 6px; font-size:24px; }} h2 {{ margin:20px 0 8px; font-size:18px; }}
p {{ line-height:1.45; }} .sub {{ color:var(--muted); margin-top:0; }}
.notice {{ border-left:4px solid #b91c1c; background:#fff7ed; padding:10px 12px; }}
#map {{ height:650px; border:1px solid var(--line); background:#e2e8f0; }}
.legend {{ display:flex; flex-wrap:wrap; gap:12px 18px; padding:10px 0; color:#334155; font-size:13px; }}
.swatch {{ display:inline-block; width:30px; height:3px; margin-right:6px; vertical-align:middle; }}
table {{ width:100%; border-collapse:collapse; background:white; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; }} th {{ color:var(--muted); font-size:12px; }}
img {{ max-width:100%; height:auto; border:1px solid var(--line); background:white; }} code {{ font-size:12px; }}
@media(max-width:800px) {{ main {{ padding:10px; }} #map {{ height:540px; }} h1 {{ font-size:21px; }} table {{ font-size:12px; }} }}
</style></head><body><main>
<h1>{html.escape(payload['title'])}</h1>
<p class="sub">Issue time: {html.escape(payload['issue_time_utc'])} | same public causal analysis source for every route</p>
<p class="notice"><b>Data boundary.</b> No future observation, positive-lead weather field, JMA/JTWC forecast, or other model forecast was passed to inference. The v62 pressure image is the causal state background; v23 supplies only a track route because it was not trained to emit a full pressure grid.</p>
<div id="map" aria-label="v23 and v62 causal route comparison map"></div>
<div class="legend"><span><i class="swatch" style="background:#111827"></i>observed input</span><span><i class="swatch" style="background:#2563eb"></i>v23 same-source patch</span><span><i class="swatch" style="background:#f59e0b"></i>v62 Pacific-domain only</span><span><i class="swatch" style="background:#b91c1c"></i>v62 full local + Pacific</span><span><i class="swatch" style="background:#047857"></i>post-issue truth, Tip only</span></div>
<h2>Route comparison</h2><table><thead><tr><th>Route</th><th>+120 h endpoint</th><th>Verification</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Pressure-state map</h2><p>The map below is rendered from v62's whole-domain analysis-tendency state. It contains 2-hPa MSLP contours, 500-hPa height and wind, 850-hPa wind structure, and the three route lines.</p><img src="{html.escape(pressure_image.name)}" alt="Causal western Pacific pressure map with v23 and v62 route comparison">
<h2>Reproduction contract</h2><p><code>{html.escape(payload['source_policy'])}</code></p>
</main><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script><script>
const DATA={data};
const map=L.map('map',{{worldCopyJump:true,zoomControl:true}}).setView([22,150],4);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:8,attribution:'&copy; OpenStreetMap contributors'}}).addTo(map);
const colors={{observed:'#111827',v23:'#2563eb',v62_pacific:'#f59e0b',v62_full:'#b91c1c',truth:'#047857'}};
const pair=p=>[p.lat,p.lon];
const addLine=(items,color,options)=>L.polyline(items.map(pair),Object.assign({{color,weight:3,opacity:.9}},options||{{}})).addTo(map);
addLine(DATA.observed,colors.observed,{{weight:4}});
DATA.observed.forEach(p=>L.circleMarker(pair(p),{{radius:3,color:colors.observed,fillColor:colors.observed,fillOpacity:1}}).bindTooltip(p.time||'input').addTo(map));
if(DATA.truth.length) addLine([DATA.observed[DATA.observed.length-1],...DATA.truth],colors.truth,{{weight:3,dashArray:'5 7'}});
for(const [key,opts] of Object.entries({{v23:{{dashArray:'7 5'}},v62_pacific:{{dashArray:'2 7'}},v62_full:{{}}}})){{
 const route=DATA.routes[key].forecast; const group=L.layerGroup().addTo(map); addLine([DATA.observed[DATA.observed.length-1],...route],colors[key],opts);
 route.forEach(p=>L.circleMarker(pair(p),{{radius:4,color:'#fff',weight:1,fillColor:colors[key],fillOpacity:.96}}).bindTooltip(DATA.routes[key].label+' +'+p.lead_hours+' h').addTo(group));
}}
const all=[...DATA.observed,...DATA.truth,...DATA.routes.v23.forecast,...DATA.routes.v62_pacific.forecast,...DATA.routes.v62_full.forecast].map(pair); map.fitBounds(all,{{padding:[20,20]}});
</script></body></html>"""
    path.write_text(html_text, encoding="utf-8")


def _case_payload(
    storm: str,
    issue_time: dt.datetime,
    rows: list[dict],
    truth: list[dict],
    source_policy: str,
    source_files: list[Path],
    v23_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
    v62: dict,
    output_stem: str,
    pressure_path: Path,
) -> dict:
    v23 = _run_v23(rows, *v23_fields, MODELS)
    routes = {"v23": v23, "v62_pacific": v62["pacific"], "v62_full": v62["full"]}
    observed = v62["observed"]
    truth_map = [
        {"time_utc": row["time_utc"], "lat": float(row["lat"]), "lon": float(row["lon"])}
        for row in truth
    ]
    scores = {
        key: _score(route["forecast"], truth_map, rows[-1])
        for key, route in routes.items()
    }
    payload = {
        "title": f"{storm}: v23 versus v62 public-data causal comparison",
        "storm": storm,
        "issue_time_utc": issue_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observed": observed,
        "truth": [{"time_utc": row["time_utc"], "lat": float(row["lat"]), "lon": float(row["lon"])} for row in truth],
        "routes": routes,
        "scores": scores,
        "source_policy": source_policy,
        "source_files": [str(path.relative_to(ROOT)) for path in source_files],
        "future_rows_used_for_inference": 0,
        "official_forecasts_used_for_inference": False,
        "forecast_products_used": [],
        "full_v62_definition": "75% local multi-level causal route + 25% broad western-Pacific causal route",
        "same_dataset_note": "v23 and both v62 routes read the same case-specific public analysis/reanalysis source and issue-time track history; v23 uses a 4-channel 17x17 patch, while v62 uses the full domain and multi-level local fields.",
        "pressure_map_note": "v23 has no full pressure-grid decoder. The pressure map is v62's causal state extrapolation, with v23 and v62 route overlays.",
    }
    world_path = ROOT / "paper" / f"{output_stem}_world_map.png"
    html_path = ROOT / "paper" / f"{output_stem}_world_map.html"
    json_path = ROOT / "track_build" / f"{output_stem}.json"
    _draw_world_map(world_path, payload)
    _html_page(html_path, payload, pressure_path)
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload | {"outputs": {"json": str(json_path), "html": str(html_path), "world_map": str(world_path)}}


def main() -> None:
    global MODELS
    MODELS = _load_v23_models()
    dolphin_source = json.loads((ROOT / "track_build" / "dolphin_v37p_v37n_same_issue_current.json").read_text(encoding="utf-8"))
    dolphin_rows = dolphin_source["current_track_history"]
    dolphin_issue = dt.datetime.fromisoformat(dolphin_source["issue_time_utc"].replace("Z", "+00:00"))
    with np.load(ROOT / "track_build" / "dolphin_analysis_causal.npz", allow_pickle=True) as archive:
        dolphin_fields = (
            np.asarray(archive["current"], dtype="float32"),
            np.asarray(archive["history"], dtype="float32"),
            np.asarray(archive["available"], dtype="float32"),
        )
    dolphin_v62 = _v62_routes(DOLPHIN_FILES, DOLPHIN_LEVELS, dolphin_rows, "time", dolphin_issue)
    dolphin_payload = _case_payload(
        "Dolphin",
        dolphin_issue,
        dolphin_rows,
        [],
        "NOAA NOMADS GFS 0.25-degree f000 analysis at the issue cycle and two earlier cycles; track history through the issue only. This is the no-key public source that a GitHub Actions job can retrieve.",
        list(DOLPHIN_FILES) + [ROOT / "track_build" / "dolphin_analysis_causal.npz"],
        dolphin_fields,
        dolphin_v62,
        "dolphin_v62_v23_public_data",
        ROOT / "paper" / "dolphin_v62_v23_public_data_pressure_map.png",
    )
    _draw_pressure_forecast(
        ROOT / "paper" / "dolphin_v62_v23_public_data_pressure_map.png",
        dolphin_v62["state_fields"],
        dolphin_v62["state_pressure"],
        dolphin_v62["latitude"],
        dolphin_v62["longitude"],
        dolphin_v62["observed"],
        dolphin_v62["full"]["forecast"],
        dolphin_v62["systems"],
        "Dolphin: v62 causal pressure state with v23 and v62 routes",
        comparison_routes={
            "v23 same-source patch": ("#2563eb", dolphin_payload["routes"]["v23"]["forecast"]),
            "v62 Pacific-domain only": ("#f59e0b", dolphin_payload["routes"]["v62_pacific"]["forecast"]),
        },
    )

    tip_case_source = json.loads((ROOT / "track_build" / "tip_v37_cfsr_causal_19791012.json").read_text(encoding="utf-8"))
    tip_rows = tip_case_source["input_history"]
    tip_truth = tip_case_source["truth_after_issue"]
    tip_issue = dt.datetime.fromisoformat(tip_case_source["issue_time_utc"].replace("Z", "+00:00"))
    tip_fields = _tip_v23_fields()
    tip_v62 = _v62_routes(TIP_FILES, TIP_LEVELS, tip_rows, "time_utc", tip_issue)
    tip_payload = _case_payload(
        "Tip",
        tip_issue,
        tip_rows,
        tip_truth,
        "NOAA CFSR historical analysis at and before the issue time plus the IBTrACS track history through the issue. Later IBTrACS rows are verification truth only. CFSR is a public downloadable analysis/reanalysis source suitable for a GitHub Actions replay.",
        list(TIP_FILES) + [ROOT / "v37_cfsr" / "tip_1979_causal" / "cfsr_tip_19791012.npz"],
        tip_fields,
        tip_v62,
        "tip_v62_v23_public_data",
        ROOT / "paper" / "tip_v62_v23_public_data_pressure_map.png",
    )
    _draw_pressure_forecast(
        ROOT / "paper" / "tip_v62_v23_public_data_pressure_map.png",
        tip_v62["state_fields"],
        tip_v62["state_pressure"],
        tip_v62["latitude"],
        tip_v62["longitude"],
        tip_v62["observed"],
        tip_v62["full"]["forecast"],
        tip_v62["systems"],
        "Tip: v62 causal pressure state with v23 and v62 routes",
        comparison_routes={
            "v23 same-source patch": ("#2563eb", tip_payload["routes"]["v23"]["forecast"]),
            "v62 Pacific-domain only": ("#f59e0b", tip_payload["routes"]["v62_pacific"]["forecast"]),
        },
    )
    print(
        json.dumps(
            {
                "dolphin": dolphin_payload["outputs"],
                "tip": tip_payload["outputs"],
                "tip_scores": tip_payload["scores"],
                "v23_seed_count": len(MODELS),
                "future_rows_used_for_inference": 0,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    MODELS: list[torch.nn.Module] = []
    main()
