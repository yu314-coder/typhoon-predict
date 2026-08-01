#!/usr/bin/env python3
"""Draw the v36A Dolphin forecast beside the saved JTWC/JMA issue-time tracks.

The official comparison points are parsed from the existing, issue-time-captured
comparison artifact so this map does not silently mix a later advisory with the
v36A forecast issue. The HTML is interactive; the PNG is self-contained apart
from the small Natural Earth coastline file downloaded on first run.
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch


ROOT = Path(__file__).resolve().parents[1]
V36_PATH = ROOT / "track_build" / "dolphin_v36_forecast.json"
OFFICIAL_HTML = ROOT / "paper" / "dolphin_latest_v23_v35_vs_official.html"
OUT_HTML = ROOT / "paper" / "dolphin_v36_vs_jma_jtwc_world_map.html"
OUT_PNG = ROOT / "paper" / "dolphin_v36_vs_jma_jtwc_world_map.png"
OUT_DATA = ROOT / "track_build" / "dolphin_v36_jma_jtwc_comparison.json"
COAST_FILE = ROOT / "paper" / "ne_110m_admin_0_countries.geojson"

COAST_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_110m_admin_0_countries.geojson"
)

# The v36 notebook uses these observations as the model input ending at its
# issue time. Keep the source values visible in the artifact metadata.
OBSERVED = [
    {"time": "2026-07-27T00:00Z", "lat": 12.8, "lon": 178.3, "vmax_kt": 30.0, "pressure_hpa": 1002.0},
    {"time": "2026-07-27T06:00Z", "lat": 13.4, "lon": 176.7, "vmax_kt": 35.0, "pressure_hpa": 1000.0},
    {"time": "2026-07-27T12:00Z", "lat": 13.6, "lon": 175.2, "vmax_kt": 43.0, "pressure_hpa": 998.0},
    {"time": "2026-07-27T18:00Z", "lat": 13.2, "lon": 173.7, "vmax_kt": 43.0, "pressure_hpa": 992.0},
    {"time": "2026-07-28T00:00Z", "lat": 13.0, "lon": 172.8, "vmax_kt": 61.0, "pressure_hpa": 990.0},
    {"time": "2026-07-28T06:00Z", "lat": 13.3, "lon": 171.7, "vmax_kt": 74.0, "pressure_hpa": 980.0},
    {"time": "2026-07-28T12:00Z", "lat": 13.4, "lon": 170.7, "vmax_kt": 100.0, "pressure_hpa": 975.0},
    {"time": "2026-07-28T18:00Z", "lat": 13.7, "lon": 169.9, "vmax_kt": 126.0, "pressure_hpa": 937.0},
    {"time": "2026-07-29T00:00Z", "lat": 14.1, "lon": 169.1, "vmax_kt": 122.0, "pressure_hpa": 941.0},
    {"time": "2026-07-29T06:00Z", "lat": 14.5, "lon": 168.4, "vmax_kt": 122.0, "pressure_hpa": 941.0},
    {"time": "2026-07-29T12:00Z", "lat": 15.2, "lon": 167.7, "vmax_kt": 130.0, "pressure_hpa": 935.0},
    {"time": "2026-07-29T18:00Z", "lat": 15.8, "lon": 166.8, "vmax_kt": 143.0, "pressure_hpa": 915.0},
    {"time": "2026-07-30T00:00Z", "lat": 16.4, "lon": 165.7, "vmax_kt": 139.0, "pressure_hpa": None},
    {"time": "2026-07-30T06:00Z", "lat": 16.9, "lon": 164.9, "vmax_kt": 139.0, "pressure_hpa": 919.0},
]


def parse_captured_official_tracks() -> dict[str, list[dict]]:
    """Read JTWC and JMA points from the saved same-issue comparison map."""

    text = OFFICIAL_HTML.read_text(encoding="utf-8")
    points: dict[str, list[dict]] = {"jtwc": [], "jma": []}
    pattern = re.compile(r"<circle\b[^>]*data-model=\"([^\"]+)\"[^>]*/>")
    for match in pattern.finditer(text):
        model, tag = match.group(1), match.group(0)
        if "JTWC official forecast" in model:
            key = "jtwc"
        elif "JMA/RSMC Tokyo official forecast" in model:
            key = "jma"
        else:
            continue
        values = dict(re.findall(r'data-(time|lat|lon|vmax|cat)="([^"]*)"', tag))
        points[key].append(
            {
                "time": values["time"],
                "lat": float(values["lat"]),
                "lon": float(values["lon"]),
                "vmax_kt": float(values["vmax"]),
                "category": values.get("cat", ""),
                "model": model,
            }
        )
    if len(points["jtwc"]) != 8 or len(points["jma"]) != 4:
        raise RuntimeError(
            "Expected 8 JTWC and 4 JMA captured points; "
            f"found {len(points['jtwc'])} and {len(points['jma'])}"
        )
    return points


def lon360(value: float) -> float:
    return float(value % 360.0)


def safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def intensity_color(vmax: float) -> str:
    if vmax >= 137:
        return "#7c3aed"  # C5 on the 1-minute scale
    if vmax >= 113:
        return "#b91c1c"  # C4
    if vmax >= 96:
        return "#dc2626"  # C3
    if vmax >= 83:
        return "#ea580c"  # C2
    if vmax >= 64:
        return "#ca8a04"  # C1
    if vmax >= 34:
        return "#0891b2"  # TS
    return "#64748b"  # TD


def category(vmax: float) -> str:
    if vmax >= 137:
        return "C5"
    if vmax >= 113:
        return "C4"
    if vmax >= 96:
        return "C3"
    if vmax >= 83:
        return "C2"
    if vmax >= 64:
        return "C1"
    if vmax >= 34:
        return "TS"
    return "TD"


def load_coastlines() -> dict:
    if not COAST_FILE.exists():
        request = urllib.request.Request(
            COAST_URL,
            headers={"User-Agent": "typhoon-predict-map/1.0"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            COAST_FILE.write_bytes(response.read())
    return json.loads(COAST_FILE.read_text(encoding="utf-8"))


def iter_rings(geometry: dict):
    kind = geometry["type"]
    coords = geometry["coordinates"]
    if kind == "Polygon":
        yield from coords
    elif kind == "MultiPolygon":
        for polygon in coords:
            yield from polygon


def draw_world(ax, coastlines: dict, *, detail: bool = False) -> None:
    for feature in coastlines["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        for ring in iter_rings(geometry):
            xs, ys = [], []
            previous = None
            for lon, lat in ring:
                x = lon360(lon)
                if previous is not None and abs(x - previous) > 180:
                    if len(xs) > 1:
                        ax.plot(xs, ys, color="#94a3b8", linewidth=0.45, zorder=1)
                    xs, ys = [], []
                xs.append(x)
                ys.append(lat)
                previous = x
            if len(xs) > 1:
                ax.plot(xs, ys, color="#94a3b8", linewidth=0.45, zorder=1)
    ax.set_facecolor("#e9f3f7")
    ax.grid(True, color="#b8cbd3", linewidth=0.45, alpha=0.65)
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude")
    if detail:
        ax.set_xlim(135, 190)
        ax.set_ylim(5, 35)
        ax.set_xticks(range(140, 191, 10))
        ax.set_yticks(range(5, 36, 5))
    else:
        ax.set_xlim(0, 360)
        ax.set_ylim(-60, 70)
        ax.set_xticks(range(0, 361, 60))
        ax.set_yticks(range(-60, 71, 20))
        ax.set_xticklabels(["0", "60E", "120E", "180", "120W", "60W", "0"])


def plot_track(ax, points: list[dict], *, color: str, linestyle: str, linewidth: float = 2.0) -> None:
    ax.plot(
        [lon360(point["lon"] if "lon" in point else point["longitude"]) for point in points],
        [point["lat"] if "lat" in point else point["latitude"] for point in points],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=4,
    )


def annotate_point(ax, x: float, y: float, label: str, color: str, *, dx: float = 0.8, dy: float = 0.8) -> None:
    ax.annotate(
        label,
        (x, y),
        xytext=(x + dx, y + dy),
        fontsize=7.3,
        color="#17202a",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": color, "alpha": 0.92, "linewidth": 0.8},
        zorder=8,
    )


def create_png(payload: dict) -> None:
    coastlines = load_coastlines()
    v36 = payload["v36"]["forecast"]
    jtwc = payload["official"]["jtwc"]
    jma = payload["official"]["jma"]

    figure, (global_ax, detail_ax) = plt.subplots(
        1,
        2,
        figsize=(19, 9.5),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
        constrained_layout=True,
    )
    for axis in (global_ax, detail_ax):
        draw_world(axis, coastlines, detail=axis is detail_ax)

    # The saved observation path is plotted in the same 0..360 longitude space.
    for axis in (global_ax, detail_ax):
        axis.plot(
            [lon360(point["lon"]) for point in OBSERVED],
            [point["lat"] for point in OBSERVED],
            color="#111827",
            linewidth=2.4,
            zorder=5,
        )
        axis.scatter(
            [lon360(point["lon"]) for point in OBSERVED],
            [point["lat"] for point in OBSERVED],
            s=13,
            color="#111827",
            zorder=6,
        )
        plot_track(axis, v36, color="#dc2626", linestyle="-", linewidth=2.6)
        plot_track(axis, jtwc, color="#c2410c", linestyle="--", linewidth=2.0)
        plot_track(axis, jma, color="#047857", linestyle="-.", linewidth=2.0)

        # Uncertainty is the mean distance of the five v36 members from the mean.
        for point in v36[::2]:
            radius_lat = max(float(point["track_spread_km"]) / 111.2, 0.2)
            radius_lon = radius_lat / max(0.15, abs(__import__("math").cos(__import__("math").radians(point["latitude"]))))
            axis.add_patch(
                Ellipse(
                    (lon360(point["longitude"]), point["latitude"]),
                    2 * radius_lon,
                    2 * radius_lat,
                    facecolor="#dc2626",
                    edgecolor="#dc2626",
                    alpha=0.06,
                    linewidth=0.8,
                    zorder=2,
                )
            )

        axis.scatter(
            [lon360(point["longitude"]) for point in v36],
            [point["latitude"] for point in v36],
            s=36,
            c=[intensity_color(point["vmax_kt"]) for point in v36],
            edgecolors="white",
            linewidths=0.75,
            zorder=7,
        )
        axis.scatter(
            [lon360(point["lon"]) for point in jtwc],
            [point["lat"] for point in jtwc],
            s=28,
            facecolors="white",
            edgecolors="#c2410c",
            linewidths=1.4,
            zorder=7,
        )
        axis.scatter(
            [lon360(point["lon"]) for point in jma],
            [point["lat"] for point in jma],
            s=28,
            facecolors="white",
            edgecolors="#047857",
            linewidths=1.4,
            zorder=7,
        )

    # Keep the global map readable while putting full intensity details on the
    # regional panel. These are the user-facing values requested on the map.
    for point in v36:
        if point["lead_hours"] % 24 == 0:
            annotate_point(
                detail_ax,
                lon360(point["longitude"]),
                point["latitude"],
                f"+{point['lead_hours']}h\n{point['vmax_kt']:.0f} kt\n{point['central_pressure_hpa']:.0f} hPa",
                "#dc2626",
            )
    for point in jtwc:
        if "(+0h)" in point["time"] or "+120h" in point["time"]:
            annotate_point(detail_ax, lon360(point["lon"]), point["lat"], f"JTWC\n{point['vmax_kt']:.0f} kt", "#c2410c", dx=-5.5, dy=-2.2)
    for point in jma:
        if "(+0h)" in point["time"] or "+69h" in point["time"]:
            annotate_point(detail_ax, lon360(point["lon"]), point["lat"], f"JMA\n{point['vmax_kt']:.0f} kt", "#047857", dx=1.0, dy=-3.0)

    global_ax.set_title("Global / Pacific-centered overview", loc="left", fontsize=13, fontweight="bold")
    detail_ax.set_title("Western Pacific detail: intensity on every v36 point", loc="left", fontsize=13, fontweight="bold")
    figure.suptitle(
        "Dolphin: TrackFormer v36A versus issue-time JTWC and JMA forecasts\n"
        f"v36 issue {payload['v36']['issue_time_utc']} | v36 mean + spread from five members",
        fontsize=16,
        fontweight="bold",
    )

    legend = [
        Line2D([0], [0], color="#111827", linewidth=2.4, label="Observed track"),
        Line2D([0], [0], color="#dc2626", linewidth=2.6, label="v36A ensemble mean"),
        Line2D([0], [0], color="#c2410c", linewidth=2.0, linestyle="--", label="JTWC Warning #14"),
        Line2D([0], [0], color="#047857", linewidth=2.0, linestyle="-.", label="JMA/RSMC Tokyo"),
        Patch(facecolor="#dc2626", alpha=0.08, edgecolor="#dc2626", label="v36 mean member spread"),
    ]
    detail_ax.legend(handles=legend, loc="lower left", fontsize=8.3, framealpha=0.96)
    figure.text(
        0.5,
        0.005,
        "Wind labels: v36/JTWC use 1-min sustained kt; JMA uses 10-min sustained kt. "
        "v36 pressure labels are central pressure; v36 pressure fields are parametric diagnostics.",
        ha="center",
        fontsize=9,
        color="#334155",
    )
    figure.savefig(OUT_PNG, dpi=190, facecolor="white")
    plt.close(figure)


def leaflet_point(point: dict, *, model: str) -> dict:
    if model == "v36":
        return {
            "model": "v36A ensemble mean",
            "lead_hours": point["lead_hours"],
            "time": point["valid_time_utc"],
            "lat": point["latitude"],
            "lon": point["longitude"],
            "vmax_kt": point["vmax_kt"],
            "pressure_hpa": point["central_pressure_hpa"],
            "vmax_spread_kt": point["vmax_spread_kt"],
            "pressure_spread_hpa": point["pressure_spread_hpa"],
            "track_spread_km": point["track_spread_km"],
            "rmw_km": point["rmw_km"],
            "wind_convention": "1-min sustained",
        }
    result = dict(point)
    result["model"] = "JTWC official forecast" if model == "jtwc" else "JMA/RSMC Tokyo official forecast"
    result["wind_convention"] = "1-min sustained" if model == "jtwc" else "10-min sustained"
    return result


def create_html(payload: dict) -> None:
    v36_points = [leaflet_point(point, model="v36") for point in payload["v36"]["forecast"]]
    jtwc_points = [leaflet_point(point, model="jtwc") for point in payload["official"]["jtwc"]]
    jma_points = [leaflet_point(point, model="jma") for point in payload["official"]["jma"]]
    obs_points = [
        {
            "model": "Observed",
            "time": point["time"],
            "lat": point["lat"],
            "lon": point["lon"],
            "vmax_kt": point["vmax_kt"],
            "pressure_hpa": point["pressure_hpa"],
            "wind_convention": "1-min input estimate",
        }
        for point in OBSERVED
    ]
    map_data = {"v36": v36_points, "jtwc": jtwc_points, "jma": jma_points, "observed": obs_points}
    data_json = safe_json(map_data).replace("</", "<\\/")
    v36_final = v36_points[-1]
    jtwc_final = jtwc_points[-1]
    jma_final = jma_points[-1]

    rows = []
    for point in v36_points:
        rows.append(
            "<tr>"
            f"<td>+{point['lead_hours']} h</td><td>{point['time']}</td>"
            f"<td>{point['lat']:.2f}</td><td>{point['lon']:.2f}E</td>"
            f"<td><b>{point['vmax_kt']:.0f} kt</b><small> +/- {point['vmax_spread_kt']:.1f}</small></td>"
            f"<td><b>{point['pressure_hpa']:.0f} hPa</b><small> +/- {point['pressure_spread_hpa']:.1f}</small></td>"
            f"<td>{point['track_spread_km']:.0f} km</td></tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dolphin v36A vs JTWC and JMA world map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
<style>
:root {{ color-scheme: light; --ink:#17202a; --muted:#5b6875; --line:#d4dde5; --surface:#fff; --bg:#eef3f6; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1500px; margin:0 auto; padding:24px; }} h1 {{ margin:0 0 6px; font-size:28px; }} h2 {{ margin:0 0 10px; font-size:17px; }} h3 {{ margin:0 0 5px; font-size:14px; }}
.sub {{ color:var(--muted); max-width:980px; margin:0 0 16px; }} .notice {{ background:#fff8e6; border-left:4px solid #d97706; padding:11px 14px; margin:12px 0 16px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:12px 0; }} .card {{ background:var(--surface); border:1px solid var(--line); border-radius:7px; padding:12px; }}
.card b {{ display:block; font-size:18px; margin-top:4px; }} .muted {{ color:var(--muted); }} .mapwrap {{ background:var(--surface); border:1px solid var(--line); border-radius:7px; overflow:hidden; }}
#map {{ height:650px; }} .legend {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px 18px; padding:13px 15px; border-top:1px solid var(--line); }} .legend span {{ display:flex; align-items:center; gap:8px; }}
.swatch {{ width:27px; height:3px; display:inline-block; flex:none; }} .dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; border:2px solid white; box-shadow:0 0 0 1px #777; flex:none; }}
.section {{ margin-top:16px; background:var(--surface); border:1px solid var(--line); border-radius:7px; padding:15px; }} table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }} th,td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; }} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }} th {{ color:var(--muted); font-size:12px; }} small {{ display:block; color:var(--muted); font-size:11px; }}
.source {{ color:var(--muted); font-size:12px; }} a {{ color:#0f5f9f; }} .leaflet-tooltip {{ font-size:12px; }} .intensity-label {{ background:transparent; border:0; color:#17202a; font-weight:700; text-shadow:0 1px 2px white,0 -1px 2px white,1px 0 2px white,-1px 0 2px white; }}
@media(max-width:900px) {{ main {{ padding:12px; }} h1 {{ font-size:22px; }} .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} #map {{ height:540px; }} .legend {{ grid-template-columns:1fr; }} table {{ font-size:12px; }} th:nth-child(7),td:nth-child(7) {{ display:none; }} }}
</style></head><body><main>
<h1>Dolphin: v36A vs JTWC + JMA</h1>
<p class="sub">World map and Western Pacific detail for the five-member TrackFormer v36A ensemble mean. The comparison is locked to the v36 issue window so the lines are not mixed with a later live advisory.</p>
<div class="notice"><b>Read the intensity labels carefully.</b> v36A and JTWC use one-minute sustained wind; JMA/RSMC Tokyo uses ten-minute sustained wind. v36 pressure is central pressure. Its pressure field is a parametric diagnostic reconstructed from predicted center pressure, wind, RMW, and wind radii, not a learned ERA5 grid.</div>
<div class="cards">
  <div class="card"><span class="muted">v36 issue</span><b>{html.escape(payload['v36']['issue_time_utc'])}</b><span class="muted">5 members, +6 to +120 h</span></div>
  <div class="card"><span class="muted">v36 at +120 h</span><b>{v36_final['lat']:.2f}N, {v36_final['lon']:.2f}E</b><span class="muted">{v36_final['vmax_kt']:.0f} kt / {v36_final['pressure_hpa']:.0f} hPa</span></div>
  <div class="card"><span class="muted">JTWC Warning #14 +120 h</span><b>{jtwc_final['lat']:.2f}N, {jtwc_final['lon']:.2f}E</b><span class="muted">{jtwc_final['vmax_kt']:.0f} kt, 1-min</span></div>
  <div class="card"><span class="muted">JMA +69 h</span><b>{jma_final['lat']:.2f}N, {jma_final['lon']:.2f}E</b><span class="muted">{jma_final['vmax_kt']:.0f} kt, 10-min</span></div>
</div>
<div class="mapwrap"><div id="map" aria-label="Dolphin world map comparing v36A, JTWC, and JMA forecasts"></div>
<div class="legend">
 <span><i class="swatch" style="background:#111827"></i>Observed track</span>
 <span><i class="swatch" style="background:#dc2626"></i>v36A ensemble mean; translucent circles are track spread</span>
 <span><i class="swatch" style="background:#c2410c;border-top:2px dashed #c2410c;height:0"></i>JTWC official Warning #14</span>
 <span><i class="swatch" style="background:#047857;border-top:2px dashed #047857;height:0"></i>JMA/RSMC Tokyo official forecast</span>
 <span><i class="dot" style="background:#7c3aed"></i>v36/JTWC C5 threshold color; marker popups include kt and hPa</span>
 <span><i class="dot" style="background:#0891b2"></i>Lower intensity colors use the 1-min category scale; JMA markers remain green outlined</span>
</div></div>
<section class="section"><h2>v36A mean forecast values shown on the map</h2><table><thead><tr><th>Lead</th><th>Valid UTC</th><th>Lat</th><th>Lon</th><th>Wind</th><th>Pressure</th><th>Track spread</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="section source"><h2>Data and provenance</h2>
<p>v36A values are read from <code>track_build/dolphin_v36_forecast.json</code>, generated by the completed Colab v36A five-member run. JTWC and JMA points are the saved same-issue capture from <code>paper/dolphin_latest_v23_v35_vs_official.html</code>: JTWC Warning #14 at 2026-07-30 06 UTC and JMA/RSMC Tokyo at 2026-07-30 09 UTC.</p>
<p>Official reference pages: <a href="https://www.metoc.navy.mil/jtwc/products/wp1226web.txt">JTWC 12W Dolphin product</a>, <a href="https://www.data.jma.go.jp/typhoon/position_table/table2026.html">JMA 2026 position table</a>, and <a href="https://www.data.jma.go.jp/typhoon/route_map/bstv2026.html">JMA 2026 route map</a>. The live JTWC product may now contain a later warning than the issue-time snapshot drawn here.</p></section>
</main>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
<script>
const DATA = {data_json};
const map = L.map('map', {{ worldCopyJump:true, zoomControl:true }}).setView([20,170],3);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom:7, attribution:'&copy; OpenStreetMap contributors' }}).addTo(map);
const colors = {{ observed:'#111827', v36:'#dc2626', jtwc:'#c2410c', jma:'#047857' }};
const windColor = (kt) => kt >= 137 ? '#7c3aed' : kt >= 113 ? '#b91c1c' : kt >= 96 ? '#dc2626' : kt >= 83 ? '#ea580c' : kt >= 64 ? '#ca8a04' : kt >= 34 ? '#0891b2' : '#64748b';
const line = (items, color, options={{}}) => L.polyline(items.map(p => [p.lat,p.lon]), Object.assign({{color,weight:3,opacity:.9}}, options)).addTo(map);
const popup = (p) => {{
  const lead = p.lead_hours === undefined ? '' : `<b>Lead:</b> +${{p.lead_hours}} h<br>`;
  const pressure = p.pressure_hpa == null ? '' : `<b>Central pressure:</b> ${{p.pressure_hpa.toFixed(0)}} hPa<br>`;
  const spread = p.track_spread_km == null ? '' : `<b>Track spread:</b> ${{p.track_spread_km.toFixed(0)}} km<br>`;
  return `<b>${{p.model}}</b><br>${{lead}}<b>Valid:</b> ${{p.time}}<br><b>Position:</b> ${{p.lat.toFixed(2)}}N, ${{p.lon.toFixed(2)}}E<br><b>Wind:</b> ${{p.vmax_kt.toFixed(0)}} kt (${{p.wind_convention}})<br>${{pressure}}${{spread}}`;
}};
line(DATA.observed, colors.observed, {{weight:3}});
DATA.observed.forEach(p => L.circleMarker([p.lat,p.lon], {{radius:3,color:colors.observed,fillColor:colors.observed,fillOpacity:.9}}).bindPopup(popup(p)).addTo(map));
line(DATA.v36, colors.v36, {{weight:4}});
DATA.v36.forEach((p,i) => {{
  if (i % 2 === 0) L.circle([p.lat,p.lon], {{radius:(p.track_spread_km||1)*1000,color:colors.v36,weight:1,opacity:.25,fillOpacity:.035}}).addTo(map);
  const marker = L.circleMarker([p.lat,p.lon], {{radius:6, color:'#fff',weight:1.5,fillColor:windColor(p.vmax_kt),fillOpacity:.98}}).bindPopup(popup(p)).addTo(map);
  if (p.lead_hours % 24 === 0) marker.bindTooltip(`+${{p.lead_hours}}h · ${{p.vmax_kt.toFixed(0)}} kt · ${{p.pressure_hpa.toFixed(0)}} hPa`, {{permanent:true,direction:'right',className:'intensity-label',offset:[7,0]}});
}});
line(DATA.jtwc, colors.jtwc, {{weight:3,dashArray:'9 7'}});
DATA.jtwc.forEach(p => {{
  const marker = L.circleMarker([p.lat,p.lon], {{radius:5,color:colors.jtwc,fillColor:'#fff',fillOpacity:1,weight:2}}).bindPopup(popup(p)).addTo(map);
  if (p.time.includes('(+120h)')) marker.bindTooltip(`JTWC ${{p.time.match(/\\(\\+(\\d+)h\\)/)?.[1] || '0'}}h · ${{p.vmax_kt.toFixed(0)}} kt`, {{permanent:true,direction:'left',className:'intensity-label',offset:[-7,0]}});
}});
line(DATA.jma, colors.jma, {{weight:3,dashArray:'3 7'}});
DATA.jma.forEach(p => {{
  const marker = L.circleMarker([p.lat,p.lon], {{radius:5,color:colors.jma,fillColor:'#fff',fillOpacity:1,weight:2}}).bindPopup(popup(p)).addTo(map);
  if (p.time.includes('(+69h)')) marker.bindTooltip(`JMA ${{p.time.match(/\\(\\+(\\d+)h\\)/)?.[1] || '0'}}h · ${{p.vmax_kt.toFixed(0)}} kt`, {{permanent:true,direction:'left',className:'intensity-label',offset:[-7,0]}});
}});
const all = DATA.observed.concat(DATA.v36,DATA.jtwc,DATA.jma).map(p => [p.lat,p.lon]);
map.fitBounds(all, {{padding:[24,24]}});
L.control.layers({{}}, {{'v36A spread':L.layerGroup([])}}).addTo(map);
</script></body></html>"""
    OUT_HTML.write_text(html_text, encoding="utf-8")


def main() -> None:
    if not V36_PATH.exists():
        raise FileNotFoundError(f"Missing v36 output: {V36_PATH}")
    v36_payload = json.loads(V36_PATH.read_text(encoding="utf-8"))
    official = parse_captured_official_tracks()
    payload = {
        "v36": v36_payload,
        "official": official,
        "observed": OBSERVED,
        "comparison_note": "Issue-time comparison; official points were captured before later advisories.",
        "sources": {
            "jtwc_product": "https://www.metoc.navy.mil/jtwc/products/wp1226web.txt",
            "jma_position_table": "https://www.data.jma.go.jp/typhoon/position_table/table2026.html",
            "jma_route_map": "https://www.data.jma.go.jp/typhoon/route_map/bstv2026.html",
        },
    }
    OUT_DATA.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    create_png(payload)
    create_html(payload)
    print(json.dumps({"html": str(OUT_HTML), "png": str(OUT_PNG), "data": str(OUT_DATA), "v36_points": len(v36_payload["forecast"])}, indent=2))


if __name__ == "__main__":
    main()
