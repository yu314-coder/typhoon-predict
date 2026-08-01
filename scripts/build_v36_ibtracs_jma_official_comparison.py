#!/usr/bin/env python3
"""Build a current Dolphin v36A / IBTrACS / JMA / JTWC comparison artifact.

The v36A forecast is read from the completed five-member run. IBTrACS and the
current JMA analysis are kept as separate provenance layers; official JTWC and
JMA forecast points are comparison data only and are never fed into v36A.
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch

# When launched as ``python scripts/<name>.py``, Python starts with ``scripts``
# on sys.path rather than the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.draw_dolphin_v36_world_map import (
    annotate_point,
    category,
    draw_world,
    intensity_color,
    load_coastlines,
    lon360,
    plot_track,
    safe_json,
)


V36_PATH = ROOT / "track_build" / "dolphin_v36_forecast.json"
IBTRACS_PATH = ROOT / "track_build" / "ibtracs_jma_only" / "dolphin_ibtracs_records.csv"
JMA_TARGET_PATH = ROOT / "data" / "jma" / "targetTc.json"
JTWC_PATH = ROOT / "data" / "jtwc" / "wp1226web_current.txt"
OUT_DATA = ROOT / "track_build" / "dolphin_v36_ibtracs_jma_official_comparison.json"
OUT_HTML = ROOT / "paper" / "dolphin_v36_ibtracs_jma_jtwc_current_world_map.html"
OUT_PNG = ROOT / "paper" / "dolphin_v36_ibtracs_jma_jtwc_current_world_map.png"
# Compatibility aliases for the older browser tab. These files are generated
# from the same v36 payload so the tab cannot silently show the stale v10 map.
LEGACY_HTML = ROOT / "paper" / "dolphin_ibtracs_jma_only_world_map.html"
LEGACY_PNG = ROOT / "paper" / "dolphin_ibtracs_jma_only_world_map.png"


def number(value: str | None) -> float | None:
    if value is None or value.strip().lower() in {"", "nan", "na", "null"}:
        return None
    return float(value)


def read_ibtracs() -> list[dict]:
    points = []
    with IBTRACS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points.append(
                {
                    "time": row["time_utc"],
                    "lat": float(row["lat"]),
                    "lon": lon360(float(row["lon"])),
                    "vmax_kt": number(row.get("vmax_kt")),
                    "pressure_hpa": number(row.get("pressure_hpa")),
                    "source": "IBTrACS WP v04r01",
                    "wind_convention": "IBTrACS USA provisional sustained wind",
                }
            )
    return points


def parse_jma() -> tuple[dict, list[dict], list[dict]]:
    target = json.loads(JMA_TARGET_PATH.read_text(encoding="utf-8"))[0]
    base = ROOT / "data" / "jma" / target["tropicalCyclone"]
    forecast = json.loads((base.parent / f"{base.name}_forecast.json").read_text(encoding="utf-8"))
    specs = json.loads((base.parent / f"{base.name}_specifications.json").read_text(encoding="utf-8"))
    spec_by_lead = {item.get("advancedHours"): item for item in specs if isinstance(item.get("advancedHours"), int)}
    points = []
    for item in forecast:
        lead = item.get("advancedHours")
        if not isinstance(lead, int):
            continue
        spec = spec_by_lead.get(lead, {})
        wind = spec.get("maximumWind", {}).get("sustained", {}).get("kt")
        points.append(
            {
                "lead_hours": lead,
                "time": item["validtime"]["UTC"].replace("+00:00", "Z"),
                "lat": float(item["center"][0]),
                "lon": lon360(float(item["center"][1])),
                "vmax_kt": number(wind),
                "pressure_hpa": number(spec.get("pressure")),
                "category": spec.get("intensity", ""),
                "model": "JMA/RSMC Tokyo official forecast (10-min sustained)",
                "wind_convention": "10-min sustained",
            }
        )
    analysis = next(point for point in points if point["lead_hours"] == 0)
    track_item = next(item for item in forecast if item.get("advancedHours") == 0)
    track = track_item.get("track", {})
    analysis_track = [
        {"lat": float(lat), "lon": lon360(float(lon)), "source": "JMA analysis track"}
        for lat, lon in track.get("preTyphoon", []) + track.get("typhoon", [])
    ]
    return (
        {
            "name": target.get("tropicalCyclone"),
            "jma_typhoon_number": forecast[0].get("typhoonNumber"),
            "name_en": forecast[0].get("name", {}).get("en"),
            "issue_time_utc": forecast[0]["issue"]["UTC"],
            "source": "JMA live typhoon data API",
        },
        points,
        analysis_track,
    )


def resolve_token(token: str, reference: dt.datetime) -> dt.datetime:
    day = int(token[:2])
    hour = int(token[2:4])
    minute = int(token[4:6])
    candidates = []
    for delta_month in (-1, 0, 1):
        month_index = reference.year * 12 + reference.month - 1 + delta_month
        year, month0 = divmod(month_index, 12)
        try:
            candidates.append(dt.datetime(year, month0 + 1, day, hour, minute, tzinfo=dt.timezone.utc))
        except ValueError:
            pass
    return min(candidates, key=lambda value: abs((value - reference).total_seconds()))


def parse_jtwc() -> tuple[dict, list[dict]]:
    text = JTWC_PATH.read_text(encoding="utf-8", errors="replace")
    remarks_date = re.search(r"(\d{2})([A-Z]{3})(\d{2})\.", text)
    if not remarks_date:
        raise RuntimeError("JTWC issue date not found")
    issue_day, issue_month, issue_year = remarks_date.groups()
    issue_date = dt.datetime.strptime(
        f"{issue_day}{issue_month}{issue_year}", "%d%b%y"
    ).replace(tzinfo=dt.timezone.utc)
    header = re.search(r"WTPN\d+\s+PGTW\s+(\d{6})", text)
    issue_time = resolve_token(header.group(1), issue_date) if header else issue_date
    warning = re.search(
        r"WARNING POSITION:\s*(\d{6})Z\s+---\s+NEAR\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])",
        text,
    )
    current_wind = re.search(
        r"PRESENT WIND DISTRIBUTION:.*?MAX SUSTAINED WINDS\s*-\s*([0-9.]+)\s*KT",
        text,
        re.S,
    )
    current_pressure = re.search(r"MINIMUM CENTRAL PRESSURE AT \d+Z IS\s+([0-9.]+)\s+MB", text)
    if not warning or not current_wind:
        raise RuntimeError("JTWC current warning position or wind not found")

    def signed(value: str, hemisphere: str) -> float:
        return float(value) * (-1 if hemisphere in {"S", "W"} else 1)

    points = [
        {
            "lead_hours": 0,
            "time": resolve_token(warning.group(1), issue_time).isoformat().replace("+00:00", "Z"),
            "lat": signed(warning.group(2), warning.group(3)),
            "lon": lon360(signed(warning.group(4), warning.group(5))),
            "vmax_kt": float(current_wind.group(1)),
            "pressure_hpa": number(current_pressure.group(1)) if current_pressure else None,
            "category": category(float(current_wind.group(1))),
            "model": "JTWC official forecast (Warning 019)",
            "wind_convention": "1-min sustained",
        }
    ]
    forecast_pattern = re.compile(
        r"(\d+)\s+HRS, VALID AT:\s*(\d{6})Z\s+---\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])"
        r".*?MAX SUSTAINED WINDS\s*-\s*([0-9.]+)\s*KT",
        re.S,
    )
    for match in forecast_pattern.finditer(text):
        lead = int(match.group(1))
        wind = float(match.group(7))
        points.append(
            {
                "lead_hours": lead,
                "time": resolve_token(match.group(2), issue_time).isoformat().replace("+00:00", "Z"),
                "lat": signed(match.group(3), match.group(4)),
                "lon": lon360(signed(match.group(5), match.group(6))),
                "vmax_kt": wind,
                "pressure_hpa": None,
                "category": category(wind),
                "model": "JTWC official forecast (Warning 019)",
                "wind_convention": "1-min sustained",
            }
        )
    if len(points) != 9:
        raise RuntimeError(f"Expected JTWC analysis plus 8 forecast points, found {len(points)}")
    return (
        {
            "warning_number": 19,
            "issue_time_utc": issue_time.isoformat().replace("+00:00", "Z"),
            "source": "JTWC current WP1226 product",
        },
        points,
    )


def draw_png(payload: dict) -> None:
    coastlines = load_coastlines()
    v36, ibtracs = payload["v36"]["forecast"], payload["ibtracs"]
    jtwc, jma = payload["official"]["jtwc"], payload["official"]["jma"]
    jma_track = payload["jma_analysis_track"]
    figure, (world, detail) = plt.subplots(1, 2, figsize=(19, 9.5), gridspec_kw={"width_ratios": [1.65, 1.0]}, constrained_layout=True)
    for axis in (world, detail):
        draw_world(axis, coastlines, detail=axis is detail)
        plot_track(axis, ibtracs, color="#111827", linestyle="-", linewidth=2.5)
        plot_track(axis, jma_track, color="#64748b", linestyle=":", linewidth=1.6)
        plot_track(axis, v36, color="#dc2626", linestyle="-", linewidth=2.7)
        plot_track(axis, jtwc, color="#c2410c", linestyle="--", linewidth=2.1)
        plot_track(axis, jma, color="#047857", linestyle="-.", linewidth=2.1)
        axis.scatter([lon360(p["lon"]) for p in ibtracs], [p["lat"] for p in ibtracs], s=14, color="#111827", zorder=6)
        for point in v36[::2]:
            radius_lat = max(float(point["track_spread_km"]) / 111.2, 0.2)
            radius_lon = radius_lat / max(0.15, abs(math.cos(math.radians(point["latitude"]))))
            axis.add_patch(Ellipse((lon360(point["longitude"]), point["latitude"]), 2 * radius_lon, 2 * radius_lat, facecolor="#dc2626", edgecolor="#dc2626", alpha=0.06, linewidth=0.8, zorder=2))
        axis.scatter([lon360(p["longitude"]) for p in v36], [p["latitude"] for p in v36], s=36, c=[intensity_color(p["vmax_kt"]) for p in v36], edgecolors="white", linewidths=0.75, zorder=7)
        for series, color in ((jtwc, "#c2410c"), (jma, "#047857")):
            axis.scatter([lon360(p["lon"]) for p in series], [p["lat"] for p in series], s=28, facecolors="white", edgecolors=color, linewidths=1.4, zorder=7)
    for point in v36:
        if point["lead_hours"] % 24 == 0:
            annotate_point(detail, lon360(point["longitude"]), point["latitude"], f"+{point['lead_hours']}h\n{point['vmax_kt']:.0f} kt\n{point['central_pressure_hpa']:.0f} hPa", "#dc2626")
    annotate_point(detail, lon360(jtwc[0]["lon"]), jtwc[0]["lat"], f"JTWC\n{jtwc[0]['vmax_kt']:.0f} kt", "#c2410c", dx=-5.5, dy=-2.2)
    annotate_point(detail, lon360(jma[0]["lon"]), jma[0]["lat"], f"JMA\n{jma[0]['vmax_kt']:.0f} kt", "#047857", dx=1.0, dy=-3.0)
    world.set_title("Current official overlay", loc="left", fontsize=13, fontweight="bold")
    detail.set_title("Western Pacific detail", loc="left", fontsize=13, fontweight="bold")
    figure.suptitle("Dolphin: v36A with IBTrACS, current JMA, and JTWC Warning 019\n" f"v36 issue {payload['v36']['issue_time_utc']} | official issue {payload['official']['jtwc_meta']['issue_time_utc']}", fontsize=16, fontweight="bold")
    detail.legend(handles=[Line2D([0], [0], color="#111827", linewidth=2.5, label="IBTrACS observations"), Line2D([0], [0], color="#64748b", linewidth=1.6, linestyle=":", label="JMA analysis track"), Line2D([0], [0], color="#dc2626", linewidth=2.7, label="v36A ensemble mean"), Line2D([0], [0], color="#c2410c", linewidth=2.1, linestyle="--", label="JTWC Warning 019"), Line2D([0], [0], color="#047857", linewidth=2.1, linestyle="-.", label="JMA official forecast"), Patch(facecolor="#dc2626", alpha=0.08, edgecolor="#dc2626", label="v36 member spread")], loc="lower left", fontsize=8.0, framealpha=0.96)
    figure.text(0.5, 0.005, "v36/JTWC wind: 1-min sustained kt. JMA wind: 10-min sustained kt. v36 pressure: central pressure; v36 fields are parametric diagnostics.", ha="center", fontsize=9, color="#334155")
    figure.savefig(OUT_PNG, dpi=190, facecolor="white")
    figure.savefig(LEGACY_PNG, dpi=190, facecolor="white")
    plt.close(figure)


def map_point(point: dict, model: str) -> dict:
    if model == "v36":
        return {"model": "v36A ensemble mean", "lead_hours": point["lead_hours"], "time": point["valid_time_utc"], "lat": point["latitude"], "lon": point["longitude"], "vmax_kt": point["vmax_kt"], "pressure_hpa": point["central_pressure_hpa"], "track_spread_km": point["track_spread_km"], "wind_convention": "1-min sustained"}
    return {key: point.get(key) for key in ("model", "lead_hours", "time", "lat", "lon", "vmax_kt", "pressure_hpa", "category", "wind_convention")}


def draw_html(payload: dict) -> None:
    data = {"v36": [map_point(p, "v36") for p in payload["v36"]["forecast"]], "ibtracs": payload["ibtracs"], "jma_analysis_track": payload["jma_analysis_track"], "jtwc": [map_point(p, "jtwc") for p in payload["official"]["jtwc"]], "jma": [map_point(p, "jma") for p in payload["official"]["jma"]]}
    data_json = safe_json(data).replace("</", "<\\/")
    v36_final, jtwc_final, jma_final = data["v36"][-1], data["jtwc"][-1], data["jma"][-1]
    rows = "".join(f"<tr><td>+{p['lead_hours']} h</td><td>{p['time']}</td><td>{p['lat']:.2f}</td><td>{p['lon']:.2f}E</td><td>{p['vmax_kt']:.0f} kt</td><td>{p['pressure_hpa']:.0f} hPa</td><td>{p['track_spread_km']:.0f} km</td></tr>" for p in data["v36"])
    official_rows = "".join(f"<tr><td>{html.escape(label)}</td><td>{p['lead_hours']} h</td><td>{p['time']}</td><td>{p['lat']:.2f}</td><td>{p['lon']:.2f}E</td><td>{p['vmax_kt']:.0f} kt</td><td>{'' if p.get('pressure_hpa') is None else f'{p["pressure_hpa"]:.0f} hPa'}</td></tr>" for label, series in (("JTWC", data["jtwc"]), ("JMA", data["jma"])) for p in series)
    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dolphin v36A current official comparison</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""><style>:root{{color-scheme:light;--ink:#17202a;--muted:#5b6875;--line:#d4dde5;--surface:#fff;--bg:#eef3f6}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1500px;margin:0 auto;padding:24px}}h1{{margin:0 0 6px;font-size:28px}}h2{{margin:0 0 10px;font-size:17px}}.sub{{color:var(--muted);max-width:1100px;margin:0 0 16px}}.notice{{background:#fff8e6;border-left:4px solid #d97706;padding:11px 14px;margin:12px 0 16px}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0}}.card,.section,.mapwrap{{background:var(--surface);border:1px solid var(--line);border-radius:7px}}.card{{padding:12px}}.card b{{display:block;font-size:17px;margin-top:4px}}.muted,.source{{color:var(--muted)}}#map{{height:670px}}.legend{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 18px;padding:13px 15px;border-top:1px solid var(--line)}}.legend span{{display:flex;align-items:center;gap:8px}}.swatch{{width:27px;height:3px;display:inline-block;flex:none}}.dot{{width:11px;height:11px;border-radius:50%;display:inline-block;border:2px solid white;box-shadow:0 0 0 1px #777;flex:none}}.section{{margin-top:16px;padding:15px;overflow:auto}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid var(--line);padding:7px 8px;text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{color:var(--muted);font-size:12px}}.source{{font-size:12px}}a{{color:#0f5f9f}}.leaflet-tooltip{{font-size:12px}}.intensity-label{{background:transparent;border:0;color:#17202a;font-weight:700;text-shadow:0 1px 2px white,0 -1px 2px white,1px 0 2px white,-1px 0 2px white}}@media(max-width:900px){{main{{padding:12px}}h1{{font-size:22px}}.cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}#map{{height:540px}}.legend{{grid-template-columns:1fr}}}}</style></head><body><main><h1>Dolphin: v36A + IBTrACS + current JMA/JTWC</h1><p class="sub">The red line is the completed five-member v36A forecast. Black points are the IBTrACS history used for source provenance. The gray dotted line is the current JMA analysis track. Official JTWC Warning 019 and the current JMA forecast are comparison lines only.</p><div class="notice"><b>Issue times are different.</b> v36A was issued at {html.escape(payload['v36']['issue_time_utc'])}; JTWC Warning 019 was issued at {html.escape(payload['official']['jtwc_meta']['issue_time_utc'])}; JMA issued its current advisory at {html.escape(payload['official']['jma_meta']['issue_time_utc'])}. This is a current overlay, not a same-issue skill score. Wind labels use 1-minute sustained kt for v36/JTWC and 10-minute sustained kt for JMA.</div><div class="cards"><div class="card"><span class="muted">v36 issue</span><b>{html.escape(payload['v36']['issue_time_utc'])}</b><span class="muted">5 members, +6 to +120 h</span></div><div class="card"><span class="muted">v36 +120 h</span><b>{v36_final['lat']:.2f}N, {v36_final['lon']:.2f}E</b><span class="muted">{v36_final['vmax_kt']:.0f} kt / {v36_final['pressure_hpa']:.0f} hPa</span></div><div class="card"><span class="muted">JTWC current +120 h</span><b>{jtwc_final['lat']:.2f}N, {jtwc_final['lon']:.2f}E</b><span class="muted">{jtwc_final['vmax_kt']:.0f} kt, 1-min</span></div><div class="card"><span class="muted">JMA current +117 h</span><b>{jma_final['lat']:.2f}N, {jma_final['lon']:.2f}E</b><span class="muted">{jma_final['vmax_kt']:.0f} kt, 10-min</span></div></div><div class="mapwrap"><div id="map" aria-label="Dolphin current v36A, IBTrACS, JMA, and JTWC comparison map"></div><div class="legend"><span><i class="swatch" style="background:#111827"></i>IBTrACS observations</span><span><i class="swatch" style="background:#64748b;border-top:2px dotted #64748b;height:0"></i>JMA analysis track</span><span><i class="swatch" style="background:#dc2626"></i>v36A ensemble mean + spread</span><span><i class="swatch" style="background:#c2410c;border-top:2px dashed #c2410c;height:0"></i>JTWC Warning 019</span><span><i class="swatch" style="background:#047857;border-top:2px dashed #047857;height:0"></i>JMA official forecast</span><span><i class="dot" style="background:#7c3aed"></i>v36 intensity marker colors</span></div></div><section class="section"><h2>v36A mean forecast</h2><table><thead><tr><th>Lead</th><th>Valid UTC</th><th>Lat</th><th>Lon</th><th>Wind</th><th>Pressure</th><th>Spread</th></tr></thead><tbody>{rows}</tbody></table></section><section class="section"><h2>Current official forecasts</h2><table><thead><tr><th>Source</th><th>Lead</th><th>Valid UTC</th><th>Lat</th><th>Lon</th><th>Wind</th><th>Pressure</th></tr></thead><tbody>{official_rows}</tbody></table></section><section class="section source"><h2>Provenance</h2><p>IBTrACS input file: <code>{html.escape(str(IBTRACS_PATH.relative_to(ROOT)))}</code>. JMA live data: <a href="https://www.jma.go.jp/bosai/map.html#3/30/140/&elem=root&typhoon=all&lang=en&contents=typhoon">RSMC Tokyo current typhoon data</a>. JTWC: <a href="https://www.metoc.navy.mil/jtwc/products/wp1226web.txt">WP1226 current product</a>. JMA official forecast points are 10-minute sustained winds; JTWC and v36 are 1-minute sustained winds.</p><p>v36A pressure fields are parametric reconstructions from predicted center pressure, wind, RMW, and wind radii; they are not learned ERA5 grids.</p></section></main><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script><script>const DATA={data_json};const map=L.map('map',{{worldCopyJump:true,zoomControl:true}}).setView([20,170],3);L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:7,attribution:'&copy; OpenStreetMap contributors'}}).addTo(map);const colors={{ibtracs:'#111827',analysis:'#64748b',v36:'#dc2626',jtwc:'#c2410c',jma:'#047857'}};const line=(items,color,options={{}})=>L.polyline(items.map(p=>[p.lat,p.lon]),Object.assign({{color,weight:3,opacity:.9}},options)).addTo(map);const windColor=kt=>kt>=137?'#7c3aed':kt>=113?'#b91c1c':kt>=96?'#dc2626':kt>=83?'#ea580c':kt>=64?'#ca8a04':kt>=34?'#0891b2':'#64748b';const popup=p=>`<b>${{p.model||'IBTrACS'}}</b><br><b>Valid:</b> ${{p.time||'JMA analysis path'}}<br><b>Position:</b> ${{p.lat.toFixed(2)}}N, ${{p.lon.toFixed(2)}}E<br>${{p.vmax_kt==null?'':`<b>Wind:</b> ${{p.vmax_kt.toFixed(0)}} kt (${{p.wind_convention||''}})<br>`}}${{p.pressure_hpa==null?'':`<b>Pressure:</b> ${{p.pressure_hpa.toFixed(0)}} hPa<br>`}}${{p.track_spread_km==null?'':`<b>Track spread:</b> ${{p.track_spread_km.toFixed(0)}} km`}}`;line(DATA.ibtracs,colors.ibtracs,{{weight:3}});DATA.ibtracs.forEach(p=>L.circleMarker([p.lat,p.lon],{{radius:3,color:colors.ibtracs,fillColor:colors.ibtracs,fillOpacity:.9}}).bindPopup(popup(p)).addTo(map));line(DATA.jma_analysis_track,colors.analysis,{{weight:2,dashArray:'2 6'}});line(DATA.v36,colors.v36,{{weight:4}});DATA.v36.forEach((p,i)=>{{if(i%2===0)L.circle([p.lat,p.lon],{{radius:(p.track_spread_km||1)*1000,color:colors.v36,weight:1,opacity:.25,fillOpacity:.035}}).addTo(map);const m=L.circleMarker([p.lat,p.lon],{{radius:6,color:'#fff',weight:1.5,fillColor:windColor(p.vmax_kt),fillOpacity:.98}}).bindPopup(popup(p)).addTo(map);if(p.lead_hours%24===0)m.bindTooltip(`+${{p.lead_hours}}h · ${{p.vmax_kt.toFixed(0)}} kt · ${{p.pressure_hpa.toFixed(0)}} hPa`,{{permanent:true,direction:'right',className:'intensity-label',offset:[7,0]}})}});line(DATA.jtwc,colors.jtwc,{{weight:3,dashArray:'9 7'}});DATA.jtwc.forEach(p=>L.circleMarker([p.lat,p.lon],{{radius:5,color:colors.jtwc,fillColor:'#fff',fillOpacity:1,weight:2}}).bindPopup(popup(p)).addTo(map));line(DATA.jma,colors.jma,{{weight:3,dashArray:'3 7'}});DATA.jma.forEach(p=>L.circleMarker([p.lat,p.lon],{{radius:5,color:colors.jma,fillColor:'#fff',fillOpacity:1,weight:2}}).bindPopup(popup(p)).addTo(map));const all=DATA.ibtracs.concat(DATA.jma_analysis_track,DATA.v36,DATA.jtwc,DATA.jma).map(p=>[p.lat,p.lon]);map.fitBounds(all,{{padding:[24,24]}});</script></body></html>'''
    OUT_HTML.write_text(html_text, encoding="utf-8")
    LEGACY_HTML.write_text(html_text, encoding="utf-8")


def main() -> None:
    v36 = json.loads(V36_PATH.read_text(encoding="utf-8"))
    jma_meta, jma, jma_track = parse_jma()
    jtwc_meta, jtwc = parse_jtwc()
    payload = {
        "v36": v36,
        "ibtracs": read_ibtracs(),
        "jma_analysis_track": jma_track,
        "official": {"jtwc_meta": jtwc_meta, "jtwc": jtwc, "jma_meta": jma_meta, "jma": jma},
        "comparison_note": "Current official overlay. v36A was issued earlier than the refreshed official advisories; this is not a same-issue skill score.",
        "sources": {
            "ibtracs": "https://www.ncei.noaa.gov/products/international-best-track-archive",
            "jma_live": "https://www.jma.go.jp/bosai/map.html#3/30/140/&elem=root&typhoon=all&lang=en&contents=typhoon",
            "jtwc_product": "https://www.metoc.navy.mil/jtwc/products/wp1226web.txt",
        },
    }
    OUT_DATA.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draw_png(payload)
    draw_html(payload)
    print(json.dumps({
        "data": str(OUT_DATA),
        "html": str(OUT_HTML),
        "html_legacy_alias": str(LEGACY_HTML),
        "png": str(OUT_PNG),
        "png_legacy_alias": str(LEGACY_PNG),
        "ibtracs_points": len(payload["ibtracs"]),
        "v36_points": len(v36["forecast"]),
        "jtwc_points": len(jtwc),
        "jma_points": len(jma),
    }, indent=2))


if __name__ == "__main__":
    main()
