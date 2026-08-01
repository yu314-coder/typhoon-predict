#!/usr/bin/env python3
"""Render current v37 against raw IBTrACS/JMA histories and official forecasts.

Only the local current v37 forecast is treated as a model output. JMA and JTWC
forecast points are read from the saved issue-time capture strictly as
comparison overlays; they are never passed to the model.
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.draw_dolphin_v36_world_map import draw_world, intensity_color, load_coastlines, lon360, safe_json
from scripts.predict_ibtracs_jma_only import parse_ibtracs

V37_PATH = ROOT / "track_build" / "dolphin_v37_current_ibtracs_jma_forecast.json"
JMA_PATH = ROOT / "data" / "jma" / "TC2615_forecast.json"
OFFICIAL_PATH = ROOT / "track_build" / "dolphin_v36_ibtracs_jma_official_comparison.json"
OUT_JSON = ROOT / "track_build" / "dolphin_v37_current_ibtracs_jma_official_comparison.json"
OUT_HTML = ROOT / "paper" / "dolphin_v37_current_ibtracs_jma_official_world_map.html"
OUT_PNG = ROOT / "paper" / "dolphin_v37_current_ibtracs_jma_official_world_map.png"


def raw_jma_analysis() -> list[dict]:
    payload = json.loads(JMA_PATH.read_text(encoding="utf-8"))
    analysis = next(
        row for row in payload
        if isinstance(row.get("part"), dict) and row["part"].get("en") == "Analysis"
    )
    track = analysis["track"]["typhoon"]
    valid_time = analysis["validtime"]["UTC"]
    from datetime import datetime, timedelta

    end = datetime.fromisoformat(valid_time.replace("Z", "+00:00"))
    points = []
    for index, pair in enumerate(track):
        when = end - timedelta(hours=3 * (len(track) - 1 - index))
        points.append({
            "time": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lat": float(pair[0]),
            "lon": float(pair[1]) % 360.0,
            "model": "JMA current analysis position",
            "vmax_kt": None,
            "pressure_hpa": None,
            "wind_convention": "analysis position only",
        })
    return points


def raw_ibtracs() -> list[dict]:
    points = []
    for row in parse_ibtracs(ROOT / "data" / "ibtracs" / "ibtracs.WP.list.v04r01.csv"):
        points.append({
            "time": row["time_utc"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]) % 360.0,
            "model": "IBTrACS observed",
            "vmax_kt": float(row["vmax_kt"]) if row.get("vmax_kt") == row.get("vmax_kt") else None,
            "pressure_hpa": float(row["pressure_hpa"]) if row.get("pressure_hpa") == row.get("pressure_hpa") else None,
            "wind_convention": "IBTrACS USA provisional",
        })
    return points


def load_data() -> dict:
    v37 = json.loads(V37_PATH.read_text(encoding="utf-8"))
    saved = json.loads(OFFICIAL_PATH.read_text(encoding="utf-8"))
    forecast = [
        {
            "time": point["valid_time_utc"],
            "lat": point["latitude"],
            "lon": point["longitude"],
            "vmax_kt": point["vmax_kt"],
            "pressure_hpa": point["central_pressure_hpa"],
            "lead_hours": point["lead_hours"],
            "track_spread_km": point["track_spread_km"],
            "model": "v37 current NOAA-steering route",
            "wind_convention": "1-min model estimate",
        }
        for point in v37["forecast"]
    ]
    return {
        "model": v37["model"],
        "issue_time_utc": v37["issue_time_utc"],
        "route_backbone": v37["route_backbone"],
        "structure_branch": v37["structure_branch"],
        "input_policy": "Current JMA analysis initializes the model; free NOAA GFS/GEFS fields provide the route steering. IBTrACS is an independent observed-history overlay. Official JMA/JTWC forecasts are comparison-only.",
        "ibtracs_observed": raw_ibtracs(),
        "jma_analysis": raw_jma_analysis(),
        "v37": forecast,
        "jma_official": saved["official"]["jma"],
        "jtwc_official": saved["official"]["jtwc"],
        "sources": {
            "ibtracs_local": str((ROOT / "data" / "ibtracs" / "ibtracs.WP.list.v04r01.csv").relative_to(ROOT)),
            "jma_current_local": str(JMA_PATH.relative_to(ROOT)),
            "gfs_gefs_route": "cached local NOAA NOMADS/GEFS GRIB-derived route fields",
            "official_comparison_capture": str(OFFICIAL_PATH.relative_to(ROOT)),
        },
        "comparison_only": ["jma_official", "jtwc_official"],
    }


def draw_png(data: dict) -> None:
    coastlines = load_coastlines()
    figure, axes = plt.subplots(
        1, 2, figsize=(20, 9.5), gridspec_kw={"width_ratios": [1.65, 1.0]}, constrained_layout=True
    )
    for axis in axes:
        draw_world(axis, coastlines, detail=axis is axes[1])
        for key, color, width, style in (
            ("ibtracs_observed", "#111827", 2.5, "-"),
            ("jma_analysis", "#64748b", 1.8, ":"),
            ("v37", "#dc2626", 3.2, "-"),
            ("jma_official", "#047857", 2.2, "--"),
            ("jtwc_official", "#c2410c", 2.2, "-."),
        ):
            points = data[key]
            axis.plot([lon360(p["lon"]) for p in points], [p["lat"] for p in points], color=color, linewidth=width, linestyle=style, zorder=5 if key == "v37" else 4)
        obs = data["ibtracs_observed"]
        axis.scatter([lon360(p["lon"]) for p in obs], [p["lat"] for p in obs], color="#111827", s=13, zorder=6)
        for point in data["v37"][::2]:
            radius_lat = max(float(point["track_spread_km"]) / 111.2, 0.2)
            radius_lon = radius_lat / max(0.15, abs(math.cos(math.radians(point["lat"]))))
            axis.add_patch(Ellipse(
                (lon360(point["lon"]), point["lat"]),
                2 * radius_lon,
                2 * radius_lat,
                facecolor="#dc2626",
                edgecolor="#dc2626",
                alpha=0.055,
                linewidth=0.7,
                zorder=2,
            ))
        axis.scatter(
            [lon360(p["lon"]) for p in data["v37"]],
            [p["lat"] for p in data["v37"]],
            s=36,
            c=[intensity_color(p["vmax_kt"]) for p in data["v37"]],
            edgecolors="white",
            linewidths=0.8,
            zorder=7,
        )
        if axis is axes[1]:
            for point in data["v37"]:
                if point["lead_hours"] % 24 == 0:
                    axis.annotate(
                        f"+{point['lead_hours']}h\n{point['vmax_kt']:.0f} kt\n{point['pressure_hpa']:.0f} hPa",
                        (lon360(point["lon"]), point["lat"]),
                        xytext=(6, 5),
                        textcoords="offset points",
                        fontsize=7,
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#dc2626", "alpha": 0.9, "linewidth": 0.7},
                        zorder=9,
                    )
    axes[1].legend(handles=[
        Line2D([0], [0], color="#111827", marker="o", linewidth=2.3, markersize=4, label="IBTrACS observations"),
        Line2D([0], [0], color="#64748b", linestyle=":", linewidth=1.8, label="JMA current analysis"),
        Line2D([0], [0], color="#dc2626", linewidth=3.2, label="v37 generated forecast"),
        Line2D([0], [0], color="#047857", linestyle="--", linewidth=2.2, label="JMA official comparison"),
        Line2D([0], [0], color="#c2410c", linestyle="-.", linewidth=2.2, label="JTWC official comparison"),
    ], loc="lower left", fontsize=7.8, framealpha=0.93)
    figure.suptitle("Dolphin: v37 current route versus JMA and JTWC forecasts", fontsize=15)
    figure.text(0.5, 0.01, "Official JMA/JTWC lines are comparison-only and are not model inputs. Red intensity markers show v37 wind estimates; labels include central pressure.", ha="center", fontsize=8.5, color="#475569")
    figure.savefig(OUT_PNG, dpi=190, facecolor="white")
    plt.close(figure)


def draw_html(data: dict) -> None:
    embedded = safe_json(data).replace("</", "<\\/")
    final = data["v37"][-1]
    rows = []
    for point in data["v37"]:
        if point["lead_hours"] % 24 == 0 or point["lead_hours"] == 120:
            rows.append(
                f"<tr><td>+{point['lead_hours']} h</td><td>{point['time']}</td><td>{point['lat']:.2f}N, {point['lon']:.2f}E</td><td>{point['vmax_kt']:.0f} kt</td><td>{point['pressure_hpa']:.0f} hPa</td><td>{point['track_spread_km']:.0f} km</td></tr>"
            )
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dolphin v37 current data versus official forecasts</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
<style>
:root {{ color-scheme:light; --ink:#17202a; --muted:#5b6875; --line:#d4dde5; --surface:#fff; --bg:#eef3f6; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1500px; margin:0 auto; padding:24px; }} h1 {{ margin:0 0 6px; font-size:28px; }} h2 {{ font-size:17px; }} .sub {{ color:var(--muted); max-width:1100px; }} .notice {{ background:#fff8e6; border-left:4px solid #d97706; padding:11px 14px; margin:12px 0 16px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:12px 0; }} .card,.section,.mapwrap {{ background:var(--surface); border:1px solid var(--line); border-radius:7px; }} .card {{ padding:12px; }} .card b {{ display:block; font-size:17px; margin-top:4px; }} .muted {{ color:var(--muted); }}
#map {{ height:680px; }} .legend {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px 18px; padding:13px 15px; border-top:1px solid var(--line); }} .legend span {{ display:flex; align-items:center; gap:8px; }} .swatch {{ width:28px; height:3px; display:inline-block; }} .section {{ margin-top:16px; padding:15px; overflow:auto; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; white-space:nowrap; }} th {{ color:var(--muted); font-size:12px; }} small {{ color:var(--muted); }} code {{ font-size:12px; }} .intensity-label {{ background:transparent; border:0; color:#17202a; font-weight:700; text-shadow:0 1px 2px white,0 -1px 2px white,1px 0 2px white,-1px 0 2px white; }}
@media(max-width:900px) {{ main {{ padding:12px; }} h1 {{ font-size:22px; }} .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} #map {{ height:560px; }} .legend {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>Dolphin: v37 current-data forecast</h1>
<p class="sub">Red is the locally generated v37 route initialized from current JMA analysis data and evaluated with the local IBTrACS history shown on the map. Green and orange are official JMA/JTWC forecast overlays for comparison only.</p>
<div class="notice"><b>Input boundary.</b> The model route uses current JMA analysis plus cached free NOAA GFS/GEFS steering fields. IBTrACS is an observed-history source shown independently. The JMA and JTWC forecast arrays are never passed to the model. Issue time: {html.escape(data["issue_time_utc"])}.</div>
<div class="cards"><div class="card"><span class="muted">v37 +120 h</span><b>{final["lat"]:.2f}N, {final["lon"]:.2f}E</b><span class="muted">{final["vmax_kt"]:.0f} kt / {final["pressure_hpa"]:.0f} hPa</span></div><div class="card"><span class="muted">Track spread +120 h</span><b>{final["track_spread_km"]:.0f} km</b><span class="muted">ensemble mean route</span></div><div class="card"><span class="muted">Route</span><b>GFS/GEFS adaptive</b><span class="muted">free public fields</span></div><div class="card"><span class="muted">Structure</span><b>v37G spatial ensemble</b><span class="muted">wind, pressure, RMW, radii</span></div></div>
<div class="mapwrap"><div id="map" aria-label="Dolphin v37 current data and official forecast comparison map"></div><div class="legend"><span><i class="swatch" style="background:#111827"></i>IBTrACS observations</span><span><i class="swatch" style="background:#64748b;border-top:2px dotted #64748b;height:0"></i>JMA current analysis</span><span><i class="swatch" style="background:#dc2626"></i>v37 generated forecast</span><span><i class="swatch" style="background:#047857;border-top:2px dashed #047857;height:0"></i>JMA official comparison</span><span><i class="swatch" style="background:#c2410c;border-top:2px dashed #c2410c;height:0"></i>JTWC official comparison</span></div></div>
<section class="section"><h2>v37 forecast values</h2><table><thead><tr><th>Lead</th><th>Valid UTC</th><th>Position</th><th>Wind</th><th>Pressure</th><th>Spread</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="section"><h2>Source files</h2><p><code>{html.escape(data["sources"]["ibtracs_local"])}</code> and <code>{html.escape(data["sources"]["jma_current_local"])}</code> provide the observed histories. Route steering uses the cached public NOAA GFS/GEFS fields. Official forecast lines are stored only under comparison keys and are not model inputs.</p></section>
</main><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script><script>
const DATA={embedded};
const map=L.map('map',{{worldCopyJump:true,zoomControl:true}}).setView([20,165],3);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:7,attribution:'&copy; OpenStreetMap contributors'}}).addTo(map);
const colors={{ib:'#111827',analysis:'#64748b',v37:'#dc2626',jma:'#047857',jtwc:'#c2410c'}};
const line=(items,color,options,target)=>L.polyline(items.map(p=>[p.lat,p.lon]),Object.assign({{color:color,weight:3,opacity:.9}},options||{{}})).addTo(target||map);
const windColor=kt=>kt>=137?'#7c3aed':kt>=113?'#b91c1c':kt>=96?'#dc2626':kt>=83?'#ea580c':kt>=64?'#ca8a04':kt>=34?'#0891b2':'#64748b';
const popup=p=>'<b>'+(p.model||'track')+'</b><br><b>Valid:</b> '+(p.time||'')+'<br><b>Position:</b> '+p.lat.toFixed(2)+'N, '+p.lon.toFixed(2)+'E<br>'+(p.vmax_kt==null?'':'<b>Wind:</b> '+p.vmax_kt.toFixed(0)+' kt<br>')+(p.pressure_hpa==null?'':'<b>Pressure:</b> '+p.pressure_hpa.toFixed(0)+' hPa<br>')+(p.track_spread_km==null?'':'<b>Spread:</b> '+p.track_spread_km.toFixed(0)+' km');
const obs=L.layerGroup().addTo(map);line(DATA.ibtracs_observed,colors.ib,{{weight:3}},obs);DATA.ibtracs_observed.forEach(p=>L.circleMarker([p.lat,p.lon],{{radius:3,color:colors.ib,fillColor:colors.ib,fillOpacity:.9}}).bindPopup(popup(p)).addTo(obs));
const analysis=L.layerGroup().addTo(map);line(DATA.jma_analysis,colors.analysis,{{weight:2,dashArray:'2 6'}},analysis);
const model=L.layerGroup().addTo(map);line(DATA.v37,colors.v37,{{weight:4}},model);DATA.v37.forEach((p,i)=>{{if(i%2===0)L.circle([p.lat,p.lon],{{radius:(p.track_spread_km||1)*1000,color:colors.v37,weight:1,opacity:.25,fillOpacity:.035}}).addTo(model);const m=L.circleMarker([p.lat,p.lon],{{radius:6,color:'#fff',weight:1.5,fillColor:windColor(p.vmax_kt),fillOpacity:.98}}).bindPopup(popup(p)).addTo(model);if(p.lead_hours%24===0)m.bindTooltip('+ '+p.lead_hours+'h · '+p.vmax_kt.toFixed(0)+' kt · '+p.pressure_hpa.toFixed(0)+' hPa',{{permanent:true,direction:'right',className:'intensity-label',offset:[7,0]}});}});
const official=L.layerGroup().addTo(map);line(DATA.jma_official,colors.jma,{{weight:3,dashArray:'3 7'}},official);line(DATA.jtwc_official,colors.jtwc,{{weight:3,dashArray:'8 6'}},official);
const all=[...DATA.ibtracs_observed,...DATA.jma_analysis,...DATA.v37,...DATA.jma_official,...DATA.jtwc_official].map(p=>[p.lat,p.lon]);map.fitBounds(all,{{padding:[24,24]}});
L.control.layers({{}},{{'IBTrACS observations':obs,'JMA current analysis':analysis,'v37 generated route':model,'JMA/JTWC comparisons':official}},{{collapsed:false}}).addTo(map);
</script></body></html>"""
    OUT_HTML.write_text(html_text, encoding="utf-8")


def main() -> None:
    data = load_data()
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    draw_png(data)
    draw_html(data)
    print(json.dumps({"json": str(OUT_JSON), "html": str(OUT_HTML), "png": str(OUT_PNG), "issue": data["issue_time_utc"], "v37_final": data["v37"][-1], "ibtracs_points": len(data["ibtracs_observed"]), "jma_analysis_points": len(data["jma_analysis"])}, indent=2))


if __name__ == "__main__":
    main()
