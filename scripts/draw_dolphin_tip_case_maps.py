#!/usr/bin/env python3
"""Render the final-policy Dolphin and Tip case forecasts as viewable maps.

This is a visualization-only script. It does not refit a model or fetch a new
forecast. The final v56 policy defaults to v37N; the v56 gate is not applied to
these case pages because Dolphin is outside the fixed feature archive and Tip
is an out-of-training 1979 hindcast.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
COAST_PATH = ROOT / "paper" / "ne_110m_admin_0_countries.geojson"
DOLPHIN_SOURCE = ROOT / "track_build" / "dolphin_v37p_v37n_same_issue_current.json"
TIP_SOURCE = ROOT / "track_build" / "tip_v37N_multi_timing_1979.json"
DOLPHIN_JSON = ROOT / "track_build" / "dolphin_final_model_map.json"
TIP_JSON = ROOT / "track_build" / "tip_final_model_map.json"
DOLPHIN_HTML = ROOT / "paper" / "dolphin_final_model_map.html"
TIP_HTML = ROOT / "paper" / "tip_final_model_map.html"
DOLPHIN_PNG = ROOT / "paper" / "dolphin_final_model_map.png"
TIP_PNG = ROOT / "paper" / "tip_final_model_map.png"


def lon180(value: float) -> float:
    value = float(value) % 360.0
    return value - 360.0 if value > 180.0 else value


def point(row: dict, *, time_key: str = "time", lon_key: str = "lon", lat_key: str = "lat") -> dict:
    result = {
        "time": row.get(time_key, row.get("valid_time_utc", row.get("time_utc", ""))),
        "lat": float(row[lat_key] if lat_key in row else row["latitude"]),
        "lon": lon180(row[lon_key] if lon_key in row else row["longitude"]),
    }
    for source, target in (("lead_hours", "lead_hours"), ("vmax_kt", "vmax_kt"), ("central_pressure_hpa", "pressure_hpa"), ("pressure_hpa", "pressure_hpa")):
        if source in row and target not in result:
            result[target] = row[source]
        elif source in row:
            result[target] = row[source]
    for source in ("central_pressure_hpa", "rmw_km", "wind_radii_km", "vmax_spread_kt", "pressure_spread_hpa", "rmw_spread_km", "wind_radii_spread_km", "pressure_map_features"):
        if source in row:
            result[source] = row[source]
    return result


def load_coastlines() -> dict:
    return json.loads(COAST_PATH.read_text(encoding="utf-8"))


def rings(geometry: dict):
    if geometry["type"] == "Polygon":
        yield from geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield from polygon


def draw_coastlines(axis, coastlines: dict) -> None:
    for feature in coastlines["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        for ring in rings(geometry):
            xs, ys = [], []
            previous = None
            for lon, lat in ring:
                x = lon180(lon)
                if previous is not None and abs(x - previous) > 180.0:
                    if len(xs) > 1:
                        axis.plot(xs, ys, color="#94a3b8", linewidth=0.45, zorder=1)
                    xs, ys = [], []
                xs.append(x)
                ys.append(lat)
                previous = x
            if len(xs) > 1:
                axis.plot(xs, ys, color="#94a3b8", linewidth=0.45, zorder=1)


def style_axis(axis, coastlines: dict, regional: bool) -> None:
    draw_coastlines(axis, coastlines)
    axis.set_facecolor("#e9f3f7")
    axis.grid(True, color="#b8cbd3", linewidth=0.45, alpha=0.7)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    if regional:
        axis.set_xlim(110, 180)
        axis.set_ylim(0, 55)
        axis.set_xticks(range(110, 181, 10))
        axis.set_yticks(range(0, 56, 10))
    else:
        axis.set_xlim(-180, 180)
        axis.set_ylim(-60, 70)
        axis.set_xticks(range(-180, 181, 60))
        axis.set_yticks(range(-60, 71, 20))


def xy(rows: list[dict]) -> tuple[list[float], list[float]]:
    return [row["lon"] for row in rows], [row["lat"] for row in rows]


def draw_line(axis, rows: list[dict], color: str, linestyle: str = "-", linewidth: float = 2.4, alpha: float = 1.0) -> None:
    xs, ys = xy(rows)
    axis.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=4)


def draw_dolphin_png(payload: dict, coastlines: dict) -> None:
    model_label = payload.get("model", "model route")
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.8), gridspec_kw={"width_ratios": [1.5, 1.0]}, constrained_layout=True)
    for axis, regional in zip(axes, (False, True)):
        style_axis(axis, coastlines, regional)
        draw_line(axis, payload["observed"], "#111827", linewidth=2.2)
        for member in payload.get("ensemble_forecasts", []):
            draw_line(axis, [payload["observed"][-1]] + member, "#ef4444", linewidth=0.55, alpha=0.12)
        draw_line(axis, [payload["observed"][-1]] + payload["forecast"], "#b91c1c", linewidth=3.0)
        draw_line(axis, payload["official"]["jtwc"], "#7c3aed", linestyle="--", linewidth=2.0)
        draw_line(axis, payload["official"]["jma"], "#2563eb", linestyle="--", linewidth=2.0)
        for item in payload["forecast"]:
            axis.scatter(item["lon"], item["lat"], s=22, color="#b91c1c", edgecolor="white", linewidth=0.8, zorder=6)
            if item.get("lead_hours", 0) % 24 == 0:
                axis.annotate(f"+{item['lead_hours']}h", (item["lon"], item["lat"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        axis.set_title("Regional view" if regional else "World view")
    axes[0].legend(handles=[
        Line2D([0], [0], color="#111827", linewidth=2.2, label="Observed input track"),
        Line2D([0], [0], color="#ef4444", linewidth=1.0, alpha=0.55, label="Causal member fan"),
        Line2D([0], [0], color="#b91c1c", linewidth=3, label=model_label),
        Line2D([0], [0], color="#7c3aed", linestyle="--", linewidth=2, label="JTWC captured track"),
        Line2D([0], [0], color="#2563eb", linestyle="--", linewidth=2, label="JMA captured track"),
    ], loc="lower left", fontsize=8)
    fig.suptitle(f"Dolphin: {payload.get('model', 'final-policy route')} · issue {payload['issue_time_utc']}", fontsize=15, fontweight="bold")
    fig.savefig(DOLPHIN_PNG, dpi=180, facecolor="white")
    plt.close(fig)


def draw_tip_png(payload: dict, coastlines: dict) -> None:
    model_label = payload.get("model", "model route")
    issue_count = len(payload.get("cases", []))
    issue_phrase = "at one issue time" if issue_count == 1 else f"at {issue_count} issue times"
    colors = ["#b91c1c", "#d97706", "#2563eb"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.8), gridspec_kw={"width_ratios": [1.5, 1.0]}, constrained_layout=True)
    for axis, regional in zip(axes, (False, True)):
        style_axis(axis, coastlines, regional)
        for case, color in zip(payload["cases"], colors):
            draw_line(axis, case["observed_before_issue"], "#475569", linewidth=1.3)
            for member in case.get("ensemble_forecasts", []):
                draw_line(axis, [case["observed_before_issue"][-1]] + member, "#f97316", linewidth=0.5, alpha=0.10)
            draw_line(axis, [case["observed_before_issue"][-1]] + case["forecast"], color, linewidth=2.6)
            draw_line(axis, [case["observed_before_issue"][-1]] + case["truth_after_issue"], "#047857", linestyle="--", linewidth=1.7)
            issue = case["observed_before_issue"][-1]
            axis.scatter(issue["lon"], issue["lat"], s=28, color=color, edgecolor="white", linewidth=0.8, zorder=7)
        axis.set_title("Regional view" if regional else "World view")
    handles = [
        Line2D([0], [0], color="#f97316", linewidth=1.0, alpha=0.55, label="Causal member fan"),
        Line2D([0], [0], color="#047857", linestyle="--", linewidth=1.7, label="IBTrACS truth after issue"),
    ]
    for case, color in zip(payload["cases"], colors):
        handles.append(Line2D([0], [0], color=color, linewidth=2.6, label=f"{model_label} · issue {case['issue_time_utc'][:13].replace('T', ' ')}Z"))
    axes[0].legend(handles=handles, loc="lower left", fontsize=8)
    fig.suptitle(f"Typhoon Tip: {model_label} {issue_phrase}", fontsize=15, fontweight="bold")
    fig.savefig(TIP_PNG, dpi=180, facecolor="white")
    plt.close(fig)


def html_shell(title: str, map_id: str, data: dict, note: str, summary: str, output_png: Path, kind: str) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
<style>
body{{margin:0;background:#eef3f7;color:#17202a;font:14px system-ui,-apple-system,sans-serif}}
main{{max-width:1500px;margin:auto;padding:18px}}
section{{background:#fff;border:1px solid #ccd6df;border-radius:7px;padding:14px;margin:12px 0}}
#{map_id}{{height:680px;border:1px solid #cbd5e1;border-radius:5px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:7px 9px;border-bottom:1px solid #d8e0e7;text-align:left}}.note{{border-left:4px solid #b91c1c;padding-left:12px}}
.legend{{background:white;padding:8px 10px;border-radius:4px;line-height:1.65;box-shadow:0 1px 4px #0003}}.swatch{{display:inline-block;width:28px;border-top:3px solid;margin-right:6px;vertical-align:middle}}.dash{{border-top-style:dashed}}
img{{max-width:100%;border:1px solid #ccd6df}}
</style></head><body><main><h1>{html.escape(title)}</h1><section class="note">{note}</section>
<section><div id="{map_id}" aria-label="{html.escape(title)} world map"></div></section>
<section><h2>Saved values</h2>{summary}</section>
<section><img src="{output_png.name}" alt="{html.escape(title)} static map"></section>
</main><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script><script>
const DATA={encoded};
const MODEL_LABEL={json.dumps(data.get("model", "model route"), ensure_ascii=False)};
const map=L.map('{map_id}').setView([20,150],3);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:8,attribution:'&copy; OpenStreetMap'}}).addTo(map);
const pt=p=>[p.lat,p.lon];
const popup=p=>{{const pressure=p.central_pressure_hpa??p.pressure_hpa; const r=p.wind_radii_km; const r34=Array.isArray(r)&&r.length>=4?(r.slice(0,4).reduce((a,b)=>a+Number(b),0)/4):null; const mf=p.pressure_map_features; const mapLine=mf===undefined?'':`<br>map min ${{Number(mf.map_min_pressure_hpa).toFixed(0)}} hPa · deficit ${{Number(mf.map_pressure_deficit_hpa).toFixed(0)}} hPa`; return `${{p.time||''}}<br>lat ${{p.lat.toFixed(2)}} · lon ${{p.lon.toFixed(2)}}${{p.lead_hours===undefined?'':`<br>lead +${{p.lead_hours}} h`}}${{p.vmax_kt===undefined?'':`<br>wind ${{Number(p.vmax_kt).toFixed(0)}} kt ±${{Number(p.vmax_spread_kt||0).toFixed(1)}}`}}${{pressure===undefined||pressure===null?'':`<br>central pressure ${{Number(pressure).toFixed(0)}} hPa ±${{Number(p.pressure_spread_hpa||0).toFixed(1)}}`}}${{p.rmw_km===undefined?'':`<br>RMW ${{Number(p.rmw_km).toFixed(0)}} km`}}${{r34===null?'':`<br>R34 mean ${{r34.toFixed(0)}} km`}}${{mapLine}}`}};
function line(rows,color,opts={{}}){{return L.polyline(rows.map(pt),{{color,weight:3,opacity:.92,...opts}}).addTo(map)}}
function markers(rows,color,label){{rows.forEach(p=>L.circleMarker(pt(p),{{radius:5,color:'#fff',weight:1.3,fillColor:color,fillOpacity:.98}}).bindPopup(`<b>${{label}}</b><br>${{popup(p)}}`).addTo(map))}}
const bounds=[];
function remember(rows){{rows.forEach(p=>bounds.push(pt(p)))}}
if('{kind}'==='dolphin'){{
  line(DATA.observed,'#111827',{{weight:2.5}}); remember(DATA.observed);
  const fanLayer=L.layerGroup();
  (DATA.ensemble_forecasts||[]).forEach(path=>L.polyline([DATA.observed.at(-1),...path].map(pt),{{color:'#ef4444',weight:.8,opacity:.16}}).addTo(fanLayer));
  fanLayer.addTo(map);
  line([DATA.observed.at(-1),...DATA.forecast],'#b91c1c',{{weight:4}}); markers(DATA.forecast,'#b91c1c',MODEL_LABEL); remember(DATA.forecast);
  line(DATA.official.jtwc,'#7c3aed',{{dashArray:'8 6',weight:2.5}}); markers(DATA.official.jtwc,'#7c3aed','JTWC captured'); remember(DATA.official.jtwc);
  line(DATA.official.jma,'#2563eb',{{dashArray:'8 6',weight:2.5}}); markers(DATA.official.jma,'#2563eb','JMA captured'); remember(DATA.official.jma);
  const routeLayer=L.polyline([DATA.observed.at(-1),...DATA.forecast].map(pt),{{color:'#b91c1c',weight:4}});
  const overlays={{'Observed input':L.polyline(DATA.observed.map(pt),{{color:'#111827',weight:2.5}}),'Causal member fan':fanLayer,[MODEL_LABEL]:routeLayer,'JTWC captured':L.polyline(DATA.official.jtwc.map(pt),{{color:'#7c3aed',dashArray:'8 6',weight:2.5}}),'JMA captured':L.polyline(DATA.official.jma.map(pt),{{color:'#2563eb',dashArray:'8 6',weight:2.5}})}};
  L.control.layers(null,overlays,{{collapsed:false}}).addTo(map);
}}else{{
  const colors=['#b91c1c','#d97706','#2563eb'];
  const fanLayer=L.layerGroup();
  DATA.cases.forEach((c,i)=>{{line(c.observed_before_issue,'#475569',{{weight:1.5,opacity:.65}}); (c.ensemble_forecasts||[]).forEach(path=>L.polyline([c.observed_before_issue.at(-1),...path].map(pt),{{color:'#f97316',weight:.7,opacity:.14}}).addTo(fanLayer)); line([c.observed_before_issue.at(-1),...c.forecast],colors[i],{{weight:3.5}}); line([c.observed_before_issue.at(-1),...c.truth_after_issue],'#047857',{{dashArray:'7 6',weight:2}}); markers(c.forecast,colors[i],`${{MODEL_LABEL}} issue ${{c.issue_time_utc}}`); remember(c.observed_before_issue); remember(c.forecast); remember(c.truth_after_issue)}});
  fanLayer.addTo(map);
}}
if(bounds.length) map.fitBounds(L.latLngBounds(bounds),{{padding:[28,28]}});
</script></body></html>"""


def build_dolphin() -> dict:
    source = json.loads(DOLPHIN_SOURCE.read_text(encoding="utf-8"))
    final_route = source["v37n"]
    return {
        "storm": "Dolphin",
        "issue_time_utc": source["issue_time_utc"],
        "model": "Final v56 policy route: v37N default",
        "observed": [point(row) for row in source["current_track_history"]],
        "forecast": [point(row, time_key="valid_time_utc") for row in final_route["forecast"]],
        "official": {
            "jtwc": [point(row) for row in source["jtwc_official"]],
            "jma": [point(row) for row in source["jma_official"]],
        },
        "input_policy": source.get("source_policy", "Saved local case artifact; visualization only."),
        "gate_applied": False,
        "gate_note": "Dolphin is outside the fixed v56 feature archive; final policy is shown using its v37N default route.",
    }


def build_tip() -> dict:
    source = json.loads(TIP_SOURCE.read_text(encoding="utf-8"))
    cases = []
    for case in source["cases"]:
        cases.append({
            "issue_time_utc": case["issue_time_utc"],
            "observed_before_issue": [point(row, time_key="time_utc") for row in case["observed_before_issue"]],
            "forecast": [point(row, time_key="valid_time_utc", lon_key="longitude", lat_key="latitude") for row in case["forecast"]],
            "truth_after_issue": [point(row, time_key="time_utc") for row in case["truth_after_issue"]],
            "score": case["score"],
        })
    return {
        "storm": "Tip",
        "model": "Final v56 policy route: v37N default",
        "cases": cases,
        "input_policy": source["input_policy"],
        "future_rows_used_for_inference": source["future_rows_used_for_inference"],
        "official_jma_jtwc_forecasts_used": source["official_jma_jtwc_forecasts_used"],
        "gate_applied": False,
        "gate_note": "Tip is an out-of-training 1979 hindcast; final policy is shown using its v37N default route.",
    }


def dolphin_summary(payload: dict) -> str:
    return (
        f"<p><b>Model:</b> {html.escape(payload['model'])}<br>"
        f"<b>Issue:</b> {html.escape(payload['issue_time_utc'])}<br>"
        f"{html.escape(payload['gate_note'])}<br>"
        "The red route is the final-policy route. Dashed lines are official tracks captured in the same local case artifact.</p>"
        "<div class=\"legend\"><span class=\"swatch\" style=\"border-color:#b91c1c\"></span>saved model &nbsp; "
        "<span class=\"swatch dash\" style=\"border-color:#7c3aed\"></span>JTWC &nbsp; "
        "<span class=\"swatch dash\" style=\"border-color:#2563eb\"></span>JMA</div>"
    )


def tip_summary(payload: dict) -> str:
    rows = []
    for case in payload["cases"]:
        score = case["score"]
        rows.append(
            f"<tr><td>{html.escape(case['issue_time_utc'])}</td><td>{score['track_error_120h_km']:.1f} km</td><td>{score['persistence_error_120h_km']:.1f} km</td></tr>"
        )
    return (
        f"<p><b>Model:</b> {html.escape(payload['model'])}<br>"
        f"{html.escape(payload['gate_note'])}<br>"
        "Each colored line is a forecast issued at a different cutoff. Green dashed lines are later IBTrACS truth.</p>"
        "<table><thead><tr><th>Issue time</th><th>v37N +120 h error</th><th>Persistence +120 h error</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def main() -> None:
    dolphin = build_dolphin()
    tip = build_tip()
    coastlines = load_coastlines()
    DOLPHIN_JSON.write_text(json.dumps(dolphin, indent=2) + "\n", encoding="utf-8")
    TIP_JSON.write_text(json.dumps(tip, indent=2) + "\n", encoding="utf-8")
    draw_dolphin_png(dolphin, coastlines)
    draw_tip_png(tip, coastlines)
    DOLPHIN_HTML.write_text(html_shell(
        "Dolphin final-model policy map", "dolphin-map", dolphin,
        "Visualization of the saved final-policy route; no new model run was performed.",
        dolphin_summary(dolphin), DOLPHIN_PNG, "dolphin"
    ), encoding="utf-8")
    TIP_HTML.write_text(html_shell(
        "Typhoon Tip final-model policy map", "tip-map", tip,
        "The map uses the saved final-policy v37N hindcast. Future Tip rows are shown only as verification truth; they were not used to create the forecast.",
        tip_summary(tip), TIP_PNG, "tip"
    ), encoding="utf-8")
    print(f"Dolphin HTML: {DOLPHIN_HTML}")
    print(f"Dolphin PNG:  {DOLPHIN_PNG}")
    print(f"Tip HTML:     {TIP_HTML}")
    print(f"Tip PNG:      {TIP_PNG}")


if __name__ == "__main__":
    main()
