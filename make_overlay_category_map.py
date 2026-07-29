"""Real track overlaid with v23/v35 mean forecasts (v10 dropped), colored by intensity category,
with interactive hover/click tooltips on every point. Tip, Bavi, Hinnamnor, Co-may use the model
overlay; Dolphin (now with real fetched GFS steering, see _fetch_dolphin_steering.py) overlays its
real partial track with v23/v35's own forecast plus JTWC's and JMA/RSMC Tokyo's official forecast
tracks (NCDR/Google DeepMind still lack a publicly extractable numeric coordinate table).
"""
import json, os
import numpy as np

R = 111.2
CATS = [("TD", 0, 34), ("TS", 34, 64), ("C1", 64, 83), ("C2", 83, 96),
        ("C3", 96, 113), ("C4", 113, 137), ("C5", 137, 999)]
COL = {"TD": ("#8a94a6", "#9fb0bd"), "TS": ("#2a78d6", "#3987e5"), "C1": ("#e0b400", "#f2c744"),
       "C2": ("#e08a1e", "#f2994a"), "C3": ("#d94f3d", "#eb5757"), "C4": ("#b8291f", "#e15347"),
       "C5": ("#8b2fb0", "#c07de0")}
MODELS = [("real", "Real (observed)", "solid", 4.2, 1.0),
          ("v23", "v23 (CoT + temporal steering)", "dashed", 2.8, 0.9),
          ("v35", "v35 (v23 + intensity-reweighted loss)", "dashdot", 2.8, 0.9)]
STROKE_STYLE = {"solid": "", "dotted": "stroke-dasharray:1.4 3.2;",
                "dashed": "stroke-dasharray:6 3;", "dashdot": "stroke-dasharray:6 2 1.4 2;",
                "loosedash": "stroke-dasharray:9 4;", "shortdash": "stroke-dasharray:3 2;"}
LCOL = {"real": "#222b33", "v10": "#7a8592", "v23": "#2a78d6", "v35": "#8b2fb0",
        "jtwc": "#c2410c", "jma": "#0f9b6c"}
LCOLD = {"real": "#e8eef4", "v10": "#9fb0bd", "v23": "#3987e5", "v35": "#c07de0",
         "jtwc": "#f0813f", "jma": "#3ecf96"}


def cat_of(v):
    for name, lo, hi in CATS:
        if lo <= v < hi:
            return name
    return "C5"


NS_PER_H = 3600 * int(1e9)


def fmt_ns(vt_ns):
    days = vt_ns // (86400 * int(1e9))
    rem = vt_ns - days * 86400 * int(1e9)
    h = rem // NS_PER_H
    # anchor: ns timestamps in this dataset are since epoch (int64 ns, numpy datetime64 convention)
    dt = np.datetime64(int(vt_ns), "ns")
    return str(dt)[:16].replace("T", " ") + "Z"


nb = json.load(open("colab_train_v17.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
body = body.replace('"/content/d/steer5_int8.npz"', '"track_build/dlm4_int8.npz"')
body = body.replace('"/content/d/track_windows_v13.npz"', '"track_build/track_windows_v13.npz"')
body = body.replace('DEVICE = torch.device("cuda")', 'DEVICE = torch.device("cpu")')
import torch, torch.nn as nn, torch.nn.functional as F
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os,
     "json": json, "time": __import__("time"), "math": __import__("math")}
exec(compile(body, "<v17-notebook>", "exec"), G)
z = G["z"]; track = G["track"]; tmean = G["tmean"]; tstd = G["tstd"]
sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64") % 360

REAL = {}
for s, nm in [("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor"), ("2026182N09163", "Bavi")]:
    k = np.where(sid == s)[0]; k = k[np.argsort(bt[k])]
    vmax = track[k, -1, 4] * tstd[4] + tmean[4]
    REAL[nm] = [(bla[k[i]], blo[k[i]], float(vmax[i]), int(bt[k[i]]))
                for i in range(len(k)) if vmax[i] > 0]

tz = np.load("track_build/tip_fixed.npz", allow_pickle=True)
ttr = tz["track"].astype("float32"); tbt = tz["base_time"].astype("int64")
tbla = tz["base_lat"].astype("float64"); tblo = tz["base_lon"].astype("float64")
o13 = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
otm, ots = o13["track_mean"].astype("float32"), o13["track_std"].astype("float32")
order = np.argsort(tbt)
tvmax = ttr[order, -1, 4] * ots[4] + otm[4]
REAL["Tip"] = [(tbla[order[i]], tblo[order[i]], float(tvmax[i]), int(tbt[order[i]]))
               for i in range(len(order)) if tvmax[i] > 0]

PRED = json.load(open("track_build/overlay_predictions.json"))
STORMS = ["Tip", "Bavi", "Hinnamnor", "Co-may"]

# ---- Dolphin: real partial track (Wunderground position/wind + Zoom Earth pressure, cross-
# checked -- see make_category_map5.py) + JTWC's official forecast track (Advisory #3, issued
# 2026-07-27 12:00 UTC -- the only one of JMA/NCDR/ECMWF/Google DeepMind with a publicly
# extractable numeric forecast table; the others were only described directionally in press
# coverage with no verifiable coordinates, so they are NOT plotted rather than guessed at). ----
KT_PER_MPH = 0.868976
DOLPHIN_TIMES = ["2026-07-27 00:00", "2026-07-27 06:00", "2026-07-27 12:00", "2026-07-27 18:00",
                 "2026-07-28 00:00", "2026-07-28 06:00", "2026-07-28 12:00", "2026-07-28 18:00",
                 "2026-07-29 00:00", "2026-07-29 06:00"]
DOLPHIN_RAW = [
    (12.8, 178.3, 35), (13.4, 176.7, 40), (13.6, 175.2, 50), (13.2, 173.7, 50),
    (13.0, 172.8, 70), (13.3, 171.7, 85), (13.4, 170.7, 115), (13.7, 169.9, 145),
    (14.1, 169.1, 140), (14.5, 168.4, 140),
]
DOLPHIN_REAL = [(la, lo, w * KT_PER_MPH, t) for (la, lo, w), t in zip(DOLPHIN_RAW, DOLPHIN_TIMES)]
JTWC_TIMES = ["2026-07-27 12:00 (issued)", "2026-07-28 00:00", "2026-07-28 12:00",
              "2026-07-29 12:00", "2026-07-30 12:00", "2026-07-31 12:00", "2026-08-01 12:00"]
JTWC_RAW = [  # (lat, lon, kt) -- TAU 0/12/24/48/72/96/120h, Advisory #3
    (13.7, 175.3, 45), (13.5, 172.9, 55), (13.7, 170.5, 70), (15.2, 167.3, 100),
    (17.0, 164.3, 120), (18.9, 161.0, 140), (21.0, 157.5, 150),
]
JTWC_FCST = [(la, lo, float(kt), t) for (la, lo, kt), t in zip(JTWC_RAW, JTWC_TIMES)]

# JMA (RSMC Tokyo) official forecast, from their CAP atom feed (data.jma.go.jp/cap-rsmctk/atom.xml),
# analysis sent 2026-07-29 12:40 UTC. NOTE: JMA's wind is 10-MINUTE sustained, a different (lower)
# convention from the 1-minute sustained wind used for every other line/point on this map -- so this
# series' category coloring is on JMA's own scale, not directly comparable to the others (flagged in
# the tooltip label and footer rather than converted, to avoid asserting an unverified factor).
JMA_TIMES = ["2026-07-29 12:00 (analysis)", "2026-07-30 12:00 (+24h)",
             "2026-07-31 12:00 (+48h)", "2026-08-01 12:00 (+72h)"]
JMA_RAW = [(15.2, 167.7, 85), (17.4, 163.8, 100), (19.2, 160.1, 110), (21.8, 156.1, 105)]
JMA_FCST = [(la, lo, float(kt), t) for (la, lo, kt), t in zip(JMA_RAW, JMA_TIMES)]

# v23/v35's own forecast on Dolphin, from the LATEST issue (2026-07-29 06:00 UTC), using REAL GFS
# steering fetched by _fetch_dolphin_steering.py -- see _dolphin_v23_v35.py.
_dv = json.load(open("track_build/dolphin_v23_v35.json"))
LEAD_TIMES_H = list(range(6, 121, 6))


def _model_fcst(tag):
    latest = _dv[tag]["issues"][-1]
    t0 = np.datetime64(latest["issue_time"])
    pts = []
    for i, h in enumerate(LEAD_TIMES_H):
        t = t0 + np.timedelta64(int(h), "h")
        pts.append((latest["lats"][i], latest["lons"][i], latest["vmax"][i],
                    str(t).replace("T", " ") + " (+" + str(h) + "h)"))
    return pts


V10_FCST = _model_fcst("v10")
V23_FCST = _model_fcst("v23")
V35_FCST = _model_fcst("v35")

LAND = json.load(open("track_build/geo/ne/ne_50m_land.geojson"))


def rings_in(lo0, lo1, la0, la1):
    out = []
    for f in LAND["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            r = poly[0]
            xs = [p[0] for p in r]; ys = [p[1] for p in r]
            if max(xs) < lo0 or min(xs) > lo1 or max(ys) < la0 or min(ys) > la1:
                continue
            tol = (lo1 - lo0) / 260.0
            simp, last = [], None
            for p in r:
                if last is None or abs(p[0] - last[0]) > tol or abs(p[1] - last[1]) > tol:
                    simp.append(p); last = p
            if len(simp) >= 3:
                out.append(simp)
    return out


_UID = [0]


def panel(nm, series, W=520, H=420):
    """series: list of (tag, label, style, dotr, opacity, pts) -- pts is [(lat,lon,vmax,time_label_or_ns), ...]"""
    m = 36
    all_pts = [p for _, _, _, _, _, pts in series for p in pts]
    lons = [p[1] % 360 for p in all_pts]; lats = [p[0] for p in all_pts]
    lo0, lo1, la0, la1 = min(lons), max(lons), min(lats), max(lats)
    px, py = (lo1 - lo0) * .07 + 1.0, (la1 - la0) * .07 + 1.0
    lo0, lo1, la0, la1 = lo0 - px, lo1 + px, la0 - py, la1 + py
    kx = np.cos(np.radians((la0 + la1) / 2))
    spanx, spany = (lo1 - lo0) * kx, la1 - la0
    sc = min((W - 2 * m) / spanx, (H - 2 * m) / spany)
    ox, oy = (W - spanx * sc) / 2, (H - spany * sc) / 2

    def PX(lo): return ox + ((lo % 360) - lo0) * kx * sc
    def PY(la): return H - oy - (la - la0) * sc

    o = [f'<svg viewBox="0 0 {W} {H}" class="map" role="img" aria-label="{nm}">',
         f'<rect x="0" y="0" width="{W}" height="{H}" class="sea"/>']
    for r in rings_in(lo0, lo1, la0, la1):
        d = "M" + " L".join(f"{PX(p[0]):.1f},{PY(p[1]):.1f}" for p in r) + " Z"
        o.append(f'<path class="land" d="{d}"/>')
    step = 10 if (la1 - la0) > 20 else 5
    g = np.ceil(la0 / step) * step
    while g <= la1:
        y = PY(g)
        o.append(f'<line class="gl" x1="0" x2="{W}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="tk" x="4" y="{y-3:.1f}">{abs(g):.0f}&deg;{"N" if g>=0 else "S"}</text>')
        g += step
    g = np.ceil(lo0 / step) * step
    while g <= lo1:
        x = PX(g)
        o.append(f'<line class="gl" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{H}"/>')
        o.append(f'<text class="tk" x="{x+3:.1f}" y="{H-5}">{((g+180)%360)-180:.0f}&deg;E</text>')
        g += step

    for tag, label, style, dotr, op, pts in series:
        d = "M" + " L".join(f"{PX(p[1]):.1f},{PY(p[0]):.1f}" for p in pts)
        o.append(f'<path d="{d}" class="ln {tag}" style="{STROKE_STYLE[style]}"/>')
        for p in pts:
            la, lo, v = p[0], p[1], p[2]
            when = p[3]
            when_s = fmt_ns(when) if isinstance(when, int) else when
            c = cat_of(v)
            _UID[0] += 1
            o.append(f'<circle cx="{PX(lo):.1f}" cy="{PY(la):.1f}" r="{dotr}" class="fx {c} pt" '
                      f'opacity="{op}" data-model="{label}" data-time="{when_s}" '
                      f'data-lat="{la:.2f}" data-lon="{lo:.2f}" data-vmax="{v:.0f}" data-cat="{c}" '
                      f'tabindex="0"/>')
    d0 = series[0][5][0]
    o.append(f'<circle cx="{PX(d0[1]):.1f}" cy="{PY(d0[0]):.1f}" r="6.5" class="genesis"/>')
    peak = max(series[0][5], key=lambda p: p[2])
    o.append(f'<text class="peakl" x="{PX(peak[1])+9:.1f}" y="{PY(peak[0])-6:.1f}">real peak {peak[2]:.0f} kt ({cat_of(peak[2])})</text>')
    o.append('</svg>')
    return "\n".join(o)


cards = []
for nm in STORMS:
    series = [(tag, label, style, dotr, op,
               REAL[nm] if tag == "real" else [tuple(p) for p in PRED[nm][tag]])
              for tag, label, style, dotr, op in MODELS]
    cards.append(f'<figure class="panel"><figcaption><h3>{nm}</h3></figcaption>{panel(nm, series)}</figure>')

dolphin_series = [("real", "Real (observed, partial)", "solid", 4.2, 1.0, DOLPHIN_REAL),
                   ("jtwc", "JTWC official forecast (Advisory #3)", "dotted", 3.4, 0.9, JTWC_FCST),
                   ("jma", "JMA/RSMC Tokyo official forecast (10-min wind convention)", "loosedash", 3.4, 0.9, JMA_FCST),
                   ("v10", "v10 forecast (track-only, no steering -- own history only)", "shortdash", 2.6, 0.85, V10_FCST),
                   ("v23", "v23 forecast (issued 2026-07-29 06:00, real GFS steering)", "dashed", 3.0, 0.9, V23_FCST),
                   ("v35", "v35 forecast (issued 2026-07-29 06:00, real GFS steering)", "dashdot", 3.0, 0.9, V35_FCST)]
cards.append(f'<figure class="panel"><figcaption><h3>Dolphin <span class="sub">2026, active</span></h3>'
             f'<p>v10 (track-only, no external data beyond its own recent positions) vs. v23/v35 '
             f'(run on REAL fetched GFS steering, NOMADS, see _fetch_dolphin_steering.py) -- all '
             f'three of this project\'s models forecast weakening; JTWC and JMA\'s official '
             f'forecasts both predict continued intensification. NCDR/Google DeepMind forecasts '
             f'exist in press coverage but no verifiable coordinate table was found for '
             f'them.</p></figcaption>{panel("Dolphin", dolphin_series)}</figure>')
cards = "".join(cards)

legend_cat = "".join(
    f'<span class="lg"><span class="sw {c}"></span>{c} <span class="rng">{lo}{"+" if hi>900 else f"-{hi-1}"}kt</span></span>'
    for c, lo, hi in CATS)
legend_model = "".join(
    f'<span class="lg"><svg width="26" height="10" class="lgsvg"><line x1="1" y1="5" x2="25" y2="5" '
    f'class="ln {tag}" style="{STROKE_STYLE[style]}"/></svg>{label}</span>'
    for tag, label, style, *_ in MODELS)
legend_model += ('<span class="lg"><svg width="26" height="10" class="lgsvg"><line x1="1" y1="5" x2="25" y2="5" '
                  f'class="ln jtwc" style="{STROKE_STYLE["dotted"]}"/></svg>JTWC official forecast (Dolphin only)</span>'
                  '<span class="lg"><svg width="26" height="10" class="lgsvg"><line x1="1" y1="5" x2="25" y2="5" '
                  f'class="ln jma" style="{STROKE_STYLE["loosedash"]}"/></svg>JMA official forecast, 10-min wind (Dolphin only)</span>'
                  '<span class="lg"><svg width="26" height="10" class="lgsvg"><line x1="1" y1="5" x2="25" y2="5" '
                  f'class="ln v10" style="{STROKE_STYLE["shortdash"]}"/></svg>v10, track-only (Dolphin only)</span>')

_palL = "".join(f".{c}{{fill:{v[0]};}}" for c, v in COL.items())
_palD = "".join(f".{c}{{fill:{v[1]};}}" for c, v in COL.items())
_lcL = "".join(f".ln.{t}{{stroke:{c};}}" for t, c in LCOL.items())
_lcD = "".join(f".ln.{t}{{stroke:{c};}}" for t, c in LCOLD.items())

HTML = f"""<meta charset="utf-8">
<title>Real vs v23/v35 (+ JTWC for Dolphin), interactive, colored by intensity category</title>
<style>
:root{{color-scheme:light;--bg:#f2f4f6;--surface:#fcfcfb;--ink:#111820;--body:#2c3a47;--muted:#5d6c7a;
 --line:#d5dce3;--sea:#eaf1f5;--land:#dfe3e0;--coast:#a8b3ba;
 --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;{_palL}{_lcL}}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;
 --sea:#101b24;--land:#26313a;--coast:#4a5a66;{_palD}{_lcD}}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0c1117;--surface:#141c25;--ink:#e8eef4;
 --body:#c2cdd8;--muted:#8697a5;--line:#26313d;--sea:#101b24;--land:#26313a;--coast:#4a5a66;{_palD}{_lcD}}}
body{{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:16px;line-height:1.6;}}
.wrap{{max-width:1180px;margin:0 auto;padding:clamp(24px,5vw,52px) clamp(16px,4vw,32px) 80px;
 display:flex;flex-direction:column;gap:26px;}}
h1{{color:var(--ink);font-size:clamp(24px,3.6vw,36px);line-height:1.14;letter-spacing:-.02em;margin:0;font-weight:660;}}
.eyebrow{{font-family:var(--mono);font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);}}
.lede{{max-width:80ch;font-size:14.5px;margin:0;}}
header{{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--line);padding-bottom:22px;}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;}}
.legend.model{{margin-top:4px;}}
.lg{{display:flex;align-items:center;gap:6px;}}
.sw{{width:11px;height:11px;border-radius:50%;display:inline-block;}}
.lgsvg{{display:block;}}
.rng{{color:var(--muted);font-family:var(--mono);font-size:11px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px;}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 14px 10px;
 margin:0;display:flex;flex-direction:column;gap:7px;}}
figcaption h3{{color:var(--ink);font-size:14.5px;font-weight:660;margin:0;display:flex;gap:8px;align-items:baseline;}}
figcaption .sub{{font-size:11px;color:var(--muted);font-weight:400;}}
figcaption p{{font-size:11.5px;color:var(--muted);margin:0;}}
.map{{width:100%;height:auto;display:block;border-radius:4px;overflow:hidden;}}
.sea{{fill:var(--sea);}} .land{{fill:var(--land);stroke:var(--coast);stroke-width:.7;}}
.gl{{stroke:var(--coast);stroke-width:.5;opacity:.4;}}
.tk{{font-family:var(--mono);font-size:8px;fill:var(--muted);}}
.ln{{fill:none;stroke-width:1.6;stroke-linejoin:round;stroke-linecap:round;opacity:.85;}}
.ln.real{{stroke-width:2.2;opacity:1;}}
.fx{{stroke:var(--surface);stroke-width:.7;cursor:pointer;}}
.fx.pt:hover,.fx.pt:focus{{stroke:var(--ink);stroke-width:1.8;outline:none;}}
.genesis{{fill:none;stroke:var(--ink);stroke-width:2;}}
.peakl{{font-family:var(--mono);font-size:9px;fill:var(--ink);font-weight:600;
 paint-order:stroke;stroke:var(--sea);stroke-width:2.4px;}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted);max-width:80ch;
 display:flex;flex-direction:column;gap:8px;}}
#tt{{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font-family:var(--mono);
 font-size:12px;padding:8px 10px;border-radius:6px;box-shadow:0 6px 18px rgba(0,0,0,.25);
 display:none;z-index:50;line-height:1.5;white-space:nowrap;}}
#tt b{{font-size:12.5px;}}
#tt .hint{{color:var(--muted);font-size:10px;margin-top:2px;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; real, overlaid with model/agency forecasts &middot; interactive</div>
  <h1>Real track vs v23 / v35 (+ JTWC for Dolphin), colored by intensity category</h1>
  <p class="lede"><b>Hover or click any point</b> to see exactly what it is: model/source, time,
  position, wind speed, and category. One map per storm: real (solid, largest dots) plus each
  model's mean-by-valid-time forecast (v23 dashed, v35 dash-dot). Dolphin's panel additionally
  shows v10 (short-dashed gray) -- track-only, no external data beyond the storm's own recent
  positions, i.e. the same IBTrACS-style history alone that all four models start from -- next to
  v23/v35 which instead run on REAL fetched GFS steering (NOMADS, current as of this run), plus
  JTWC's (dotted orange) and JMA/RSMC Tokyo's (loose-dashed green) official forecasts for
  comparison. Notably, all three of this project's models predict Dolphin <b>weakening</b> over the
  next several days while both agencies predict continued intensification. JMA reports 10-minute
  sustained wind, a different (lower) convention from the 1-minute convention used everywhere else
  on this page, so its point colors are on its own scale -- not directly comparable
  category-for-category to the other lines.</p>
  <div class="legend">{legend_cat}</div>
  <div class="legend model">{legend_model}</div>
 </header>
 <div class="grid">{cards}</div>
 <footer>
  <p>v23/v35 paths for Tip/Bavi/Hinnamnor/Co-may are mean-by-valid-time: at each 6h valid time, the
  mean position and mean vmax across every forecast whose lead lands on that moment (bins under 3
  contributing forecasts dropped). For Dolphin, v10/v23/v35 instead each show the single forecast
  issued from the LATEST real observation (2026-07-29 06:00 UTC). v10 uses only that observed track
  history -- no steering field at all, the IBTrACS-style "own data only" baseline -- while v23/v35
  run on real GFS deep-layer-mean steering (850/500/200 hPa u/v) fetched from NOAA/NOMADS for each
  of Dolphin's 10 issue times
  (<code>_fetch_dolphin_steering.py</code>, <code>_dolphin_v23_v35.py</code>) -- the same real-data
  discipline used for Noul, not a zero-filled fallback. JTWC's Dolphin forecast is the official TAU
  0/12/24/48/72/96/120h track from Advisory #3 (issued 2026-07-27 12:00 UTC) -- a single forecast,
  not an ensemble mean, and already several advisory cycles old relative to the latest observation
  shown. JMA's forecast is from RSMC Tokyo's official CAP atom feed
  (<code>data.jma.go.jp/cap-rsmctk/atom.xml</code>), analysis sent 2026-07-29 12:40 UTC, forecast to
  +72h only (the feed does not carry a 96h/120h lead) -- the most recent official analysis/forecast
  of the four sources on this map, but on JMA's own 10-minute-sustained-wind convention rather than
  the 1-minute convention used for real/JTWC/v23/v35.</p>
  <p><b>On NCDR/Google DeepMind for Dolphin.</b> Public reporting describes DeepMind's WeatherLab
  output qualitatively (trending northwest) but its live interactive tool has no fetchable
  coordinate table, and NCDR itself aggregates JMA/CWA/JTWC rather than issuing an independent
  forecast. Rather than approximate their tracks from a qualitative description, they are left out
  entirely.</p>
 </footer>
</div>
<div id="tt"></div>
<script>
(function(){{
  var tt = document.getElementById('tt');
  var pinned = null;
  function render(el){{
    var m = el.dataset.model, t = el.dataset.time, la = el.dataset.lat, lo = el.dataset.lon,
        v = el.dataset.vmax, c = el.dataset.cat;
    tt.innerHTML = '<b>' + m + '</b><br>' + t + '<br>' + la + '&deg;, ' + lo + '&deg;E<br>' +
                   v + ' kt &mdash; <b>' + c + '</b><div class="hint">click to pin / unpin</div>';
  }}
  function place(evt){{
    var x = evt.clientX + 14, y = evt.clientY + 14;
    var vw = window.innerWidth, vh = window.innerHeight;
    tt.style.left = Math.min(x, vw - 200) + 'px';
    tt.style.top = Math.min(y, vh - 90) + 'px';
  }}
  document.querySelectorAll('.pt').forEach(function(el){{
    el.addEventListener('mouseenter', function(e){{ if(!pinned){{ render(el); tt.style.display='block'; place(e); }} }});
    el.addEventListener('mousemove', function(e){{ if(!pinned) place(e); }});
    el.addEventListener('mouseleave', function(){{ if(!pinned) tt.style.display='none'; }});
    el.addEventListener('click', function(e){{
      e.stopPropagation();
      if(pinned === el){{ pinned = null; tt.style.display='none'; return; }}
      pinned = el; render(el); tt.style.display='block'; place(e);
    }});
    el.addEventListener('focus', function(e){{ render(el); tt.style.display='block';
      var r = el.getBoundingClientRect(); place({{clientX:r.left, clientY:r.top}}); }});
  }});
  document.addEventListener('click', function(){{ pinned = null; tt.style.display='none'; }});
}})();
</script>"""

os.makedirs("paper", exist_ok=True)
open("paper/overlay_real_vs_models.html", "w").write(HTML)
print(f"wrote paper/overlay_real_vs_models.html ({len(HTML)/1000:.0f} KB)")
