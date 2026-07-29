"""Real track overlaid with v10/v23/v35 mean-forecast tracks, all on the SAME map per storm, all
dots colored by intensity category (TD/TS/C1-C5) -- the real line and each model's line use
different strokes (solid vs. dotted/dashed/dash-dot) so they're visually distinct even where paths
cross. Tip, Bavi, Co-may, Hinnamnor only: Dolphin has no real steering-field data yet, so v23/v35
cannot run on it (see _overlay_predictions.py docstring).

Reads track_build/overlay_predictions.json (model paths) and rebuilds each storm's real track the
same way make_category_map.py does.
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
          ("v10", "v10 (no environment)", "dotted", 2.6, 0.85),
          ("v23", "v23 (CoT + temporal steering)", "dashed", 2.6, 0.85),
          ("v35", "v35 (v23 + intensity-reweighted loss)", "dashdot", 2.6, 0.85)]


def cat_of(v):
    for name, lo, hi in CATS:
        if lo <= v < hi:
            return name
    return "C5"


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
    REAL[nm] = [(bla[k[i]], blo[k[i]], float(vmax[i])) for i in range(len(k)) if vmax[i] > 0]

tz = np.load("track_build/tip_fixed.npz", allow_pickle=True)
ttr = tz["track"].astype("float32"); tbt = tz["base_time"].astype("int64")
tbla = tz["base_lat"].astype("float64"); tblo = tz["base_lon"].astype("float64")
o13 = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
otm, ots = o13["track_mean"].astype("float32"), o13["track_std"].astype("float32")
order = np.argsort(tbt)
tvmax = ttr[order, -1, 4] * ots[4] + otm[4]
REAL["Tip"] = [(tbla[order[i]], tblo[order[i]], float(tvmax[i])) for i in range(len(order)) if tvmax[i] > 0]

PRED = json.load(open("track_build/overlay_predictions.json"))
STORMS = ["Tip", "Bavi", "Hinnamnor", "Co-may"]

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


STROKE_STYLE = {"solid": "", "dotted": "stroke-dasharray:1.4 3.2;",
                "dashed": "stroke-dasharray:6 3;", "dashdot": "stroke-dasharray:6 2 1.4 2;"}
LCOL = {"real": "#222b33", "v10": "#7a8592", "v23": "#2a78d6", "v35": "#8b2fb0"}
LCOLD = {"real": "#e8eef4", "v10": "#9fb0bd", "v23": "#3987e5", "v35": "#c07de0"}


def panel(nm, W=520, H=420):
    m = 36
    all_pts = REAL[nm] + [tuple(p) for tag, _, *_ in MODELS if tag != "real" for p in PRED[nm][tag]]
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

    o = [f'<svg viewBox="0 0 {W} {H}" class="map" role="img" aria-label="{nm} real vs v10/v23/v35">',
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

    for tag, label, style, dotr, op in MODELS:
        pts = REAL[nm] if tag == "real" else [tuple(p) for p in PRED[nm][tag]]
        d = "M" + " L".join(f"{PX(p[1]):.1f},{PY(p[0]):.1f}" for p in pts)
        o.append(f'<path d="{d}" class="ln {tag}" style="{STROKE_STYLE[style]}"/>')
        for la, lo, v in pts:
            c = cat_of(v)
            o.append(f'<circle cx="{PX(lo):.1f}" cy="{PY(la):.1f}" r="{dotr}" class="fx {c}" opacity="{op}"/>')

    rla, rlo, rv0 = REAL[nm][0]
    o.append(f'<circle cx="{PX(rlo):.1f}" cy="{PY(rla):.1f}" r="6.5" class="genesis"/>')
    peak = max(REAL[nm], key=lambda p: p[2])
    o.append(f'<text class="peakl" x="{PX(peak[1])+9:.1f}" y="{PY(peak[0])-6:.1f}">real peak {peak[2]:.0f} kt ({cat_of(peak[2])})</text>')
    o.append('</svg>')
    return "\n".join(o)


cards = "".join(
    f'<figure class="panel"><figcaption><h3>{nm}</h3></figcaption>{panel(nm)}</figure>'
    for nm in STORMS)

legend_cat = "".join(
    f'<span class="lg"><span class="sw {c}"></span>{c} <span class="rng">{lo}{"+" if hi>900 else f"-{hi-1}"}kt</span></span>'
    for c, lo, hi in CATS)
legend_model = "".join(
    f'<span class="lg"><svg width="26" height="10" class="lgsvg"><line x1="1" y1="5" x2="25" y2="5" '
    f'class="ln {tag}" style="{STROKE_STYLE[style]}"/></svg>{label}</span>'
    for tag, label, style, *_ in MODELS)

_palL = "".join(f".{c}{{fill:{v[0]};}}" for c, v in COL.items())
_palD = "".join(f".{c}{{fill:{v[1]};}}" for c, v in COL.items())
_lcL = "".join(f".ln.{t}{{stroke:{c};}}" for t, c in LCOL.items())
_lcD = "".join(f".ln.{t}{{stroke:{c};}}" for t, c in LCOLD.items())

HTML = f"""<meta charset="utf-8">
<title>Real vs v10/v23/v35, overlaid, colored by intensity category</title>
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
figcaption h3{{color:var(--ink);font-size:14.5px;font-weight:660;margin:0;}}
.map{{width:100%;height:auto;display:block;border-radius:4px;overflow:hidden;}}
.sea{{fill:var(--sea);}} .land{{fill:var(--land);stroke:var(--coast);stroke-width:.7;}}
.gl{{stroke:var(--coast);stroke-width:.5;opacity:.4;}}
.tk{{font-family:var(--mono);font-size:8px;fill:var(--muted);}}
.ln{{fill:none;stroke-width:1.6;stroke-linejoin:round;stroke-linecap:round;opacity:.85;}}
.ln.real{{stroke-width:2.2;opacity:1;}}
.fx{{stroke:var(--surface);stroke-width:.7;}}
.genesis{{fill:none;stroke:var(--ink);stroke-width:2;}}
.peakl{{font-family:var(--mono);font-size:9px;fill:var(--ink);font-weight:600;
 paint-order:stroke;stroke:var(--sea);stroke-width:2.4px;}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted);max-width:80ch;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; real, overlaid with model mean forecasts</div>
  <h1>Real track vs v10 / v23 / v35, overlaid, colored by intensity category</h1>
  <p class="lede">One map per storm: the real track (solid line, largest dots) and each model's
  mean-by-valid-time forecast (v10 dotted, v23 dashed, v35 dash-dot, smaller dots) drawn together.
  Every dot -- real or predicted -- is colored by ITS OWN intensity category, so a model drifting
  into the wrong color band is a wrong category call, independent of position error. Dolphin isn't
  shown here: no real steering-field data exists for it yet, so v23/v35 can't run on it (unlike
  Tip, which has a dedicated real record).</p>
  <div class="legend">{legend_cat}</div>
  <div class="legend model">{legend_model}</div>
 </header>
 <div class="grid">{cards}</div>
 <footer>
  <p>Model paths are mean-by-valid-time: at each 6h valid time, the mean position and mean vmax
  across every forecast (v10/v23/v35, 1/10/5-seed ensembles respectively) whose lead lands on that
  moment, dropping bins with fewer than 3 contributing forecasts. Bavi, Co-may, and Hinnamnor use
  the in-dataset forward pass (real steering fields from the training data); Tip uses its dedicated
  real 3-hourly steering record. v10 has no steering input at all (track history only).</p>
 </footer>
</div>"""

os.makedirs("paper", exist_ok=True)
open("paper/overlay_real_vs_models.html", "w").write(HTML)
print(f"wrote paper/overlay_real_vs_models.html ({len(HTML)/1000:.0f} KB)")
