"""Typhoon Tip (1979): real track vs. v23's and v35's mean-by-valid-time predicted tracks, all
colored by intensity category (TD/TS/C1-C5) -- so both position error and intensity-category error
are visible on the same map, extending make_category_map.py's real-only view with model output.

Reads track_build/tip_v23_v35_meanpath.json (from _tip_v23_v35_meanpath.py).
"""
import json, os
import numpy as np

R = 111.2
CATS = [("TD", 0, 34), ("TS", 34, 64), ("C1", 64, 83), ("C2", 83, 96),
        ("C3", 96, 113), ("C4", 113, 137), ("C5", 137, 999)]
COL = {"TD": ("#8a94a6", "#9fb0bd"), "TS": ("#2a78d6", "#3987e5"), "C1": ("#e0b400", "#f2c744"),
       "C2": ("#e08a1e", "#f2994a"), "C3": ("#d94f3d", "#eb5757"), "C4": ("#b8291f", "#e15347"),
       "C5": ("#8b2fb0", "#c07de0")}


def cat_of(v):
    for name, lo, hi in CATS:
        if lo <= v < hi:
            return name
    return "C5"


# ---- real Tip track (verbatim from make_category_map.py) ----
tz = np.load("track_build/tip_fixed.npz", allow_pickle=True)
ttr = tz["track"].astype("float32"); tbt = tz["base_time"].astype("int64")
tbla = tz["base_lat"].astype("float64"); tblo = tz["base_lon"].astype("float64")
o13 = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
otm, ots = o13["track_mean"].astype("float32"), o13["track_std"].astype("float32")
order = np.argsort(tbt)
tvmax = ttr[order, -1, 4] * ots[4] + otm[4]
real_pts = [(tbla[order[i]], tblo[order[i]], float(tvmax[i])) for i in range(len(order)) if tvmax[i] > 0]

pred = json.load(open("track_build/tip_v23_v35_meanpath.json"))
PANELS = [("Real (observed)", real_pts, "obs"),
          ("v23 -- mean forecast", [tuple(p) for p in pred["v23"]], "pred"),
          ("v35 -- mean forecast", [tuple(p) for p in pred["v35"]], "pred")]

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
            if len(simp) >= 3:
                pass
    return out


# shared bounding box (union of all three tracks) so the three panels are directly comparable
all_pts = real_pts + PANELS[1][1] + PANELS[2][1]
_lons = [p[1] % 360 for p in all_pts]; _lats = [p[0] for p in all_pts]
LO0, LO1, LA0, LA1 = min(_lons), max(_lons), min(_lats), max(_lats)
_px, _py = (LO1 - LO0) * .06 + 1.0, (LA1 - LA0) * .06 + 1.0
LO0, LO1, LA0, LA1 = LO0 - _px, LO1 + _px, LA0 - _py, LA1 + _py


def panel(title, pts, kind, W=420, H=460):
    m = 34
    kx = np.cos(np.radians((LA0 + LA1) / 2))
    spanx, spany = (LO1 - LO0) * kx, LA1 - LA0
    sc = min((W - 2 * m) / spanx, (H - 2 * m) / spany)
    ox, oy = (W - spanx * sc) / 2, (H - spany * sc) / 2

    def PX(lo): return ox + ((lo % 360) - LO0) * kx * sc
    def PY(la): return H - oy - (la - LA0) * sc

    o = [f'<svg viewBox="0 0 {W} {H}" class="map" role="img" aria-label="{title} on Typhoon Tip">',
         f'<rect x="0" y="0" width="{W}" height="{H}" class="sea"/>']
    for r in rings_in(LO0, LO1, LA0, LA1):
        d = "M" + " L".join(f"{PX(p[0]):.1f},{PY(p[1]):.1f}" for p in r) + " Z"
        o.append(f'<path class="land" d="{d}"/>')
    step = 10
    g = np.ceil(LA0 / step) * step
    while g <= LA1:
        y = PY(g)
        o.append(f'<line class="gl" x1="0" x2="{W}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="tk" x="4" y="{y-3:.1f}">{abs(g):.0f}&deg;{"N" if g>=0 else "S"}</text>')
        g += step
    g = np.ceil(LO0 / step) * step
    while g <= LO1:
        x = PX(g)
        o.append(f'<line class="gl" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{H}"/>')
        o.append(f'<text class="tk" x="{x+3:.1f}" y="{H-5}">{((g+180)%360)-180:.0f}&deg;E</text>')
        g += step
    d = "M" + " L".join(f"{PX(p[1]):.1f},{PY(p[0]):.1f}" for p in pts)
    o.append(f'<path class="track{" dashed" if kind=="pred" else ""}" d="{d}"/>')
    for la, lo, v in pts:
        c = cat_of(v)
        o.append(f'<circle cx="{PX(lo):.1f}" cy="{PY(la):.1f}" r="3.4" class="fx {c}"/>')
    la0p, lo0p, v0 = pts[0]
    o.append(f'<circle cx="{PX(lo0p):.1f}" cy="{PY(la0p):.1f}" r="6" class="genesis"/>')
    peak = max(pts, key=lambda p: p[2])
    o.append(f'<text class="peakl" x="{PX(peak[1])+8:.1f}" y="{PY(peak[0])-6:.1f}">peak {peak[2]:.0f} kt ({cat_of(peak[2])})</text>')
    o.append('</svg>')
    return "\n".join(o)


cards = "".join(
    f'<figure class="panel"><figcaption><h3>{title}</h3>'
    f'<p>{len(pts)} points, {min(p[2] for p in pts):.0f}&ndash;{max(p[2] for p in pts):.0f} kt</p>'
    f'</figcaption>{panel(title, pts, kind)}</figure>'
    for title, pts, kind in PANELS)

legend = "".join(
    f'<span class="lg"><span class="sw {c}"></span>{c} <span class="rng">{lo}{"+" if hi>900 else f"-{hi-1}"}kt</span></span>'
    for c, lo, hi in CATS)

_palL = "".join(f".{c}{{fill:{v[0]};}}" for c, v in COL.items())
_palD = "".join(f".{c}{{fill:{v[1]};}}" for c, v in COL.items())

HTML = f"""<meta charset="utf-8">
<title>Typhoon Tip: real vs v23 vs v35, colored by intensity category</title>
<style>
:root{{color-scheme:light;--bg:#f2f4f6;--surface:#fcfcfb;--ink:#111820;--body:#2c3a47;--muted:#5d6c7a;
 --line:#d5dce3;--sea:#eaf1f5;--land:#dfe3e0;--coast:#a8b3ba;--track:#333d47;
 --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;
 --sea:#101b24;--land:#26313a;--coast:#4a5a66;--track:#c2cdd8;}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0c1117;--surface:#141c25;--ink:#e8eef4;
 --body:#c2cdd8;--muted:#8697a5;--line:#26313d;--sea:#101b24;--land:#26313a;--coast:#4a5a66;--track:#c2cdd8;}}
body{{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:16px;line-height:1.6;}}
.wrap{{max-width:1180px;margin:0 auto;padding:clamp(24px,5vw,52px) clamp(16px,4vw,32px) 80px;
 display:flex;flex-direction:column;gap:26px;}}
h1{{color:var(--ink);font-size:clamp(24px,3.6vw,36px);line-height:1.14;letter-spacing:-.02em;margin:0;font-weight:660;}}
.eyebrow{{font-family:var(--mono);font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);}}
.lede{{max-width:80ch;font-size:14.5px;margin:0;}}
header{{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--line);padding-bottom:22px;}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;}}
.lg{{display:flex;align-items:center;gap:6px;}}
.sw{{width:11px;height:11px;border-radius:50%;display:inline-block;}}
.rng{{color:var(--muted);font-family:var(--mono);font-size:11px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 14px 10px;
 margin:0;display:flex;flex-direction:column;gap:7px;}}
figcaption{{display:flex;flex-direction:column;gap:2px;}}
figcaption h3{{color:var(--ink);font-size:14.5px;font-weight:660;margin:0;}}
figcaption p{{font-size:12px;color:var(--muted);margin:0;}}
.map{{width:100%;height:auto;display:block;border-radius:4px;overflow:hidden;}}
.sea{{fill:var(--sea);}} .land{{fill:var(--land);stroke:var(--coast);stroke-width:.7;}}
.gl{{stroke:var(--coast);stroke-width:.5;opacity:.4;}}
.tk{{font-family:var(--mono);font-size:8px;fill:var(--muted);}}
.track{{fill:none;stroke:var(--track);stroke-width:1;opacity:.5;}}
.track.dashed{{stroke-dasharray:3 2.4;}}
.fx{{stroke:var(--surface);stroke-width:.8;}}
{_palL}
@media (prefers-color-scheme:dark){{{_palD}}}
.genesis{{fill:none;stroke:var(--ink);stroke-width:2;}}
.peakl{{font-family:var(--mono);font-size:9px;fill:var(--ink);font-weight:600;
 paint-order:stroke;stroke:var(--sea);stroke-width:2.4px;}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted);max-width:80ch;
 display:flex;flex-direction:column;gap:8px;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; real vs. predicted, category-colored</div>
  <h1>Typhoon Tip (1979): real track vs. v23 and v35 mean forecasts</h1>
  <p class="lede">Same three-panel layout and map extent so position and intensity-category error
  are both visible at a glance. The two model panels are each model's mean forecast by valid time
  (solid line, category-colored dots -- dashed stroke marks it as a forecast, not observed).
  Tip is the most intense tropical cyclone ever recorded; both models are known to
  systematically under-predict its peak intensity (see the mean-forecast-vs-real vmax chart from
  earlier this session) -- this map shows where that shows up as a wrong CATEGORY, not just a
  wrong number.</p>
  <div class="legend">{legend}</div>
 </header>
 <div class="grid">{cards}</div>
 <footer>
  <p>Real track: <code>track_build/tip_fixed.npz</code>, the dedicated real 3-hourly record used
  for Tip's out-of-training test. Model panels: mean position and mean vmax across every forecast
  whose lead lands on that 6-hour valid time (bins with fewer than 3 contributing forecasts are
  dropped), the same convention <code>make_map_html.py</code> uses for track-only maps elsewhere
  in this project -- extended here to also average predicted vmax, not just position.</p>
 </footer>
</div>"""

os.makedirs("paper", exist_ok=True)
open("paper/tip_v23_v35_category_map.html", "w").write(HTML)
print(f"wrote paper/tip_v23_v35_category_map.html ({len(HTML)/1000:.0f} KB)")
