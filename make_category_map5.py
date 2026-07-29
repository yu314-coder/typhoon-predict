"""Five storms, real track colored by intensity category: Tip, Bavi, Co-may, Hinnamnor, and the
currently-active Typhoon Dolphin (2026) -- a requested subset of make_category_map.py's six.

Dolphin sourcing (as of 2026-07-29 ~06:00 UTC -- STILL ACTIVE, this is a partial record, not a
complete life cycle): position + wind from Wunderground's western-Pacific tracker, pressure
cross-checked against Zoom Earth's independent table at matching timestamps (both derive from the
same JTWC/JMA advisories; wind-speed conversion checks landed within ~5 km/h of each other at
every matched time, confirming the two sources agree). Only points through the last confirmed
observation are included -- both sources continue past this point into FORECAST territory
(smoothly increasing to a round-number peak days out), which is excluded per this project's
never-fabricate/never-treat-forecast-as-observed convention.
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


STORMS = []

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

for s, nm, yr in [("2025203N20124", "Co-may", "2025"), ("2022239N22150", "Hinnamnor", "2022"),
                   ("2026182N09163", "Bavi", "2026")]:
    k = np.where(sid == s)[0]
    k = k[np.argsort(bt[k])]
    vmax = track[k, -1, 4] * tstd[4] + tmean[4]
    pts = [(bla[k[i]], blo[k[i]], float(vmax[i])) for i in range(len(k)) if vmax[i] > 0]
    STORMS.append((nm, yr, pts))

# ---- Tip 1979 ----
tz = np.load("track_build/tip_fixed.npz", allow_pickle=True)
ttr = tz["track"].astype("float32"); tbt = tz["base_time"].astype("int64")
tbla = tz["base_lat"].astype("float64"); tblo = tz["base_lon"].astype("float64")
o13 = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
otm, ots = o13["track_mean"].astype("float32"), o13["track_std"].astype("float32")
order = np.argsort(tbt)
tvmax = ttr[order, -1, 4] * ots[4] + otm[4]
tip_pts = [(tbla[order[i]], tblo[order[i]], float(tvmax[i])) for i in range(len(order)) if tvmax[i] > 0]
STORMS.append(("Tip", "1979", tip_pts))

# ---- Dolphin 2026 -- ACTIVE, partial record, see module docstring for sourcing ----
KT_PER_MPH = 0.868976
# (lat, lon, wind_mph [Wunderground], pressure_hPa [Zoom Earth, matched by nearest timestamp])
DOLPHIN_RAW = [
    (12.8, 178.3, 35, 1002), (13.4, 176.7, 40, 1000), (13.6, 175.2, 50, 998),
    (13.2, 173.7, 50, 992), (13.0, 172.8, 70, 990), (13.3, 171.7, 85, 980),
    (13.4, 170.7, 115, 975), (13.7, 169.9, 145, 937), (14.1, 169.1, 140, 941),
    (14.5, 168.4, 140, 941),
]
dolphin_pts = [(la, lo, w * KT_PER_MPH) for la, lo, w, p in DOLPHIN_RAW]
STORMS.append(("Dolphin", "2026 (active)", dolphin_pts))

print("storms:", [(nm, len(pts)) for nm, yr, pts in STORMS])


def rings_in(lo0, lo1, la0, la1, land):
    out = []
    for f in land["features"]:
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


LAND = json.load(open("track_build/geo/ne/ne_50m_land.geojson"))


def panel(nm, yr, pts, W=460, H=360):
    m = 36
    lons = [p[1] % 360 for p in pts]; lats = [p[0] for p in pts]
    lo0, lo1, la0, la1 = min(lons), max(lons), min(lats), max(lats)
    px, py = (lo1 - lo0) * .08 + 1.2, (la1 - la0) * .08 + 1.2
    lo0, lo1, la0, la1 = lo0 - px, lo1 + px, la0 - py, la1 + py
    kx = np.cos(np.radians((la0 + la1) / 2))
    spanx, spany = (lo1 - lo0) * kx, la1 - la0
    sc = min((W - 2 * m) / spanx, (H - 2 * m) / spany)
    ox, oy = (W - spanx * sc) / 2, (H - spany * sc) / 2

    def PX(lo): return ox + ((lo % 360) - lo0) * kx * sc
    def PY(la): return H - oy - (la - la0) * sc

    o = [f'<svg viewBox="0 0 {W} {H}" class="map" role="img" aria-label="{nm} real track by category">',
         f'<rect x="0" y="0" width="{W}" height="{H}" class="sea"/>']
    for r in rings_in(lo0, lo1, la0, la1, LAND):
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
    d = "M" + " L".join(f"{PX(p[1]):.1f},{PY(p[0]):.1f}" for p in pts)
    o.append(f'<path class="track" d="{d}"/>')
    for la, lo, v in pts:
        c = cat_of(v)
        o.append(f'<circle cx="{PX(lo):.1f}" cy="{PY(la):.1f}" r="3.6" class="fx {c}"/>')
    la0p, lo0p, v0 = pts[0]
    o.append(f'<circle cx="{PX(lo0p):.1f}" cy="{PY(la0p):.1f}" r="6.5" class="genesis"/>')
    o.append(f'<text class="startl" x="{PX(lo0p)+9:.1f}" y="{PY(la0p)+3.5:.1f}">genesis</text>')
    peak = max(pts, key=lambda p: p[2])
    o.append(f'<text class="peakl" x="{PX(peak[1])+9:.1f}" y="{PY(peak[0])-6:.1f}">peak {peak[2]:.0f} kt ({cat_of(peak[2])})</text>')
    last = pts[-1]
    if nm == "Dolphin":
        o.append(f'<text class="nowl" x="{PX(last[1])+9:.1f}" y="{PY(last[0])+14:.1f}">latest ob, still active</text>')
    o.append('</svg>')
    return "\n".join(o)


cards = "".join(
    f'<figure class="panel"><figcaption><h3>{nm} <span class="sub">{yr}</span></h3>'
    f'<p>{len(pts)} real fixes, {min(p[2] for p in pts):.0f}&ndash;{max(p[2] for p in pts):.0f} kt</p>'
    f'</figcaption>{panel(nm, yr, pts)}</figure>'
    for nm, yr, pts in STORMS)

legend = "".join(
    f'<span class="lg"><span class="sw {c}"></span>{c} <span class="rng">{lo}{"+" if hi>900 else f"-{hi-1}"}kt</span></span>'
    for c, lo, hi in CATS)

_palL = "".join(f".{c}{{fill:{v[0]};}}" for c, v in COL.items())
_palD = "".join(f".{c}{{fill:{v[1]};}}" for c, v in COL.items())

HTML = f"""<meta charset="utf-8">
<title>Tip, Bavi, Co-may, Hinnamnor, Dolphin -- real tracks by intensity category</title>
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
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px;}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 14px 10px;
 margin:0;display:flex;flex-direction:column;gap:7px;}}
figcaption{{display:flex;flex-direction:column;gap:2px;}}
figcaption h3{{color:var(--ink);font-size:14.5px;font-weight:660;margin:0;display:flex;gap:8px;align-items:baseline;}}
figcaption .sub{{font-size:11px;color:var(--muted);font-weight:400;}}
figcaption p{{font-size:12px;color:var(--muted);margin:0;}}
.map{{width:100%;height:auto;display:block;border-radius:4px;overflow:hidden;}}
.sea{{fill:var(--sea);}} .land{{fill:var(--land);stroke:var(--coast);stroke-width:.7;}}
.gl{{stroke:var(--coast);stroke-width:.5;opacity:.4;}}
.tk{{font-family:var(--mono);font-size:8px;fill:var(--muted);}}
.track{{fill:none;stroke:var(--track);stroke-width:1;opacity:.5;}}
.fx{{stroke:var(--surface);stroke-width:.8;}}
{_palL}
@media (prefers-color-scheme:dark){{{_palD}}}
.genesis{{fill:none;stroke:var(--ink);stroke-width:2;}}
.startl,.peakl{{font-family:var(--mono);font-size:9px;fill:var(--ink);font-weight:600;
 paint-order:stroke;stroke:var(--sea);stroke-width:2.4px;}}
.nowl{{font-family:var(--mono);font-size:8.5px;fill:#c0392b;font-weight:700;
 paint-order:stroke;stroke:var(--sea);stroke-width:2.4px;}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted);max-width:80ch;
 display:flex;flex-direction:column;gap:8px;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; real storm history, not model output</div>
  <h1>Tip, Bavi, Co-may, Hinnamnor, Dolphin &mdash; real tracks by intensity category</h1>
  <p class="lede">Every fix is the storm's actual recorded position and wind speed &mdash; ground
  truth. Color marks the JTWC/Saffir-Simpson category at that moment (1-min sustained wind, knots).
  Hollow ring marks genesis; peak intensity and its category are labeled on each panel.
  <b>Dolphin is a currently active, still-intensifying storm</b> (as of 2026-07-29) &mdash; its
  panel is a partial record through the latest confirmed observation, not a complete life cycle.</p>
  <div class="legend">{legend}</div>
 </header>
 <div class="grid">{cards}</div>
 <footer>
  <p>Co-may (2025), Hinnamnor (2022), and Bavi (2026) are reconstructed from this project's own
  training windows (real IBTrACS-derived vmax at each issue time). Tip (1979) is from the dedicated
  real 3-hourly record used for its out-of-training test.</p>
  <p><b>Dolphin (2026), sourcing.</b> Position and wind: Wunderground's western-Pacific tracker.
  Pressure: Zoom Earth's independent table, cross-checked against Wunderground's wind at matching
  timestamps (both derive from JTWC/JMA advisories; converted wind speeds agreed within ~5 km/h at
  every matched point). Forecast values from both sources (a smooth multi-day climb to a round-number
  peak) are excluded &mdash; only confirmed observations through 2026-07-29 ~06:00 UTC are shown, per
  this project's rule that forecasts are never plotted as if they were observed. Public reporting at
  the time indicates JMA/JTWC forecasts for further rapid intensification toward Category 5, with
  meteorologists noting Dolphin could rival Typhoon Tip (1979) as one of the most intense tropical
  cyclones on record &mdash; not yet observed, not shown here.</p>
 </footer>
</div>"""

os.makedirs("paper", exist_ok=True)
open("paper/storms_by_category_5.html", "w").write(HTML)
print(f"wrote paper/storms_by_category_5.html ({len(HTML)/1000:.0f} KB)")
