"""v23's IBTrACS-only forecast for Typhoon Dolphin (2026): the real observed track, plus v23's
predicted track/intensity using ONLY the storm's own history (position/wind/pressure) -- no
steering field at all, the same ablation documented in models/README.md and run via the
released standalone package (models/run_v23.py, no --steering flag).

This is the actual public package running -- not the internal dev pipeline -- so the numbers here
are exactly reproducible by anyone who clones the repo. See the "How to run this" section in the
generated page, or just run:

    python3 models/run_v23.py --track dolphin_track.json --out forecast.json

Writes paper/dolphin_v23_ibtracs_only.html.
"""
import json, os
import numpy as np

R = 111.2
CATS = [("TD", 0, 34), ("TS", 34, 64), ("C1", 64, 83), ("C2", 83, 96),
        ("C3", 96, 113), ("C4", 113, 137), ("C5", 137, 999)]
COL = {"TD": ("#8a94a6", "#9fb0bd"), "TS": ("#2a78d6", "#3987e5"), "C1": ("#e0b400", "#f2c744"),
       "C2": ("#e08a1e", "#f2994a"), "C3": ("#d94f3d", "#eb5757"), "C4": ("#b8291f", "#e15347"),
       "C5": ("#8b2fb0", "#c07de0")}
LCOL = {"real": "#222b33", "v23_ibt": "#2a78d6", "jtwc": "#c2410c"}
LCOLD = {"real": "#e8eef4", "v23_ibt": "#3987e5", "jtwc": "#f0813f"}
STROKE_STYLE = {"solid": "", "dashed": "stroke-dasharray:6 3;", "dotted": "stroke-dasharray:1.4 3.2;"}


def cat_of(v):
    for name, lo, hi in CATS:
        if lo <= v < hi:
            return name
    return "C5"


# ---- real observed track (Wunderground position/wind + Zoom Earth pressure, cross-checked --
# see make_category_map5.py) ----
KT_PER_MPH = 0.868976
DOLPHIN_TIMES = ["2026-07-27T00:00", "2026-07-27T06:00", "2026-07-27T12:00", "2026-07-27T18:00",
                 "2026-07-28T00:00", "2026-07-28T06:00", "2026-07-28T12:00", "2026-07-28T18:00",
                 "2026-07-29T00:00", "2026-07-29T06:00"]
RAW = [(12.8, 178.3, 35, 1002), (13.4, 176.7, 40, 1000), (13.6, 175.2, 50, 998),
       (13.2, 173.7, 50, 992), (13.0, 172.8, 70, 990), (13.3, 171.7, 85, 980),
       (13.4, 170.7, 115, 975), (13.7, 169.9, 145, 937), (14.1, 169.1, 140, 941),
       (14.5, 168.4, 140, 941)]
REAL = [(la, lo, w * KT_PER_MPH, t) for (la, lo, w, p), t in zip(RAW, DOLPHIN_TIMES)]

# ---- v23 IBTrACS-only forecast, from the actual released package (models/run_v23.py, no
# --steering) -- see module docstring for the exact reproduction command ----
FCST = json.load(open("track_build/dolphin_v23_ibtracs_forecast.json"))
V23_IBT = [(FCST["lats"][i], FCST["lons"][i], FCST["vmax_kt"][i],
            str(np.datetime64(FCST["issue_time"]) + np.timedelta64(int(h), "h")).replace("T", " ") +
            f" (+{h}h)")
           for i, h in enumerate(FCST["lead_hours"])]

# ---- JTWC official forecast (Advisory #3, issued 2026-07-27 12:00 UTC) -- TAU 0/12/24/48/72/96/120h ----
JTWC_TIMES = ["2026-07-27 12:00 (issued)", "2026-07-28 00:00", "2026-07-28 12:00",
              "2026-07-29 12:00", "2026-07-30 12:00", "2026-07-31 12:00", "2026-08-01 12:00"]
JTWC_TAU_H = [0, 12, 24, 48, 72, 96, 120]
JTWC_RAW = [(13.7, 175.3, 45), (13.5, 172.9, 55), (13.7, 170.5, 70), (15.2, 167.3, 100),
            (17.0, 164.3, 120), (18.9, 161.0, 140), (21.0, 157.5, 150)]
JTWC_FCST = [(la, lo, float(kt), t) for (la, lo, kt), t in zip(JTWC_RAW, JTWC_TIMES)]

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


def panel(series, W=760, H=520):
    m = 42
    all_pts = [p for _, _, _, _, _, pts in series for p in pts]
    lons = [p[1] % 360 for p in all_pts]; lats = [p[0] for p in all_pts]
    lo0, lo1, la0, la1 = min(lons), max(lons), min(lats), max(lats)
    px, py = (lo1 - lo0) * .08 + 1.0, (la1 - la0) * .08 + 1.0
    lo0, lo1, la0, la1 = lo0 - px, lo1 + px, la0 - py, la1 + py
    kx = np.cos(np.radians((la0 + la1) / 2))
    spanx, spany = (lo1 - lo0) * kx, la1 - la0
    sc = min((W - 2 * m) / spanx, (H - 2 * m) / spany)
    ox, oy = (W - spanx * sc) / 2, (H - spany * sc) / 2

    def PX(lo): return ox + ((lo % 360) - lo0) * kx * sc
    def PY(la): return H - oy - (la - la0) * sc

    o = [f'<svg viewBox="0 0 {W} {H}" class="map" role="img" aria-label="Dolphin: v23 IBTrACS-only forecast">',
         f'<rect x="0" y="0" width="{W}" height="{H}" class="sea"/>']
    for r in rings_in(lo0, lo1, la0, la1):
        d = "M" + " L".join(f"{PX(p[0]):.1f},{PY(p[1]):.1f}" for p in r) + " Z"
        o.append(f'<path class="land" d="{d}"/>')
    step = 5
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
            la, lo, v, when = p
            c = cat_of(v)
            o.append(f'<circle cx="{PX(lo):.1f}" cy="{PY(la):.1f}" r="{dotr}" class="fx {c} pt" '
                      f'opacity="{op}" data-model="{label}" data-time="{when}" '
                      f'data-lat="{la:.2f}" data-lon="{lo:.2f}" data-vmax="{v:.0f}" data-cat="{c}" '
                      f'tabindex="0"/>')
    d0 = series[0][5][0]
    o.append(f'<circle cx="{PX(d0[1]):.1f}" cy="{PY(d0[0]):.1f}" r="7" class="genesis"/>')
    now_pt = series[0][5][-1]
    o.append(f'<text class="nowl" x="{PX(now_pt[1])+10:.1f}" y="{PY(now_pt[0])-8:.1f}">now: '
              f'{now_pt[2]:.0f} kt ({cat_of(now_pt[2])})</text>')
    o.append('</svg>')
    return "\n".join(o)


series = [("real", "Real (observed)", "solid", 5.0, 1.0, REAL),
          ("v23_ibt", "v23, IBTrACS-only (steering zeroed)", "dashed", 3.6, 0.9, V23_IBT),
          ("jtwc", "JTWC official forecast (Advisory #3)", "dotted", 4.0, 0.9, JTWC_FCST)]
map_svg = panel(series)

rows = "".join(
    f'<tr><td>+{h}h</td><td>{str(np.datetime64(FCST["issue_time"]) + np.timedelta64(int(h), "h")).replace("T", " ")}Z</td>'
    f'<td>{FCST["lats"][i]:.2f}&deg;N</td><td>{FCST["lons"][i]:.2f}&deg;E</td>'
    f'<td>{FCST["vmax_kt"][i]:.0f} kt</td><td class="cat{cat_of(FCST["vmax_kt"][i])}">{cat_of(FCST["vmax_kt"][i])}</td>'
    f'<td>{FCST["pres_hpa"][i]:.0f} hPa</td></tr>'
    for i, h in enumerate(FCST["lead_hours"]))

jtwc_rows = "".join(
    f'<tr><td>TAU {h}h</td><td>{t}</td><td>{la:.2f}&deg;N</td><td>{lo:.2f}&deg;E</td>'
    f'<td>{kt:.0f} kt</td><td class="cat{cat_of(kt)}">{cat_of(kt)}</td><td>&mdash;</td></tr>'
    for h, t, (la, lo, kt, _) in zip(JTWC_TAU_H, JTWC_TIMES, JTWC_FCST))

_palL = "".join(f".{c}{{fill:{v[0]};}} .cat{c}{{color:{v[0]};}}" for c, v in COL.items())
_palD = "".join(f".{c}{{fill:{v[1]};}} .cat{c}{{color:{v[1]};}}" for c, v in COL.items())
_lcL = "".join(f".ln.{t}{{stroke:{c};}}" for t, c in LCOL.items())
_lcD = "".join(f".ln.{t}{{stroke:{c};}}" for t, c in LCOLD.items())

RUN_CODE = """# 1. clone the repo (checkpoints are committed with Git LFS)
git clone https://github.com/yu314-coder/typhoon-predict.git
cd typhoon-predict
git lfs pull   # fetches models/v23_seed*.pt (10 seeds, ~526 MB total)
pip install torch numpy

# 2. write your storm's track history as JSON -- oldest to newest, 6h apart, ending at "now"
#    (pres_hpa may be omitted per-fix if unknown -- it's zero-filled and flagged, not guessed)
cat > dolphin_track.json <<'EOF'
[
  {"time": "2026-07-29T00:00", "lat": 14.1, "lon": 169.1, "vmax_kt": 121.7, "pres_hpa": 941},
  {"time": "2026-07-29T06:00", "lat": 14.5, "lon": 168.4, "vmax_kt": 121.7, "pres_hpa": 941}
]
EOF
# (the model uses up to the last 9 fixes; fewer than 9 is fine -- see models/run_v23.py --help)

# 3. run it -- IBTrACS-only mode is the default (omit --steering entirely)
python3 models/run_v23.py --track dolphin_track.json --out forecast.json

# forecast.json: 20 leads (6h..120h), each with lat/lon/vmax_kt/pres_hpa, averaged across
# all 10 seeds in the ensemble. To ALSO use real steering data (full-data mode, this project's
# headline 434.96 km number), fetch a deep-layer-mean wind NPZ (see _fetch_dolphin_steering.py
# for a working NOAA/NOMADS recipe) and add: --steering my_steering.npz"""

HTML = f"""<meta charset="utf-8">
<title>TrackFormer v23 -- IBTrACS-only forecast for Typhoon Dolphin</title>
<style>
:root{{color-scheme:light;--bg:#f2f4f6;--surface:#fcfcfb;--ink:#111820;--body:#2c3a47;--muted:#5d6c7a;
 --line:#d5dce3;--sea:#eaf1f5;--land:#dfe3e0;--coast:#a8b3ba;--code-bg:#0d1117;--code-ink:#c9d1d9;
 --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;{_palL}{_lcL}}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;
 --sea:#101b24;--land:#26313a;--coast:#4a5a66;{_palD}{_lcD}}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0c1117;--surface:#141c25;--ink:#e8eef4;
 --body:#c2cdd8;--muted:#8697a5;--line:#26313d;--sea:#101b24;--land:#26313a;--coast:#4a5a66;{_palD}{_lcD}}}
body{{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:16px;line-height:1.6;}}
.wrap{{max-width:900px;margin:0 auto;padding:clamp(24px,5vw,52px) clamp(16px,4vw,32px) 80px;
 display:flex;flex-direction:column;gap:26px;}}
h1{{color:var(--ink);font-size:clamp(22px,3.2vw,32px);line-height:1.16;letter-spacing:-.02em;margin:0;font-weight:660;}}
h2{{color:var(--ink);font-size:17px;font-weight:640;margin:0 0 4px;}}
.eyebrow{{font-family:var(--mono);font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);}}
.lede{{max-width:72ch;font-size:14.5px;margin:0;}}
header{{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--line);padding-bottom:22px;}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;}}
.lg{{display:flex;align-items:center;gap:6px;}}
.lgsvg{{display:block;}}
section{{display:flex;flex-direction:column;gap:10px;}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px;}}
.map{{width:100%;height:auto;display:block;border-radius:4px;overflow:hidden;}}
.sea{{fill:var(--sea);}} .land{{fill:var(--land);stroke:var(--coast);stroke-width:.7;}}
.gl{{stroke:var(--coast);stroke-width:.5;opacity:.4;}}
.tk{{font-family:var(--mono);font-size:9px;fill:var(--muted);}}
.ln{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round;opacity:.9;}}
.ln.real{{stroke-width:2.6;opacity:1;}}
.fx{{stroke:var(--surface);stroke-width:.7;cursor:pointer;}}
.fx.pt:hover,.fx.pt:focus{{stroke:var(--ink);stroke-width:1.8;outline:none;}}
.genesis{{fill:none;stroke:var(--ink);stroke-width:2;}}
.nowl{{font-family:var(--mono);font-size:10px;fill:var(--ink);font-weight:600;
 paint-order:stroke;stroke:var(--sea);stroke-width:2.6px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);font-family:var(--mono);}}
th{{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em;}}
td:first-child,th:first-child{{padding-left:0;}}
.tablewrap{{overflow-x:auto;}}
pre{{background:var(--code-bg);color:var(--code-ink);border-radius:8px;padding:16px 18px;
 overflow-x:auto;font-family:var(--mono);font-size:12.5px;line-height:1.55;margin:0;}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted);max-width:72ch;}}
#tt{{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font-family:var(--mono);
 font-size:12px;padding:8px 10px;border-radius:6px;box-shadow:0 6px 18px rgba(0,0,0,.25);
 display:none;z-index:50;line-height:1.5;white-space:nowrap;}}
#tt .hint{{color:var(--muted);font-size:10px;margin-top:2px;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer v23 &middot; IBTrACS-only ablation vs. JTWC &middot; Typhoon Dolphin, 2026</div>
  <h1>v23's forecast for Dolphin using ONLY its own track history, vs. JTWC</h1>
  <p class="lede">No steering field, no environmental data of any kind -- just Dolphin's own
  observed position, wind, and pressure history (the same information IBTrACS or any best-track
  record gives you for a past storm), fed through v23's 10-seed ensemble via the actual released
  package (<code>models/run_v23.py</code>, no <code>--steering</code> flag), issued from the latest
  real observation (2026-07-29 06:00 UTC) -- layered against JTWC's official Advisory #3 forecast
  (issued 2026-07-27 12:00 UTC) for comparison. v23 forecasts weakening; JTWC forecasts continued
  intensification toward Category 5.</p>
  <div class="legend">
   <span class="lg"><svg width="26" height="10" class="lgsvg"><line x1="1" y1="5" x2="25" y2="5" class="ln real"/></svg>Real (observed)</span>
   <span class="lg"><svg width="26" height="10" class="lgsvg"><line x1="1" y1="5" x2="25" y2="5" class="ln v23_ibt" style="{STROKE_STYLE['dashed']}"/></svg>v23, IBTrACS-only forecast</span>
   <span class="lg"><svg width="26" height="10" class="lgsvg"><line x1="1" y1="5" x2="25" y2="5" class="ln jtwc" style="{STROKE_STYLE['dotted']}"/></svg>JTWC official forecast (Advisory #3)</span>
  </div>
 </header>

 <section>
  <h2>Predicted track</h2>
  <div class="panel">{map_svg}</div>
 </section>

 <section>
  <h2>v23 forecast table (IBTrACS-only)</h2>
  <div class="panel tablewrap">
   <table>
    <thead><tr><th>Lead</th><th>Valid time (UTC)</th><th>Lat</th><th>Lon</th><th>vmax</th><th>Cat</th><th>Pressure</th></tr></thead>
    <tbody>{rows}</tbody>
   </table>
  </div>
 </section>

 <section>
  <h2>JTWC official forecast table (Advisory #3, wind only -- no per-lead pressure published)</h2>
  <div class="panel tablewrap">
   <table>
    <thead><tr><th>Lead</th><th>Valid time (UTC)</th><th>Lat</th><th>Lon</th><th>vmax</th><th>Cat</th><th>Pressure</th></tr></thead>
    <tbody>{jtwc_rows}</tbody>
   </table>
  </div>
 </section>

 <section>
  <h2>How to run this yourself</h2>
  <p class="lede">This is exactly the command that produced the forecast above -- same checkpoints,
  same code, no internal-only tooling.</p>
  <pre>{RUN_CODE}</pre>
 </section>

 <footer>
  <p>v23 also supports a full-data mode (<code>--steering my_steering.npz</code>) that additionally
  feeds a real deep-layer-mean steering-wind patch around the storm -- required for the project's
  headline 434.96 km aggregate result. On Dolphin specifically, adding real steering shifts the
  120h forecast position by roughly 600 km (further west, less recurved) and firms up the intensity
  forecast somewhat (90 &rarr; 99 kt at 120h) without changing the overall call: v23 forecasts
  weakening from here either way, in contrast to JTWC's and JMA's official forecasts of continued
  intensification. See <code>models/README.md</code> and
  <a href="overlay_real_vs_models.html">the full multi-model/multi-agency comparison map</a> for
  that side-by-side.</p>
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
open("paper/dolphin_v23_ibtracs_only.html", "w").write(HTML)
print(f"wrote paper/dolphin_v23_ibtracs_only.html ({len(HTML)/1000:.0f} KB)")
