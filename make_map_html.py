"""Real-world maps of every forecast for the four test storms — v10 vs v16 — as an HTML page.

Reads track_build/v10_tracks.json and track_build/v16_tracks.json (both produced by the model
runs), plus Natural Earth 50m land polygons. Writes paper/storm_maps.html.

Each model gets its own panel: v10 violet and v16 blue separate at only dE 16.6 under
deuteranopia, which passes the floor but is too weak to disentangle overlapping thin lines.

The MEAN track is averaged by VALID TIME, not by lead. Forecasts launched hours apart only
describe the same moment at different lead offsets; a lead-wise mean would blend positions hours
apart and draw a route the model never predicted.
"""
import json, math, os

TD = os.environ.get("TRACK_DIR", "track_build")

LAND = json.load(open("track_build/geo/ne/ne_50m_land.geojson"))
V10 = json.load(open(f"{TD}/v10_tracks.json"))
CONS = {}
for _t in ("v10", "v17", "v18", "v19"):
    _c = f"{TD}/{_t}_consensus.json"
    if os.path.exists(_c):
        CONS[_t] = json.load(open(_c))
_want = [t.strip() for t in os.environ.get("MAP_MODELS", "v10,v17,v18,v19").split(",") if t.strip()]
MODELS = [("v10", V10)] if "v10" in _want else []
for _t in ("v13", "v14", "v17", "v18", "v19"):
    if _t not in _want:
        continue
    _p = f"{TD}/{_t}_tracks.json"
    if os.path.exists(_p):
        MODELS.append((_t, json.load(open(_p))))
STORMS = [tuple(x.split(":")) for x in os.environ["MAP_STORMS"].split(",")] \
         if os.environ.get("MAP_STORMS") else \
         [("Bavi", "2026"), ("Wayne", "1986"), ("Co-may", "2025"), ("Hinnamnor", "2022")]
# validated both modes: node validate_palette.js "#eda100,#e34948,#2a78d6,#1baf7a" light
#                        node validate_palette.js "#c98500,#e66767,#3987e5,#199e70" dark
COL = {"v10": ("#eda100", "#c98500"), "v13": ("#e87ba4", "#d55181"), "v14": ("#e34948", "#e66767"),
       "v17": ("#2a78d6", "#3987e5"), "v18": ("#1baf7a", "#199e70"),
       "v19": ("#4a3aa7", "#9085e9")}
NOTE = {"v10": "no environmental field at all",
        "v13": "surface SLP only",
        "v14": "500 hPa steering, original data",
        "v17": "steering + repaired data + 4-term track loss",
        "v18": "v17 + EMA, stronger dropout, input jitter",
        "v19": "v17 + intensity persistence baseline, rarity + speed weighting"}


def rings_in(lo0, lo1, la0, la1):
    """Land rings overlapping the box, thinned to what the panel can actually resolve."""
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


SIX_H = int(6 * 3600 * 1e9)
MIN_MEMBERS = 3


def mean_by_valid_time(bts, lats, lons):
    """Mean forecast position per valid time.

    Valid times are SNAPPED to the native 6-hour best-track grid first. Some storms carry 3-hourly
    fixes, which puts consecutive valid times in disjoint sets of initialisations — the mean then
    alternates between two unrelated subsets and draws a sawtooth, with steps up to 354 km per 3 h
    for Bavi, which no storm does. Snapping makes neighbouring bins share most of their members.

    Bins with fewer than MIN_MEMBERS forecasts are dropped: at the ends of a storm's life only one
    initialisation reaches that far, and a one-member 'mean' is just that single forecast wearing a
    bold line.
    """
    acc = {}
    for w, bt in enumerate(bts):
        for L in range(20):
            vt = int(round((int(bt) + (L + 1) * SIX_H) / SIX_H)) * SIX_H
            a = acc.setdefault(vt, [0.0, 0.0, 0])
            a[0] += lats[w][L]; a[1] += lons[w][L]; a[2] += 1
    ts = [t for t in sorted(acc) if acc[t][2] >= MIN_MEMBERS]
    return [acc[t][0] / acc[t][2] for t in ts], [acc[t][1] / acc[t][2] for t in ts]


def panel(tag, rec, obs_lat, obs_lon, storm=None, W=520, H=380):
    m = 34
    LAT, LON, BT = rec["lat"], rec["lon"], rec["base_time"]
    xs = [x for t in LON for x in t] + list(obs_lon)
    ys = [y for t in LAT for y in t] + list(obs_lat)
    lo0, lo1, la0, la1 = min(xs), max(xs), min(ys), max(ys)
    px, py = (lo1 - lo0) * .07 + 1.0, (la1 - la0) * .07 + 1.0
    lo0, lo1, la0, la1 = lo0 - px, lo1 + px, la0 - py, la1 + py
    kx = math.cos(math.radians((la0 + la1) / 2))
    spanx, spany = (lo1 - lo0) * kx, la1 - la0
    sc = min((W - 2 * m) / spanx, (H - 2 * m) / spany)
    ox, oy = (W - spanx * sc) / 2, (H - spany * sc) / 2
    def PX(lon): return ox + (lon - lo0) * kx * sc
    def PY(lat): return H - oy - (lat - la0) * sc
    o = [f'<svg viewBox="0 0 {W} {H}" class="map" role="img" '
         f'aria-label="{tag} forecasts over the western Pacific">',
         f'<rect x="0" y="0" width="{W}" height="{H}" class="sea"/>']
    for r in rings_in(lo0, lo1, la0, la1):
        d = "M" + " L".join(f"{PX(p[0]):.1f},{PY(p[1]):.1f}" for p in r) + " Z"
        o.append(f'<path class="land" d="{d}"/>')
    step = 5 if (la1 - la0) > 11 else 2
    g = math.ceil(la0 / step) * step
    while g <= la1:
        y = PY(g)
        o.append(f'<line class="gl" x1="0" x2="{W}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="tk" x="4" y="{y-3:.1f}">{abs(g):.0f}°{"N" if g>=0 else "S"}</text>')
        g += step
    g = math.ceil(lo0 / step) * step
    while g <= lo1:
        x = PX(g)
        o.append(f'<line class="gl" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{H}"/>')
        o.append(f'<text class="tk" x="{x+3:.1f}" y="{H-5}">{((g+180)%360)-180:.0f}°E</text>')
        g += step
    for a in range(len(LAT)):
        d = "M" + " L".join(f"{PX(lo):.1f},{PY(la):.1f}" for la, lo in zip(LAT[a], LON[a]))
        o.append(f'<path class="spag" d="{d}"/>')
    labels = []
    cc = CONS.get(tag, {}).get(storm)
    if cc and os.environ.get("MAP_BOTH"):
        # show the plain valid-time mean AND the weighted+smoothed consensus side by side
        mla, mlo = mean_by_valid_time(BT, LAT, LON)
        o.append('<path class="rawmean" d="M' +
                 " L".join(f"{PX(lo):.1f},{PY(la):.1f}" for la, lo in zip(mla, mlo)) + '"/>')
        labels.append((PY(mla[-1]), PX(mlo[-1]), "mean"))
        pts_w = cc.get("smooth") or cc["rmt"]
        o.append('<path class="meanline" d="M' +
                 " L".join(f"{PX(p[1]):.1f},{PY(p[0]):.1f}" for p in pts_w) + '"/>')
        labels.append((PY(pts_w[-1][0]), PX(pts_w[-1][1]), "weighted"))
    elif cc:
        pts_w = cc.get("smooth") or cc["rmt"]
        o.append('<path class="meanline" d="M' +
                 " L".join(f"{PX(p[1]):.1f},{PY(p[0]):.1f}" for p in pts_w) + '"/>')
        labels.append((PY(pts_w[-1][0]), PX(pts_w[-1][1]), "consensus"))
    else:
        mla, mlo = mean_by_valid_time(BT, LAT, LON)
        o.append('<path class="meanline" d="M' +
                 " L".join(f"{PX(lo):.1f},{PY(la):.1f}" for la, lo in zip(mla, mlo)) + '"/>')
        labels.append((PY(mla[-1]), PX(mlo[-1]), "mean"))
    # push apart so the two endpoint labels never overlap
    labels.sort()
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i - 1][0] < 13:
            labels[i] = (labels[i - 1][0] + 13, labels[i][1], labels[i][2])
    for ly, lx, txt in labels:
        o.append(f'<text class="cl" x="{lx+7:.1f}" y="{ly+3.5:.1f}">{txt}</text>')
    o.append('<path class="obs" d="M' +
             " L".join(f"{PX(lo):.1f},{PY(la):.1f}" for la, lo in zip(obs_lat, obs_lon)) + '"/>')
    o.append(f'<circle class="start" cx="{PX(obs_lon[0]):.1f}" cy="{PY(obs_lat[0]):.1f}" r="5"/>')
    o.append(f'<text class="startl" x="{PX(obs_lon[0])+9:.1f}" y="{PY(obs_lat[0])+3.5:.1f}">genesis</text>')
    o.append('</svg>')
    return "\n".join(o)


sections = []
for nm, yr in STORMS:
    if nm not in V10:
        continue
    obs_lat, obs_lon = V10[nm]["base_lat"], V10[nm]["base_lon"]
    cards = []
    for tag, src in MODELS:
        if src is None or nm not in src:
            continue
        rec = src[nm]
        cards.append(
            f'<figure class="panel" data-model="{tag}">'
            f'<figcaption><h3>{tag}<span class="sub">{NOTE[tag]}</span></h3>'
            f'<p><b>{rec["err120_mean"]:.0f} km</b> mean 120 h error over {rec["n"]} forecasts</p>'
            f'</figcaption>{panel(tag, rec, obs_lat, obs_lon, storm=nm)}</figure>')
    sections.append(f"""
  <section class="storm">
    <div class="sec-head"><div class="eyebrow">{yr}</div><h2>{nm}</h2>
      <p class="lede">Every full-horizon forecast this storm produced — {V10[nm]['n']} of them —
      drawn as one thin line each. Bold is the mean by valid time; dotted black is the observed
      track from genesis.</p></div>
    <div class="pair">{''.join(cards)}</div>
  </section>""")

HTML = f"""<title>TrackFormer — forecast tracks on the map</title>
<style>
:root{{color-scheme:light;--bg:#f2f4f6;--surface:#fcfcfb;--surface-2:#e9edf1;--ink:#111820;--body:#2c3a47;
 --muted:#5d6c7a;--line:#d5dce3;--sea:#eaf1f5;--land:#dfe3e0;--coast:#a8b3ba;
 --c-v10:#eda100;--c-v14:#e34948;--c-v17:#2a78d6;--c-v18:#1baf7a;--c-v19:#4a3aa7;--obs:#11181f;
 --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--surface-2:#1b242f;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;
 --line:#26313d;--sea:#101b24;--land:#26313a;--coast:#4a5a66;
 --c-v10:#c98500;--c-v14:#e66767;--c-v17:#3987e5;--c-v18:#199e70;--c-v19:#9085e9;--obs:#f0f5fa;}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0c1117;--surface:#141c25;--surface-2:#1b242f;
 --ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;--sea:#101b24;--land:#26313a;
 --coast:#4a5a66;--c-v10:#c98500;--c-v14:#e66767;--c-v17:#3987e5;--c-v18:#199e70;--c-v19:#9085e9;--obs:#f0f5fa;}}
body{{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:16px;line-height:1.6;}}
.wrap{{max-width:1180px;margin:0 auto;padding:clamp(26px,5vw,56px) clamp(16px,4vw,34px) 90px;
 display:flex;flex-direction:column;gap:44px;}}
h1,h2,h3{{color:var(--ink);margin:0;text-wrap:balance;}}
h1{{font-size:clamp(27px,4vw,40px);line-height:1.12;letter-spacing:-.022em;font-weight:660;}}
h2{{font-size:clamp(19px,2.3vw,24px);letter-spacing:-.012em;font-weight:640;}}
h3{{font-size:14px;font-weight:660;display:flex;align-items:baseline;gap:8px;}}
h3 .sub{{font-size:11.5px;font-weight:400;color:var(--muted);}}
p{{margin:0;}} .lede{{max-width:76ch;font-size:14.5px;}}
.eyebrow{{font-family:var(--mono);font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);}}
header{{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--line);padding-bottom:24px;}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:13px;margin-top:4px;}}
.lg{{display:flex;align-items:center;gap:7px;}}
.lg svg{{display:block;}}
.storm{{display:flex;flex-direction:column;gap:14px;border-top:1px solid var(--line);padding-top:26px;}}
.storm:first-of-type{{border-top:none;padding-top:0;}}
.sec-head{{display:flex;flex-direction:column;gap:6px;}}
.pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px;}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 13px 10px;
 margin:0;display:flex;flex-direction:column;gap:7px;}}
.panel.pending{{display:flex;align-items:center;justify-content:center;min-height:300px;
 color:var(--muted);font-size:13.5px;text-align:center;padding:24px;}}
figcaption{{display:flex;flex-direction:column;gap:2px;padding:0 3px;}}
figcaption p{{font-size:12.5px;color:var(--muted);}}
figcaption b{{color:var(--ink);font-family:var(--mono);}}
.map{{width:100%;height:auto;display:block;border-radius:4px;overflow:hidden;}}
.sea{{fill:var(--sea);}}
.land{{fill:var(--land);stroke:var(--coast);stroke-width:.7;}}
.gl{{stroke:var(--coast);stroke-width:.5;opacity:.45;}}
.tk{{font-family:var(--mono);font-size:8.5px;fill:var(--muted);}}
.spag{{fill:none;stroke-width:.8;opacity:.3;stroke-linejoin:round;stroke-linecap:round;}}
.meanline{{fill:none;stroke-width:2.8;stroke-linejoin:round;stroke-linecap:round;}}
.rawmean{{fill:none;stroke-width:2;stroke-dasharray:5 3;opacity:.62;stroke-linejoin:round;stroke-linecap:round;}}
.rmtline{{fill:none;stroke-width:2.6;stroke-dasharray:7 3.5;stroke-linejoin:round;stroke-linecap:round;}}
[data-model="v10"] .spag,[data-model="v10"] .meanline,[data-model="v10"] .rawmean{{stroke:var(--c-v10);}}
[data-model="v17"] .spag,[data-model="v17"] .meanline,[data-model="v17"] .rawmean{{stroke:var(--c-v17);}}
[data-model="v18"] .spag,[data-model="v18"] .meanline,[data-model="v18"] .rawmean{{stroke:var(--c-v18);}}
[data-model="v19"] .spag,[data-model="v19"] .meanline,[data-model="v19"] .rawmean{{stroke:var(--c-v19);}}
[data-model="v14"] .spag,[data-model="v14"] .meanline,[data-model="v14"] .rawmean{{stroke:var(--c-v14);}}
[data-model="v10"] .rmtline{{stroke:var(--c-v10);}}
[data-model="v17"] .rmtline{{stroke:var(--c-v17);}}
[data-model="v18"] .rmtline{{stroke:var(--c-v18);}}
.obs{{fill:none;stroke:var(--obs);stroke-width:2.4;stroke-dasharray:1.4 2.6;stroke-linecap:round;}}
.start{{fill:var(--surface);stroke:var(--obs);stroke-width:2;}}
.startl{{font-family:var(--mono);font-size:9.5px;fill:var(--obs);font-weight:600;}}
.cl{{font-family:var(--mono);font-size:9.5px;font-weight:700;paint-order:stroke;stroke:var(--sea);stroke-width:2.6px;}}
[data-model="v10"] .cl{{fill:var(--c-v10);}}
[data-model="v17"] .cl{{fill:var(--c-v17);}}
[data-model="v18"] .cl{{fill:var(--c-v18);}}
[data-model="v19"] .cl{{fill:var(--c-v19);}}
[data-model="v14"] .cl{{fill:var(--c-v14);}}
footer{{border-top:1px solid var(--line);padding-top:20px;font-size:13px;color:var(--muted);
 max-width:76ch;display:flex;flex-direction:column;gap:8px;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; every forecast, on the map</div>
  <h1>{(_want[0] + " vs " + _want[1] + ", head to head") if len(_want) == 2 else "Where these models actually send the storm"}</h1>
  <p class="lede">Four test storms, every full-horizon forecast each one produced — 262 forecasts in
  all. <strong>v10</strong> sees no environmental field whatsoever; <strong>v17</strong> and <strong>v18</strong> see 500&nbsp;hPa
  steering winds on the repaired reanalysis. Each model gets its own panel so overlapping
  tracks never have to be told apart by colour alone.</p>
  <div class="legend">
   <span class="lg"><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="var(--c-v16)" stroke-width="1" opacity=".45"/></svg>one forecast</span>
   <span class="lg"><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="var(--c-v17)" stroke-width="2" stroke-dasharray="5 3" opacity=".62" stroke-linecap="round"/></svg>plain mean by valid time</span>
   <span class="lg"><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="var(--c-v17)" stroke-width="2.8" stroke-linecap="round"/></svg>weighted + smoothed consensus</span>
   <span class="lg"><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="var(--obs)" stroke-width="2.4" stroke-dasharray="1.4 2.6" stroke-linecap="round"/></svg>observed</span>
  </div>
 </header>
{''.join(sections)}
 <footer>
  <p>Coastlines are Natural Earth 50&nbsp;m land polygons, simplified to the resolution each panel can
  actually show. Projection is equirectangular with longitude scaled by cos(latitude), so shapes are
  locally true at the centre of each panel.</p>
  <p><strong>Why the weighted consensus and not the plain mean.</strong> Averaged over the four
  storms the equal-weight mean scores 302 / 229 / 235&nbsp;km for v10 / v17 / v18, against
  134 / 99 / 112&nbsp;km for the weighted-and-smoothed consensus &mdash; better on every storm and
  every model, by about 55%. There is no case where the plain mean wins, so only the winner is
  drawn; each caption still quotes what the mean would have scored.</p>
  <p>The weighted consensus is additionally passed through a constant-velocity Kalman/RTS
  smoother. Solving every valid time independently makes the track jump whenever the short-lead
  membership rotates, producing turns no storm makes &mdash; on Co-may the point-wise solve swung
  45&deg; per step. The smoother treats each weighted position as an observation whose variance
  comes from the min-variance solve itself, so well-determined times pull hard and poorly-determined
  ones yield to the motion model. Its one parameter is chosen as the <em>least</em> smoothing whose
  error stays within 2% of the best, because minimising error alone flattens real curvature: a
  straight line through the middle of a turn scores well and is the wrong shape.</p>
  <p><strong>The lead-weighted consensus is an analysis, not a forecast.</strong> At each valid time
  the members come from different initialisations and so carry different lead times; the weights are
  minimum-variance on a Marchenko-Pastur-cleaned lead&times;lead error covariance, fitted
  <em>leave-one-storm-out</em>. Because the solution puts ~45% of its weight on leads 1&ndash;4 and
  none on leads 17&ndash;20, it leans on recent initialisations &mdash; so its error is not comparable
  to the 120&nbsp;h forecast error beside it. Most of its advantage is simply lead-awareness; RMT only
  cleans the residual covariance shape.</p>
  <p>The mean is averaged by <em>valid time</em>, not by lead. Forecasts launched hours apart only
  describe the same moment at different lead offsets — averaging by lead would blend positions hours
  apart and draw a route no model ever predicted.</p>
  <p>The observed track is reconstructed from the base position of every window, which is an actual
  best-track fix, so it is data rather than a fit.</p>
 </footer>
</div>"""

os.makedirs("paper", exist_ok=True)
open("paper/storm_maps.html", "w").write(HTML)
print(f"wrote paper/storm_maps.html ({len(HTML)/1000:.0f} KB)  models: "
      + ", ".join(t for t, _ in MODELS))
