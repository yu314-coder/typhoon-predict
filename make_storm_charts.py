"""Per-storm forecast-vs-observed page: track on a map, plus the five quantities that describe
how a storm behaves. Reads track_build/storm_forecasts.json, writes paper/storm_forecasts.html.
"""
import json, math, os

D = json.load(open("track_build/storm_forecasts.json"))
LEADS = [6 * (i + 1) for i in range(20)]
SERIES = [("observed", "observed", "obs"), ("v10", "v10", "v10"), ("v14", "v14", "v14")]
# validated: node scripts/validate_palette.js "#2a78d6,#e34948" (light) / "#3987e5,#e66767" (dark)
COL = {"observed": ("#111820", "#e8eef4"), "v10": ("#2a78d6", "#3987e5"), "v14": ("#e34948", "#e66767")}

PANELS = [("vmax", "Peak wind", "kt"), ("pressure", "Central pressure", "hPa"),
          ("rmw", "Radius of max wind", "km"), ("speed", "Forward speed", "km/h"),
          ("bearing", "Heading", "deg")]


def unwrap(v):
    """Bearings are circular; unwrap so a track crossing north doesn't draw a 360-degree cliff."""
    o, prev = [], None
    for x in v:
        if x is None:
            o.append(None); continue
        if prev is not None:
            while x - prev > 180: x -= 360
            while x - prev < -180: x += 360
        o.append(x); prev = x
    return o


def axis(lo, hi, n=4):
    return [lo + (hi - lo) * t / n for t in range(n + 1)]


def linechart(rec, key, unit):
    W, H = 430, 210
    ml, mr, mt, mb = 50, 66, 12, 32
    pw, ph = W - ml - mr, H - mt - mb
    data = {}
    for sk, _, _ in SERIES:
        v = rec[sk][key] if sk != "observed" else rec["observed"][key]
        data[sk] = unwrap(v) if key == "bearing" else v
    vals = [x for s in data.values() for x in s if x is not None]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * .14 or 1
    lo, hi = lo - pad, hi + pad
    def X(i): return ml + pw * i / 19
    def Y(v): return mt + ph * (1 - (v - lo) / (hi - lo))
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{key}">']
    for v in axis(lo, hi):
        y = Y(v)
        o.append(f'<line class="gl" x1="{ml}" x2="{ml+pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="tk" x="{ml-7}" y="{y+3.4:.1f}" text-anchor="end">{v:.0f}</text>')
    for h in (24, 72, 120):
        o.append(f'<text class="tk" x="{X(h//6-1):.1f}" y="{H-12}" text-anchor="middle">{h}h</text>')
    o.append(f'<text class="ax" x="{ml-38}" y="{mt+ph/2:.0f}" transform="rotate(-90 {ml-38} {mt+ph/2:.0f})" text-anchor="middle">{unit}</text>')
    ends = []
    for sk, lab, _ in SERIES:
        pts = [(X(i), Y(v)) for i, v in enumerate(data[sk]) if v is not None]
        if not pts: continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="1 3.2" stroke-width="2.4"' if sk == "observed" else ""
        o.append(f'<path class="ln" d="{d}"{dash} style="stroke:var(--c-{sk})"/>')
        ends.append([pts[-1][1], sk, pts[-1][0], lab])
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 11.5:
            ends[i][0] = ends[i - 1][0] + 11.5
    for ly, sk, lx, lab in ends:
        o.append(f'<text class="dl" x="{lx+8:.1f}" y="{ly+3.4:.1f}" style="fill:var(--c-{sk})">{lab}</text>')
    for i in range(20):
        rows = "".join(f"<span class='k'><i style='background:var(--c-{sk})'></i>{lab}</span>"
                       f"<span class='v'>{'—' if data[sk][i] is None else format(data[sk][i], '.1f')}</span>"
                       for sk, lab, _ in SERIES)
        o.append(f'<rect class="hit" x="{X(i)-pw/38:.1f}" y="{mt}" width="{pw/19:.1f}" height="{ph}" '
                 f'data-h="{LEADS[i]}" data-rows="{rows}"/>')
    o.append('</svg>')
    return "\n".join(o)


def mapchart(rec):
    W, H = 430, 330
    m = 30
    lats = [rec["base"]["lat"]] + [x for sk, _, _ in SERIES for x in rec[sk]["lat"]]
    lons = [rec["base"]["lon"]] + [x for sk, _, _ in SERIES for x in rec[sk]["lon"]]
    la0, la1, lo0, lo1 = min(lats), max(lats), min(lons), max(lons)
    pla, plo = (la1 - la0) * .16 + .6, (lo1 - lo0) * .16 + .6
    la0, la1, lo0, lo1 = la0 - pla, la1 + pla, lo0 - plo, lo1 + plo
    kx = math.cos(math.radians((la0 + la1) / 2))          # equirectangular: shrink lon by cos(lat)
    spanx, spany = (lo1 - lo0) * kx, (la1 - la0)
    sc = min((W - 2 * m) / spanx, (H - 2 * m) / spany)
    ox, oy = (W - spanx * sc) / 2, (H - spany * sc) / 2
    def PX(lon): return ox + (lon - lo0) * kx * sc
    def PY(lat): return H - oy - (lat - la0) * sc
    o = [f'<svg viewBox="0 0 {W} {H}" class="map" role="img" aria-label="forecast tracks">']
    step = 5 if (la1 - la0) > 12 else 2
    g0 = math.floor(la0 / step) * step
    while g0 <= la1:
        if g0 >= la0:
            y = PY(g0)
            o.append(f'<line class="gl" x1="{m*.4:.0f}" x2="{W-m*.4:.0f}" y1="{y:.1f}" y2="{y:.1f}"/>')
            o.append(f'<text class="tk" x="{m*.4+2:.0f}" y="{y-3:.1f}">{abs(g0):.0f}°{"N" if g0>=0 else "S"}</text>')
        g0 += step
    g1 = math.floor(lo0 / step) * step
    while g1 <= lo1:
        if g1 >= lo0:
            x = PX(g1)
            o.append(f'<line class="gl" x1="{x:.1f}" x2="{x:.1f}" y1="{m*.4:.0f}" y2="{H-m*.4:.0f}"/>')
            o.append(f'<text class="tk" x="{x+3:.1f}" y="{H-m*.4-3:.0f}">{g1:.0f}°E</text>')
        g1 += step
    for sk, lab, _ in SERIES:
        pts = [(PX(lo), PY(la)) for la, lo in zip(rec[sk]["lat"], rec[sk]["lon"])]
        pts = [(PX(rec["base"]["lon"]), PY(rec["base"]["lat"]))] + pts
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="1 3.6" stroke-width="3"' if sk == "observed" else ""
        o.append(f'<path class="ln" d="{d}"{dash} style="stroke:var(--c-{sk})"/>')
        for i, (x, y) in enumerate(pts[1:]):
            if (i + 1) % 4 == 0:
                o.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="3" style="fill:var(--c-{sk})"/>')
        x, y = pts[-1]
        o.append(f'<text class="dl" x="{x+7:.1f}" y="{y+3.4:.1f}" style="fill:var(--c-{sk})">{lab}</text>')
    bx, by = PX(rec["base"]["lon"]), PY(rec["base"]["lat"])
    o.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4.5" class="orig"/>')
    o.append(f'<text class="dl orig-l" x="{bx-7:.1f}" y="{by-8:.1f}" text-anchor="end">start</text>')
    o.append('</svg>')
    return "\n".join(o)


sections = []
for nm in ["Bavi", "Wayne", "Co-may", "Hinnamnor"]:
    if nm not in D: continue
    r = D[nm]
    e10, e14 = r["v10"]["err120"], r["v14"]["err120"]
    win = "v10" if e10 < e14 else "v14"
    panels = "".join(f'<figure class="panel"><figcaption><h3>{t}</h3></figcaption>{linechart(r, k, u)}</figure>'
                     for k, t, u in PANELS)
    sections.append(f"""
  <section class="storm">
    <div class="sec-head">
      <div class="eyebrow">{r['year']} &middot; {r['sid']}</div>
      <h2>{r['name']}</h2>
      <p class="lede">Initialised {r['base']['time']} UTC at {r['base']['lat']:.1f}&deg;N
      {r['base']['lon']:.1f}&deg;E — the earliest window with a full 5-day horizon, one of
      {r['n_windows']} for this storm. Dotted line is observed.</p>
      <div class="chips">
        <span class="chip"><i style="background:var(--c-v10)"></i>v10 &nbsp;<b>{e10:.0f} km</b> at 120 h</span>
        <span class="chip"><i style="background:var(--c-v14)"></i>v14 &nbsp;<b>{e14:.0f} km</b> at 120 h</span>
        <span class="chip win">{win} closer here</span>
      </div>
    </div>
    <div class="storm-grid">
      <figure class="panel map-panel"><figcaption><h3>Track</h3><p>Forecast vs observed position.
        Dots mark every 24 h.</p></figcaption>{mapchart(r)}</figure>
      <div class="mini">{panels}</div>
    </div>
  </section>""")

HTML = f"""<title>TrackFormer — four storms, forecast vs observed</title>
<style>
:root{{color-scheme:light;--bg:#f2f4f6;--surface:#fcfcfb;--surface-2:#e9edf1;--ink:#111820;--body:#2c3a47;
 --muted:#5d6c7a;--line:#d5dce3;--c-observed:#111820;--c-v10:#2a78d6;--c-v14:#e34948;
 --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--surface-2:#1b242f;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;
 --line:#26313d;--c-observed:#e8eef4;--c-v10:#3987e5;--c-v14:#e66767;}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0c1117;--surface:#141c25;--surface-2:#1b242f;
 --ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;
 --c-observed:#e8eef4;--c-v10:#3987e5;--c-v14:#e66767;}}
body{{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:16px;line-height:1.6;}}
.wrap{{max-width:1180px;margin:0 auto;padding:clamp(26px,5vw,58px) clamp(16px,4vw,34px) 90px;
 display:flex;flex-direction:column;gap:46px;}}
h1,h2,h3{{color:var(--ink);margin:0;text-wrap:balance;}}
h1{{font-size:clamp(27px,4vw,40px);line-height:1.12;letter-spacing:-.022em;font-weight:660;}}
h2{{font-size:clamp(19px,2.3vw,24px);letter-spacing:-.012em;font-weight:640;}}
h3{{font-size:13.5px;font-weight:640;}}
p{{margin:0;}} .lede{{max-width:74ch;font-size:14.5px;}}
.eyebrow{{font-family:var(--mono);font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);}}
header{{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--line);padding-bottom:24px;}}
.storm{{display:flex;flex-direction:column;gap:16px;border-top:1px solid var(--line);padding-top:26px;}}
.storm:first-of-type{{border-top:none;padding-top:0;}}
.sec-head{{display:flex;flex-direction:column;gap:7px;}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin-top:3px;}}
.chip{{display:flex;align-items:center;gap:6px;background:var(--surface-2);border-radius:20px;
 padding:3px 11px;font-size:12.5px;color:var(--body);}}
.chip b{{color:var(--ink);font-family:var(--mono);}}
.chip i{{width:9px;height:9px;border-radius:50%;display:inline-block;}}
.chip.win{{background:transparent;border:1px dashed var(--line);color:var(--muted);}}
.storm-grid{{display:grid;grid-template-columns:minmax(330px,430px) 1fr;gap:14px;align-items:start;}}
.mini{{display:grid;grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:12px;}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:5px;padding:13px 12px 6px;
 margin:0;display:flex;flex-direction:column;gap:4px;position:relative;}}
figcaption{{padding:0 3px;display:flex;flex-direction:column;gap:2px;}}
figcaption p{{font-size:11.5px;color:var(--muted);line-height:1.4;}}
.chart,.map{{width:100%;height:auto;overflow:visible;stroke:none;}}
.gl{{stroke:var(--line);stroke-width:1;}}
.tk{{font-family:var(--mono);font-size:9px;fill:var(--muted);}}
.ax{{font-size:9.5px;fill:var(--muted);}}
.ln{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round;}}
.pt{{stroke:var(--surface);stroke-width:1.6;}}
.dl{{font-family:var(--mono);font-size:10px;font-weight:600;}}
.orig{{fill:var(--surface);stroke:var(--muted);stroke-width:1.8;}}
.orig-l{{fill:var(--muted);font-weight:500;}}
.hit{{fill:transparent;stroke:none;}}
.tip{{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);
 border-radius:4px;padding:6px 9px;font-size:11px;box-shadow:0 4px 14px rgba(0,0,0,.14);opacity:0;
 transition:opacity .1s;min-width:126px;z-index:6;}}
.tip b{{color:var(--ink);font-family:var(--mono);font-size:10.5px;display:block;margin-bottom:3px;}}
.tip .row{{display:flex;justify-content:space-between;gap:12px;align-items:center;}}
.tip .k{{display:flex;align-items:center;gap:5px;}}
.tip .k i{{width:8px;height:8px;border-radius:2px;display:inline-block;}}
.tip .v{{font-family:var(--mono);color:var(--ink);font-variant-numeric:tabular-nums;}}
footer{{border-top:1px solid var(--line);padding-top:20px;font-size:13px;color:var(--muted);max-width:74ch;
 display:flex;flex-direction:column;gap:8px;}}
@media (max-width:900px){{.storm-grid{{grid-template-columns:1fr;}}}}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important;}}}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; case studies</div>
  <h1>Four storms, forecast against what actually happened</h1>
  <p class="lede">The same four storms the project has always tested on. Each panel is a single
  5-day forecast launched from the earliest full-horizon window, so nothing is averaged away —
  this is one forecast, start to finish, next to the truth. <strong>v10</strong> uses no
  environmental field at all; <strong>v14</strong> adds 500&nbsp;hPa steering winds. Both are shown
  on the repaired reanalysis.</p>
 </header>
{''.join(sections)}
 <footer>
  <p>Heading is unwrapped so a track turning through north reads as a smooth curve rather than a
  360&deg; cliff. Forward speed and heading are derived from the per-step displacements the model
  actually predicts, not fitted afterwards.</p>
  <p>Single-window errors here differ from the per-storm means quoted elsewhere, which average over
  every initialisation for that storm — 49 windows for Bavi, 121 for Wayne.</p>
 </footer>
</div>
<script>
document.querySelectorAll('.panel').forEach(function(p){{
  var svg=p.querySelector('svg'); if(!svg) return;
  var hits=svg.querySelectorAll('.hit'); if(!hits.length) return;
  var tip=document.createElement('div'); tip.className='tip'; p.appendChild(tip);
  hits.forEach(function(h){{
    h.addEventListener('mouseenter',function(){{
      var rows=h.getAttribute('data-rows');
      tip.innerHTML='<b>+'+h.getAttribute('data-h')+' h</b>'+
        rows.replace(/<span class="k">/g,'<div class="row"><span class="k">')
            .replace(/<\\/span><span class="v">/g,'</span><span class="v">')
            .replace(/<\\/span>(?=<span class="k">|$)/g,'</span></div>');
      tip.style.opacity=1;
      var r=h.getBoundingClientRect(), pr=p.getBoundingClientRect();
      var x=r.left-pr.left+r.width/2, left=x+12;
      if(left+140>pr.width) left=x-140;
      tip.style.left=Math.max(4,left)+'px'; tip.style.top='26px';
    }});
  }});
  svg.addEventListener('mouseleave',function(){{tip.style.opacity=0;}});
}});
</script>"""

os.makedirs("paper", exist_ok=True)
open("paper/storm_forecasts.html", "w").write(HTML)
print(f"wrote paper/storm_forecasts.html ({len(HTML)/1000:.0f} KB)")
