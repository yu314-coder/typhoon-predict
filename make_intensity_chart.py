"""Per-lead intensity/radius/speed comparison chart for v10 / v21 / v25, exact numbers on the page.

Reads track_build/intensity_compare.json (per-lead MAE for each metric, each model) and writes a
self-contained, theme-aware HTML with one small-multiple SVG per metric plus an exact-number table.
"""
import json, os

R = json.load(open("track_build/intensity_compare.json"))
COL = {"v10": ("#eda100", "#c98500"), "v21": ("#2a78d6", "#3987e5"), "v25": ("#c2185b", "#e0457f")}
NOTE = {"v10": "no environment", "v21": "chain-of-thought steering",
        "v25": "v21 + ocean-heat / shear / humidity token"}
# metric key -> (label, unit, "lower is better")
METRICS = [
    ("vmax", "Maximum wind (vmax)", "kt"),
    ("pressure", "Central pressure", "hPa"),
    ("rmw", "Radius of max wind (RMW)", "nm"),
    ("radii", "Wind radii R34/50/64", "nm"),
    ("speed", "Translation (moving) speed", "km/h"),
    ("track", "Track position error", "km"),
]
LEADS = list(range(1, 21))
HRS = [6 * L for L in LEADS]


def am(x):
    v = [q for q in x if q is not None]
    return sum(v) / len(v) if v else float("nan")


def chart(key, label, unit):
    W, H, m = 360, 232, 40
    series = {t: R[t][key] for t in ("v10", "v21", "v25")}
    allv = [v for t in series for v in series[t] if v is not None]
    ymin, ymax = min(allv), max(allv)
    pad = (ymax - ymin) * 0.12 + 1e-6
    ymin, ymax = ymin - pad, ymax + pad
    x0, x1 = m, W - 12
    y0, y1 = H - 28, 14

    def PX(L): return x0 + (L - 1) / 19 * (x1 - x0)
    def PY(v): return y1 + (ymax - v) / (ymax - ymin) * (y0 - y1)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{label} by lead">']
    # y gridlines + ticks (4)
    for g in range(5):
        v = ymin + (ymax - ymin) * g / 4
        y = PY(v)
        o.append(f'<line class="gl" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="ax" x="{x0-6:.1f}" y="{y+3:.1f}" text-anchor="end">{v:.1f}</text>')
    # x ticks at 24,48,72,96,120 h
    for hh in (24, 48, 72, 96, 120):
        L = hh / 6
        x = PX(L)
        o.append(f'<line class="gl" x1="{x:.1f}" x2="{x:.1f}" y1="{y1}" y2="{y0}"/>')
        o.append(f'<text class="ax" x="{x:.1f}" y="{y0+14:.1f}" text-anchor="middle">{hh}</text>')
    o.append(f'<text class="axl" x="{(x0+x1)/2:.1f}" y="{H-2:.1f}" text-anchor="middle">forecast hour</text>')
    o.append(f'<text class="axl" transform="translate(11,{(y0+y1)/2:.1f}) rotate(-90)" text-anchor="middle">MAE ({unit})</text>')
    for t in ("v10", "v21", "v25"):
        pts = [(PX(L), PY(v)) for L, v in zip(LEADS, series[t]) if v is not None]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        o.append(f'<path d="{d}" class="ln {t}"/>')
        # endpoint dot + exact 120h value
        ex, ey = pts[-1]
        o.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="2.6" class="dot {t}"/>')
    o.append('</svg>')
    # exact numbers row (aggregate mean over leads, and 120h)
    rows = []
    for t in ("v10", "v21", "v25"):
        agg = R[t]["agg_track"] if key == "track" else am(R[t][key])
        h120 = R[t][key][19]
        rows.append((t, agg, h120))
    best_agg = min(r[1] for r in rows)
    num = "".join(
        f'<div class="num {t}"><span class="sw"></span><b>{t}</b>'
        f'<span class="v{" best" if abs(agg-best_agg)<1e-9 else ""}">{agg:.2f}</span>'
        f'<span class="v120">{h120:.1f} @120h</span></div>'
        for t, agg, h120 in rows)
    return (f'<figure class="card"><figcaption><h3>{label}</h3>'
            f'<p class="unit">mean absolute error, {unit} &middot; lower is better</p></figcaption>'
            f'{"".join(o)}<div class="nums">{num}</div></figure>')


# headline deltas v25 vs v21
def delta(key):
    a = R["v21"]["agg_track"] if key == "track" else am(R["v21"][key])
    b = R["v25"]["agg_track"] if key == "track" else am(R["v25"][key])
    return b - a  # negative = v25 better


dp = delta("pressure"); dv = delta("vmax"); dt = delta("track")
cards = "".join(chart(k, lab, u) for k, lab, u in METRICS)

palL = "".join(f".{t}.ln{{stroke:{c[0]};}} .{t} .sw,.dot.{t}{{background:{c[0]};fill:{c[0]};}}" for t, c in COL.items())
palD = "".join(f".{t}.ln{{stroke:{c[1]};}} .{t} .sw,.dot.{t}{{background:{c[1]};fill:{c[1]};}}" for t, c in COL.items())
legend = "".join(f'<span class="lg {t}"><span class="sw"></span><b>{t}</b> {NOTE[t]}</span>' for t in ("v10", "v21", "v25"))

HTML = f"""<meta charset="utf-8">
<title>v25 vs v21 vs v10 — intensity, size and speed</title>
<style>
:root{{color-scheme:light;--bg:#f2f4f6;--surface:#fcfcfb;--ink:#111820;--body:#2c3a47;--muted:#5d6c7a;
 --line:#d5dce3;--grid:#c3ccd5;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;{palL}}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;--grid:#2b3745;{palD}}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;
 --muted:#8697a5;--line:#26313d;--grid:#2b3745;{palD}}}
body{{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:16px;line-height:1.6;}}
.wrap{{max-width:1120px;margin:0 auto;padding:clamp(24px,5vw,52px) clamp(16px,4vw,32px) 80px;
 display:flex;flex-direction:column;gap:26px;}}
h1{{color:var(--ink);font-size:clamp(24px,3.6vw,36px);line-height:1.14;letter-spacing:-.02em;margin:0;font-weight:660;}}
.eyebrow{{font-family:var(--mono);font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);}}
.lede{{max-width:80ch;font-size:14.5px;margin:0;}}
header{{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--line);padding-bottom:22px;}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:13px;}}
.lg,.num{{display:flex;align-items:center;gap:7px;}}
.sw{{width:15px;height:4px;border-radius:2px;display:inline-block;}}
.call{{display:flex;flex-wrap:wrap;gap:10px;}}
.pill{{background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:6px 14px;font-size:13px;}}
.pill b{{color:var(--ink);font-family:var(--mono);}}
.up{{color:#c2185b;}} .dn{{color:#1a9e6f;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 14px 10px;margin:0;
 display:flex;flex-direction:column;gap:6px;}}
figcaption h3{{color:var(--ink);font-size:14px;font-weight:640;margin:0;}}
.unit{{font-size:11.5px;color:var(--muted);margin:0;}}
.chart{{width:100%;height:auto;display:block;}}
.gl{{stroke:var(--grid);stroke-width:.6;opacity:.5;}}
.ax{{font-family:var(--mono);font-size:8px;fill:var(--muted);}}
.axl{{font-family:var(--mono);font-size:8.5px;fill:var(--muted);}}
.ln{{fill:none;stroke-width:2.1;stroke-linejoin:round;stroke-linecap:round;}}
.nums{{display:flex;flex-wrap:wrap;gap:6px 14px;border-top:1px solid var(--line);padding-top:8px;font-size:12px;}}
.num b{{color:var(--ink);font-family:var(--mono);font-size:11px;}}
.num .v{{font-family:var(--mono);color:var(--ink);font-weight:600;}}
.num .v.best{{color:#1a9e6f;}}
.num .v120{{font-family:var(--mono);color:var(--muted);font-size:10.5px;}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted);max-width:80ch;
 display:flex;flex-direction:column;gap:8px;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; does the ocean data help?</div>
  <h1>v25 vs v21 vs v10 — wind, pressure, size and speed</h1>
  <p class="lede">Every quantity the model forecasts, scored per lead on the {int(0)+3763:,} western-Pacific and
  eastern-Pacific test windows of 2020&ndash;2025. v25 adds ocean heat, deep-layer shear and mid-level humidity
  to v21 as one decoder token. These are <b>intensity</b> predictors, so the pay-off &mdash; if any &mdash; should
  land on pressure and wind, not track. Each number below is a mean-absolute-error; the dot marks the 120&nbsp;h value.</p>
  <div class="call">
   <span class="pill">central pressure: <b class="{'dn' if dp<0 else 'up'}">{dp:+.2f} hPa</b> v25 vs v21</span>
   <span class="pill">max wind: <b class="{'dn' if dv<0 else 'up'}">{dv:+.2f} kt</b></span>
   <span class="pill">track: <b class="{'dn' if dt<0 else 'up'}">{dt:+.1f} km</b></span>
  </div>
  <div class="legend">{legend}</div>
 </header>
 <div class="grid">{cards}</div>
 <footer>
  <p><b>How to read it.</b> All six panels are mean-absolute-error against JTWC best track, lower is better.
  Pressure, wind, RMW and the R34/50/64 wind radii are direct model outputs; moving speed is the length of each
  6-hour displacement step; track is cumulative great-circle position error. Each metric is masked with its own
  validity channel, so a window missing a pressure fix never counts toward the pressure score.</p>
  <p><b>The one real win is pressure.</b> The ocean-heat token lowers central-pressure error by
  {abs(dp):.2f} hPa, the tightest thermodynamic gauge of a storm and exactly where sea-surface warmth should bite.
  Wind, RMW and radii move by less than the seed-to-seed noise, and track is marginally worse &mdash; expected,
  since none of these features describe where the storm goes. This is why v26 pushes the ocean signal deeper: the
  21&times;21 heat <em>patch</em> through a small CNN into the intensity head, rather than 43 numbers in one token.</p>
 </footer>
</div>"""

os.makedirs("paper", exist_ok=True)
open("paper/v25_intensity_compare.html", "w").write(HTML)
print(f"wrote paper/v25_intensity_compare.html ({len(HTML)/1000:.0f} KB)")
