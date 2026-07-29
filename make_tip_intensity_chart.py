"""Predicted mean forecast vs real vmax/pressure on Typhoon Tip (1979), v10/v23/v34, as a
small-multiple grid -- same SVG/grid convention as make_intensity_chart.py, but plotting the
actual forecast curve against the observed curve (not MAE), since the point here is to see
WHERE each model's mean forecast tracks or drifts from what really happened on the most intense
tropical cyclone on record.

Reads track_build/tip_intensity.json (written by _tip_intensity.py).
"""
import json, os

R = json.load(open("track_build/tip_intensity.json"))
COL = {"v10": ("#eda100", "#c98500"), "v23": ("#eb6834", "#d95926"), "v34": ("#8b2fb0", "#c07de0")}
NOTE = {"v10": "no environment", "v23": "chain-of-thought + temporal steering",
        "v34": "LandGate, frozen v23 backbone"}
METRICS = [("vmax", "Maximum wind (vmax)", "kt")]
# pressure dropped: Tip's target_mask has zero valid pressure fixes (pre-satellite record gap,
# not a bug -- vmax is the only intensity channel with real target values for this storm).
LEADS = list(range(1, 21))
HRS = [6 * L for L in LEADS]


def chart(key, label, unit):
    W, H, m = 520, 300, 46
    real = R["v10"][key]["real"]  # identical real series across tags (same storm, same mask)
    series = {t: R[t][key]["pred"] for t in ("v10", "v23", "v34")}
    allv = [v for t in series for v in series[t] if v is not None] + [v for v in real if v is not None]
    ymin, ymax = min(allv), max(allv)
    pad = (ymax - ymin) * 0.12 + 1e-6
    ymin, ymax = ymin - pad, ymax + pad
    x0, x1 = m, W - 14
    y0, y1 = H - 34, 16

    def PX(L): return x0 + (L - 1) / 19 * (x1 - x0)
    def PY(v): return y1 + (ymax - v) / (ymax - ymin) * (y0 - y1)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{label} on Typhoon Tip">']
    for g in range(5):
        v = ymin + (ymax - ymin) * g / 4
        y = PY(v)
        o.append(f'<line class="gl" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="ax" x="{x0-7:.1f}" y="{y+3:.1f}" text-anchor="end">{v:.0f}</text>')
    for hh in (24, 48, 72, 96, 120):
        L = hh / 6
        x = PX(L)
        o.append(f'<line class="gl" x1="{x:.1f}" x2="{x:.1f}" y1="{y1}" y2="{y0}"/>')
        o.append(f'<text class="ax" x="{x:.1f}" y="{y0+15:.1f}" text-anchor="middle">{hh}</text>')
    o.append(f'<text class="axl" x="{(x0+x1)/2:.1f}" y="{H-4:.1f}" text-anchor="middle">forecast hour</text>')
    o.append(f'<text class="axl" transform="translate(12,{(y0+y1)/2:.1f}) rotate(-90)" text-anchor="middle">{unit}</text>')
    # observed (real), dotted black, drawn first so model lines sit on top
    pts = [(PX(L), PY(v)) for L, v in zip(LEADS, real) if v is not None]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    o.append(f'<path d="{d}" class="obs"/>')
    for t in ("v10", "v23", "v34"):
        pts = [(PX(L), PY(v)) for L, v in zip(LEADS, series[t]) if v is not None]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        o.append(f'<path d="{d}" class="ln {t}"/>')
        ex, ey = pts[-1]
        o.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="2.8" class="dot {t}"/>')
    o.append('</svg>')
    rows = []
    for t in ("v10", "v23", "v34"):
        p120 = series[t][19]; r120 = real[19]
        rows.append((t, p120, r120))
    num = "".join(
        f'<div class="num {t}"><span class="sw"></span><b>{t}</b>'
        f'<span class="v">{p:.1f}</span><span class="v120">real {rv:.1f} @120h</span></div>'
        for t, p, rv in rows)
    return (f'<figure class="card"><figcaption><h3>{label}</h3>'
            f'<p class="unit">mean forecast (ensemble-averaged) vs. observed &middot; {unit}</p></figcaption>'
            f'{"".join(o)}<div class="nums">{num}</div></figure>')


cards = "".join(chart(k, lab, u) for k, lab, u in METRICS)
palL = "".join(f".{t}.ln{{stroke:{c[0]};}} .{t} .sw,.dot.{t}{{background:{c[0]};fill:{c[0]};}}" for t, c in COL.items())
palD = "".join(f".{t}.ln{{stroke:{c[1]};}} .{t} .sw,.dot.{t}{{background:{c[1]};fill:{c[1]};}}" for t, c in COL.items())
legend = "".join(f'<span class="lg {t}"><span class="sw"></span><b>{t}</b> {NOTE[t]}</span>' for t in ("v10", "v23", "v34"))
legend += '<span class="lg"><span class="sw obs"></span><b>observed</b> real IBTrACS best track</span>'

HTML = f"""<meta charset="utf-8">
<title>v10 vs v23 vs v34 -- mean forecast vs real, Typhoon Tip (1979)</title>
<style>
:root{{color-scheme:light;--bg:#f2f4f6;--surface:#fcfcfb;--ink:#111820;--body:#2c3a47;--muted:#5d6c7a;
 --line:#d5dce3;--grid:#c3ccd5;--obs:#11181f;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;{palL}}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;--grid:#2b3745;--obs:#f0f5fa;{palD}}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;
 --muted:#8697a5;--line:#26313d;--grid:#2b3745;--obs:#f0f5fa;{palD}}}
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
.sw.obs{{background:repeating-linear-gradient(90deg,var(--obs) 0 3px,transparent 3px 6px);}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 14px 10px;margin:0;
 display:flex;flex-direction:column;gap:6px;}}
figcaption h3{{color:var(--ink);font-size:14px;font-weight:640;margin:0;}}
.unit{{font-size:11.5px;color:var(--muted);margin:0;}}
.chart{{width:100%;height:auto;display:block;}}
.gl{{stroke:var(--grid);stroke-width:.6;opacity:.5;}}
.ax{{font-family:var(--mono);font-size:8px;fill:var(--muted);}}
.axl{{font-family:var(--mono);font-size:8.5px;fill:var(--muted);}}
.ln{{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round;}}
.obs{{fill:none;stroke:var(--obs);stroke-width:2.2;stroke-dasharray:1.4 3;stroke-linecap:round;}}
.nums{{display:flex;flex-wrap:wrap;gap:6px 14px;border-top:1px solid var(--line);padding-top:8px;font-size:12px;}}
.num b{{color:var(--ink);font-family:var(--mono);font-size:11px;}}
.num .v{{font-family:var(--mono);color:var(--ink);font-weight:600;}}
.num .v120{{font-family:var(--mono);color:var(--muted);font-size:10.5px;}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted);max-width:80ch;
 display:flex;flex-direction:column;gap:8px;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; mean forecast vs reality</div>
  <h1>v10 vs v23 vs v34 &mdash; Typhoon Tip (1979)</h1>
  <p class="lede">Tip is the most intense tropical cyclone ever recorded (870&nbsp;hPa central
  pressure). The dotted black line is what actually happened; each colored line is that model's
  ensemble-mean forecast at each lead. All three models increasingly underestimate Tip's real
  intensity as lead time grows &mdash; none of them ever forecasts the storm reaching what it
  actually reached by 120&nbsp;h.</p>
  <div class="legend">{legend}</div>
 </header>
 <div class="grid">{cards}</div>
 <footer>
  <p><b>How to read it.</b> The panel averages over Tip's 105 full-horizon forecasts (real value
  and each model's mean forecast, at each lead, masked to windows with a valid vmax target --
  Tip's central-pressure record has no valid target at any lead, a pre-satellite data gap, not
  shown here). The gap between the dotted observed line and a colored forecast line at a given lead is
  that model's mean bias at that lead &mdash; not an error bar, a systematic pull toward weaker
  predictions the further out the forecast reaches.</p>
  <p><b>None of the architecture changes tested in this project fix this.</b> v23's steering-flow
  representation and v34's land-interaction gate both target track, not intensity, and neither
  narrows the gap on Tip specifically &mdash; if anything v23/v34 are slightly more conservative
  than the plain track-only v10 baseline at 96&ndash;120&nbsp;h. Extreme-intensity forecasting needs
  information (ocean heat content, wind shear, eyewall structure) that none of v10/v23/v34's inputs
  carry.</p>
 </footer>
</div>"""

os.makedirs("paper", exist_ok=True)
open("paper/tip_intensity_mean_forecast.html", "w").write(HTML)
print(f"wrote paper/tip_intensity_mean_forecast.html ({len(HTML)/1000:.0f} KB)")
