"""Predicted vs real central pressure (mbar/hPa) for Typhoon Dolphin: real observed pressure
(solid black, through the latest confirmed observation) vs v23's and v35's forecast pressure
(issued from that same latest observation, run on real fetched GFS steering -- see
_dolphin_v23_v35.py), plotted against calendar time so the forecast visibly continues from where
the real line stops. vmax is included alongside for the same reason the Tip chart did: pressure and
wind together are the fuller "how strong is/will this storm be" picture.

Reads track_build/dolphin_v23_v35.json.
"""
import json, os
import numpy as np

R = 111.2
COL = {"real": ("#222b33", "#e8eef4"), "v10": ("#7a8592", "#9fb0bd"),
       "v23": ("#2a78d6", "#3987e5"), "v35": ("#8b2fb0", "#c07de0"), "jtwc": ("#c2410c", "#f0813f")}
NOTE = {"v10": "no environment (track-only)", "v23": "CoT + temporal steering",
        "v35": "v23 + intensity-reweighted loss"}

DOLPHIN_TIMES = ["2026-07-27T00:00", "2026-07-27T06:00", "2026-07-27T12:00", "2026-07-27T18:00",
                 "2026-07-28T00:00", "2026-07-28T06:00", "2026-07-28T12:00", "2026-07-28T18:00",
                 "2026-07-29T00:00", "2026-07-29T06:00"]
DOLPHIN_RAW = [  # (wind_mph, pressure_hPa)
    (35, 1002), (40, 1000), (50, 998), (50, 992), (70, 990),
    (85, 980), (115, 975), (145, 937), (140, 941), (140, 941),
]
KT_PER_MPH = 0.868976
real_t = [np.datetime64(t) for t in DOLPHIN_TIMES]
real_vmax = [w * KT_PER_MPH for w, p in DOLPHIN_RAW]
real_pres = [float(p) for w, p in DOLPHIN_RAW]

_dv = json.load(open("track_build/dolphin_v23_v35.json"))
LEAD_H = list(range(6, 121, 6))


def model_series(tag):
    latest = _dv[tag]["issues"][-1]
    t0 = np.datetime64(latest["issue_time"])
    ts = [t0 + np.timedelta64(int(h), "h") for h in LEAD_H]
    return ts, latest["vmax"], latest["pressure"]


# JTWC official forecast (Advisory #3, issued 2026-07-27 12:00 UTC) -- wind only, JTWC does not
# publish forecast pressure at each TAU (confirmed by checking the advisory text directly; it only
# gives pressure for the CURRENT position, not future ones).
JTWC_TAU_H = [0, 12, 24, 48, 72, 96, 120]
JTWC_ISSUE = np.datetime64("2026-07-27T12:00")
JTWC_KT = [45, 55, 70, 100, 120, 140, 150]
JTWC_T = [JTWC_ISSUE + np.timedelta64(int(h), "h") for h in JTWC_TAU_H]


T0 = real_t[0]


def hours_since(t):
    return float((t - T0) / np.timedelta64(1, "h"))


def chart(metric_key, label, unit, real_vals, model_vals, invert=False, raw_series=None):
    """model_vals: {tag: (ts, vals)} anchored to the real/"now" point.
    raw_series: {tag: (ts, vals)} drawn independently, own issue time (e.g. JTWC)."""
    raw_series = raw_series or {}
    W, H, m = 560, 300, 48
    all_t = real_t + [t for ts, _ in model_vals.values() for t in ts] + [t for ts, _ in raw_series.values() for t in ts]
    all_v = real_vals + [v for _, vals in model_vals.values() for v in vals] + [v for _, vals in raw_series.values() for v in vals]
    hmin, hmax = 0, hours_since(max(all_t))
    vmin, vmax_ = min(all_v), max(all_v)
    pad = (vmax_ - vmin) * 0.12 + 1e-6
    vmin, vmax_ = vmin - pad, vmax_ + pad
    x0, x1 = m, W - 14
    y0, y1 = H - 40, 16

    def PX(t): return x0 + hours_since(t) / hmax * (x1 - x0)
    def PY(v):
        f = (v - vmin) / (vmax_ - vmin)
        if invert:
            f = 1 - f
        return y1 + (1 - f) * (y0 - y1)

    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{label} for Dolphin">']
    for g in range(5):
        v = vmin + (vmax_ - vmin) * g / 4
        y = PY(v)
        o.append(f'<line class="gl" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="ax" x="{x0-7:.1f}" y="{y+3:.1f}" text-anchor="end">{v:.0f}</text>')
    step_h = 24
    gh = 0
    while gh <= hmax:
        x = PX(T0 + np.timedelta64(int(gh), "h"))
        o.append(f'<line class="gl" x1="{x:.1f}" x2="{x:.1f}" y1="{y1}" y2="{y0}"/>')
        lbl = str(T0 + np.timedelta64(int(gh), "h"))[5:10]
        o.append(f'<text class="ax" x="{x:.1f}" y="{y0+13:.1f}" text-anchor="middle">{lbl}</text>')
        gh += step_h
    now_x = PX(real_t[-1])
    o.append(f'<line x1="{now_x:.1f}" x2="{now_x:.1f}" y1="{y1}" y2="{y0}" class="nowln"/>')
    o.append(f'<text class="nowlbl" x="{now_x+4:.1f}" y="{y1+9:.1f}">now</text>')
    o.append(f'<text class="axl" x="{(x0+x1)/2:.1f}" y="{H-4:.1f}" text-anchor="middle">date (UTC)</text>')
    o.append(f'<text class="axl" transform="translate(13,{(y0+y1)/2:.1f}) rotate(-90)" text-anchor="middle">{unit}</text>')

    pts = [(PX(t), PY(v)) for t, v in zip(real_t, real_vals)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    o.append(f'<path d="{d}" class="ln real"/>')
    for x, y in pts:
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" class="dot real"/>')

    for tag, (ts, vals) in model_vals.items():
        pts = [(PX(real_t[-1]), PY(real_vals[-1]))] + [(PX(t), PY(v)) for t, v in zip(ts, vals)]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        o.append(f'<path d="{d}" class="ln {tag}"/>')
        ex, ey = pts[-1]
        o.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3" class="dot {tag}"/>')
    for tag, (ts, vals) in raw_series.items():
        pts = [(PX(t), PY(v)) for t, v in zip(ts, vals)]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        o.append(f'<path d="{d}" class="ln {tag}"/>')
        for x, y in pts:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" class="dot {tag}"/>')
    o.append('</svg>')

    rows = []
    for tag, (ts, vals) in model_vals.items():
        rows.append((tag, vals[-1], "@120h forecast"))
    for tag, (ts, vals) in raw_series.items():
        rows.append((tag, vals[-1], f"@{JTWC_TAU_H[-1]}h official forecast"))
    num = "".join(
        f'<div class="num {t}"><span class="sw"></span><b>{t}</b><span class="v">{v:.0f}{unit}</span>'
        f'<span class="v120">{lbl}</span></div>' for t, v, lbl in rows)
    num = (f'<div class="num real"><span class="sw"></span><b>real</b>'
           f'<span class="v">{real_vals[-1]:.0f}{unit}</span><span class="v120">latest ob</span></div>') + num
    return (f'<figure class="card"><figcaption><h3>{label}</h3>'
            f'<p class="unit">real (through latest ob) vs. forecast issued from that ob</p></figcaption>'
            f'{"".join(o)}<div class="nums">{num}</div></figure>')


v10_t, v10_vmax, v10_pres = model_series("v10")
v23_t, v23_vmax, v23_pres = model_series("v23")
v35_t, v35_vmax, v35_pres = model_series("v35")

MVALS = {"v10": (v10_t, v10_vmax), "v23": (v23_t, v23_vmax), "v35": (v35_t, v35_vmax)}
PVALS = {"v10": (v10_t, v10_pres), "v23": (v23_t, v23_pres), "v35": (v35_t, v35_pres)}

card_pres = chart("pressure", "Central pressure", "hPa", real_pres, PVALS, invert=True)
card_vmax = chart("vmax", "Maximum wind (vmax)", "kt", real_vmax, MVALS,
                   raw_series={"jtwc": (JTWC_T, JTWC_KT)})

legend = ('<span class="lg real"><span class="sw"></span><b>real</b> observed, through latest ob</span>'
          + "".join(f'<span class="lg {t}"><span class="sw"></span><b>{t}</b> {NOTE[t]}, forecast from latest ob</span>'
                    for t in ("v10", "v23", "v35"))
          + '<span class="lg jtwc"><span class="sw"></span><b>JTWC</b> official forecast, wind only '
            '(no pressure published per-lead), Advisory #3</span>')

palL = "".join(f".{t}.ln{{stroke:{c[0]};}} .{t} .sw,.dot.{t}{{background:{c[0]};fill:{c[0]};}}" for t, c in COL.items())
palD = "".join(f".{t}.ln{{stroke:{c[1]};}} .{t} .sw,.dot.{t}{{background:{c[1]};fill:{c[1]};}}" for t, c in COL.items())

HTML = f"""<meta charset="utf-8">
<title>Dolphin: predicted vs real central pressure and vmax</title>
<style>
:root{{color-scheme:light;--bg:#f2f4f6;--surface:#fcfcfb;--ink:#111820;--body:#2c3a47;--muted:#5d6c7a;
 --line:#d5dce3;--grid:#c3ccd5;--now:#c0392b;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;{palL}}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;--grid:#2b3745;--now:#e0574a;{palD}}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;
 --muted:#8697a5;--line:#26313d;--grid:#2b3745;--now:#e0574a;{palD}}}
body{{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:16px;line-height:1.6;}}
.wrap{{max-width:1180px;margin:0 auto;padding:clamp(24px,5vw,52px) clamp(16px,4vw,32px) 80px;
 display:flex;flex-direction:column;gap:26px;}}
h1{{color:var(--ink);font-size:clamp(24px,3.6vw,36px);line-height:1.14;letter-spacing:-.02em;margin:0;font-weight:660;}}
.eyebrow{{font-family:var(--mono);font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);}}
.lede{{max-width:80ch;font-size:14.5px;margin:0;}}
header{{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--line);padding-bottom:22px;}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:13px;}}
.lg,.num{{display:flex;align-items:center;gap:7px;}}
.sw{{width:15px;height:4px;border-radius:2px;display:inline-block;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px;}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 14px 10px;margin:0;
 display:flex;flex-direction:column;gap:6px;}}
figcaption h3{{color:var(--ink);font-size:14px;font-weight:640;margin:0;}}
.unit{{font-size:11.5px;color:var(--muted);margin:0;}}
.chart{{width:100%;height:auto;display:block;}}
.gl{{stroke:var(--grid);stroke-width:.6;opacity:.5;}}
.nowln{{stroke:var(--now);stroke-width:1;stroke-dasharray:2 2;opacity:.7;}}
.nowlbl{{font-family:var(--mono);font-size:8.5px;fill:var(--now);}}
.ax{{font-family:var(--mono);font-size:8px;fill:var(--muted);}}
.axl{{font-family:var(--mono);font-size:8.5px;fill:var(--muted);}}
.ln{{fill:none;stroke-width:2.1;stroke-linejoin:round;stroke-linecap:round;}}
.ln.real{{stroke-width:2.4;}}
.nums{{display:flex;flex-wrap:wrap;gap:6px 14px;border-top:1px solid var(--line);padding-top:8px;font-size:12px;}}
.num b{{color:var(--ink);font-family:var(--mono);font-size:11px;}}
.num .v{{font-family:var(--mono);color:var(--ink);font-weight:600;}}
.num .v120{{font-family:var(--mono);color:var(--muted);font-size:10.5px;}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted);max-width:80ch;
 display:flex;flex-direction:column;gap:8px;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; predicted vs real, Typhoon Dolphin</div>
  <h1>Dolphin: predicted central pressure and vmax vs real</h1>
  <p class="lede">The dotted "now" line marks the latest real observation (2026-07-29 06:00 UTC).
  Left of it: what actually happened. Right of it: each model's forecast issued from that same
  observation -- v10 (track-only, no steering) and v23/v35 (real fetched GFS steering) -- plus
  JTWC's official Advisory #3 forecast (wind only; JTWC does not publish forecast pressure at each
  lead, only for the current position). Pressure is inverted on its axis (lower = stronger) so both
  panels read "up and to the right = weakening" the same way.</p>
  <div class="legend">{legend}</div>
 </header>
 <div class="grid">{card_pres}{card_vmax}</div>
 <footer>
  <p><b>All three of this project's models forecast weakening from here -- JTWC forecasts the
  opposite.</b> Real pressure bottomed at 937 hPa (Jul 28 18:00), already recovering slightly to
  941 hPa by the latest observation. v10, v23, and v35 all extrapolate that recovery forward --
  pressure rising (weakening) and wind falling through the 120h horizon -- v10 most aggressively
  (57 kt by 120h) since it has no steering/environmental information to reason from beyond the raw
  kinematic trend, while v23/v35 (99/98 kt by 120h) are less extreme. JTWC's official forecast
  points the opposite direction entirely: continued intensification to 150 kt / ~915 hPa. Whether
  the real storm follows the models' extrapolation-of-recent-trend or JTWC's synoptic reasoning
  will be visible in the next few days of real observations.</p>
 </footer>
</div>"""

os.makedirs("paper", exist_ok=True)
open("paper/dolphin_pressure_vmax_chart.html", "w").write(HTML)
print(f"wrote paper/dolphin_pressure_vmax_chart.html ({len(HTML)/1000:.0f} KB)")
