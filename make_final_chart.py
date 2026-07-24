"""Final scoreboard, every model on ONE convention -- and the correction that changed the answer.

WHY THIS PAGE REPLACES THE EARLIER ONE. Two different averages had been compared as if they were
the same number:

    pooled      np.abs(O-T)[mask].mean()      over all (window, lead) pairs   <- Colab
    unweighted  mean of the 20 per-lead means                                 <- local scripts

Later leads carry larger error AND fewer valid targets, so the unweighted form up-weights them.
v10/v21/v23/v25 were quoted unweighted while v26/v26abl/v27/v27abl were quoted pooled, and the gap
is about 0.3 kt -- the same size as the effects being claimed.

Recomputed pooled throughout, one earlier headline REVERSES: v23 wins maximum wind at 16.12 kt,
not v26 at 16.26. "The ocean patch wins wind" was an artefact of the mismatch. What survives is
narrower and still true: WITHIN the v26 run, switching the ocean patch on improved wind by 0.86 kt
against its own ablation -- it simply never reached v23's level.

Reads track_build/pooled_metrics.json.
"""
import json, os

D = json.load(open("track_build/pooled_metrics.json"))
ROWS = ["v10", "v21", "v23", "v25", "v26abl", "v26", "v27abl", "v27"]
ROWS = [t for t in ROWS if t in D]
NOTE = {"v10": "no environment at all",
        "v21": "chain-of-thought steering",
        "v23": "+ temporal steering stack (t-24 h, t-12 h, now)",
        "v25": "+ 43 environmental scalars as one token",
        "v26abl": "v26 with the ocean patch OFF",
        "v26": "+ ocean-heat patch CNN on the intensity head (AOML)",
        "v27abl": "v27 with env + ocean OFF (v23-shaped, retrained)",
        "v27": "v23 + env token + ocean patch CNN (GODAS)"}
COL = {"v10": ("#eda100", "#c98500"), "v21": ("#2a78d6", "#3987e5"),
       "v23": ("#eb6834", "#d95926"), "v25": ("#c2185b", "#e0457f"),
       "v26abl": ("#8f9aa6", "#7c8894"), "v26": ("#1a9e6f", "#22b880"),
       "v27abl": ("#7a8794", "#98a6b4"), "v27": ("#3d4852", "#9fb0bd")}
METRICS = [("track", "Track position error", "km"),
           ("vmax", "Maximum wind", "kt"),
           ("pres", "Central pressure", "hPa"),
           ("rmw", "Radius of max wind", "nm"),
           ("radii", "Wind radii R34/50/64", "nm")]


def bars(key, label, unit):
    W, H = 380, 232
    m_l, m_r, m_t = 78, 62, 14
    vals = [D[t][key] for t in ROWS]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    x0, x1 = m_l, W - m_r
    bh, gap = 18, 7
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{label} by model">']
    for i, t in enumerate(ROWS):
        v = D[t][key]; y = m_t + i * (bh + gap)
        frac = (v - (lo - .18 * span)) / (span * 1.36)
        w = max(3.0, frac * (x1 - x0))
        best = abs(v - lo) < 1e-9
        o.append(f'<text class="ml" x="{m_l-8}" y="{y+bh*0.72:.0f}" text-anchor="end">{t}</text>')
        o.append(f'<rect class="bar {t}" x="{x0}" y="{y}" width="{w:.1f}" height="{bh}" rx="2"/>')
        o.append(f'<text class="val{" best" if best else ""}" x="{x0+w+7:.1f}" y="{y+bh*0.72:.0f}">{v:.2f}</text>')
    o.append('</svg>')
    win = [t for t in ROWS if abs(D[t][key] - lo) < 1e-9][0]
    return (f'<figure class="card"><figcaption><h3>{label}</h3>'
            f'<p class="unit">mean absolute error, {unit} &middot; lower is better &middot; pooled</p></figcaption>'
            f'{"".join(o)}<div class="delta">best: <b>{win} {lo:.2f} {unit}</b></div></figure>')


cards = "".join(bars(k, lab, u) for k, lab, u in METRICS)
palL = "".join(f".bar.{t}{{fill:{c[0]};}} .sw.{t}{{background:{c[0]};}}" for t, c in COL.items() if t in D)
palD = "".join(f".bar.{t}{{fill:{c[1]};}} .sw.{t}{{background:{c[1]};}}" for t, c in COL.items() if t in D)
legend = "".join(f'<span class="lg"><span class="sw {t}"></span><b>{t}</b> {NOTE[t]}</span>' for t in ROWS)

d27 = {k: D["v27"][k] - D["v27abl"][k] for k, _, _ in METRICS}
d26 = D["v26"]["vmax"] - D["v26abl"]["vmax"]

HTML = f"""<meta charset="utf-8">
<title>TrackFormer — the corrected scoreboard</title>
<style>
:root{{color-scheme:light;--bg:#f2f4f6;--surface:#fcfcfb;--ink:#111820;--body:#2c3a47;--muted:#5d6c7a;
 --line:#d5dce3;--good:#1a9e6f;--bad:#c2185b;
 --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;{palL}}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;
 --good:#22b880;--bad:#e0457f;{palD}}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;
 --muted:#8697a5;--line:#26313d;--good:#22b880;--bad:#e0457f;{palD}}}
body{{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:16px;line-height:1.6;}}
.wrap{{max-width:1120px;margin:0 auto;padding:clamp(24px,5vw,52px) clamp(16px,4vw,32px) 80px;
 display:flex;flex-direction:column;gap:24px;}}
h1{{color:var(--ink);font-size:clamp(24px,3.6vw,36px);line-height:1.14;letter-spacing:-.02em;margin:0;font-weight:660;}}
h2{{color:var(--ink);font-size:19px;margin:0;font-weight:640;}}
.eyebrow{{font-family:var(--mono);font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);}}
.lede{{max-width:82ch;font-size:14.5px;margin:0;}}
header{{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--line);padding-bottom:20px;}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;}}
.lg{{display:flex;align-items:center;gap:7px;}}
.sw{{width:15px;height:11px;border-radius:3px;display:inline-block;}}
.call{{display:flex;flex-wrap:wrap;gap:10px;}}
.pill{{background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:6px 14px;font-size:13px;}}
.pill b{{font-family:var(--mono);}}
.good{{color:var(--good);}} .bad{{color:var(--bad);}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px;}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 14px 10px;margin:0;
 display:flex;flex-direction:column;gap:4px;}}
figcaption h3{{color:var(--ink);font-size:14px;font-weight:640;margin:0;}}
.unit{{font-size:11.5px;color:var(--muted);margin:0;}}
.chart{{width:100%;height:auto;display:block;}}
.ml{{font-family:var(--mono);font-size:10px;fill:var(--muted);}}
.val{{font-family:var(--mono);font-size:10px;fill:var(--ink);}}
.val.best{{fill:var(--good);font-weight:700;}}
.delta{{border-top:1px solid var(--line);padding-top:7px;font-size:11.5px;font-family:var(--mono);}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px 18px;
 display:flex;flex-direction:column;gap:10px;}}
.warn{{border-left:3px solid var(--bad);padding-left:14px;}}
table{{border-collapse:collapse;width:100%;font-size:13px;}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;}}
th{{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;}}
td.n{{font-family:var(--mono);text-align:right;}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted);max-width:82ch;
 display:flex;flex-direction:column;gap:8px;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; every model, one convention</div>
  <h1>v23 wins track and wind. Neither thing added after it helped.</h1>
  <p class="lede">Eight models, 10-seed ensembles, 3,763 western- and eastern-Pacific test windows of
  2020&ndash;2025. Every figure here is <b>pooled</b> over all window&times;lead pairs, each intensity channel
  masked with its own validity flag. That matters: the earlier version of this page mixed pooled and
  per-lead-averaged numbers, and fixing it reverses one of its headlines.</p>
  <div class="call">
   <span class="pill">best track: <b class="good">v23 {D['v23']['track']:.2f} km</b></span>
   <span class="pill">best wind: <b class="good">v23 {D['v23']['vmax']:.2f} kt</b></span>
   <span class="pill">v27 vs its own control: <b class="bad">worse on all five</b></span>
  </div>
  <div class="legend">{legend}</div>
 </header>

 <div class="grid">{cards}</div>

 <section class="panel warn">
  <h2>The correction: "the ocean patch wins wind" does not survive</h2>
  <p class="lede">v26 was reported at 16.26 kt against v23's 16.43 and called the wind winner. But 16.26 was
  <em>pooled</em> and 16.43 was <em>per-lead averaged</em>. Recomputed pooled, <b>v23 scores
  {D['v23']['vmax']:.2f} kt and beats v26's {D['v26']['vmax']:.2f}</b>. The ranking flips.</p>
  <p class="lede">What still stands is the narrower, within-run claim: inside the v26 experiment, turning the
  ocean patch on improved wind by <b>{abs(d26):.2f} kt</b> against its own matched ablation. That comparison was
  always pooled-against-pooled. The ocean patch did something real to its own baseline &mdash; it just never
  reached the level v23 already had.</p>
 </section>

 <section class="panel">
  <h2>v27: stacking v23 and v26 made everything worse</h2>
  <p class="lede">v27 put the environmental token and the ocean CNN on top of v23's steering stack. v27abl is the
  identical model with both switched off, trained in the same run &mdash; the matched control. v27 lost on every
  metric, on both the full test set and the ocean-covered subset, with no exceptions:</p>
  <div style="overflow-x:auto"><table>
   <tr><th>metric</th><th style="text-align:right">v27abl</th><th style="text-align:right">v27</th>
       <th style="text-align:right">delta</th></tr>
   {"".join(f'<tr><td>{lab}</td><td class="n">{D["v27abl"][k]:.2f}</td><td class="n">{D["v27"][k]:.2f}</td>'
            f'<td class="n bad">{d27[k]:+.2f}</td></tr>' for k, lab, _ in METRICS)}
  </table></div>
  <p class="lede">An init assertion proved the ocean token cannot reach the track decoder, so the 6.8&nbsp;km track
  loss cannot be direct interference. The two paths share the encoders, and the extra intensity objectives evidently
  pull that shared representation away from what the steering stack needs. <b>The parts are not independent.</b></p>
 </section>

 <footer>
  <p>Bars are scaled within each panel's own range so differences of a few tenths stay visible; read the printed
  number, not the bar length. v26, v26abl and v27abl come from their Colab JSONs &mdash; already pooled, and their
  checkpoints no longer exist to recompute. Everything else was recomputed locally from checkpoints, with v27
  reproducing its Colab track to 0.00&nbsp;km and its pooled wind to 16.76 kt exactly, which is what confirms the
  two harnesses now agree.</p>
  <p>None of these deltas has been through a paired-storm bootstrap except v23's original -9.09&nbsp;km, which
  remains the only significance-tested result in the project.</p>
 </footer>
</div>"""

os.makedirs("paper", exist_ok=True)
open("paper/final_scoreboard.html", "w").write(HTML)
print(f"wrote paper/final_scoreboard.html ({len(HTML)/1000:.0f} KB)")
for k, lab, u in METRICS:
    lo = min(D[t][k] for t in ROWS)
    win = [t for t in ROWS if abs(D[t][k] - lo) < 1e-9][0]
    print(f"  {lab:24s} best {win:7s} {lo:8.2f} {u}")
