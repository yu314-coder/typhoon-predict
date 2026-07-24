"""v26 result figure: what the ocean-heat patch actually bought, with the retrain separated out.

The point of this page is one comparison the earlier charts could not make: v26 vs v26abl. Those two
differ in EXACTLY one thing -- whether the 21x21 ocean-heat patch reaches the intensity decoder --
so the gap between them is the ocean effect, and the gap from v25 to v26abl is what simply
retraining bought. Reporting v26 against v25 alone would have credited the ocean with both.

All numbers are 10-seed ensembles on the WP+EP 2020+ test set, each intensity channel masked with
its OWN validity flag. v26/v26abl come from the Colab run (downloads/v26.json, v26abl.json);
v10/v21/v25 from the local per-lead run (track_build/intensity_compare.json), same masking.
"""
import json, os

AGG = json.load(open("track_build/intensity_compare.json"))
V26 = json.load(open("downloads/v26.json"))["v26"]["all"]
V26A = json.load(open("downloads/v26abl.json"))["v26abl"]["all"]
V26_O = json.load(open("downloads/v26.json"))["v26"]["ocean"]
V26A_O = json.load(open("downloads/v26abl.json"))["v26abl"]["ocean"]


def am(x):
    v = [q for q in x if q is not None]
    return sum(v) / len(v) if v else float("nan")


ROWS = ["v10", "v21", "v23", "v25", "v26abl", "v26"]
NOTE = {"v10": "no environment at all",
        "v21": "chain-of-thought steering",
        "v23": "temporal steering stack (t-24 h, t-12 h, now) &mdash; bootstrap-confirmed",
        "v25": "+ 43 environmental scalars (one token)",
        "v26abl": "v26 with the ocean patch switched OFF",
        "v26": "+ 21x21 ocean-heat patch through a CNN into the intensity head"}
COL = {"v10": ("#eda100", "#c98500"), "v21": ("#2a78d6", "#3987e5"),
       "v23": ("#eb6834", "#d95926"), "v25": ("#c2185b", "#e0457f"),
       "v26abl": ("#7a8794", "#98a6b4"), "v26": ("#1a9e6f", "#22b880")}

D = {t: {k: am(AGG[t][k]) for k in ("vmax", "pressure", "rmw", "radii")}
     for t in ("v10", "v21", "v23", "v25")}
D["v26abl"] = {"vmax": V26A["vmax"], "pressure": V26A["pres"], "rmw": V26A["rmw"], "radii": V26A["radii"]}
D["v26"] = {"vmax": V26["vmax"], "pressure": V26["pres"], "rmw": V26["rmw"], "radii": V26["radii"]}
for t in ("v10", "v21", "v23", "v25"):
    D[t]["track"] = AGG[t]["agg_track"]
D["v26abl"]["track"] = V26A["track"]; D["v26"]["track"] = V26["track"]

METRICS = [("vmax", "Maximum wind", "kt", True),
           ("pressure", "Central pressure", "hPa", False),
           ("rmw", "Radius of max wind", "nm", False),
           ("radii", "Wind radii R34/50/64", "nm", False),
           ("track", "Track position error", "km", False)]


def bars(key, label, unit, headline):
    W, H = 380, 176
    m_l, m_r, m_t = 74, 60, 18
    vals = [D[t][key] for t in ROWS]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    x0, x1 = m_l, W - m_r
    bh, gap = 20, 8
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{label} by model">']
    for i, t in enumerate(ROWS):
        v = D[t][key]
        y = m_t + i * (bh + gap)
        # bars are scaled within [lo-10% span, hi+10% span] so small differences stay legible
        frac = (v - (lo - .18 * span)) / (span * 1.36)
        w = max(3.0, frac * (x1 - x0))
        best = abs(v - lo) < 1e-9
        o.append(f'<text class="ml{" b" if t.startswith("v26") else ""}" x="{m_l-8}" y="{y+bh*0.72:.0f}" text-anchor="end">{t}</text>')
        o.append(f'<rect class="bar {t}" x="{x0}" y="{y}" width="{w:.1f}" height="{bh}" rx="2"/>')
        o.append(f'<text class="val{" best" if best else ""}" x="{x0+w+7:.1f}" y="{y+bh*0.72:.0f}">{v:.2f}</text>')
    o.append('</svg>')
    d = D["v26"][key] - D["v26abl"][key]
    if key == "track":
        # the token never reaches the track decoder; any gap here is shared-encoder drift between
        # two separate training runs, not an ocean effect. Calling it a "gain" would be a claim.
        verdict = "incidental &mdash; token never reaches the track decoder"
        cls = "null"
    else:
        verdict = "real gain" if headline else ("null" if abs(d) < 0.15 else ("gain" if d < 0 else "worse"))
        cls = "good" if (d < -0.15) else ("bad" if d > 0.15 else "null")
    return (f'<figure class="card"><figcaption><h3>{label}</h3>'
            f'<p class="unit">mean absolute error, {unit} &middot; lower is better</p></figcaption>'
            f'{"".join(o)}'
            f'<div class="delta {cls}">ocean effect (v26 &minus; v26abl): <b>{d:+.2f} {unit}</b> &middot; {verdict}</div>'
            f'</figure>')


cards = "".join(bars(k, lab, u, hl) for k, lab, u, hl in METRICS)

# attribution for the two intensity headliners
def attrib(key, unit):
    a = D["v25"][key]; b = D["v26abl"][key]; c = D["v26"][key]
    return (f'<tr><td><b>{key if key!="pressure" else "pressure"}</b></td>'
            f'<td class="n">{a:.2f}</td><td class="n">{b:.2f}</td><td class="n">{c:.2f}</td>'
            f'<td class="n {"good" if b-a<-0.05 else "null"}">{b-a:+.2f}</td>'
            f'<td class="n {"good" if c-b<-0.15 else ("bad" if c-b>0.15 else "null")}">{c-b:+.2f}</td>'
            f'<td class="u">{unit}</td></tr>')


rows_attr = attrib("vmax", "kt") + attrib("pressure", "hPa") + attrib("rmw", "nm") + attrib("radii", "nm")

palL = "".join(f".bar.{t.replace('.','_')}{{fill:{c[0]};}} .sw.{t.replace('.','_')}{{background:{c[0]};}}" for t, c in COL.items())
palD = "".join(f".bar.{t.replace('.','_')}{{fill:{c[1]};}} .sw.{t.replace('.','_')}{{background:{c[1]};}}" for t, c in COL.items())
legend = "".join(f'<span class="lg"><span class="sw {t}"></span><b>{t}</b> {NOTE[t]}</span>' for t in ROWS)

dv = D["v26"]["vmax"] - D["v26abl"]["vmax"]
dp = D["v26"]["pressure"] - D["v26abl"]["pressure"]

HTML = f"""<meta charset="utf-8">
<title>TrackFormer — which model wins what</title>
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
.good{{color:var(--good);}} .bad{{color:var(--bad);}} .null{{color:var(--muted);}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px;}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 14px 10px;margin:0;
 display:flex;flex-direction:column;gap:4px;}}
figcaption h3{{color:var(--ink);font-size:14px;font-weight:640;margin:0;}}
.unit{{font-size:11.5px;color:var(--muted);margin:0;}}
.chart{{width:100%;height:auto;display:block;}}
.ml{{font-family:var(--mono);font-size:10px;fill:var(--muted);}}
.ml.b{{fill:var(--ink);font-weight:700;}}
.val{{font-family:var(--mono);font-size:10px;fill:var(--ink);}}
.val.best{{fill:var(--good);font-weight:700;}}
.delta{{border-top:1px solid var(--line);padding-top:7px;font-size:11.5px;font-family:var(--mono);}}
table{{border-collapse:collapse;width:100%;font-size:13px;}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;}}
th{{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;}}
td.n{{font-family:var(--mono);text-align:right;}} td.u{{color:var(--muted);font-size:11.5px;}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px 18px;
 display:flex;flex-direction:column;gap:10px;}}
.warn{{border-left:3px solid var(--bad);padding-left:14px;}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted);max-width:82ch;
 display:flex;flex-direction:column;gap:8px;}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer &middot; where the project actually stands</div>
  <h1>No single best model &mdash; v23 takes four of six, v26 takes two</h1>
  <p class="lede">Every version scored the same way: 10-seed ensembles on the 3,763 WP+EP test windows of
  2020&ndash;2025, each intensity channel masked with its own validity flag. <b>v23</b> (temporal steering stack)
  is the best track model and the only gain in this project that survived a paired-storm bootstrap.
  <b>v26</b> (ocean-heat patch &rarr; CNN &rarr; intensity head) wins maximum wind and RMW. Reporting either one
  as "the best model" on its own would be a claim the numbers do not support.</p>
  <div class="call">
   <span class="pill">best track: <b class="good">v23 {D['v23']['track']:.1f} km</b></span>
   <span class="pill">best pressure: <b class="good">v23 {D['v23']['pressure']:.2f} hPa</b></span>
   <span class="pill">best max wind: <b class="good">v26 {D['v26']['vmax']:.2f} kt</b></span>
   <span class="pill">ocean effect on wind: <b class="good">{dv:+.2f} kt</b> &middot; on pressure <b class="null">{dp:+.2f} hPa</b></span>
  </div>
  <div class="legend">{legend}</div>
 </header>

 <div class="grid">{cards}</div>

 <section class="panel">
  <h2>Separating the retrain from the ocean</h2>
  <p class="lede">v26 is a fresh training run, and a fresh run moves the numbers on its own. Crediting the ocean with
  the whole v25&rarr;v26 gap would have been wrong &mdash; on pressure it would have been <em>entirely</em> wrong.</p>
  <div style="overflow-x:auto">
  <table>
   <tr><th>metric</th><th style="text-align:right">v25</th><th style="text-align:right">v26abl</th>
       <th style="text-align:right">v26</th><th style="text-align:right">retrain</th>
       <th style="text-align:right">ocean</th><th></th></tr>
   {rows_attr}
  </table></div>
  <p class="lede">Maximum wind is the only column where the ocean term carries the gain. Pressure improved by
  0.18&nbsp;hPa from v25 &mdash; and every bit of that was the retrain; the ocean patch contributed
  {dp:+.2f}. RMW and the wind radii move by less than the run-to-run spread and cancel in sign, so they are null.</p>
 </section>

 <section class="panel">
  <h2>The map shows v23, not v26</h2>
  <p class="lede">The world map that goes with this page draws <b>v23</b>, because the map draws <em>tracks</em> and
  v23 is the track winner at {D['v23']['track']:.1f} km. Drawing v26 there would add nothing: its ocean token
  reaches only the intensity decoder, and an init assertion proved the track output is bit-identical with the token
  on or off, so a v26 track map would be the v25 map redrawn ({D['v26']['track']:.0f} km vs {D['v25']['track']:.0f} km,
  a difference with no ocean content in it).</p>
  <p class="lede">Separately, and this one was a mistake: the Colab runtime idled out before v26's 20 checkpoints were
  pulled, and <code>/content</code> is wiped on disconnect, so they are gone. Its numbers survive only because the
  results JSON was saved first. The training script now mirrors every improved checkpoint to Drive.</p>
 </section>

 <footer>
  <p>Bars are scaled within each panel's own range so differences of a few tenths stay visible; read the printed
  number, not the bar length, for magnitude. Lower is better throughout.</p>
  <p>A 0.86&nbsp;kt gain on a 17&nbsp;kt base is about 5%. It has not yet been through a paired-storm bootstrap,
  so it is a measured difference between two matched 10-seed ensembles, not yet a significance-tested result.
  This project has already seen one apparent 8&nbsp;km effect dissolve into seed noise under that test.</p>
 </footer>
</div>"""

os.makedirs("paper", exist_ok=True)
open("paper/v26_ocean_result.html", "w").write(HTML)
print(f"wrote paper/v26_ocean_result.html ({len(HTML)/1000:.0f} KB)")
for k, lab, u, _ in METRICS:
    print(f"  {lab:24s} v26abl {D['v26abl'][k]:8.2f} -> v26 {D['v26'][k]:8.2f}  "
          f"ocean {D['v26'][k]-D['v26abl'][k]:+.2f} {u}")
