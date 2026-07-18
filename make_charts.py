"""Build the per-lead-time model comparison page from measured metrics.

Reads track_build/perlead_local.json (v10/v13/v14/v14.1, measured locally on the repaired
fields) and, if present, track_build/perlead_v16.json (v16 with/without SST, from Colab).
Writes paper/model_metrics.html.
"""
import json, os

LOCAL = json.load(open("track_build/perlead_local.json"))
V16 = json.load(open("track_build/perlead_v16.json")) if os.path.exists("track_build/perlead_v16.json") else None
LEADS = [6 * (i + 1) for i in range(20)]

PANELS = [
    ("track",    "Track error",    "km",  "Great-circle distance between forecast and actual position, cumulative."),
    ("vmax",     "Peak wind",      "kt",  "Mean absolute error in maximum sustained wind."),
    ("pressure", "Central pressure", "hPa", "Mean absolute error in minimum sea-level pressure."),
    ("rmw",      "Radius of max wind", "km", "Mean absolute error in the radius where peak wind occurs."),
    ("radii",    "Wind radii",     "km",  "Mean absolute error across all 12 wind-extent radii (34/50/64 kt x 4 quadrants)."),
]
# categorical slots, validated: node scripts/validate_palette.js "#2a78d6,#1baf7a,#eda100,#4a3aa7,#e34948" --mode light
LINE = {
    "v10":   ("#4a3aa7", "#9085e9"),
    "v13":   ("#1baf7a", "#199e70"),
    "v14":   ("#eda100", "#c98500"),
    "v14.1": ("#e34948", "#e66767"),
    "v16":   ("#2a78d6", "#3987e5"),
}
ABL = {"with_sst": ("#2a78d6", "#3987e5"), "no_sst": ("#e34948", "#e66767")}

series = dict(LOCAL)
if V16:
    series["v16"] = {k: V16[k]["with_sst"] for k in V16}
ORDER = [k for k in ["v10", "v13", "v14", "v14.1", "v16"] if k in series]


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def chart(sid, key, unit, data, colors, labels):
    """One SVG line chart. data: {name: [20 values]}"""
    W, H = 460, 250
    ml, mr, mt, mb = 52, 96, 14, 34
    pw, ph = W - ml - mr, H - mt - mb
    vals = [x for s in data.values() for x in s if x is not None]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * .12 or 1
    lo, hi = max(0, lo - pad), hi + pad
    def X(i): return ml + pw * i / 19
    def Y(v): return mt + ph * (1 - (v - lo) / (hi - lo))
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{key} by lead time" '
           f'class="chart" data-metric="{key}">']
    for t in range(5):
        v = lo + (hi - lo) * t / 4
        y = Y(v)
        out.append(f'<line class="grid" x1="{ml}" x2="{ml+pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{ml-8}" y="{y+3.5:.1f}" text-anchor="end">{v:.0f}</text>')
    for h in (24, 48, 72, 96, 120):
        i = h // 6 - 1
        out.append(f'<text class="tick" x="{X(i):.1f}" y="{H-14}" text-anchor="middle">{h}</text>')
    out.append(f'<text class="axlab" x="{ml+pw/2:.0f}" y="{H-1}" text-anchor="middle">lead time (hours)</text>')
    out.append(f'<text class="axlab" x="{ml-40}" y="{mt+ph/2:.0f}" transform="rotate(-90 {ml-40} {mt+ph/2:.0f})" text-anchor="middle">{unit}</text>')
    ends = []
    for name, vs in data.items():
        pts = [(X(i), Y(v)) for i, v in enumerate(vs) if v is not None]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<path class="ln" d="{d}" style="stroke:var(--c-{name.replace(".","_")})"/>')
        ends.append((pts[-1][1], name, pts[-1][0]))
    # de-collide the direct labels: sort by y, then push apart to a minimum spacing
    ends.sort()
    MINGAP = 12.0
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < MINGAP:
            ends[i] = (ends[i - 1][0] + MINGAP, ends[i][1], ends[i][2])
    for ly, name, lx in ends:
        v = name.replace(".", "_")
        out.append(f'<circle class="end" cx="{lx:.1f}" cy="{Y(data[name][-1]):.1f}" r="3.4" style="fill:var(--c-{v})"/>')
        out.append(f'<text class="dlab" x="{lx+9:.1f}" y="{ly+3.5:.1f}" style="fill:var(--c-{v})">{labels[name]}</text>')
    for i in range(20):
        rows = "".join(f"<span class='k'><i style='background:var(--c-{n.replace('.','_')})'></i>{labels[n]}</span>"
                       f"<span class='v'>{(vs[i] if vs[i] is not None else float('nan')):.1f}</span>"
                       for n, vs in data.items())
        out.append(f'<rect class="hit" x="{X(i)-pw/38:.1f}" y="{mt}" width="{pw/19:.1f}" height="{ph}" '
                   f'data-h="{LEADS[i]}" data-rows="{rows.replace(chr(34), chr(39))}"/>')
    out.append(f'<line class="cross" x1="0" x2="0" y1="{mt}" y2="{mt+ph}" style="opacity:0"/>')
    out.append('</svg>')
    return "\n".join(out)


def table(data, labels, unit_map):
    head = "".join(f"<th class='num'>{labels[n]}</th>" for n in data)
    rows = []
    for i, h in enumerate(LEADS):
        cells = "".join(f"<td class='num'>{(vs[i] if vs[i] is not None else float('nan')):.1f}</td>"
                        for vs in data.values())
        rows.append(f"<tr><td class='num'>+{h}</td>{cells}</tr>")
    return (f"<table><thead><tr><th class='num'>lead (h)</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


cssvars = "".join(f"--c-{k.replace('.','_')}:{v[0]};" for k, v in LINE.items())
cssvars_d = "".join(f"--c-{k.replace('.','_')}:{v[1]};" for k, v in LINE.items())
if V16:
    cssvars += "--c-with_sst:#2a78d6;--c-no_sst:#e34948;"
    cssvars_d += "--c-with_sst:#3987e5;--c-no_sst:#e66767;"

labels = {k: k for k in ORDER}
lineage_legend = ''.join(f'<span class="lg"><i style="background:var(--c-{k.replace(chr(46),chr(95))})"></i>{k}</span>' for k in ORDER)
lineage_panels, lineage_tables = [], []
for key, title, unit, blurb in PANELS:
    d = {n: series[n][key] for n in ORDER if key in series[n]}
    if not d:
        continue
    lineage_panels.append(
        f'<figure class="panel"><figcaption><h3>{title}</h3><p>{blurb}</p></figcaption>'
        f'{chart("lin", key, unit, d, {n: LINE[n] for n in d}, labels)}</figure>')
    lineage_tables.append(f"<h4>{title} ({unit})</h4>" + table(d, labels, unit))

abl_html = ""
if V16:
    al = {"with_sst": "with SST", "no_sst": "SST removed"}
    ap, at = [], []
    for key, title, unit, blurb in PANELS:
        d = {"with_sst": V16[key]["with_sst"], "no_sst": V16[key]["no_sst"]}
        ap.append(f'<figure class="panel"><figcaption><h3>{title}</h3><p>{blurb}</p></figcaption>'
                  f'{chart("abl", key, unit, d, ABL, al)}</figure>')
        at.append(f"<h4>{title} ({unit})</h4>" + table(d, al, unit))
    deltas = []
    for key, title, unit, _ in PANELS:
        a, b = mean(V16[key]["with_sst"]), mean(V16[key]["no_sst"])
        pct = 100 * (b - a) / b
        deltas.append(f"<tr><td>{title}</td><td class='num'>{a:.2f}</td><td class='num'>{b:.2f}</td>"
                      f"<td class='num' style='color:{'var(--good)' if a<b else 'var(--bad)'}'>{a-b:+.2f} {unit}</td>"
                      f"<td class='num' style='color:{'var(--good)' if a<b else 'var(--bad)'}'>{pct:+.1f}%</td></tr>")
    abl_html = f"""
  <section>
    <div class="sec-head"><div class="eyebrow">Ablation</div>
      <h2>Does the ocean actually earn its channel?</h2>
      <p class="lede">Same architecture, same parameter count, same three seeds — the SST channel is
      zeroed rather than removed, so the only thing that changes is the information. Zero is also how
      the pipeline encodes an unavailable field, so the ablated model sees a state it was trained to
      handle. Negative delta means SST helps.</p></div>
    <div class="tablewrap"><table><thead><tr><th>metric</th><th class="num">with SST</th>
      <th class="num">SST removed</th><th class="num">delta</th><th class="num">relative</th></tr></thead>
      <tbody>{''.join(deltas)}</tbody></table></div>
    <div class="panels">{''.join(ap)}</div>
    <details><summary>Ablation data table</summary><div class="tablewrap">{''.join(at)}</div></details>
  </section>"""

HTML = f"""<title>TrackFormer — forecast skill by lead time</title>
<style>
:root{{color-scheme:light;
 --bg:#f2f4f6;--surface:#fcfcfb;--surface-2:#e9edf1;--ink:#111820;--body:#2c3a47;--muted:#5d6c7a;--line:#d5dce3;
 --good:#1baf7a;--bad:#e34948;{cssvars}
 --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--surface-2:#1b242f;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;
 --good:#199e70;--bad:#e66767;{cssvars_d}}}}}
:root[data-theme="dark"]{{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--surface-2:#1b242f;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;
 --good:#199e70;--bad:#e66767;{cssvars_d}}}
body{{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:16px;line-height:1.6;}}
.wrap{{max-width:1120px;margin:0 auto;padding:clamp(26px,5vw,60px) clamp(16px,4vw,36px) 90px;
 display:flex;flex-direction:column;gap:48px;}}
h1,h2,h3,h4{{color:var(--ink);margin:0;text-wrap:balance;}}
h1{{font-size:clamp(28px,4vw,42px);line-height:1.1;letter-spacing:-.022em;font-weight:660;}}
h2{{font-size:clamp(20px,2.4vw,25px);letter-spacing:-.012em;font-weight:640;}}
h3{{font-size:15px;font-weight:640;}} h4{{font-size:13px;font-weight:640;margin:18px 0 6px;}}
p{{margin:0;}} .lede{{max-width:70ch;}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);}}
header{{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--line);padding-bottom:26px;}}
section{{display:flex;flex-direction:column;gap:20px;}}
.sec-head{{display:flex;flex-direction:column;gap:7px;}}
.panels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:5px;padding:16px 14px 8px;margin:0;
 display:flex;flex-direction:column;gap:6px;position:relative;}}
figcaption{{display:flex;flex-direction:column;gap:3px;padding:0 4px;}}
figcaption p{{font-size:12.5px;color:var(--muted);line-height:1.45;}}
.chart{{width:100%;height:auto;overflow:visible;stroke:none;}}
.grid-line,.grid{{stroke:var(--line);stroke-width:1;}}
.tick{{font-family:var(--mono);font-size:9.5px;fill:var(--muted);}}
.axlab{{font-family:var(--sans);font-size:10px;fill:var(--muted);}}
.ln{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round;}}
.end{{stroke:var(--surface);stroke-width:2;}}
.dlab{{font-family:var(--mono);font-size:10.5px;font-weight:600;}}
.hit{{fill:transparent;stroke:none;}} .cross{{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3;pointer-events:none;}}
.tip{{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);
 border-radius:4px;padding:7px 9px;font-size:11.5px;box-shadow:0 4px 14px rgba(0,0,0,.13);opacity:0;
 transition:opacity .1s;min-width:132px;z-index:5;}}
.tip b{{color:var(--ink);font-family:var(--mono);font-size:11px;display:block;margin-bottom:4px;}}
.tip .row{{display:flex;justify-content:space-between;gap:12px;align-items:center;}}
.tip .k{{display:flex;align-items:center;gap:5px;color:var(--body);}}
.tip .k i{{width:8px;height:8px;border-radius:2px;display:inline-block;}}
.tip .v{{font-family:var(--mono);color:var(--ink);font-variant-numeric:tabular-nums;}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:5px;background:var(--surface);}}
table{{border-collapse:collapse;width:100%;font-size:13px;}}
th,td{{padding:7px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap;}}
thead th{{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
 color:var(--muted);font-weight:500;background:var(--surface-2);}}
tbody tr:last-child td{{border-bottom:none;}}
.num{{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;}}
details{{border:1px solid var(--line);border-radius:5px;background:var(--surface);padding:12px 14px;}}
summary{{cursor:pointer;font-size:13.5px;color:var(--ink);font-weight:600;}}
summary:focus-visible{{outline:2px solid var(--c-v16);outline-offset:2px;}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--body);}}
.lg{{display:flex;align-items:center;gap:6px;}}
.lg i{{width:11px;height:3px;border-radius:2px;display:inline-block;}}
footer{{border-top:1px solid var(--line);padding-top:20px;font-size:13px;color:var(--muted);max-width:72ch;}}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important;}}}}
</style>
<div class="wrap">
 <header>
  <div class="eyebrow">TrackFormer · forecast skill</div>
  <h1>What each model actually predicts, hour by hour</h1>
  <p class="lede">Every model scored on the same set: Western Pacific and Eastern Pacific storms from
  2020 on, restricted to the {len(LEADS)}-lead full horizon — 3,763 windows — using the
  <strong>repaired</strong> reanalysis fields. Track error is cumulative; the other four are mean
  absolute error at each lead.</p>
 </header>

 <section>
  <div class="sec-head"><div class="eyebrow">Lineage</div>
   <h2>v10 to v16, across all five predicted quantities</h2>
   <p class="lede">Track is the headline number, but the model predicts intensity and storm size too —
   and those tell a different story. Lines are direct-labelled; hover any chart for exact values.</p></div>
  <div class="legend">{lineage_legend}</div>
  <div class="panels">{''.join(lineage_panels)}</div>
  <details><summary>Lineage data table</summary><div class="tablewrap">{''.join(lineage_tables)}</div></details>
 </section>
{abl_html}
 <footer>
  <p>Measured {os.popen('date +%Y-%m-%d').read().strip()} from the saved checkpoints. v10/v13/v14/v14.1
  scored locally; v16 and its ablation on the Colab L4. All figures use the repaired steering and SLP
  fields, in which windows with no reanalysis timestep inside tolerance are marked unavailable rather
  than silently snapped to the nearest one.</p>
 </footer>
</div>
<script>
document.querySelectorAll('.panel').forEach(function(p){{
  var svg=p.querySelector('svg'); if(!svg) return;
  var tip=document.createElement('div'); tip.className='tip'; p.appendChild(tip);
  var cross=svg.querySelector('.cross');
  svg.querySelectorAll('.hit').forEach(function(h){{
    h.addEventListener('mouseenter',function(){{
      var rows=h.getAttribute('data-rows').replace(/'/g,'"');
      tip.innerHTML='<b>+'+h.getAttribute('data-h')+' h</b>'+
        rows.replace(/<span class="k">/g,'<div class="row"><span class="k">')
            .replace(/<\\/span><span class="v">/g,'</span><span class="v">')
            .replace(/<\\/span>(?=<span class="k">|$)/g,'</span></div>');
      tip.style.opacity=1;
      var r=h.getBoundingClientRect(), pr=p.getBoundingClientRect();
      var x=r.left-pr.left+r.width/2, left=x+12;
      if(left+150>pr.width) left=x-150;
      tip.style.left=Math.max(4,left)+'px'; tip.style.top='30px';
      var bb=h.getBBox(); cross.setAttribute('x1',bb.x+bb.width/2); cross.setAttribute('x2',bb.x+bb.width/2);
      cross.style.opacity=.6;
    }});
  }});
  svg.addEventListener('mouseleave',function(){{tip.style.opacity=0;cross.style.opacity=0;}});
}});
</script>"""

os.makedirs("paper", exist_ok=True)
open("paper/model_metrics.html", "w").write(HTML)
print(f"wrote paper/model_metrics.html ({len(HTML)/1000:.0f} KB)"
      + ("" if V16 else "  [ablation section pending — perlead_v16.json not found]"))
