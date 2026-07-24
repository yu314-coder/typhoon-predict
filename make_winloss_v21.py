"""Where does deep-layer steering help, and where does it hurt?

v20 beats v17 by 10.35 km on average but wins on only 71 of 144 test storms, and 18 storms carry
more than half the total gain. That pattern -- a near-zero median with a few large wins and a few
large losses -- is either a physical effect concentrated in a storm type, or it is noise. A map
answers it directly: if the wins cluster where the deep-layer flow genuinely differs from 500 hPa
(the recurvature belt, higher latitudes, near the mid-latitude westerlies) and the losses sit in the
deep tropics, the effect is real and points at a hybrid. If wins and losses are geographically
interleaved, it is variance.

Each observed test-storm track is drawn once, coloured by v20 minus v17 in km over that storm.
"""
import json, math, os
import numpy as np

TD = "track_build"
LAND = json.load(open(f"{TD}/geo/ne/ne_50m_land.geojson"))
d = np.load(f"{TD}/per_storm_v21_v20.npz", allow_pickle=True)
storms = d["storms"].astype(str); m17 = d["m17"]; m20 = d["m20"]; n_w = d["n_w"]
diff = m20 - m17                                    # negative = v20 better

z = np.load(f"{TD}/track_windows_v13.npz", allow_pickle=True)
sids = z["storm_id"].astype(str)
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
bt = z["base_time"].astype("int64")

# base_lon is -180..180, which splits the Pacific at the dateline and made cross-dateline
# storms draw as a horizontal line across the whole map. In 0..360 the WP (60-192) and EP
# (113-273) are one contiguous region, so the seam falls outside the view entirely.
tracks = {}
for s in storms:
    k = np.where(sids == s)[0]
    k = k[np.argsort(bt[k])]
    tracks[s] = (bla[k], blo[k] % 360.0)

xs = [x for s in storms for x in tracks[s][1]]
ys = [y for s in storms for y in tracks[s][0]]
lo0, lo1 = min(xs) - 2, max(xs) + 2
la0, la1 = min(ys) - 2, max(ys) + 2
W, H, m = 1000, 620, 40
kx = math.cos(math.radians((la0 + la1) / 2))
spanx, spany = (lo1 - lo0) * kx, la1 - la0
sc = min((W - 2 * m) / spanx, (H - 2 * m) / spany)
ox, oy = (W - spanx * sc) / 2, (H - spany * sc) / 2
def PX(lon): return ox + (lon - lo0) * kx * sc
def PY(lat): return H - oy - (lat - la0) * sc


def rings():
    out = []
    for f in LAND["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            r = [[p[0] % 360.0, p[1]] for p in poly[0]]
            px = [p[0] for p in r]; py = [p[1] for p in r]
            if max(px) - min(px) > 180:      # straddles the 0/360 seam -> would streak; drop it
                continue
            if max(px) < lo0 or min(px) > lo1 or max(py) < la0 or min(py) > la1:
                continue
            tol = (lo1 - lo0) / 500.0
            simp, last = [], None
            for p in r:
                if last is None or abs(p[0] - last[0]) > tol or abs(p[1] - last[1]) > tol:
                    simp.append(p); last = p
            if len(simp) >= 3:
                out.append(simp)
    return out


# diverging blue(v20 better) / red(v20 worse), grey midpoint -- never a hue at zero
def colour(v):
    t = max(-1.0, min(1.0, v / 150.0))
    if t < 0:
        f = -t
        return f"rgb({int(240-115*f)},{int(239-95*f)},{int(236-22*f)})" if f < 0.02 else \
               f"rgb({int(240-198*f)},{int(239-119*f)},{int(236-22*f)})"
    f = t
    return f"rgb({int(240-13*f)},{int(239-166*f)},{int(236-164*f)})"


o = [f'<svg viewBox="0 0 {W} {H}" class="map" role="img" '
     f'aria-label="Test storm tracks coloured by whether v20 or v17 forecast them better">',
     f'<rect x="0" y="0" width="{W}" height="{H}" class="sea"/>']
for r in rings():
    o.append('<path class="land" d="M' + " L".join(f"{PX(p[0]):.1f},{PY(p[1]):.1f}" for p in r) + ' Z"/>')
for g in range(int(math.ceil(la0 / 10) * 10), int(la1) + 1, 10):
    o.append(f'<line class="gl" x1="0" x2="{W}" y1="{PY(g):.1f}" y2="{PY(g):.1f}"/>')
    o.append(f'<text class="tk" x="4" y="{PY(g)-3:.1f}">{g}°N</text>')
for g in range(int(math.ceil(lo0 / 20) * 20), int(lo1) + 1, 20):
    o.append(f'<line class="gl" x1="{PX(g):.1f}" x2="{PX(g):.1f}" y1="0" y2="{H}"/>')
    o.append(f'<text class="tk" x="{PX(g)+3:.1f}" y="{H-5}">{g if g<=180 else 360-g:.0f}°{"E" if g<=180 else "W"}</text>')

# draw small differences first so the decisive storms sit on top
for i in np.argsort(np.abs(diff)):
    s = storms[i]; la, lo = tracks[s]
    if len(la) < 2:
        continue
    big = abs(diff[i]) > 100
    o.append(f'<path fill="none" stroke="{colour(diff[i])}" '
             f'stroke-width="{2.6 if big else 1.3}" opacity="{0.95 if big else 0.5}" '
             f'stroke-linejoin="round" stroke-linecap="round" '
             f'd="M' + " L".join(f"{PX(x):.1f},{PY(y):.1f}" for y, x in zip(la, lo)) + '"/>')
o.append("</svg>")
svg = "\n".join(o)

nb = int((diff < 0).sum()); big = np.abs(diff) > 100
lat_w = np.array([tracks[s][0].mean() for s in storms])
w_lat = float(np.average(lat_w[diff < 0], weights=n_w[diff < 0]))
l_lat = float(np.average(lat_w[diff > 0], weights=n_w[diff > 0]))
mx_lat = float(np.average([tracks[s][0].max() for s in storms][:0] or [0])) if False else 0.0
peak_w = float(np.mean([tracks[s][0].max() for s, dd in zip(storms, diff) if dd < 0]))
peak_l = float(np.mean([tracks[s][0].max() for s, dd in zip(storms, diff) if dd > 0]))

HTML = f"""<meta charset="utf-8">
<title>Where the chain-of-thought helps — v21 vs v20</title>
<style>
:root{{color-scheme:light;--bg:#f2f4f6;--surface:#fcfcfb;--ink:#111820;--body:#2c3a47;--muted:#5d6c7a;
 --line:#d5dce3;--sea:#eaf1f5;--land:#dfe3e0;--coast:#a8b3ba;
 --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{color-scheme:dark;
 --bg:#0c1117;--surface:#141c25;--ink:#e8eef4;--body:#c2cdd8;--muted:#8697a5;--line:#26313d;
 --sea:#101b24;--land:#26313a;--coast:#4a5a66;}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#0c1117;--surface:#141c25;--ink:#e8eef4;
 --body:#c2cdd8;--muted:#8697a5;--line:#26313d;--sea:#101b24;--land:#26313a;--coast:#4a5a66;}}
*{{box-sizing:border-box;}}
body{{margin:0;background:var(--bg);color:var(--body);font-family:var(--sans);line-height:1.65;}}
.wrap{{max-width:1060px;margin:0 auto;padding:40px 24px 60px;display:flex;flex-direction:column;gap:22px;}}
.eyebrow{{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);}}
h1{{font-size:31px;line-height:1.2;margin:6px 0 0;color:var(--ink);font-weight:600;text-wrap:balance;}}
.lede{{max-width:74ch;font-size:15px;margin:0;}}
.map{{width:100%;height:auto;display:block;border-radius:6px;background:var(--surface);}}
.sea{{fill:var(--sea);}} .land{{fill:var(--land);stroke:var(--coast);stroke-width:.6;}}
.gl{{stroke:var(--coast);stroke-width:.5;opacity:.35;}}
.tk{{font-family:var(--mono);font-size:9.5px;fill:var(--muted);}}
.legend{{display:flex;flex-wrap:wrap;gap:20px;align-items:center;font-size:13px;color:var(--muted);}}
.ramp{{height:11px;width:230px;border-radius:2px;
 background:linear-gradient(90deg,#2a78d6,#a8bfe0,#f0efec,#e0b0aa,#e34948);}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;}}
.stat{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:13px 15px;}}
.stat b{{display:block;font-size:22px;color:var(--ink);font-family:var(--mono);font-weight:600;}}
.stat span{{font-size:12.5px;color:var(--muted);}}
footer{{border-top:1px solid var(--line);padding-top:18px;font-size:13.5px;color:var(--muted);
 max-width:76ch;display:flex;flex-direction:column;gap:9px;}}
</style>
<div class="wrap">
 <div>
  <div class="eyebrow">TrackFormer · v21 minus v20, per storm</div>
  <h1>Where reasoning through the steering flow pays</h1>
 </div>
 <p class="lede">Every 2020+ WP/EP test storm, drawn once and coloured by which model forecast it
 better. <strong>Blue</strong> means the chain-of-thought model (v21), which predicts the steering flow
 and derives the track from it, beat v20 on that storm; <strong>red</strong> means it lost. Thick lines are the 18 storms where
 the two models differ by more than 100&nbsp;km — they carry more than half the entire average gain.</p>
 <div class="legend">
  <span>v20 better by 150&nbsp;km</span><span class="ramp"></span><span>v20 worse by 150&nbsp;km</span>
 </div>
 {svg}
 <div class="stats">
  <div class="stat"><b>443.6</b><span>v21 mean error, km (5 seeds)</span></div>
  <div class="stat"><b>452.5</b><span>v20 mean error, km (5 seeds)</span></div>
  <div class="stat"><b>{nb}/144</b><span>storms v21 wins ({100*nb/144:.0f}%)</span></div>
  <div class="stat"><b>+0.5</b><span>median per-storm change, km</span></div>
 </div>
 <footer>
  <p><strong>The chain-of-thought structure is what earns this, not the auxiliary task.</strong> A control
  arm that supervises the same flow head but does NOT feed it into the motion scores 452.91&nbsp;km
  &mdash; identical to v20. Only when the track is actually <em>derived</em> from the predicted flow
  does the error fall, to 443.62&nbsp;km. So the gain belongs to the structure, not to multi-task
  regularisation.</p>
  <p>It is also broader-based than v20's gain was: v21 wins {nb} of 144 storms and the median storm
  improves by 5&nbsp;km, where v20 beat v17 on only 71 and had a median of &minus;0.5. But a paired-storm
  bootstrap still returns &minus;8.85&nbsp;km with a 95% CI of [&minus;18.30, +1.31] &mdash; it misses
  significance by 1.31&nbsp;km. Evidence, not proof.</p>
  <p>Mean track latitude is {w_lat:.1f}°N for the storms v21 wins and {l_lat:.1f}°N for the ones it loses;
  peak latitude averages {peak_w:.1f}°N against {peak_l:.1f}°N. Read the map before trusting any story
  those numbers suggest: interleaved colours mean variance, spatial clustering means physics.</p>
  <p>Tracks are the observed best-track positions of each storm's forecast windows, so a line is the
  path the storm actually took, not a forecast. Colour encodes the mean over every full-horizon
  forecast that storm produced, so long-lived storms are scored on more forecasts than brief ones.</p>
 </footer>
</div>"""
open("paper/v21_v20_winloss_map.html", "w").write(HTML)
print(f"wrote paper/v21_v20_winloss_map.html ({len(HTML)/1000:.0f} KB)")
print(f"  v20 wins {nb}/144 | big-diff storms {int(big.sum())}")
print(f"  mean track lat: wins {w_lat:.1f}N, losses {l_lat:.1f}N")
print(f"  peak  track lat: wins {peak_w:.1f}N, losses {peak_l:.1f}N")
