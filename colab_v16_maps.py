"""Real-world maps of every forecast for the four test storms, v10 vs v16.

Run INSIDE the finished colab_train_v16 session (after the ablation, so the SST channel is back):

    !wget -q -O maps.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v16_maps.py
    exec(open('maps.py').read())

v10 has no checkpoint on this machine, so its tracks are precomputed locally and fetched as JSON;
v16 is evaluated here from CKPTS. Both are drawn the same way, from every full-horizon window.

The MEAN track is averaged by VALID TIME, not by lead. Forecasts launched at different hours are
only comparable when they describe the same moment, and averaging by lead would smear together
positions hours apart and invent a track the model never predicted.
"""
import json, math, os, urllib.request
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"
for url, dst in [(f"{RAW}/track_build/v10_tracks.json", "v10_tracks.json"), (NE, "ne_land.geojson")]:
    if not os.path.exists(dst):
        urllib.request.urlretrieve(url, dst)
V10 = json.load(open("v10_tracks.json"))
LAND = json.load(open("ne_land.geojson"))
print(f"v10 tracks for {len(V10)} storms | {len(LAND['features'])} land features", flush=True)

R = 111.2
sid_all = z["storm_id"].astype(str)
nl_all = z["n_leads"].astype(int)
bt_all = z["base_time"].astype("int64")
bla_all = z["base_lat"].astype("float64")
blo_all = z["base_lon"].astype("float64")
STORMS = [("2026182N09163", "Bavi", "2026"), ("1986228N19120", "Wayne", "1986"),
          ("2025203N20124", "Co-may", "2025"), ("2022239N22150", "Hinnamnor", "2022")]
COL = {"v10": "#4a3aa7", "v16": "#2a78d6"}


def to_ll(lat0, lon0, cE, cN):
    lat = lat0 + cN / R
    lon = lon0 + cE / (R * np.cos(np.radians((lat0 + lat) / 2)))
    return lat, lon


@torch.no_grad()
def v16_tracks(k):
    """Ensemble-mean prediction from the 3 v16 seeds, for window indices k."""
    P = []
    for i in range(0, len(k), 128):
        j = k[i:i + 128]
        tr = torch.from_numpy(track[j]).to(DEVICE)
        vp = torch.from_numpy(vpair[j]).to(DEVICE)
        sp = torch.from_numpy(SLP[j]).to(DEVICE)
        s = torch.stack([mm(tr, vp, sp)[0] for mm in models]).mean(0)
        P.append((s * TARGET_SCALE).float().cpu().numpy())
    P = np.concatenate(P)
    return np.cumsum(P[..., 0], 1), np.cumsum(P[..., 1], 1)


def mean_by_valid_time(bts, lats, lons):
    """Average forecast positions that describe the SAME moment. Returns (t, lat, lon) sorted."""
    acc = {}
    for w, bt in enumerate(bts):
        for L in range(20):
            vt = int(bt) + int((L + 1) * 6 * 3600 * 1e9)
            a = acc.setdefault(vt, [0.0, 0.0, 0])
            a[0] += lats[w][L]; a[1] += lons[w][L]; a[2] += 1
    ts = sorted(acc)
    return (np.array(ts),
            np.array([acc[t][0] / acc[t][2] for t in ts]),
            np.array([acc[t][1] / acc[t][2] for t in ts]))


def draw_land(ax, lo0, lo1, la0, la1):
    for f in LAND["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.asarray(poly[0], dtype="float64")
            if ring[:, 0].max() < lo0 - 3 or ring[:, 0].min() > lo1 + 3: continue
            if ring[:, 1].max() < la0 - 3 or ring[:, 1].min() > la1 + 3: continue
            ax.add_patch(PathPatch(Path(ring), facecolor="#dfe3e0", edgecolor="#9aa5ac",
                                   lw=.6, zorder=1))


fig, axes = plt.subplots(4, 2, figsize=(12.4, 21))
plt.rcParams.update({"font.size": 9})
summary = []
for row, (s, nm, yr) in enumerate(STORMS):
    k = np.where((sid_all == s) & (nl_all == 20))[0]
    k = k[np.argsort(bt_all[k])]
    obs_lat, obs_lon = bla_all[k], blo_all[k]          # each window's base IS an observed fix
    tE = np.cumsum(target[k][..., 0], 1); tN = np.cumsum(target[k][..., 1], 1)
    cE16, cN16 = v16_tracks(k)
    lat16 = np.empty_like(cE16); lon16 = np.empty_like(cE16)
    for a in range(len(k)):
        lat16[a], lon16[a] = to_ll(bla_all[k[a]], blo_all[k[a]], cE16[a], cN16[a])
    err16 = np.hypot(cE16[:, 19] - tE[:, 19], cN16[:, 19] - tN[:, 19]).mean()
    lat10 = np.asarray(V10[nm]["lat"]); lon10 = np.asarray(V10[nm]["lon"])
    err10 = V10[nm]["err120_mean"]
    summary.append((nm, err10, err16, len(k)))

    for col, (tag, LAT, LON, err) in enumerate([("v10", lat10, lon10, err10),
                                                ("v16", lat16, lon16, err16)]):
        ax = axes[row, col]
        lo0 = min(LON.min(), obs_lon.min()); lo1 = max(LON.max(), obs_lon.max())
        la0 = min(LAT.min(), obs_lat.min()); la1 = max(LAT.max(), obs_lat.max())
        px, py = (lo1 - lo0) * .08 + 1.2, (la1 - la0) * .08 + 1.2
        lo0, lo1, la0, la1 = lo0 - px, lo1 + px, la0 - py, la1 + py
        ax.set_facecolor("#eef4f7")
        draw_land(ax, lo0, lo1, la0, la1)
        for a in range(len(LAT)):
            ax.plot(LON[a], LAT[a], color=COL[tag], lw=.7, alpha=.28, zorder=2, solid_capstyle="round")
        _, mla, mlo = mean_by_valid_time(bt_all[k], LAT, LON)
        ax.plot(mlo, mla, color=COL[tag], lw=2.6, zorder=4, label=f"{tag} mean (by valid time)",
                solid_capstyle="round")
        ax.plot(obs_lon, obs_lat, color="#11181f", lw=2.4, ls=(0, (1, 1.9)), zorder=5, label="observed")
        ax.plot(obs_lon[0], obs_lat[0], "o", ms=7, mfc="#fcfcfb", mec="#11181f", mew=1.8, zorder=6)
        ax.set_xlim(lo0, lo1); ax.set_ylim(la0, la1)
        ax.set_aspect(1 / math.cos(math.radians((la0 + la1) / 2)))
        ax.grid(color="#c6d2da", lw=.5, alpha=.8, zorder=0)
        ax.set_title(f"{nm} ({yr}) — {tag}   ·   {len(LAT)} forecasts   ·   mean 120 h error {err:.0f} km",
                     fontsize=10, loc="left", pad=6)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("longitude (°E)", fontsize=8); ax.set_ylabel("latitude (°N)", fontsize=8)
        ax.legend(loc="best", fontsize=8, framealpha=.9)
        for sp in ax.spines.values(): sp.set_color("#b9c4cc")

fig.suptitle("Every full-horizon forecast for the four test storms — v10 (no environmental field) "
             "vs v16 (steering + observed SST)\nThin lines are individual initialisations; bold is the "
             "mean by valid time; dotted black is observed.", fontsize=11.5, x=.012, ha="left", y=.998)
fig.tight_layout(rect=[0, 0, 1, .985])
fig.savefig("/content/v16_storm_maps.png", dpi=125, bbox_inches="tight", facecolor="#ffffff")
try: fig.savefig(f"{DATA}/v16_storm_maps.png", dpi=125, bbox_inches="tight", facecolor="#ffffff")
except Exception as e: print("(drive copy failed:", e, ")")
plt.show()

print("\n" + "=" * 60)
print(f"{'storm':12s} {'n':>4s} {'v10':>9s} {'v16':>9s} {'delta':>9s}")
for nm, a, b, n in summary:
    print(f"{nm:12s} {n:4d} {a:9.0f} {b:9.0f} {b-a:+9.0f}")
print("=" * 60)
print("mean 120 h track error, km, averaged over every full-horizon window")
print("saved /content/v16_storm_maps.png")
