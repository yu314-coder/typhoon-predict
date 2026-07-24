"""Fair eval of v6 (v3 architecture, trained on 3.6x-expanded IBTrACS data) vs v3.

Evaluates ONLY on full-20-lead 2020+ windows (n_leads==20) so the test set matches v3 exactly.
Reports WP-only and all-basin metrics, and a paired storm-level bootstrap of the track difference.
"""
import re, math, json
import numpy as np
import torch
import torch.nn as nn

dev = torch.device("cpu")
TS = np.array([100., 100., 35., 20., 50.] + [50.] * 12, dtype="float32")


def build_v3():
    g = {"torch": torch, "nn": nn, "F": __import__("torch.nn.functional", fromlist=["x"]), "math": math, "np": np}
    src = open("train_track_v3.py").read()
    for blk in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM = len\(KIN_COLS\), len\(THERMO_COLS\)",
                r"def sinusoidal.*?return e", r"def encoder.*?return nn\.TransformerEncoder\(layer, depth\)",
                r"def decoder.*?return nn\.TransformerDecoder\(layer, depth\)",
                r"class TrackFormerV3.*?return state, logscale"]:
        exec(re.search(blk, src, re.S).group(0), g)
    return g["TrackFormerV3"]


def load(model_cls, ckpt):
    m = model_cls().to(dev).eval()
    sd = torch.load(ckpt, map_location=dev, weights_only=False)["model"]
    m.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()})
    return m


def metrics(m, z, idx, perwin=False):
    tmean, tstd = z["track_mean"].astype("float32"), z["track_std"].astype("float32")
    v0 = (z["track"][:, -1, 2:4] * tstd[2:4] + tmean[2:4]).astype("float32")
    trk, tgt, msk = z["track"].astype("float32"), z["target"], z["target_mask"].astype("float32")
    P = []
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            st, _ = m(torch.from_numpy(trk[b]), torch.from_numpy(v0[b]))
            P.append(st.numpy() * TS)
    P = np.concatenate(P); T = tgt[idx]; M = msk[idx]
    pt, tt = np.cumsum(P[..., :2], 1), np.cumsum(T[..., :2], 1)
    perr = np.sqrt(((pt - tt) ** 2).sum(-1)).mean(1)   # per-window mean over leads
    o = {"track": round(float(perr.mean()), 1)}
    for i, nm in [(2, "vmax"), (3, "pres"), (4, "rmw")]:
        v = M[..., i] > 0.5
        o[nm] = round(float(np.abs(P[..., i][v] - T[..., i][v]).mean()), 2)
    rm = M[..., 5:17] > 0.5
    o["radius"] = round(float(np.abs(P[..., 5:17] - T[..., 5:17])[rm].mean()), 2)
    return (o, perr) if perwin else o


V3 = build_v3()
zc = np.load("track_build/track_windows_v3data.npz", allow_pickle=True)
nl, yr, bs, sid = zc["n_leads"].astype(int), zc["year"].astype(int), zc["basin"].astype(str), zc["storm_id"].astype(str)
full_wp = np.where((nl == 20) & (yr >= 2020) & (bs == "WP"))[0]
full_ab = np.where((nl == 20) & (yr >= 2020))[0]
print(f"full-20-lead 2020+ test: WP {len(full_wp)}, all-basin {len(full_ab)}")

m6 = load(V3, "track_build/track_v6_best.pt")
print("\n=== v6 (expanded data) on full-20-lead 2020+ ===")
o_wp, perr6 = metrics(m6, zc, full_wp, perwin=True)
o_ab = metrics(m6, zc, full_ab)
print("WP-only    :", json.dumps(o_wp))
print("all-basin  :", json.dumps(o_ab))
print("\nv3 baseline: WP track 659.0/vmax 21.61/pres 18.06/rmw 11.81/radius 28.83 ; "
      "all-basin track 592.1/vmax 18.59/pres 14.61/rmw 14.87/radius 27.85")

# paired storm-bootstrap of v6 track vs v3's recorded 659 (need v3 per-window on same windows)
m3 = load(V3, "track_build/track_v3_best.pt")
# v3 was trained on the smaller npz but architecture identical; run it on these same windows
_, perr3 = metrics(m3, zc, full_wp, perwin=True)
diff = perr6.mean() - perr3.mean()
storms = sid[full_wp]; uniq = np.unique(storms); rng = np.random.RandomState(0); diffs = []
for _ in range(2000):
    pick = rng.choice(len(uniq), len(uniq), replace=True); mult = {u: 0 for u in uniq}
    for pi in pick:
        mult[uniq[pi]] += 1
    w = np.array([mult[s] for s in storms], float)
    diffs.append(np.average(perr6, weights=w) - np.average(perr3, weights=w))
diffs = np.array(diffs); lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\nWP track: v6 {perr6.mean():.1f} vs v3(rerun) {perr3.mean():.1f} | diff {diff:+.1f} km")
print(f"storm-bootstrap (v6-v3): mean {diffs.mean():+.1f}, 95% CI [{lo:+.1f},{hi:+.1f}], P(v6 better)={(diffs<0).mean():.3f}")
