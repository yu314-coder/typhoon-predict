"""Add Bavi's v10 entry to track_build/v10_tracks.json, MERGING (not overwriting) -- Noul's entry
was added separately by _noul_predict.py and must survive. _v10tracks.py's own STORMS list
already includes Bavi but the checked-in v10_tracks.json predates that and was never rerun; this
is the minimal fix instead of blindly rerunning a script that would clobber Noul.
"""
import re, math, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def build(fn):
    src = open(fn).read()
    g = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
         "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
    for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(pat, src, re.S).group(0), g)
    return g["TrackFormerV9"]


z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
sid = z["storm_id"].astype(str); nl = z["n_leads"].astype(int); bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
tm = z["track_mean"].astype("float32"); ts = z["track_std"].astype("float32")
v0 = track[:, -1, 2:4] * ts[2:4] + tm[2:4]; vp = track[:, -2, 2:4] * ts[2:4] + tm[2:4]
vpair = np.concatenate([v0, vp], 1).astype("float32"); SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12)
m = build("train_track_v10.py")()
m.load_state_dict(torch.load("track_build/track_v10_best.pt", map_location="cpu", weights_only=False)["model"]); m.eval()
R = 111.2


def to_ll(lat0, lon0, cE, cN):
    lat = lat0 + cN / R; lon = lon0 + cE / (R * np.cos(np.radians((lat0 + lat) / 2))); return lat, lon


s, nm = "2026182N09163", "Bavi"
k = np.where((sid == s) & (nl == 20))[0]; k = k[np.argsort(bt[k])]
assert len(k), "Bavi not found in track_windows_v13.npz"
with torch.no_grad():
    sv, _ = m(torch.from_numpy(track[k]), torch.from_numpy(vpair[k]))
P = (sv * SC).numpy()
cE, cN = np.cumsum(P[..., 0], 1), np.cumsum(P[..., 1], 1)
lats, lons, errs = [], [], []
for a in range(len(k)):
    la, lo = to_ll(bla[k[a]], blo[k[a]], cE[a], cN[a])
    lats.append(np.round(la, 3).tolist()); lons.append(np.round(lo, 3).tolist())
    tE, tN = np.cumsum(target[k[a], :, 0]), np.cumsum(target[k[a], :, 1])
    errs.append(float(np.hypot(cE[a, 19] - tE[19], cN[a, 19] - tN[19])))

p = "track_build/v10_tracks.json"
out = json.load(open(p))
out[nm] = {"widx": k.tolist(), "lat": lats, "lon": lons,
           "base_time": bt[k].tolist(), "base_lat": np.round(bla[k], 3).tolist(),
           "base_lon": np.round(blo[k], 3).tolist(),
           "err120_mean": float(np.mean(errs)), "n": int(len(k))}
json.dump(out, open(p, "w"))
print(f"{nm:10s} {len(k):3d} windows | v10 mean 120h err {np.mean(errs):6.0f} km -- merged into {p}")
