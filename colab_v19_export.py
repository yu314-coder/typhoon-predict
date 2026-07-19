"""Standalone v17 track export -- runs in the Colab TERMINAL on CPU, so it does not touch the
GPU or interrupt a training cell. Rebuilds everything from files; uses no notebook globals.

    cd /content && wget -q -O s17.py <raw-url> && python3 s17.py
"""
import re, os, sys, json, math, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
DEVICE = torch.device("cpu")
STEER_CLIP, STEER_DROP = 4.0, 0.0

# The v17 class differs from train_track_v14_1.py (adapter is Linear(d,d) not Linear(d,64), and
# int_dec has 3 layers not 5), so the definition is taken from the notebook that produced the
# checkpoints rather than assumed.
nb_path = "/content/_v17.ipynb"
if not os.path.exists(nb_path):
    urllib.request.urlretrieve(f"{RAW}/colab_train_v17.ipynb", nb_path)
import json as _json
_nb = _json.load(open(nb_path))
src = "\n".join("".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code")
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
     "DEVICE": DEVICE, "STEER_DROP": STEER_DROP, "STEER_CLIP": STEER_CLIP}
for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
            r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
            r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    m = re.search(pat, src, re.S)
    assert m, f"pattern not found: {pat[:40]}"
    exec(m.group(0), G)
Net = G["TrackFormerV17"]

z = np.load("/content/d/track_windows_v13.npz", allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
sid = z["storm_id"].astype(str); nl = z["n_leads"].astype(int)
bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
tmean = z["track_mean"].astype("float32"); tstd = z["track_std"].astype("float32")
_q = np.load("/content/d/steer5_int8.npz")
SLP = np.clip(_q["q"][:, :4].astype("float32") / 31.75, -STEER_CLIP, STEER_CLIP)
del _q
v0 = track[:, -1, 2:4] * tstd[2:4] + tmean[2:4]
vp = track[:, -2, 2:4] * tstd[2:4] + tmean[2:4]
vpair = np.concatenate([v0, vp], 1).astype("float32")
SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12)

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DRV = "/content/drive/MyDrive/typhoon"
CK = [f"/content/v19_seed{i}.pt" for i in range(NS)]
CK = [c if os.path.exists(c) else f"{DRV}/{os.path.basename(c)}" for c in CK]
CK = [c for c in CK if os.path.exists(c)]
assert CK, "no v19 checkpoints found"
M = []
for c in CK:
    sd = torch.load(c, map_location="cpu", weights_only=False)["model"]
    sd = {k[len("inner."):]: v for k, v in sd.items() if k.startswith("inner.")}
    m = Net(); m.load_state_dict(sd); m.eval(); M.append(m)

# v19's intensity head predicts a DELTA on the current observed values -- add it back
INT_SRC = [4, 5, 7] + list(range(8, 20))
_SRC = torch.tensor(INT_SRC)
_tmn = torch.tensor(tmean); _tsd = torch.tensor(tstd)

def int_base(tr):
    cur = tr[:, -1, :][:, _SRC] * _tsd[_SRC] + _tmn[_SRC]
    cur = torch.where(cur > 0, cur, torch.zeros_like(cur))
    return (cur / SC[2:]).unsqueeze(1).expand(-1, 20, -1)
print(f"v19: {len(M)} seeds on CPU  <- {[os.path.basename(c) for c in CK]}", flush=True)

R = 111.2
STORMS = [("2026182N09163", "Bavi"), ("1986228N19120", "Wayne"),
          ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]

@torch.no_grad()
def predict(k):
    out = []
    for i in range(0, len(k), 64):
        j = k[i:i + 64]
        tr_ = torch.from_numpy(track[j])
        s = torch.stack([mm(tr_, torch.from_numpy(vpair[j]),
                            torch.from_numpy(SLP[j]))[0] for mm in M]).mean(0)
        s = torch.cat([s[..., :2], s[..., 2:] + int_base(tr_)], -1)
        out.append((s * SC).numpy())
    return np.concatenate(out)

def sp_bear(E, N):
    return np.hypot(E, N) / 6.0, (np.degrees(np.arctan2(E, N)) + 360) % 360

tracks_out, series_out = {}, {}
for s, nm in STORMS:
    k = np.where((sid == s) & (nl == 20))[0]; k = k[np.argsort(bt[k])]
    P = predict(k)
    cE, cN = np.cumsum(P[..., 0], 1), np.cumsum(P[..., 1], 1)
    T = target[k]; tE, tN = np.cumsum(T[..., 0], 1), np.cumsum(T[..., 1], 1)
    lats, lons = [], []
    for a in range(len(k)):
        la = bla[k[a]] + cN[a] / R
        lo = blo[k[a]] + cE[a] / (R * np.cos(np.radians((bla[k[a]] + la) / 2)))
        lats.append(np.round(la, 3).tolist()); lons.append(np.round(lo, 3).tolist())
    err = float(np.hypot(cE[:, 19] - tE[:, 19], cN[:, 19] - tN[:, 19]).mean())
    tracks_out[nm] = {"lat": lats, "lon": lons, "base_time": bt[k].tolist(),
                      "base_lat": np.round(bla[k], 3).tolist(),
                      "base_lon": np.round(blo[k], 3).tolist(),
                      "err120_mean": err, "n": int(len(k))}
    psp, pbr = sp_bear(P[0, :, 0], P[0, :, 1])
    series_out[nm] = {"lat": lats[0], "lon": lons[0],
                      "vmax": P[0, :, 2].tolist(), "pressure": P[0, :, 3].tolist(),
                      "rmw": P[0, :, 4].tolist(), "speed": psp.tolist(), "bearing": pbr.tolist(),
                      "err120": float(np.hypot(cE[0, 19] - tE[0, 19], cN[0, 19] - tN[0, 19]))}
    print(f"{nm:11s} {len(k):3d} windows | v19 mean 120h {err:6.0f} km", flush=True)

json.dump(tracks_out, open("/content/v19_tracks.json", "w"))
json.dump(series_out, open("/content/v19_series.json", "w"))
print("wrote /content/v19_tracks.json and /content/v19_series.json")
