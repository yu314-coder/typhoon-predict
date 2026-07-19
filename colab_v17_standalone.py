"""Standalone v17 track export -- runs in the Colab TERMINAL on CPU, so it does not touch the
GPU or interrupt a training cell. Rebuilds everything from files; uses no notebook globals.

    cd /content && wget -q -O s17.py <raw-url> && python3 s17.py
"""
import re, os, json, math, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
DEVICE = torch.device("cpu")
STEER_CLIP, STEER_DROP = 4.0, 0.0

# v17's architecture is byte-identical to train_track_v14_1.py's (4-channel steer CNN with
# Dropout2d, same curved baseline) -- so its class definition loads v17 state dicts directly.
src_path = "/content/_v141.py"
if not os.path.exists(src_path):
    urllib.request.urlretrieve(f"{RAW}/train_track_v14_1.py", src_path)
src = open(src_path).read()
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
     "DEVICE": DEVICE, "STEER_DROP": STEER_DROP, "STEER_CLIP": STEER_CLIP}
for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
            r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
            r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(pat, src, re.S).group(0), G)
Net = G["TrackFormerV9"]

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

DRV = "/content/drive/MyDrive/typhoon"
CK = [f"{DRV}/v17_seed{i}.pt" for i in range(5)]
CK = [c for c in CK if os.path.exists(c)]
assert CK, f"no v17 checkpoints under {DRV}"
M = []
for c in CK:
    m = Net(); m.load_state_dict(torch.load(c, map_location="cpu", weights_only=False)["model"])
    m.eval(); M.append(m)
print(f"v17: {len(M)} seeds on CPU", flush=True)

R = 111.2
STORMS = [("2026182N09163", "Bavi"), ("1986228N19120", "Wayne"),
          ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]

@torch.no_grad()
def predict(k):
    out = []
    for i in range(0, len(k), 64):
        j = k[i:i + 64]
        s = torch.stack([mm(torch.from_numpy(track[j]), torch.from_numpy(vpair[j]),
                            torch.from_numpy(SLP[j]))[0] for mm in M]).mean(0)
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
    print(f"{nm:11s} {len(k):3d} windows | v17 mean 120h {err:6.0f} km", flush=True)

json.dump(tracks_out, open("/content/v17_tracks.json", "w"))
json.dump(series_out, open("/content/v17_series.json", "w"))
print("wrote /content/v17_tracks.json and /content/v17_series.json")
