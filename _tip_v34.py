"""Add v34 (MoE gate on frozen v23_seed0 backbone) to Typhoon Tip (1979), same real steering
pipeline _tip_v33.py used for v31/v33 (tip_dlm4.npy: real SLPanom/SLPtend + real deep-layer-mean
u/v). v23/v31/v33 tracks already exist in track_build/{tag}_tracks.json from that run -- this only
computes and appends v34, using the same terrain_wp.npz coastline grid.
"""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from make_rmt_tracks import observed_at, SIX_H

torch.set_num_threads(8); DEVICE = torch.device("cpu"); R = 111.2; KM6H = 6 * 3600 / 1000.0

nb = json.load(open("colab_train_v17.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os, "DEVICE": DEVICE,
     "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p, src, re.S).group(0), G)
Base = G["TrackFormerV17"]
DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

v28src = open("colab_v28_train.py").read()
hs_src = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28src, re.S).group(0)
tf_src = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28src, re.S).group(0)
g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(hs_src, g23); exec(tf_src, g23)
V23 = g23["TrackFormerHist"]

v34src = open("colab_v39_train.py").read()
gate_src = re.search(r"class LandGate\(nn\.Module\):.*?\n        return g \* raw\n", v34src, re.S).group(0)
tfm_src = re.search(r"class TrackFormerMoELand\(V23\):.*?\n        return s, ls, fp\n", v34src, re.S).group(0)
g34 = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "A_MAX": 1.0,
       "GATE_BIAS_INIT": -5.0, "USE_LAND": 1}
exec(gate_src, g34); exec(tfm_src, g34)
TrackFormerMoELand = g34["TrackFormerMoELand"]


def load(ck, cls):
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("inner.")} or sd
    m = cls().eval(); m.load_state_dict(sd)
    return m


z = np.load("track_build/tip_fixed.npz", allow_pickle=True)
tr = z["track"].astype("float32"); tgt = z["target"].astype("float32"); nl = z["n_leads"].astype(int)
bt = z["base_time"].astype("int64"); bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
o13 = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
tm, ts = o13["track_mean"].astype("float32"), o13["track_std"].astype("float32")
vp = np.concatenate([tr[:, -1, 2:4] * ts[2:4] + tm[2:4], tr[:, -2, 2:4] * ts[2:4] + tm[2:4]], 1).astype("float32")
S20 = np.load("track_build/tip_dlm4.npy").astype("float32")
SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12); obs = observed_at(bt, bla, blo)
K = np.where(nl == 20)[0]; K = K[np.argsort(bt[K])]

SIX = int(6 * 3600 * 1e9)
_key = {int(bt[i]): i for i in range(len(bt))}
HIST = np.full((len(bt), 2), -1, np.int64)
for i in range(len(bt)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = _key.get(int(bt[i]) - back * SIX, -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(bt))[:, None])


def build_hist(j):
    h = np.zeros((len(j), 8, 17, 17), "float32")
    for c in range(2):
        idx = HIST_S[j, c]
        h[:, c * 4:(c + 1) * 4] = S20[idx]
    return h, HAVE[j]


_T = np.load("track_build/terrain_wp.npz")
T_LAT, T_LON, LSM, ELEV = _T["lat"], _T["lon"], _T["lsm"], _T["elev"]
_LAND = LSM > 0.5
LAND_LAT = T_LAT[np.where(_LAND)[0]]; LAND_LON = T_LON[np.where(_LAND)[1]]; LAND_ELEV = ELEV[_LAND]


def land_features_np(lat, lon, heading_rad):
    n = len(lat)
    dlat = LAND_LAT[None, :] - lat[:, None]
    dlon = (LAND_LON[None, :] - lon[:, None]) * np.cos(np.radians(lat[:, None]))
    dy = dlat * R; dx = dlon * R
    dist = np.hypot(dx, dy); bearing = np.arctan2(dx, dy)
    dphi = np.arctan2(np.sin(bearing - heading_rad[:, None]), np.cos(bearing - heading_rad[:, None]))
    cone = np.abs(dphi) <= np.radians(30.0)
    ahead = np.where(cone, dist, np.inf)
    dist_nearest = dist.min(1); dist_ahead = ahead.min(1)
    dist_ahead = np.where(np.isfinite(dist_ahead), dist_ahead, 3000.0)
    near_ahead = ahead < 500.0
    terr_ahead = np.array([LAND_ELEV[near_ahead[k]].max() if near_ahead[k].any() else 0.0
                            for k in range(n)])
    return np.stack([dist_nearest, dist_ahead, terr_ahead], 1).astype("float32")


def build_lf5(j):
    phi = np.arctan2(vp[j, 1], vp[j, 0])
    lf3 = land_features_np(bla[j], blo[j] % 360, phi)
    spd = np.hypot(vp[j, 0], vp[j, 1])
    latv = tr[j, -1, 48] * ts[48] + tm[48]
    lf5 = np.concatenate([lf3 / np.array([1000., 1000., 1000.], "float32"),
                          (spd / 100.0)[:, None].astype("float32"),
                          (np.abs(latv) / 30.0)[:, None].astype("float32")], 1)
    return lf5, np.ones(len(j), "float32")


import glob
M34 = [load(f, TrackFormerMoELand) for f in sorted(glob.glob("downloads/v34ck/**/v34_seed*.pt", recursive=True))]
print(f"v34: {len(M34)} seeds")


@torch.no_grad()
def full_storm(ms):
    P = []
    for i in range(0, len(K), 64):
        j = K[i:i + 64]
        h, hv = build_hist(j)
        lf5, lok = build_lf5(j)
        a = [torch.from_numpy(tr[j]), torch.from_numpy(vp[j]), torch.from_numpy(S20[j]),
             torch.from_numpy(h), torch.from_numpy(hv), torch.from_numpy(lf5), torch.from_numpy(lok)]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).numpy())
    A = np.concatenate(P); cE, cN = np.cumsum(A[..., 0], 1), np.cumsum(A[..., 1], 1)
    T = tgt[K]; tE, tN = np.cumsum(T[..., 0], 1), np.cumsum(T[..., 1], 1)
    return float(np.hypot(cE[:, 19] - tE[:, 19], cN[:, 19] - tN[:, 19]).mean())


print("Typhoon Tip 1979 -- 105 full-horizon forecasts, mean 120 h error")
print(f"  v34   (MoE gate, frozen v23_seed0)   {full_storm(M34):8.2f} km")

for tag, ms in (("v34", M34),):
    with torch.no_grad():
        A = []
        for i in range(0, len(K), 64):
            j = K[i:i + 64]
            h, hv = build_hist(j)
            lf5, lok = build_lf5(j)
            a = [torch.from_numpy(tr[j]), torch.from_numpy(vp[j]), torch.from_numpy(S20[j]),
                 torch.from_numpy(h), torch.from_numpy(hv), torch.from_numpy(lf5), torch.from_numpy(lok)]
            A.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).numpy())
    A = np.concatenate(A)
    cE, cN = np.cumsum(A[..., 0], 1), np.cumsum(A[..., 1], 1)
    tE, tN = np.cumsum(tgt[K][..., 0], 1), np.cumsum(tgt[K][..., 1], 1)
    lats, lons = [], []
    for a2 in range(len(K)):
        la = bla[K[a2]] + cN[a2] / R
        lo = blo[K[a2]] + cE[a2] / (R * np.cos(np.radians((bla[K[a2]] + la) / 2)))
        lats.append(np.round(la, 3).tolist()); lons.append(np.round(lo, 3).tolist())
    err = float(np.hypot(cE[:, 19] - tE[:, 19], cN[:, 19] - tN[:, 19]).mean())
    out = {"lat": lats, "lon": lons, "base_time": bt[K].tolist(),
           "base_lat": np.round(bla[K], 3).tolist(), "base_lon": np.round(blo[K], 3).tolist(),
           "err120_mean": err, "lead_h": 120, "n": int(len(K))}
    p = f"track_build/{tag}_tracks.json"
    d = json.load(open(p)) if os.path.exists(p) else {}
    d["Tip"] = out
    json.dump(d, open(p, "w"))
    print(f"added Tip to {p}  (mean 120h {err:.0f} km)")
