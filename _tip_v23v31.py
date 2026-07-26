"""Tip (1979) with v23 (temporal steering stack) and v31 (+ land-drag), beside v21/v20/v17/v10.

Tip is the other real landfall test in this repo (Honshu, Japan, ~Oct 19 1979) -- same idea as
Noul's landfall check, but on a storm the model has never seen either way (pre-satellite,
pre-dates all training data by decades) and with a MUCH better steering input: tip_dlm4.npy
carries real SLPanom/SLPtend (from tip_steer4.npy) AND real deep-layer-mean u/v (extracted by
extract_tip_dlm.py from 1979 NCEP daily reanalysis), all four channels normalised with the
TRAINING scale, never Tip's own statistics.

v23's t-12h/t-24h history comes from Tip's own record: the grid is 3-hourly (finer than the 6h
cadence used elsewhere), so historical channels are found by exact time-matching (bt[i] - 12h,
bt[i] - 24h) against tip_fixed.npz's own base_time array, not by a fixed step count. Genuinely
missing only near the very start of the record; falls back to the window's own current steering
with have=0, matching HistStem's actual training-time convention (self-repeat + an explicit
have-channel), not zero -- the same fix just applied to _noul_predict.py.

v31's land geometry is real: track_build/terrain_wp.npz, the same WP coastline grid v31 trained
on (covers 3-47N, 100-148E -- Tip's post-recurve track drifts outside that box to the north/east;
the distance-to-land calc still returns a real geographic distance to whatever land IS in the
grid, which is what v31 saw for any storm during training that left the box the same way, not a
bug introduced here).
"""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from make_rmt_tracks import observed_at, SIX_H

torch.set_num_threads(8); DEVICE = torch.device("cpu"); R = 111.2; KM6H = 6 * 3600 / 1000.0


def km(a1, o1, a2, o2):
    return math.hypot((o2 - o1) * R * math.cos(math.radians((a1 + a2) / 2)), (a2 - a1) * R)


nb = json.load(open("colab_train_v17.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os, "DEVICE": DEVICE,
     "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p, src, re.S).group(0), G)
Base = G["TrackFormerV17"]; KC, TC, EC = G["KIN_COLS"], G["THERMO_COLS"], G["ENV_COLS"]
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

v31src = open("colab_v36_train.py").read()
land_src = re.search(r"class LandDrag\(nn\.Module\):.*?\n        return self\.a_max \* torch\.tanh\(knots @ self\.Wi\.t\(\)\)\n", v31src, re.S).group(0)
tfl_src = re.search(r"class TrackFormerLand\(V23\):.*?\n        return s, ls, fp\n", v31src, re.S).group(0)
A_MAX = 0.65
g31 = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "A_MAX": A_MAX, "USE_LAND": 1}
exec(land_src, g31); exec(tfl_src, g31)
TrackFormerLand = g31["TrackFormerLand"]


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
S20 = np.load("track_build/tip_dlm4.npy").astype("float32")             # [N,4,17,17], real SLP+DLM
S4 = np.load("track_build/tip_steer4.npy").astype("float32")
S17 = np.clip(S4 / np.load("track_build/steer5_scale.npy")[:4][None, :, None, None], -4, 4).astype("float32")
SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12); obs = observed_at(bt, bla, blo)
K = np.where(nl == 20)[0]; K = K[np.argsort(bt[K])]

# ---- v23 temporal steering: exact time-matched t-12h/t-24h within Tip's own 3-hourly record.
# Missing only near the very start; falls back to the window's own current steering with have=0,
# HistStem's actual training-time convention (see module docstring).
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


# ---- real land geometry, same WP coastline grid v31 trained on ----
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
    return lf5, np.ones(len(j), "float32")   # Tip is 100% WP, no basin gate needed


@torch.no_grad()
def full_storm(ms, S, cls, use_hist=False, use_land=False):
    P = []
    for i in range(0, len(K), 64):
        j = K[i:i + 64]
        a = [torch.from_numpy(tr[j]), torch.from_numpy(vp[j]), torch.from_numpy(S[j])]
        if use_hist:
            h, hv = build_hist(j)
            a += [torch.from_numpy(h), torch.from_numpy(hv)]
        if use_land:
            lf5, lok = build_lf5(j)
            a += [torch.from_numpy(lf5), torch.from_numpy(lok)]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).numpy())
    A = np.concatenate(P); cE, cN = np.cumsum(A[..., 0], 1), np.cumsum(A[..., 1], 1)
    T = tgt[K]; tE, tN = np.cumsum(T[..., 0], 1), np.cumsum(T[..., 1], 1)
    return float(np.hypot(cE[:, 19] - tE[:, 19], cN[:, 19] - tN[:, 19]).mean())


import glob
M21 = [load(f"downloads/x/v21_seed{i}.pt", V21) for i in range(5)]
M20 = [load(f"downloads/x/v20_seed{i}.pt", Base) for i in range(5)]
M17 = [load(f"downloads/x/v17_seed{i}.pt", Base) for i in range(5)]
M23 = [load(f, V23) for f in sorted(glob.glob("downloads/x/v23_seed*.pt"))]
M31 = [load(f, TrackFormerLand) for f in sorted(glob.glob("downloads/v31ck/**/v31_seed*.pt", recursive=True))]

print(f"steering inputs differ: {float(np.abs(S17 - S20).max())} max abs diff")
print(f"models: v21 {len(M21)} | v20 {len(M20)} | v17 {len(M17)} | v23 {len(M23)} | v31 {len(M31)} seeds")
print("Typhoon Tip 1979 -- 105 full-horizon forecasts, mean 120 h error")
print(f"  v10   (no fields)            1250 km   [from the earlier export]")
print(f"  v17   (500 hPa steering)   {full_storm(M17, S17, Base):8.2f} km")
print(f"  v20   (deep-layer mean)    {full_storm(M20, S20, Base):8.2f} km")
print(f"  v21   (chain-of-thought)   {full_storm(M21, S20, V21):8.2f} km")
print(f"  v23   (+temporal steer)    {full_storm(M23, S20, V23, use_hist=True):8.2f} km")
print(f"  v31   (+land-drag)         {full_storm(M31, S20, TrackFormerLand, use_hist=True, use_land=True):8.2f} km")


# ---- export v23 vs v31 tracks on Tip for the map, same schema _v31tracks.py used for
# Wayne/Co-may/Hinnamnor (and _noul_predict.py used for Noul) so make_map_html.py just works.
@torch.no_grad()
def run_paths(tag, ms, use_land):
    P = []
    for i in range(0, len(K), 64):
        j = K[i:i + 64]
        h, hv = build_hist(j)
        a = [torch.from_numpy(tr[j]), torch.from_numpy(vp[j]), torch.from_numpy(S20[j]),
             torch.from_numpy(h), torch.from_numpy(hv)]
        if use_land:
            lf5, lok = build_lf5(j)
            a += [torch.from_numpy(lf5), torch.from_numpy(lok)]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).numpy())
    return np.concatenate(P)


for tag, ms, cls, use_land in (("v23", M23, V23, False), ("v31", M31, TrackFormerLand, True)):
    A = run_paths(tag, ms, use_land)
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

p10 = "track_build/v10_tracks.json"
d10 = json.load(open(p10))
obs_lat = np.round(bla[K], 3).tolist(); obs_lon = np.round(blo[K], 3).tolist()
d10["Tip"] = {"lat": [obs_lat] * len(K), "lon": [obs_lon] * len(K),
              "base_time": bt[K].tolist(), "base_lat": obs_lat, "base_lon": obs_lon, "n": int(len(K))}
json.dump(d10, open(p10, "w"))
print(f"added Tip to {p10} (observed reference)")
