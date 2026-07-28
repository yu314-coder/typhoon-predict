"""v34 (MoE gate on frozen v23_seed0 backbone) vs v23_seed0 ALONE (the fair, matched-backbone
baseline) and vs v33 (storm-normalized oversampling, the least-bad from-scratch retrain), on the
WP-only test set, broken down by land proximity/terrain exactly like _v33eval.py -- since the
aggregate Colab number (-0.33 km vs v23_seed0, essentially null) doesn't rule out a targeted
trade-off (helps in mountainous-near-land, washes out elsewhere in the aggregate).

RESULTS ALREADY KNOWN from the Colab prints:
  v23_seed0 ALONE (v34's own frozen backbone)         460.85 km full  (the fair comparison)
  v34 ENSEMBLE (5 seeds, gate on frozen v23_seed0)     460.52 km full  (-0.33 vs v23_seed0)
  v23 (10-seed ensemble, HEADLINE, not matched)        434.96 km full

This script checks WHERE that -0.33 km comes from -- specifically whether the land gate is
actually doing something (positive in mountainous-near-land, even if offset by noise elsewhere)
or whether it is truly inert everywhere, matching the near-zero aggregate.
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R_KM = 111.2

nb = json.load(open("colab_train_v17.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
body = body.replace('"/content/d/steer5_int8.npz"', '"track_build/dlm4_int8.npz"')
body = body.replace('"/content/d/track_windows_v13.npz"', '"track_build/track_windows_v13.npz"')
body = body.replace('DEVICE = torch.device("cuda")', 'DEVICE = torch.device("cpu")')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os,
     "json": json, "time": __import__("time"), "math": math}
exec(compile(body, "<v17-notebook>", "exec"), G)
Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
vpair = G["vpair"]; te_idx = G["te_idx"]; basins = G["basins"].astype(str); z = G["z"]; SC = G["TARGET_SCALE"]

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))
KM6H = 6 * 3600 / 1000.0

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

v33src = open("colab_v38_train.py").read()
land33_src = re.search(r"class LandDrag\(nn\.Module\):.*?\n        return self\.a_max \* torch\.tanh\(knots @ self\.Wi\.t\(\)\)\n", v33src, re.S).group(0)
tfl33_src = re.search(r"class TrackFormerLand\(V23\):.*?\n        return s, ls, fp\n", v33src, re.S).group(0)
g33 = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "A_MAX": 1.0, "USE_LAND": 1}
exec(land33_src, g33); exec(tfl33_src, g33)
TrackFormerLand33 = g33["TrackFormerLand"]

v34src = open("colab_v39_train.py").read()
gate_src = re.search(r"class LandGate\(nn\.Module\):.*?\n        return g \* raw\n", v34src, re.S).group(0)
tfm_src = re.search(r"class TrackFormerMoELand\(V23\):.*?\n        return s, ls, fp\n", v34src, re.S).group(0)
g34 = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "A_MAX": 1.0,
       "GATE_BIAS_INIT": -5.0, "USE_LAND": 1}
exec(gate_src, g34); exec(tfm_src, g34)
TrackFormerMoELand = g34["TrackFormerMoELand"]

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64") % 360
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

_T = np.load("track_build/terrain_wp.npz")
T_LAT, T_LON, LSM, ELEV = _T["lat"], _T["lon"], _T["lsm"], _T["elev"]
_LAND = LSM > 0.5
LAND_LAT = T_LAT[np.where(_LAND)[0]]; LAND_LON = T_LON[np.where(_LAND)[1]]; LAND_ELEV = ELEV[_LAND]


def land_features_np(lat, lon, heading_rad):
    n = len(lat)
    dlat = LAND_LAT[None, :] - lat[:, None]
    dlon = (LAND_LON[None, :] - lon[:, None]) * np.cos(np.radians(lat[:, None]))
    dy = dlat * R_KM; dx = dlon * R_KM
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


def landfeat_of(idx, lat_np, lon_np, heading_np, basin_np):
    wp = basin_np == "WP"; n = len(idx); out = np.zeros((n, 3), "float32")
    if wp.any():
        out[wp] = land_features_np(lat_np[wp], lon_np[wp] % 360, heading_np[wp])
    return out, wp.astype("float32")


def build_lf5(idx):
    phi = np.arctan2(vpair[idx, 1], vpair[idx, 0])
    lf3, lok = landfeat_of(idx, bla[idx], blo[idx], phi, basins[idx])
    spd = np.hypot(vpair[idx, 0], vpair[idx, 1])
    latv = track[idx, -1, 48] * G["tstd"][48] + G["tmean"][48]
    lf5 = np.concatenate([lf3 / np.array([1000., 1000., 1000.], "float32"),
                          (spd / 100.0)[:, None].astype("float32"),
                          (np.abs(latv) / 30.0)[:, None].astype("float32")], 1)
    return lf5, lok, lf3[:, 1], lf3[:, 2]   # + raw dist_ahead, terr_ahead for bucketing


def load(cls, paths):
    ms = []
    for p in paths:
        m = cls().eval()
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        m.load_state_dict(sd)
        ms.append(m)
    return ms


MS = {"v23_seed0": load(V23, ["track_build/v23_seed0_ref.pt"]),
      "v33": load(TrackFormerLand33, sorted(glob.glob("downloads/v33ck/**/v33_seed*.pt", recursive=True))),
      "v34": load(TrackFormerMoELand, sorted(glob.glob("downloads/v34ck/**/v34_seed*.pt", recursive=True)))}
print(f"v23_seed0: {len(MS['v23_seed0'])} | v33: {len(MS['v33'])} | v34: {len(MS['v34'])} seeds")

TAGS = ("v23_seed0", "v33", "v34")


@torch.no_grad()
def errors(tag, idx):
    """[n,20,2] cumulative E/N error, km."""
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        hv = torch.from_numpy(HAVE[j])
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]), h, hv]
        if tag in ("v33", "v34"):
            lf5, lok, _, _ = build_lf5(j)
            a += [torch.from_numpy(lf5), torch.from_numpy(lok)]
        P.append((torch.stack([m(*a)[0] for m in MS[tag]]).mean(0)[..., :2] * SC[:2]).float().numpy())
    return np.cumsum(np.concatenate(P), 1) - np.cumsum(target[idx][..., :2], 1)


def along_cross(idx, E):
    obs = np.cumsum(target[idx][..., :2], 1)
    prev = np.concatenate([np.zeros((len(idx), 1, 2)), obs[:, :-1]], 1)
    step = obs - prev
    hd = np.arctan2(step[..., 1], step[..., 0])
    uE, uN = np.cos(hd), np.sin(hd)
    along = E[..., 0] * uE + E[..., 1] * uN
    cross = -E[..., 0] * uN + E[..., 1] * uE
    return along, cross


full = z["n_leads"].astype(int) == 20
TE = np.array([i for i in te_idx if full[i] and basins[i] == "WP"])
print(f"WP-only, full 20-lead test set: {len(TE)} windows")

phi = np.arctan2(vpair[TE, 1], vpair[TE, 0])
_, _, dist_ahead, terr_ahead = build_lf5(TE)
NEAR_KM = 500.0; MOUNTAIN_M = 500.0
near_t = dist_ahead <= NEAR_KM
mountain_t = near_t & (terr_ahead >= MOUNTAIN_M)
flat_t = near_t & ~mountain_t
print(f"buckets: open_ocean {(~near_t).sum()} | flat_near_land {flat_t.sum()} | "
      f"mountainous_near_land {mountain_t.sum()}")

E = {tag: errors(tag, TE) for tag in TAGS}
AC = {tag: along_cross(TE, E[tag]) for tag in TAGS}

LEADS = (0, 3, 7, 11, 15, 19)  # 6,24,48,72,96,120h
print(f"\nALONG-TRACK RMS (km) -- the axis the land gate/drag actually corrects")
print(f"{'bucket':>20s} {'lead_h':>7s} | {'v23_seed0':>10s} {'v33':>8s} {'v34':>8s}")
for name, mask in (("open_ocean", ~near_t), ("flat_near_land", flat_t), ("mountainous_near_land", mountain_t)):
    if mask.sum() == 0:
        continue
    for L in LEADS:
        vals = [float(np.sqrt((AC[tag][0][mask, L] ** 2)).mean()) for tag in TAGS]
        print(f"{name:>20s} {6*(L+1):7d} | {vals[0]:10.1f} {vals[1]:8.1f} {vals[2]:8.1f}")

print(f"\noverall track RMS error (km) by bucket, v23_seed0 vs v33 vs v34:")
for name, mask in (("open_ocean", ~near_t), ("flat_near_land", flat_t), ("mountainous_near_land", mountain_t)):
    if mask.sum() == 0:
        continue
    for L, lbl in ((7, "48h"), (19, "120h")):
        row = [float(np.sqrt((E[tag][mask, L] ** 2).sum(-1)).mean()) for tag in TAGS]
        print(f"  {name:22s} {lbl:5s} | v23_seed0 {row[0]:7.1f}  v33 {row[1]:7.1f}  v34 {row[2]:7.1f}  "
              f"| v34-v23_seed0 {row[2]-row[0]:+7.1f}  v34-v33 {row[2]-row[1]:+7.1f}")

print(f"\nALONG vs CROSS split, mountainous_near_land only (n={mountain_t.sum()}):")
for L, lbl in ((7, "48h"), (19, "120h")):
    print(f"  -- {lbl} --")
    for tag in TAGS:
        al, cr = AC[tag]
        a = float(np.sqrt((al[mountain_t, L] ** 2)).mean())
        c = float(np.sqrt((cr[mountain_t, L] ** 2)).mean())
        print(f"    {tag}: along-RMS {a:7.1f}  cross-RMS {c:7.1f}")
