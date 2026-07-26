"""Export v31 tracks for the map, gated on reproducing the Colab ensemble first.

v31 = v23 + LandDrag (land/kinematic features -> zero-init MLP -> along-track push), built
directly on v23. RESULT (Colab, 5 seeds): full WP+EP 443.07 km vs v23 434.96 km (+8.11, worse),
WP-only 473.38 km -- every seed landed above the baseline (461.5-479.5 km), worse than v29's own
null (+3.74 km). The map is drawn anyway: v23 vs v31 makes the (lack of) difference visible on
real storms, which is itself the finding worth showing.

Nothing is written until the local ensemble reproduces the Colab number (443.07 km, full WP+EP).
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R = 111.2; KM6H = 6 * 3600 / 1000.0
EXPECT = {"full": 443.07, "wp_only": 473.38}
A_MAX = 0.65

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
mask = z["target_mask"].astype(bool)

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
USE_LAND = 1
g31 = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "A_MAX": A_MAX, "USE_LAND": USE_LAND}
exec(land_src, g31); exec(tfl_src, g31)
TrackFormerLand = g31["TrackFormerLand"]

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
nl = z["n_leads"].astype(int)
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64") % 360
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

# ---- land geometry ----
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
    return lf5, lok


def load(cls, paths):
    ms = []
    for p in paths:
        m = cls().eval()
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        m.load_state_dict(sd)
        ms.append(m)
    return ms


MS = {"v31": load(TrackFormerLand, sorted(glob.glob("downloads/v31ck/**/v31_seed*.pt", recursive=True))),
      "v23": load(V23, sorted(glob.glob("downloads/x/v23_seed*.pt", recursive=True)))}
print(f"v31: {len(MS['v31'])} seeds | v23: {len(MS['v23'])} seeds")


@torch.no_grad()
def run(tag, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        hv = torch.from_numpy(HAVE[j])
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]), h, hv]
        if tag == "v31":
            lf5, lok = build_lf5(j)
            a += [torch.from_numpy(lf5), torch.from_numpy(lok)]
        sv = torch.stack([m(*a)[0] for m in MS[tag]]).mean(0)
        P.append((sv * SC).float().numpy())
    return np.concatenate(P)


full = nl == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
wp_only = np.array([i for i in wpep if basins[i] == "WP"])
T = target[wpep]; TC = np.cumsum(T[..., :2], 1)

ok = True
for tag, pop, key_ in (("v31", wpep, "full"), ("v31", wp_only, "wp_only")):
    P = run(tag, pop)
    C = np.cumsum(P[..., :2], 1)
    Tp = np.cumsum(target[pop][..., :2], 1)
    agg = float(np.sqrt(((C - Tp) ** 2).sum(-1)).mean())
    d = agg - EXPECT[key_]
    print(f"  {tag:5s} {key_:8s} local {agg:7.2f} km | colab {EXPECT[key_]:7.2f} | diff {d:+.2f}  "
          f"{'OK' if abs(d) < 0.5 else '*** MISMATCH ***'}")
    ok &= abs(d) < 0.5
if not ok:
    sys.exit("local forward does not reproduce Colab -- refusing to export tracks")
print("forward validated\n")

IC = json.load(open("track_build/intensity_compare.json")) if os.path.exists("track_build/intensity_compare.json") else {}
for tag in ("v31", "v23"):
    Pfull = run(tag, wpep)
    Cfull = np.cumsum(Pfull[..., :2], 1)
    agg = float(np.sqrt(((Cfull - TC) ** 2).sum(-1)).mean())
    IC[tag] = {"track": np.sqrt(((Cfull - TC) ** 2).sum(-1)).mean(0).tolist(), "agg_track": agg}
json.dump(IC, open("track_build/intensity_compare.json", "w"))

STORMS = [("1986228N19120", "Wayne"), ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]
for tag in ("v31", "v23"):
    out = {}
    for s, nm in STORMS:
        k = np.where((sid == s) & (nl == 20))[0]; k = k[np.argsort(bt[k])]
        if not len(k):
            continue
        A = run(tag, k)
        cE, cN = np.cumsum(A[..., 0], 1), np.cumsum(A[..., 1], 1)
        tE, tN = np.cumsum(target[k][..., 0], 1), np.cumsum(target[k][..., 1], 1)
        lats, lons = [], []
        for a2 in range(len(k)):
            la = bla[k[a2]] + cN[a2] / R
            lo = blo[k[a2]] + cE[a2] / (R * np.cos(np.radians((bla[k[a2]] + la) / 2)))
            lats.append(np.round(la, 3).tolist()); lons.append(np.round(lo, 3).tolist())
        err = float(np.hypot(cE[:, 19] - tE[:, 19], cN[:, 19] - tN[:, 19]).mean())
        out[nm] = {"lat": lats, "lon": lons, "base_time": bt[k].tolist(),
                   "base_lat": np.round(bla[k], 3).tolist(),
                   "base_lon": np.round(blo[k], 3).tolist(),
                   "err120_mean": err, "n": int(len(k))}
        print(f"  {tag:5s} {nm:11s} {len(k):3d} fc | mean 120h {err:6.0f} km", flush=True)
    json.dump(out, open(f"track_build/{tag}_tracks.json", "w"))
print("\nwrote track_build/{v31,v23}_tracks.json and merged both into intensity_compare.json")
