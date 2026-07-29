"""Add vmax/pressure/rmw intensity metrics for v23 and v34 to track_build/intensity_compare.json,
alongside the v10 baseline already there (see _intensity_compare.py's docstring for channel layout
and masking convention -- reused verbatim here). This answers "how do v10/v23/v34 compare on
storm STRENGTH prediction", not just track: track error alone says nothing about vmax/pressure.

v23 and v34 both output the full 17-dim state (TARGET_SCALE = [100,100,35,20,50]+[50]*12) even
though every prior v23/v34 script only ever evaluated the track dims -- the intensity channels were
never dropped, just never reported.
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
DEVICE = torch.device("cpu"); R = 111.2

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


MS23 = load(V23, sorted(glob.glob("downloads/x/v23_seed*.pt")))
MS34 = load(TrackFormerMoELand, sorted(glob.glob("downloads/v34ck/**/v34_seed*.pt", recursive=True)))
print(f"v23: {len(MS23)} seeds | v34: {len(MS34)} seeds")

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
print(f"WP+EP 2020+ test windows: {len(wpep)}")


@torch.no_grad()
def predict(tag, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        hv = torch.from_numpy(HAVE[j])
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]), h, hv]
        if tag == "v34":
            lf5, lok = build_lf5(j)
            a += [torch.from_numpy(lf5), torch.from_numpy(lok)]
            sv = torch.stack([m(*a)[0] for m in MS34]).mean(0)
        else:
            sv = torch.stack([m(*a)[0] for m in MS23]).mean(0)
        P.append((sv * SC).float().numpy())
    return np.concatenate(P)


T = target[wpep]; K = mask[wpep]
KM6 = 6.0

res = json.load(open("track_build/intensity_compare.json")) if os.path.exists("track_build/intensity_compare.json") else {}
EXPECT = {"v23": 434.96, "v34": 460.52}
for tag in ("v23", "v34"):
    P = predict(tag, wpep)
    pt, tt = np.cumsum(P[..., :2], 1), np.cumsum(T[..., :2], 1)
    trackL = np.sqrt(((pt - tt) ** 2).sum(-1)).mean(0)
    agg_track = float(np.sqrt(((pt - tt) ** 2).sum(-1)).mean())
    if abs(agg_track - EXPECT[tag]) > 0.5:
        sys.exit(f"{tag} track {agg_track:.2f} != expected {EXPECT[tag]} -- refusing")
    r = {"track": trackL.tolist(), "agg_track": agg_track}
    for ci, nm in [(2, "vmax"), (3, "pressure"), (4, "rmw")]:
        r[nm] = [float(np.abs(P[:, L, ci] - T[:, L, ci])[K[:, L, ci]].mean()) if K[:, L, ci].any() else None
                 for L in range(20)]
    rm = K[..., 5:17]
    r["radii"] = [float(np.abs(P[:, L, 5:17] - T[:, L, 5:17])[rm[:, L]].mean()) if rm[:, L].any() else None
                  for L in range(20)]
    Pspd = np.hypot(P[..., 0], P[..., 1]) / KM6
    Tspd = np.hypot(T[..., 0], T[..., 1]) / KM6
    km0 = K[..., 0]
    r["speed"] = [float(np.abs(Pspd[:, L] - Tspd[:, L])[km0[:, L]].mean()) if km0[:, L].any() else None
                  for L in range(20)]
    res[tag] = r

    def am(x):
        v = [q for q in x if q is not None]; return sum(v) / len(v) if v else float("nan")
    print(f"{tag:4s} | track {agg_track:6.1f} km | vmax {am(r['vmax']):5.2f} kt | pres {am(r['pressure']):5.2f} hPa "
          f"| rmw {am(r['rmw']):5.2f} nm | radii {am(r['radii']):5.2f} nm | speed {am(r['speed']):5.2f} km/h")

json.dump(res, open("track_build/intensity_compare.json", "w"))
print("\nupdated track_build/intensity_compare.json")

print("\n--- v10 vs v23 vs v34, strength (vmax/pressure/rmw) ---")


def am(x):
    v = [q for q in x if q is not None]; return sum(v) / len(v) if v else float("nan")


for tag in ("v10", "v23", "v34"):
    r = res[tag]
    print(f"{tag:4s} | track {r['agg_track']:6.1f} km | vmax MAE {am(r['vmax']):5.2f} kt | "
          f"pres MAE {am(r['pressure']):5.2f} hPa | rmw MAE {am(r['rmw']):5.2f} nm")
