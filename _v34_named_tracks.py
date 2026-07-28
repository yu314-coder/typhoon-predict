"""Export v34 (MoE gate on frozen v23_seed0 backbone) tracks for the map (Wayne, Co-may,
Hinnamnor, Bavi), gated on reproducing the Colab ensemble numbers first, exactly like
_v33_named_tracks.py did for v33.

v34 result (Colab, 5 seeds, frozen v23_seed0 backbone): full WP+EP 460.52 km vs v23_seed0 ALONE
460.85 km (-0.33, essentially null) vs v23 10-seed ensemble 434.96 km (+25.56, not the fair
comparison -- see land-interaction-lever memory). `_v34eval.py`'s bucketed breakdown confirms the
null holds in every bucket including mountainous-near-land. Drawn on the map anyway, since seeing
v34 sit almost exactly on top of v23 -- rather than v33's visibly different path -- is the
clearest way to show what "the land gate does nothing" looks like on an actual track.

Nothing is written until the local ensemble reproduces the Colab number (v34 full WP+EP, 5 seeds).
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R = 111.2; KM6H = 6 * 3600 / 1000.0
EXPECT = {"v34": {"full": 460.52, "wp_only": 494.55}}

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

v34src = open("colab_v39_train.py").read()
gate_src = re.search(r"class LandGate\(nn\.Module\):.*?\n        return g \* raw\n", v34src, re.S).group(0)
tfm_src = re.search(r"class TrackFormerMoELand\(V23\):.*?\n        return s, ls, fp\n", v34src, re.S).group(0)
g34 = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "A_MAX": 1.0,
       "GATE_BIAS_INIT": -5.0, "USE_LAND": 1}
exec(gate_src, g34); exec(tfm_src, g34)
TrackFormerMoELand = g34["TrackFormerMoELand"]

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


MS = {"v34": load(TrackFormerMoELand, sorted(glob.glob("downloads/v34ck/**/v34_seed*.pt", recursive=True)))}
print(f"v34: {len(MS['v34'])} seeds")


@torch.no_grad()
def run(tag, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        hv = torch.from_numpy(HAVE[j])
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]), h, hv]
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
for tag in ("v34",):
    for pop, key_ in ((wpep, "full"), (wp_only, "wp_only")):
        P = run(tag, pop)
        C = np.cumsum(P[..., :2], 1)
        Tp = np.cumsum(target[pop][..., :2], 1)
        agg = float(np.sqrt(((C - Tp) ** 2).sum(-1)).mean())
        d = agg - EXPECT[tag][key_]
        print(f"  {tag:5s} {key_:8s} local {agg:7.2f} km | colab {EXPECT[tag][key_]:7.2f} | diff {d:+.2f}  "
              f"{'OK' if abs(d) < 0.5 else '*** MISMATCH ***'}")
        ok &= abs(d) < 0.5
if not ok:
    sys.exit("local forward does not reproduce Colab -- refusing to export tracks")
print("forward validated\n")

# ---- MERGE into existing track_build/v34_tracks.json, never overwrite
STORMS = [("1986228N19120", "Wayne"), ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor"),
          ("2026182N09163", "Bavi")]
p = "track_build/v34_tracks.json"
out = json.load(open(p)) if os.path.exists(p) else {}
for s, nm in STORMS:
    k = np.where((sid == s) & (nl == 20))[0]; k = k[np.argsort(bt[k])]
    if not len(k):
        continue
    A = run("v34", k)
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
    print(f"  v34   {nm:11s} {len(k):3d} fc | mean 120h {err:6.0f} km", flush=True)
json.dump(out, open(p, "w"))
print(f"\nwrote track_build/v34_tracks.json")
