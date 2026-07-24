"""Add v23 -- the best TRACK model -- to both the map and the intensity comparison.

v23 is the temporal steering stack (t-24 h, t-12 h, now) and the only version whose gain survived
a paired-storm bootstrap (-9.09 km vs its own ablation, p=0.011). It is the best track model this
project has: 435.0 km against v21's 443.6.

It is NOT the best intensity model -- v26's ocean patch owns that (16.26 kt vmax). The two crowns
sit on different heads, which is why this script fills in v23's intensity numbers too rather than
letting the summary imply one model wins everything.

Model classes are extracted verbatim from the training scripts. Nothing is written until the local
ensemble reproduces the Colab track number.
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R = 111.2; KM6H = 6 * 3600 / 1000.0
EXPECT_V23 = 435.0

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
vpair = G["vpair"]; te_idx = G["te_idx"]; basins = G["basins"]; z = G["z"]; SC = G["TARGET_SCALE"]
mask = z["target_mask"].astype(bool)

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

# ---- v23: history stack ----------------------------------------------------------------------
v28 = open("colab_v28_train.py").read()
hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28, re.S).group(0)
tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28, re.S).group(0)
gh = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
      "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(hs, gh); exec(tf, gh)
V23 = gh["TrackFormerHist"]

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
nl = z["n_leads"].astype(int)
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])


def load(cls, p):
    m = cls().eval()
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"])
    return m


CK = sorted(glob.glob("downloads/x/v23_seed*.pt"))
MS = [load(V23, p) for p in CK]
print(f"v23: {len(MS)} seeds")


@torch.no_grad()
def run(idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]),
             h, torch.from_numpy(HAVE[j])]
        sv = torch.stack([m(*a)[0] for m in MS]).mean(0)
        P.append((sv * SC).float().numpy())
    return np.concatenate(P)


full = nl == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
P = run(wpep); T = target[wpep]; K = mask[wpep]
C = np.cumsum(P[..., :2], 1); TC = np.cumsum(T[..., :2], 1)
agg = float(np.sqrt(((C - TC) ** 2).sum(-1)).mean())
print(f"v23 local {agg:.2f} km | colab {EXPECT_V23:.1f} | diff {agg-EXPECT_V23:+.2f}")
if abs(agg - EXPECT_V23) > 0.6:
    sys.exit("v23 does not reproduce the Colab number -- refusing to export")
print("forward validated\n")

# ---- intensity metrics, merged into the existing comparison ----------------------------------
IC = json.load(open("track_build/intensity_compare.json"))
r = {"track": np.sqrt(((C - TC) ** 2).sum(-1)).mean(0).tolist(), "agg_track": agg}
for ci, nm in ((2, "vmax"), (3, "pressure"), (4, "rmw")):
    r[nm] = [float(np.abs(P[:, L, ci] - T[:, L, ci])[K[:, L, ci]].mean()) if K[:, L, ci].any() else None
             for L in range(20)]
rm = K[..., 5:17]
r["radii"] = [float(np.abs(P[:, L, 5:17] - T[:, L, 5:17])[rm[:, L]].mean()) if rm[:, L].any() else None
              for L in range(20)]
Pspd = np.hypot(P[..., 0], P[..., 1]) / 6.0; Tspd = np.hypot(T[..., 0], T[..., 1]) / 6.0
k0 = K[..., 0]
r["speed"] = [float(np.abs(Pspd[:, L] - Tspd[:, L])[k0[:, L]].mean()) if k0[:, L].any() else None
              for L in range(20)]
IC["v23"] = r
json.dump(IC, open("track_build/intensity_compare.json", "w"))


def am(x):
    v = [q for q in x if q is not None]
    return sum(v) / len(v) if v else float("nan")


print(f"v23 | track {agg:6.1f} km | vmax {am(r['vmax']):5.2f} kt | pres {am(r['pressure']):5.2f} hPa "
      f"| rmw {am(r['rmw']):5.2f} nm | radii {am(r['radii']):5.2f} nm | speed {am(r['speed']):5.2f} km/h")

# ---- tracks for the map ----------------------------------------------------------------------
STORMS = [("1986228N19120", "Wayne"), ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]
out = {}
for s, nm in STORMS:
    k = np.where((sid == s) & (nl == 20))[0]; k = k[np.argsort(bt[k])]
    if not len(k):
        continue
    A = run(k)
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
    print(f"  v23 {nm:11s} {len(k):3d} fc | mean 120h {err:6.0f} km", flush=True)
json.dump(out, open("track_build/v23_tracks.json", "w"))
print("\nwrote track_build/v23_tracks.json and merged v23 into intensity_compare.json")
