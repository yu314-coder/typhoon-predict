"""Export v28 and v28abl tracks for the map, gated on reproducing the Colab ensembles first.

WHAT THIS MAP IS FOR. v28 = v23 + an N-only meridional drift adapter. v28abl = the SAME code with
the drift switched off, i.e. v23's architecture retrained in the same run. Drawing v23, v28abl and
v28 together shows the real finding:

    v23      N@120h  -18.6 km      (original run)
    v28abl   N@120h  +23.7 km      (same architecture, fresh run)   <- 42 km swing, opposite sign
    v28      N@120h   +6.8 km      (drift adapter on)

The gap between the two CONTROLS is larger than the gap the drift adapter was built to close. The
map makes that visible: v23 and v28abl are the same model, so any daylight between their tracks is
pure run-to-run variation, and the v28 line has to be judged against that spread, not against zero.

Nothing is written until both local ensembles reproduce their Colab numbers (v28 440.27,
v28abl 434.31).
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R = 111.2; KM6H = 6 * 3600 / 1000.0
EXPECT = {"v28": 440.27, "v28abl": 434.31}

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
tmean = G["tmean"]; tstd = G["tstd"]
TM = torch.tensor(tmean); TS = torch.tensor(tstd)

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

v28src = open("colab_v28_train.py").read()
hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28src, re.S).group(0)
tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28src, re.S).group(0)
g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(hs, g23); exec(tf, g23)
V23 = g23["TrackFormerHist"]

# v28 classes, extracted verbatim from the training script
src = open("colab_v34_train.py").read()
cd = re.search(r"class MeridionalDrift\(nn\.Module\):.*?return sign\[:, None\] \* self\.a_max \* mag", src, re.S).group(0)
wd = re.search(r"class TrackFormerDrift\(V23\):.*?return s, ls, fp\n", src, re.S).group(0)
assert "def forward" in wd and "_z0" in wd, "TrackFormerDrift extraction truncated"
gd = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "TM": TM, "TS": TS,
      "A_MAX": 0.65, "USE_DRIFT": 1}
exec(cd, gd); exec(wd, gd)
V28 = gd["TrackFormerDrift"]

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


def load(paths):
    ms = []
    for p in paths:
        m = V28().eval()
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"])
        ms.append(m)
    return ms


MS = {"v28": load(sorted(glob.glob("downloads/v28ck/**/v28_seed*.pt", recursive=True))),
      "v28abl": load(sorted(glob.glob("downloads/v28ablck/**/v28abl_seed*.pt", recursive=True)))}
print(f"v28: {len(MS['v28'])} seeds | v28abl: {len(MS['v28abl'])} seeds")


@torch.no_grad()
def run(tag, idx):
    """v28abl is the SAME class with the drift disabled -- the global the class closes over."""
    gd["USE_DRIFT"] = 1 if tag == "v28" else 0
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]),
             h, torch.from_numpy(HAVE[j])]
        sv = torch.stack([m(*a)[0] for m in MS[tag]]).mean(0)
        P.append((sv * SC).float().numpy())
    gd["USE_DRIFT"] = 1
    return np.concatenate(P)


full = nl == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = target[wpep]; K = mask[wpep]; TC = np.cumsum(T[..., :2], 1)

IC = json.load(open("track_build/intensity_compare.json"))
ok = True
for tag in ("v28", "v28abl"):
    P = run(tag, wpep)
    C = np.cumsum(P[..., :2], 1)
    agg = float(np.sqrt(((C - TC) ** 2).sum(-1)).mean())
    d = agg - EXPECT[tag]
    print(f"  {tag:7s} local {agg:7.2f} km | colab {EXPECT[tag]:7.2f} | diff {d:+.2f}  "
          f"{'OK' if abs(d) < 0.1 else '*** MISMATCH ***'}")
    ok &= abs(d) < 0.1
    r = {"track": np.sqrt(((C - TC) ** 2).sum(-1)).mean(0).tolist(), "agg_track": agg}
    for ci, nm in ((2, "vmax"), (3, "pressure"), (4, "rmw")):
        r[nm] = [float(np.abs(P[:, L, ci] - T[:, L, ci])[K[:, L, ci]].mean()) if K[:, L, ci].any() else None
                 for L in range(20)]
    rm = K[..., 5:17]
    r["radii"] = [float(np.abs(P[:, L, 5:17] - T[:, L, 5:17])[rm[:, L]].mean()) if rm[:, L].any() else None
                  for L in range(20)]
    Ps = np.hypot(P[..., 0], P[..., 1]) / 6.0; Ts = np.hypot(T[..., 0], T[..., 1]) / 6.0
    k0 = K[..., 0]
    r["speed"] = [float(np.abs(Ps[:, L] - Ts[:, L])[k0[:, L]].mean()) if k0[:, L].any() else None
                  for L in range(20)]
    r["mN120"] = float((C[:, 19, 1] - TC[:, 19, 1]).mean())
    IC[tag] = r
if not ok:
    sys.exit("local forward does not reproduce Colab -- refusing to export tracks")
print("forward validated\n")
json.dump(IC, open("track_build/intensity_compare.json", "w"))

STORMS = [("1986228N19120", "Wayne"), ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]
for tag in ("v28", "v28abl"):
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
        print(f"  {tag:7s} {nm:11s} {len(k):3d} fc | mean 120h {err:6.0f} km", flush=True)
    json.dump(out, open(f"track_build/{tag}_tracks.json", "w"))
print("\nwrote track_build/{v28,v28abl}_tracks.json and merged both into intensity_compare.json")
