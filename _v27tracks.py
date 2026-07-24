"""Export v27 tracks for the map, gated on reproducing the Colab ensemble first.

v27 is v23's temporal steering stack plus v25's env token plus v26's ocean-patch CNN, trained on
GODAS. Its Colab ensemble came in at 445.90 km on all WP+EP -- 10.9 km WORSE than v23's 435.0 --
so what this draws is a version that did NOT keep the track gain it was built to keep. Whether
that is the combination interfering or just run-to-run difference is what v27abl decides; this
script only makes the tracks drawable.

Model classes are extracted verbatim from the training scripts. Nothing is written until the local
forward reproduces 445.90.
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R = 111.2; KM6H = 6 * 3600 / 1000.0
EXPECT = 445.90

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

v28 = open("colab_v28_train.py").read()
hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28, re.S).group(0)
tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28, re.S).group(0)
g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(hs, g23); exec(tf, g23)
V23 = g23["TrackFormerHist"]

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

_E = np.load("track_build/env_features.npz", allow_pickle=True)
EFEAT = _E["feat"].astype("float32"); EGOT = _E["got"].astype("float32"); NENV = EFEAT.shape[1]
_p = EGOT > 0
_mu = np.array([EFEAT[_p[:, c], c].mean() if _p[:, c].any() else 0.0 for c in range(NENV)], "float32")
_sd = np.array([EFEAT[_p[:, c], c].std() + 1e-6 if _p[:, c].any() else 1.0 for c in range(NENV)], "float32")
ENORM = ((EFEAT - _mu[None]) / _sd[None]) * EGOT

_O = np.load("track_build/ocean_patch.npz", allow_pickle=True)
OQ = _O["q"]; OSC = _O["scale"].astype("float32"); OGOT = _O["got"].astype("float32")
_s = (OQ[OGOT > 0][::17].astype("float32") * OSC[None, :, None, None])
OM = np.array([float(_s[:, c][_s[:, c] != 0].mean()) for c in range(3)], "float32")
OS = np.array([float(_s[:, c][_s[:, c] != 0].std()) + 1e-6 for c in range(3)], "float32")
del _s


def ocean_in(j):
    p = OQ[j].astype("float32") * OSC[None, :, None, None]
    v = (p != 0).astype("float32")
    p = (p - OM[None, :, None, None]) / OS[None, :, None, None] * v
    return np.concatenate([p, v[:, :1]], 1)


v31 = open("colab_v31_train.py").read()
ed = re.search(r"class _EnvDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", v31, re.S).group(0)
v32 = open("colab_v32_train.py").read()
oc = re.search(r"class OceanCNN\(nn\.Module\):.*?return self\.net\(x\)\n", v32, re.S).group(0)
od = re.search(r"class _OceanDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", v32, re.S).group(0)
v33 = open("colab_v33_train.py").read()
al = re.search(r"class TrackFormerAll\(V23\):.*?finally:\n            "
               r"self\._envn = self\._envg = self\._op = self\._og = None\n", v33, re.S).group(0)
assert "def forward" in al, "TrackFormerAll extraction truncated"
gA = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "NENV": NENV,
      "USE_ENV": 1, "USE_OCEAN": 1}
exec(ed, gA); exec(oc, gA); exec(od, gA); exec(al, gA)
V27 = gA["TrackFormerAll"]

CK = sorted(glob.glob("downloads/v27ck/**/v27_seed*.pt", recursive=True))
MS = []
for p in CK:
    m = V27().eval()
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"])
    MS.append(m)
print(f"v27: {len(MS)} seeds loaded")


@torch.no_grad()
def run(idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]),
             h, torch.from_numpy(HAVE[j]), torch.from_numpy(ENORM[j]), torch.from_numpy(EGOT[j]),
             torch.from_numpy(ocean_in(j)), torch.from_numpy(OGOT[j])]
        sv = torch.stack([m(*a)[0] for m in MS]).mean(0)
        P.append((sv * SC).float().numpy())
    return np.concatenate(P)


full = nl == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
P = run(wpep); T = target[wpep]; K = mask[wpep]
C = np.cumsum(P[..., :2], 1); TC = np.cumsum(T[..., :2], 1)
agg = float(np.sqrt(((C - TC) ** 2).sum(-1)).mean())
print(f"v27 local {agg:.2f} km | colab {EXPECT:.2f} | diff {agg-EXPECT:+.2f}")
if abs(agg - EXPECT) > 0.1:
    sys.exit("v27 does not reproduce the Colab number -- refusing to export")
print("forward validated\n")

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
IC["v27"] = r
json.dump(IC, open("track_build/intensity_compare.json", "w"))


def am(x):
    v = [q for q in x if q is not None]
    return sum(v) / len(v) if v else float("nan")


print(f"v27 | track {agg:6.1f} km | vmax {am(r['vmax']):5.2f} kt | pres {am(r['pressure']):5.2f} hPa "
      f"| rmw {am(r['rmw']):5.2f} nm | radii {am(r['radii']):5.2f} nm | speed {am(r['speed']):5.2f} km/h")

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
    print(f"  v27 {nm:11s} {len(k):3d} fc | mean 120h {err:6.0f} km", flush=True)
json.dump(out, open("track_build/v27_tracks.json", "w"))
print("\nwrote track_build/v27_tracks.json and merged v27 into intensity_compare.json")
