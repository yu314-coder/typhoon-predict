"""Export v10 / v21 / v25 tracks for the map, validating each forward against Colab first.

v25 is v21 plus one environmental token (ocean heat + shear + humidity) appended to both decoder
memories. It needs the 43 env features AND their presence mask threaded through -- exactly the
metrics() path in colab_v31_train.py. The env features exist for only ~50% of WP+EP windows; where
absent the token is not appended, so v25 == v21 there (e.g. 1986 Wayne, pre-satellite ocean data).

Model classes are extracted VERBATIM from the training scripts, never retyped. Nothing is written
until the local ensemble error reproduces the Colab number for every arm (v25 451.86, v21 443.62).
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
DEVICE = torch.device("cpu"); R = 111.2; KM6H = 6 * 3600 / 1000.0

# ---- v17 notebook: data arrays + Base -------------------------------------------------------
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

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))

# ---- v21 (chain-of-thought steering) --------------------------------------------------------
CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

# ---- environmental features (replicate colab_v31_train.py exactly) ---------------------------
_E = np.load("track_build/env_features.npz", allow_pickle=True)
EFEAT = _E["feat"].astype("float32"); EGOT = _E["got"].astype("float32")
NENV = EFEAT.shape[1]
_present = EGOT > 0
_mu = np.array([EFEAT[_present[:, c], c].mean() if _present[:, c].any() else 0.0
                for c in range(NENV)], "float32")
_sd = np.array([EFEAT[_present[:, c], c].std() + 1e-6 if _present[:, c].any() else 1.0
                for c in range(NENV)], "float32")
ENORM = ((EFEAT - _mu[None]) / _sd[None]) * EGOT
print(f"env features: {NENV} predictors | present on {100*(EGOT.sum(1) > 0).mean():.1f}% of windows")

# ---- v25 (v21 + env token) ------------------------------------------------------------------
USE_ENV = 1
v31 = open("colab_v31_train.py").read()
ed = re.search(r"class _EnvDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n",
               v31, re.S).group(0)
te = re.search(r"class TrackFormerEnv\(V21\):.*?self\._envn = self\._envg = None\n            return super\(\)\.forward\(tr, vp, slp\)\n        finally:\n            self\._envn = self\._envg = None\n",
               v31, re.S)
if te is None:
    te = re.search(r"class TrackFormerEnv\(V21\):.*?finally:\n            self\._envn = self\._envg = None\n",
                   v31, re.S)
g25 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "NENV": NENV, "USE_ENV": USE_ENV}
exec(ed, g25); exec(te.group(0), g25)
V25 = g25["TrackFormerEnv"]

# ---- checkpoints ----------------------------------------------------------------------------
def load(cls, path):
    m = cls().eval()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return m

V25_CK = sorted(glob.glob("downloads/v25ck/v25_seed*.pt"))
V21_CK = sorted(glob.glob("downloads/x/v21_seed*.pt"))
print(f"v25: {len(V25_CK)} seeds | v21: {len(V21_CK)} seeds")

MS25 = [load(V25, p) for p in V25_CK]
MS21 = [load(V21, p) for p in V21_CK]

# v10 (single best checkpoint)
def build_v10():
    s = open("train_track_v10.py").read()
    g = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
         "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
    for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(pat, s, re.S).group(0), g)
    return g["TrackFormerV9"]

m10 = build_v10()()
m10.load_state_dict(torch.load("track_build/track_v10_best.pt", map_location="cpu",
                               weights_only=False)["model"]); m10.eval()

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
nl = z["n_leads"].astype(int)
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")


@torch.no_grad()
def run(tag, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        if tag == "v10":
            sv, _ = m10(torch.from_numpy(track[j]), torch.from_numpy(vpair[j]))
        elif tag == "v21":
            a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
            sv = torch.stack([m(*a)[0] for m in MS21]).mean(0)
        else:  # v25
            a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]),
                 torch.from_numpy(ENORM[j]), torch.from_numpy(EGOT[j])]
            sv = torch.stack([m(*a)[0] for m in MS25]).mean(0)
        P.append((sv * SC).float().numpy())
    return np.concatenate(P)


# ---- reproduction gate ----------------------------------------------------------------------
full = nl == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = np.cumsum(target[wpep][..., :2], 1)
EXPECT = {"v21": 443.62, "v25": 451.86}
print(f"\nWP+EP 2020+, {len(wpep)} windows")
ok = True
for tag, exp in EXPECT.items():
    C = np.cumsum(run(tag, wpep)[..., :2], 1)
    e = float(np.sqrt(((C - T) ** 2).sum(-1)).mean())
    d = e - exp
    print(f"  {tag:4s} local {e:8.2f} km | colab {exp:7.2f} | diff {d:+.2f}  "
          f"{'OK' if abs(d) < 0.1 else '*** MISMATCH ***'}")
    ok &= abs(d) < 0.1
if not ok:
    sys.exit("local forward does not reproduce Colab -- refusing to export tracks")
print("forward validated\n")

# ---- export tracks --------------------------------------------------------------------------
STORMS = [("1986228N19120", "Wayne"), ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]
for tag in ("v10", "v21", "v25"):
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
        ecov = float(np.mean([EGOT[i].sum() > 0 for i in k]))
        out[nm] = {"lat": lats, "lon": lons, "base_time": bt[k].tolist(),
                   "base_lat": np.round(bla[k], 3).tolist(),
                   "base_lon": np.round(blo[k], 3).tolist(),
                   "err120_mean": err, "n": int(len(k)), "env_cov": ecov}
        print(f"  {tag:4s} {nm:11s} {len(k):3d} fc | mean 120h {err:6.0f} km | env cov {100*ecov:3.0f}%",
              flush=True)
    json.dump(out, open(f"track_build/{tag}_tracks.json", "w"))
print("\nwrote track_build/{v10,v21,v25}_tracks.json")
