"""Recompute every locally-held model's intensity on ONE convention: POOLED.

THE BUG THIS FIXES. Two different averages were being compared as if they were the same number:

    Colab metrics():  np.abs(O-T)[mask].mean()      pooled over all (window, lead) pairs
    local scripts:    mean of the 20 per-lead means  unweighted by lead

Later leads carry larger error AND fewer valid targets, so the unweighted form up-weights them.
For v27 the same model scores vmax 16.76 pooled against 17.13 unweighted -- a 0.37 kt gap, the
same size as the effects being reported. v10/v21/v23/v25 were quoted unweighted (local) while
v26/v26abl/v27/v27abl were quoted pooled (Colab), so several cross-model claims were invalid --
including "v26 wins max wind at 16.26 against v23's 16.43".

Pooled is the convention kept, because it is what the Colab-only models (v26, v26abl, v27abl)
report and their checkpoints no longer exist to recompute.

Writes track_build/pooled_metrics.json.
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)

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
KM6H = 6 * 3600 / 1000.0

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
te = re.search(r"class TrackFormerEnv\(V21\):.*?finally:\n            self\._envn = self\._envg = None\n", v31, re.S).group(0)
g25 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "NENV": NENV, "USE_ENV": 1}
exec(ed, g25); exec(te, g25)
V25 = g25["TrackFormerEnv"]

v32 = open("colab_v32_train.py").read()
oc = re.search(r"class OceanCNN\(nn\.Module\):.*?return self\.net\(x\)\n", v32, re.S).group(0)
od = re.search(r"class _OceanDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", v32, re.S).group(0)
v33 = open("colab_v33_train.py").read()
al = re.search(r"class TrackFormerAll\(V23\):.*?finally:\n            "
               r"self\._envn = self\._envg = self\._op = self\._og = None\n", v33, re.S).group(0)
gA = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "NENV": NENV,
      "USE_ENV": 1, "USE_OCEAN": 1}
exec(ed, gA); exec(oc, gA); exec(od, gA); exec(al, gA)
V27 = gA["TrackFormerAll"]


def build_v10():
    s = open("train_track_v10.py").read()
    g = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
         "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
    for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(pat, s, re.S).group(0), g)
    return g["TrackFormerV9"]


def load(cls, p):
    m = cls().eval(); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"]); return m


m10 = build_v10()(); m10.load_state_dict(torch.load("track_build/track_v10_best.pt", map_location="cpu", weights_only=False)["model"]); m10.eval()
MS21 = [load(V21, p) for p in sorted(glob.glob("downloads/x/v21_seed*.pt"))]
MS23 = [load(V23, p) for p in sorted(glob.glob("downloads/x/v23_seed*.pt"))]
MS25 = [load(V25, p) for p in sorted(glob.glob("downloads/v25ck/v25_seed*.pt"))]
MS27 = [load(V27, p) for p in sorted(glob.glob("downloads/v27ck/**/v27_seed*.pt", recursive=True))]
print(f"seeds  v21 {len(MS21)} | v23 {len(MS23)} | v25 {len(MS25)} | v27 {len(MS27)}")


@torch.no_grad()
def predict(tag, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        tr = torch.from_numpy(track[j]); vp = torch.from_numpy(vpair[j]); sp = torch.from_numpy(SLP[j])
        if tag == "v10":
            sv, _ = m10(tr, vp)
        elif tag == "v21":
            sv = torch.stack([m(tr, vp, sp)[0] for m in MS21]).mean(0)
        elif tag == "v23":
            h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
            sv = torch.stack([m(tr, vp, sp, h, torch.from_numpy(HAVE[j]))[0] for m in MS23]).mean(0)
        elif tag == "v25":
            sv = torch.stack([m(tr, vp, sp, torch.from_numpy(ENORM[j]),
                                torch.from_numpy(EGOT[j]))[0] for m in MS25]).mean(0)
        else:
            h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
            sv = torch.stack([m(tr, vp, sp, h, torch.from_numpy(HAVE[j]),
                                torch.from_numpy(ENORM[j]), torch.from_numpy(EGOT[j]),
                                torch.from_numpy(ocean_in(j)), torch.from_numpy(OGOT[j]))[0]
                              for m in MS27]).mean(0)
        P.append((sv * SC).float().numpy())
    return np.concatenate(P)


full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = target[wpep]; K = mask[wpep]
TC = np.cumsum(T[..., :2], 1)

out = {}
for tag in ("v10", "v21", "v23", "v25", "v27"):
    O = predict(tag, wpep)
    C = np.cumsum(O[..., :2], 1)
    r = {"track": float(np.sqrt(((C - TC) ** 2).sum(-1)).mean())}
    for ci, nm in ((2, "vmax"), (3, "pres"), (4, "rmw")):
        m_ = K[..., ci]
        r[nm] = float(np.abs(O[..., ci] - T[..., ci])[m_].mean())      # POOLED
    m_ = K[..., 5:17]
    r["radii"] = float(np.abs(O[..., 5:17] - T[..., 5:17])[m_].mean())
    out[tag] = r
    print(f"{tag:7s} track {r['track']:7.2f} | vmax {r['vmax']:5.2f} | pres {r['pres']:5.2f} "
          f"| rmw {r['rmw']:5.2f} | radii {r['radii']:5.2f}")

# Colab-only models: their checkpoints are gone, but their reported figures are already POOLED
for f, tag in (("downloads/v26.json", "v26"), ("downloads/v26abl.json", "v26abl"),
               ("downloads/v27abl.json", "v27abl")):
    if os.path.exists(f):
        d = json.load(open(f))[tag]["all"]
        out[tag] = {"track": d["track"], "vmax": d["vmax"], "pres": d["pres"],
                    "rmw": d["rmw"], "radii": d["radii"], "source": "colab (checkpoints lost)"}
        print(f"{tag:7s} track {d['track']:7.2f} | vmax {d['vmax']:5.2f} | pres {d['pres']:5.2f} "
              f"| rmw {d['rmw']:5.2f} | radii {d['radii']:5.2f}   <- from Colab JSON")

json.dump(out, open("track_build/pooled_metrics.json", "w"), indent=1)
print("\nwrote track_build/pooled_metrics.json  (all figures POOLED, one convention)")
