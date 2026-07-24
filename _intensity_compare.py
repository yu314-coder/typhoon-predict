"""Per-lead intensity / radius / moving-speed comparison for v10, v21, v25.

Channels of the 17-dim target (TARGET_SCALE = [100,100,35,20,50]+[50]*12):
    0,1  E,N displacement per 6 h step (km)      -> track error and MOVING SPEED
    2    vmax   (kt)                              -> wind speed / wind max
    3    pressure (hPa)                           -> central pressure
    4    rmw    (nm)                              -> radius of maximum wind
    5:17 R34/R50/R64 in NE/SE/SW/NW quadrants (nm) -> wind radii

Every intensity metric is masked with its OWN validity channel (target_mask[...,i]) -- the earlier
"140 hPa" figure came from reusing the position mask, which counts windows whose pressure target is
absent. Moving speed at lead L is the length of that step's displacement vector over 6 h.

Reproduction of the track number (v21 443.62, v25 451.86) is asserted before any metric is trusted.
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
vpair = G["vpair"]; te_idx = G["te_idx"]; basins = G["basins"]; z = G["z"]; SC = G["TARGET_SCALE"]
mask = z["target_mask"].astype(bool)

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": 6*3600/1000.0, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

_E = np.load("track_build/env_features.npz", allow_pickle=True)
EFEAT = _E["feat"].astype("float32"); EGOT = _E["got"].astype("float32"); NENV = EFEAT.shape[1]
_present = EGOT > 0
_mu = np.array([EFEAT[_present[:, c], c].mean() if _present[:, c].any() else 0.0 for c in range(NENV)], "float32")
_sd = np.array([EFEAT[_present[:, c], c].std() + 1e-6 if _present[:, c].any() else 1.0 for c in range(NENV)], "float32")
ENORM = ((EFEAT - _mu[None]) / _sd[None]) * EGOT

USE_ENV = 1
v31 = open("colab_v31_train.py").read()
ed = re.search(r"class _EnvDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", v31, re.S).group(0)
te = re.search(r"class TrackFormerEnv\(V21\):.*?finally:\n            self\._envn = self\._envg = None\n", v31, re.S).group(0)
g25 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "NENV": NENV, "USE_ENV": USE_ENV}
exec(ed, g25); exec(te, g25)
V25 = g25["TrackFormerEnv"]


def load(cls, path):
    m = cls().eval(); m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"]); return m


MS25 = [load(V25, p) for p in sorted(glob.glob("downloads/v25ck/v25_seed*.pt"))]
MS21 = [load(V21, p) for p in sorted(glob.glob("downloads/x/v21_seed*.pt"))]


def build_v10():
    s = open("train_track_v10.py").read()
    g = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os, "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
    for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(pat, s, re.S).group(0), g)
    return g["TrackFormerV9"]


m10 = build_v10()(); m10.load_state_dict(torch.load("track_build/track_v10_best.pt", map_location="cpu", weights_only=False)["model"]); m10.eval()

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
print(f"WP+EP 2020+ test windows: {len(wpep)} | v25 seeds {len(MS25)} | v21 seeds {len(MS21)}")


@torch.no_grad()
def predict(tag, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        if tag == "v10":
            sv, _ = m10(torch.from_numpy(track[j]), torch.from_numpy(vpair[j]))
        elif tag == "v21":
            a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
            sv = torch.stack([m(*a)[0] for m in MS21]).mean(0)
        else:
            a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]),
                 torch.from_numpy(ENORM[j]), torch.from_numpy(EGOT[j])]
            sv = torch.stack([m(*a)[0] for m in MS25]).mean(0)
        P.append((sv * SC).float().numpy())
    return np.concatenate(P)


T = target[wpep]; K = mask[wpep]
KM6 = 6.0  # hours per step
res = {}
EXPECT = {"v10": None, "v21": 443.62, "v25": 451.86}
for tag in ("v10", "v21", "v25"):
    P = predict(tag, wpep)
    pt, tt = np.cumsum(P[..., :2], 1), np.cumsum(T[..., :2], 1)
    trackL = np.sqrt(((pt - tt) ** 2).sum(-1)).mean(0)          # per-lead km
    if EXPECT[tag] is not None:
        d = float(trackL[19]) - 0  # 120h not the aggregate; verify aggregate below
    # aggregate track over leads (matches the 443.62 / 451.86 definition = mean over all leads&windows)
    agg_track = float(np.sqrt(((pt - tt) ** 2).sum(-1)).mean())
    if EXPECT[tag] is not None and abs(agg_track - EXPECT[tag]) > 0.1:
        sys.exit(f"{tag} track {agg_track:.2f} != expected {EXPECT[tag]} -- refusing")
    r = {"track": trackL.tolist(), "agg_track": agg_track}
    # vmax(2) kt, pressure(3) hPa, rmw(4) nm -- per-channel mask
    for ci, nm in [(2, "vmax"), (3, "pressure"), (4, "rmw")]:
        r[nm] = [float(np.abs(P[:, L, ci] - T[:, L, ci])[K[:, L, ci]].mean()) if K[:, L, ci].any() else None
                 for L in range(20)]
    # wind radii: mean over the 12 quadrant/threshold channels, nm
    rm = K[..., 5:17]
    r["radii"] = [float(np.abs(P[:, L, 5:17] - T[:, L, 5:17])[rm[:, L]].mean()) if rm[:, L].any() else None
                  for L in range(20)]
    # moving speed: |displacement step| / 6h, km/h. mask = position valid (channel 0)
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
print("\nwrote track_build/intensity_compare.json")
