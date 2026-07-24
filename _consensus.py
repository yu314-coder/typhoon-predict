"""Output-level consensus: can combining models beat the best single model (v23, 434.96 km)?

This is the operational practice the project has never tried, and the redirect the signed-N bias
check pointed to: consensus attacks VARIANCE, and the meridional error is variance, not bias.

METHOD (leakage-controlled):
  - each model's SEED-ENSEMBLE cumulative E/N track is computed on validation (2016-2019) and
    test (2020+), WP+EP, full 20-lead horizon.
  - EQUAL-WEIGHT consensus: the operational baseline; simple averaging captures most of the gain.
  - FITTED consensus: nonnegative weights on the simplex, fit to MINIMISE validation squared
    position error, then applied UNCHANGED to test. Weights are fit on validation ONLY -- the
    2020+ set is never used to choose them.
  - also a per-lead-GROUP fit (short 6-24 h, med 30-72 h, long 78-120 h), as the v28 proposal
    suggested, since the useful members differ by horizon.

DIVERSITY: consensus only helps if members make DECORRELATED errors. The per-window 120 h error
correlation matrix is reported first -- if everything correlates > 0.95 there is nothing to gain.

The honest test is the TEST number against v23's 434.96 km. Validation is the fitting set.
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
vpair = G["vpair"]; basins = G["basins"]; z = G["z"]; SC = G["TARGET_SCALE"]
va_idx, te_idx = G["va_idx"], G["te_idx"]
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

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64"); nl = z["n_leads"].astype(int)
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
    return np.concatenate([(p - OM[None, :, None, None]) / OS[None, :, None, None] * v, v[:, :1]], 1)


ed = re.search(r"class _EnvDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", open("colab_v31_train.py").read(), re.S).group(0)
te_ = re.search(r"class TrackFormerEnv\(V21\):.*?finally:\n            self\._envn = self\._envg = None\n", open("colab_v31_train.py").read(), re.S).group(0)
g25 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "NENV": NENV, "USE_ENV": 1}
exec(ed, g25); exec(te_, g25)
V25 = g25["TrackFormerEnv"]
v32 = open("colab_v32_train.py").read()
oc = re.search(r"class OceanCNN\(nn\.Module\):.*?return self\.net\(x\)\n", v32, re.S).group(0)
od = re.search(r"class _OceanDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", v32, re.S).group(0)
al = re.search(r"class TrackFormerAll\(V23\):.*?finally:\n            self\._envn = self\._envg = self\._op = self\._og = None\n", open("colab_v33_train.py").read(), re.S).group(0)
gA = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "NENV": NENV, "USE_ENV": 1, "USE_OCEAN": 1}
exec(ed, gA); exec(oc, gA); exec(od, gA); exec(al, gA)
V27 = gA["TrackFormerAll"]


def build_v10():
    s = open("train_track_v10.py").read()
    g = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os, "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
    for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(pat, s, re.S).group(0), g)
    return g["TrackFormerV9"]


def load(cls, p):
    m = cls().eval(); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"]); return m


m10 = build_v10()(); m10.load_state_dict(torch.load("track_build/track_v10_best.pt", map_location="cpu", weights_only=False)["model"]); m10.eval()
MODELS = {
    "v10": (m10, None),
    "v21": ([load(V21, p) for p in sorted(glob.glob("downloads/x/v21_seed*.pt"))], "v21"),
    "v23": ([load(V23, p) for p in sorted(glob.glob("downloads/x/v23_seed*.pt"))], "v23"),
    "v25": ([load(V25, p) for p in sorted(glob.glob("downloads/v25ck/v25_seed*.pt"))], "v25"),
    "v27": ([load(V27, p) for p in sorted(glob.glob("downloads/v27ck/**/v27_seed*.pt", recursive=True))], "v27"),
}
print("seeds:", {k: (1 if k == "v10" else len(v[0])) for k, v in MODELS.items()})


@torch.no_grad()
def disp(tag, idx):
    """seed-ensemble E/N per-lead DISPLACEMENT (km), shape [n,20,2]."""
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        tr = torch.from_numpy(track[j]); vp = torch.from_numpy(vpair[j]); sp = torch.from_numpy(SLP[j])
        if tag == "v10":
            sv = m10(tr, vp)[0]
        elif tag == "v21":
            sv = torch.stack([m(tr, vp, sp)[0] for m in MODELS["v21"][0]]).mean(0)
        elif tag == "v23":
            h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
            sv = torch.stack([m(tr, vp, sp, h, torch.from_numpy(HAVE[j]))[0] for m in MODELS["v23"][0]]).mean(0)
        elif tag == "v25":
            sv = torch.stack([m(tr, vp, sp, torch.from_numpy(ENORM[j]), torch.from_numpy(EGOT[j]))[0] for m in MODELS["v25"][0]]).mean(0)
        else:
            h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
            sv = torch.stack([m(tr, vp, sp, h, torch.from_numpy(HAVE[j]), torch.from_numpy(ENORM[j]),
                                torch.from_numpy(EGOT[j]), torch.from_numpy(ocean_in(j)), torch.from_numpy(OGOT[j]))[0]
                              for m in MODELS["v27"][0]]).mean(0)
        P.append((sv[..., :2] * SC[:2]).float().numpy())
    return np.concatenate(P)


full = nl == 20; wpep = np.isin(basins, ["WP", "EP"])
VA = np.array([i for i in va_idx if full[i] and wpep[i]])
TE = np.array([i for i in te_idx if full[i] and wpep[i]])
NAMES = list(MODELS)


def positions(idx):
    """cumulative positions per model: dict name -> [n,20,2], plus obs [n,20,2]."""
    obs = np.cumsum(target[idx][..., :2], 1)
    return {nm: np.cumsum(disp(nm, idx), 1) for nm in NAMES}, obs


print("predicting validation ..."); PV, OV = positions(VA)
print("predicting test ...");       PT, OT = positions(TE)


def err(pos, obs):
    return float(np.sqrt(((pos - obs) ** 2).sum(-1)).mean())          # all-lead mean km


print("\n--- single models (sanity: v23 should be ~434.96 on test) ---")
for nm in NAMES:
    print(f"  {nm:5s}  val {err(PV[nm], OV):7.2f}  test {err(PT[nm], OT):7.2f}")

# diversity: per-window 120 h error-vector correlation
E120 = np.stack([np.sqrt(((PT[nm][:, 19] - OT[:, 19]) ** 2).sum(-1)) for nm in NAMES])
C = np.corrcoef(E120)
print("\n--- per-window 120 h error correlation (test) ---")
print("        " + " ".join(f"{n:>6s}" for n in NAMES))
for a, n in enumerate(NAMES):
    print(f"  {n:5s} " + " ".join(f"{C[a, b]:6.2f}" for b in range(len(NAMES))))

# stack positions -> [K, n, 40]
def stack(P):
    return np.stack([P[nm].reshape(len(P[nm]), -1) for nm in NAMES])       # [K,n,40]
SV, SVo = stack(PV), OV.reshape(len(OV), -1)
ST, STo = stack(PT), OT.reshape(len(OT), -1)


def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u) - 1
    rho = np.nonzero(u - css / (np.arange(len(u)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


def fit_weights(S, o, iters=4000, lr=0.02):
    """min over simplex w of mean_n ||sum_k w_k S[k,n] - o[n]||^2, projected gradient."""
    K = S.shape[0]; w = np.full(K, 1.0 / K)
    for _ in range(iters):
        pred = np.einsum("k,knd->nd", w, S)          # [n,40]
        r = pred - o
        grad = np.einsum("knd,nd->k", S, r) * (2.0 / len(o))
        w = proj_simplex(w - lr * grad)
    return w


def consensus_err(S, o, w):
    pred = np.einsum("k,knd->nd", w, S).reshape(len(o), 20, 2)
    return err(pred, o.reshape(len(o), 20, 2))


w_eq = np.full(len(NAMES), 1.0 / len(NAMES))
w_fit = fit_weights(SV, SVo)                                            # fit on VALIDATION only
print("\n--- consensus ---")
print(f"  equal-weight        val {consensus_err(SV, SVo, w_eq):7.2f}  test {consensus_err(ST, STo, w_eq):7.2f}")
print(f"  fitted (val)        val {consensus_err(SV, SVo, w_fit):7.2f}  test {consensus_err(ST, STo, w_fit):7.2f}")
print(f"    weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, w_fit)))

# per-lead-group fit: short 1-4, med 5-12, long 13-20 (as the v28 proposal suggested)
groups = {"short_6-24h": range(0, 4), "med_30-72h": range(4, 12), "long_78-120h": range(12, 20)}
predV = np.zeros_like(OV); predT = np.zeros_like(OT); gw = {}
for gname, leads in groups.items():
    ml = list(leads)
    Sg = SV.reshape(len(NAMES), len(VA), 20, 2)[:, :, ml].reshape(len(NAMES), len(VA), -1)
    og = OV[:, ml].reshape(len(VA), -1)
    wg = fit_weights(Sg, og); gw[gname] = {n: float(x) for n, x in zip(NAMES, wg)}
    predV[:, ml] = np.einsum("k,knld->nld", wg, SV.reshape(len(NAMES), len(VA), 20, 2)[:, :, ml])
    predT[:, ml] = np.einsum("k,knld->nld", wg, ST.reshape(len(NAMES), len(TE), 20, 2)[:, :, ml])
print(f"  fitted per-group    val {err(predV, OV):7.2f}  test {err(predT, OT):7.2f}")
for gname, w in gw.items():
    print(f"    {gname:12s}: " + ", ".join(f"{n}={x:.2f}" for n, x in w.items()))

best_single = min(err(PT[nm], OT) for nm in NAMES)
best_cons = min(consensus_err(ST, STo, w_eq), consensus_err(ST, STo, w_fit), err(predT, OT))
print("\n" + "=" * 66)
print(f"best single model (test):   {best_single:.2f} km")
print(f"best consensus    (test):   {best_cons:.2f} km   delta {best_cons - best_single:+.2f} km")
print("=" * 66)
json.dump({"names": NAMES, "single_test": {nm: err(PT[nm], OT) for nm in NAMES},
           "corr": C.tolist(), "w_fit": {n: float(x) for n, x in zip(NAMES, w_fit)},
           "equal_test": consensus_err(ST, STo, w_eq), "fit_test": consensus_err(ST, STo, w_fit),
           "group_test": err(predT, OT), "group_weights": gw},
          open("track_build/consensus.json", "w"), indent=1)
print("wrote track_build/consensus.json")
