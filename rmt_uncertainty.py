"""Lever 2 (zero-download): does the two-point generalized-MP edge (the Yau paper) give a
better-calibrated forecast covariance than classical MP / Ledoit-Wolf / diagonal?

Regime: per-window MC-dropout ENSEMBLE covariance of the track block (20 leads x 2 axes = 40-dim
cumulative position) from N members. With N ~ p this is the high-dimensional / noisy-sample regime
the paper targets (classical MP assumes homogeneous unit noise; forecast errors are heterogeneous).

Protocol (matches the paper: fit params on a split, validate on held-out storms):
  * MC-dropout ensemble (N passes) of TrackFormer v3 on WP-2020+ windows.
  * Split test STORMS 50/50 -> FIT (estimate global a,beta and a per-estimator variance-inflation
    lambda) and EVAL (report).
  * Score each covariance estimator by mean Gaussian NLL of the truth and by 90% coverage on EVAL.
    A per-estimator lambda (fitted on FIT) removes the trivial MC-dropout under-dispersion so the
    comparison isolates the COVARIANCE SHAPE (correlation structure), which is what RMT cleans.
"""
import re, math, json, os, sys
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---- inlined pure-numpy helpers (from covariance_denoise.py; avoid sklearn/scipy deps) ----
def mp_support(y):
    root = math.sqrt(y)
    return [(max(0.0, (1.0 - root) ** 2), (1.0 + root) ** 2)]


def _companion_cubic(zz, a, beta, y):
    return np.array([a * zz, a * (zz - y + 1.0) + zz,
                     a + zz - y + 1.0 - y * beta * (a - 1.0), 1.0], dtype=np.float64)


def generalized_support(a, beta, y, grid_points=6000):
    if a <= 0.0 or not 0.0 < beta < 1.0 or y <= 0.0:
        raise ValueError("need a>0, 0<beta<1, y>0")
    if abs(a - 1.0) < 1e-7:
        return mp_support(y)
    scale = max(1.0, a) * (1.0 + math.sqrt(max(1.0, y))) ** 2
    zmax = max(10.0, 8.0 * scale)
    vals = np.linspace(1e-7, zmax, grid_points)
    inside = np.array([bool(np.any(np.imag(np.roots(_companion_cubic(float(zz), a, beta, y))) > 1e-7)) for zz in vals])
    intervals, start = [], None
    for i, ins in enumerate(inside):
        if ins and start is None:
            start = i
        if start is not None and (not ins or i == len(inside) - 1):
            end = i if ins else i - 1
            if end >= start:
                intervals.append((float(vals[start]), float(vals[end])))
            start = None
    return intervals or mp_support(y)


def ledoit_wolf_cov(X):
    """Ledoit-Wolf shrinkage toward scaled identity (assume_centered=False)."""
    n, p = X.shape
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    b2 = 0.0
    for i in range(n):
        xi = Xc[i][:, None]
        b2 += np.sum((xi @ xi.T - S) ** 2)
    b2 = min(d2, b2 / (n * n * p))
    shr = b2 / d2 if d2 > 0 else 0.0
    return (1 - shr) * S + shr * mu * np.eye(p)


def chi2_ppf(q, dof):
    z = {0.90: 1.2815515594, 0.95: 1.6448536269}.get(q, 1.2815515594)
    return dof * (1.0 - 2.0 / (9 * dof) + z * math.sqrt(2.0 / (9 * dof))) ** 3
TS = np.array([100., 100.] + [0.] * 15, dtype="float32")  # only track dims matter here
N_ENS = int(os.environ.get("N_ENS", "100"))
EPS = 1e-6

# ---- rebuild v3 ----
g = {"torch": torch, "nn": nn, "F": __import__("torch.nn.functional", fromlist=["x"]), "math": math, "np": np}
src = open("train_track_v3.py").read()
for blk in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM = len\(KIN_COLS\), len\(THERMO_COLS\)",
            r"def sinusoidal.*?return e", r"def encoder.*?return nn\.TransformerEncoder\(layer, depth\)",
            r"def decoder.*?return nn\.TransformerDecoder\(layer, depth\)",
            r"class TrackFormerV3.*?return state, logscale"]:
    exec(re.search(blk, src, re.S).group(0), g)
model = g["TrackFormerV3"]().to(DEV)
ck = torch.load("models/trackformer_v3_15M_fp16.pt", map_location="cpu", weights_only=False)
model.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["model"].items()})
model.eval()

z = np.load("track_build/track_windows_v2.npz", allow_pickle=True)
tmean, tstd = z["track_mean"].astype("float32"), z["track_std"].astype("float32")
v0 = (z["track"][:, -1, 2:4] * tstd[2:4] + tmean[2:4]).astype("float32")
trk, tgt = z["track"].astype("float32"), z["target"].astype("float32")
yr, bs, sid = z["year"].astype(int), z["basin"].astype(str), z["storm_id"].astype(str)
wp = np.where((yr >= 2020) & (bs == "WP"))[0]


def enable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()


MULTIVAR = os.environ.get("MULTIVAR", "0") == "1"


def build_vec(st_np, physical=False):
    """[n,20,17] state -> feature vector. Track (cumulative pos, km) + optionally vmax (kt) +
    pressure (hPa) = a genuine multi-scale (two-population) vector. physical=True for raw targets;
    otherwise the model's normalized output is rescaled by TARGET_SCALE."""
    sc = np.array([1., 1., 1., 1.] if physical else [100., 100., 35., 20.])
    pos = np.cumsum(st_np[..., :2] * sc[:2], 1).reshape(len(st_np), 40)  # 40 track km
    if not MULTIVAR:
        return pos
    vmax = (st_np[..., 2] * sc[2]).reshape(len(st_np), 20)               # 20 kt
    pres = (st_np[..., 3] * sc[3]).reshape(len(st_np), 20)               # 20 hPa
    return np.concatenate([pos, vmax, pres], axis=1)                     # 80-dim


@torch.no_grad()
def ensemble_positions(idx):
    model.eval(); enable_dropout(model)
    tr_t = torch.from_numpy(trk[idx]).to(DEV); v0_t = torch.from_numpy(v0[idx]).to(DEV)
    d = 80 if MULTIVAR else 40
    samples = np.zeros((len(idx), N_ENS, d), dtype="float32")
    for s in range(N_ENS):
        st, _ = model(tr_t, v0_t)
        samples[:, s] = build_vec(st.float().cpu().numpy())
    model.eval()
    truth = build_vec(tgt[idx], physical=True)
    return samples, truth


def clean_spectrum(vals, vecs, edge):
    signal = vals > edge
    bulk = vals[~signal]
    floor = float(np.median(bulk)) if bulk.size else float(max(EPS, edge / 2))
    cv = np.where(signal, vals, floor)
    C = (vecs * cv) @ vecs.T
    return (C + C.T) / 2


def prep_window(ens):
    """Precompute the (a,beta)-independent pieces for one window's ensemble [N,p]."""
    n, p = ens.shape
    m = ens.mean(0); X = ens - m
    S = (X.T @ X) / n
    vals, vecs = np.linalg.eigh((S + S.T) / 2); vals = np.clip(vals, EPS, None)
    scale = float(np.mean(np.diag(S))) + EPS
    lw = ledoit_wolf_cov(ens)
    return {"S": S, "vals": vals, "vecs": vecs, "scale": scale, "p": p,
            "diag": np.diag(np.diag(S)), "lw": (lw + lw.T) / 2}


def estimators(pw, mp_unit, gen_unit):
    """Build the 5 covariance estimators from precomputed pieces + unit MP edges."""
    p, sc, vals, vecs = pw["p"], pw["scale"], pw["vals"], pw["vecs"]
    I = EPS * np.eye(p)
    return {"diagonal": pw["diag"] + I, "sample": pw["S"] + I, "ledoit_wolf": pw["lw"] + I,
            "classical_mp": clean_spectrum(vals, vecs, sc * mp_unit) + I,
            "generalized_mp": clean_spectrum(vals, vecs, sc * gen_unit) + I}


def nll(truth, mean, Sigma, lam):
    Sig = lam * Sigma
    sign, logdet = np.linalg.slogdet(Sig)
    if sign <= 0:
        Sig = Sig + 1e-3 * np.eye(len(Sig)); sign, logdet = np.linalg.slogdet(Sig)
    d = truth - mean
    sol = np.linalg.solve(Sig, d)
    return 0.5 * (d @ sol + logdet + len(d) * math.log(2 * math.pi))


def coverage(truth, mean, Sigma, lam, q=0.90):
    Sig = lam * Sigma
    d = truth - mean
    md2 = d @ np.linalg.solve(Sig, d)
    return md2 <= chi2_ppf(q, len(d))


print(f"device {DEV} | N_ENS {N_ENS} | WP-2020+ windows {len(wp)}")
ens, truth = ensemble_positions(wp)
means = ens.mean(1)
P = ens.shape[2]; Y = P / N_ENS
mp_unit = mp_support(Y)[0][1]
# split storms 50/50
ust = np.unique(sid[wp]); rng = np.random.RandomState(0); rng.shuffle(ust)
fit_st = set(ust[:len(ust) // 2])
fit = np.array([i for i, s in enumerate(sid[wp]) if s in fit_st])
evl = np.array([i for i in range(len(wp)) if i not in set(fit)])
print(f"fit windows {len(fit)} | eval windows {len(evl)} | y=p/N={Y:.3f}")

# precompute (a,beta)-independent pieces once per window
FIT = list(fit[:700]); EVL = list(evl[:1400])
pw = {i: prep_window(ens[i]) for i in FIT + EVL}

# fit global (a,beta): grid search minimizing mean generalized_mp NLL on FIT (unit edge computed once/combo)
best = (1e18, 2.0, 0.25, mp_unit)
for a in [2., 4., 6., 10., 20.]:
    for beta in [0.05, 0.1, 0.25, 0.5]:
        gu = max(hi for _, hi in generalized_support(a, beta, Y))
        tot = sum(nll(truth[i], means[i], estimators(pw[i], mp_unit, gu)["generalized_mp"], 1.0) for i in FIT[:350])
        if tot < best[0]:
            best = (tot, a, beta, gu)
_, A, BETA, GEN_UNIT = best
print(f"fitted two-point params: a={A}, beta={BETA} (gen edge {GEN_UNIT:.2f} vs MP edge {mp_unit:.2f})")

names = ["diagonal", "sample", "ledoit_wolf", "classical_mp", "generalized_mp"]
cache = {i: estimators(pw[i], mp_unit, GEN_UNIT) for i in FIT + EVL}
def fit_lambda(name, idxs):
    md = [float((truth[i] - means[i]) @ np.linalg.solve(cache[i][name], truth[i] - means[i])) for i in idxs]
    return max(1e-6, float(np.mean(md)) / P)

print(f"\n{'estimator':16s} {'eval_NLL':>10s} {'cover@90%':>10s}  (lower NLL better; coverage near 0.90)")
rows = {}
for nm in names:
    lam = fit_lambda(nm, FIT)
    ns = [nll(truth[i], means[i], cache[i][nm], lam) for i in EVL]
    cov = float(np.mean([coverage(truth[i], means[i], cache[i][nm], lam) for i in EVL]))
    rows[nm] = (float(np.mean(ns)), cov)
    print(f"{nm:16s} {np.mean(ns):>10.2f} {cov:>10.3f}")

best_nm = min(rows, key=lambda k: rows[k][0])
print(f"\nBEST calibration: {best_nm}")
print(json.dumps({k: {"nll": round(v[0], 2), "coverage": round(v[1], 3)} for k, v in rows.items()}))
open("track_build/rmt_uncertainty_result.json", "w").write(json.dumps(
    {"a": A, "beta": BETA, "N_ENS": N_ENS, "results": {k: {"nll": v[0], "coverage": v[1]} for k, v in rows.items()},
     "best": best_nm}, indent=2))
print("saved track_build/rmt_uncertainty_result.json")
