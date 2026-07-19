"""RMT-weighted ensemble combination for v17.

    !wget -q -O rmt.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v17_rmt.py
    exec(open('rmt.py').read())

Run after the notebook's definition cells (data / model / loss / eval). It rebuilds CKPTS from the
Drive mirror, so nothing is retrained.

WHAT RMT CAN AND CANNOT DO HERE — worth stating plainly, because it is easy to oversell:

  * Cleaning a covariance does NOT move an equal-weight mean. For exchangeable members the mean is
    the mean, whatever you do to the covariance. RMT only bites when the weights are allowed to be
    unequal, or when the covariance itself is the answer (uncertainty).
  * The 5x5 between-seed error covariance is estimated from ~16k validation windows x 40 dims.
    The aspect ratio q = 5/16342 is ~0.0003, so the Marchenko-Pastur noise band is negligible and
    cleaning it changes essentially nothing. We compute it anyway and REPORT the size of the
    correction, rather than claiming a benefit we did not measure.
  * Where RMT genuinely earns its place is the PER-WINDOW MC-dropout covariance: 40 dimensions from
    M members, q = 40/M ~ 1. There the sample covariance is rank-deficient and its small
    eigenvalues are pure noise. Eigenvalue clipping (Laloux et al.) fixes the shape, which is what
    makes the uncertainty calibrated.

So: seed weights come from a validation-fitted min-variance solve (RMT checked, reported), and RMT
does the work on the MC-dropout covariance for calibration.
"""
import json, math, os
import numpy as np, torch

MC_MEMBERS = int(os.environ.get("MC_MEMBERS", "24"))
SEEDS = 5

# ---- rebuild CKPTS from the Drive mirror; do not retrain ------------------
cands = [f"{DATA}/v17_seed{i}.pt" for i in range(SEEDS)]
CKPTS = [c for c in cands if os.path.exists(c)]
assert len(CKPTS) >= 2, f"need >=2 checkpoints, found {CKPTS}"
print(f"using {len(CKPTS)} checkpoints from Drive", flush=True)
mods = [load_model(c) for c in CKPTS]
K = len(mods)

full = z["n_leads"].astype(int) == 20
VAL = np.array([i for i in va_idx if full[i]])                       # fit weights here
TEST = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])   # report here
print(f"fit on {len(VAL)} validation windows, report on {len(TEST)} WP+EP test windows\n", flush=True)


@torch.no_grad()
def cum_positions(model, idx, train_mode=False):
    """Cumulative (E,N) per lead -> [n,20,2] in km."""
    was = model.training
    model.train(train_mode)                      # train_mode=True keeps dropout live for MC
    P = []
    for i in range(0, len(idx), 256):
        j = idx[i:i + 256]
        s = model(torch.from_numpy(track[j]).to(DEVICE),
                  torch.from_numpy(vpair[j]).to(DEVICE),
                  torch.from_numpy(SLP[j]).to(DEVICE))[0]
        P.append((s * TARGET_SCALE).float().cpu().numpy()[..., :2])
    model.train(was)
    return np.cumsum(np.concatenate(P), 1)


def truth(idx):
    return np.cumsum(target[idx][..., :2], 1)


def err_km(pred, idx):
    return np.sqrt(((pred - truth(idx)) ** 2).sum(-1)).mean()


# ---- 1. between-seed error covariance on validation ----------------------
print("computing per-seed validation errors ...", flush=True)
EV = np.stack([cum_positions(m, VAL) - truth(VAL) for m in mods])     # [K,n,20,2]
E = EV.reshape(K, -1)                                                  # [K, n*40]
C = (E @ E.T) / E.shape[1]                                             # [K,K] mean squared error / cross-error
print("between-seed error covariance (km^2), diagonal = each seed's MSE:")
for i in range(K):
    print("   " + "  ".join(f"{C[i, j]:9.0f}" for j in range(K)))

# Marchenko-Pastur check on C: is any of this structure noise?
q = K / E.shape[1]
ev = np.linalg.eigvalsh(C)
sigma2 = ev.mean()
mp_hi = sigma2 * (1 + math.sqrt(q)) ** 2
print(f"\nMP check: q = K/n = {q:.2e}, noise edge lambda+ = {mp_hi:.0f}, "
      f"largest eigenvalue = {ev.max():.0f}")
print(f"  eigenvalues above the edge: {(ev > mp_hi).sum()}/{K}"
      f"  -> at this aspect ratio MP cleaning is a no-op; reporting it rather than claiming a gain")

# ---- 2. minimum-variance weights ----------------------------------------
one = np.ones(K)
Cinv = np.linalg.pinv(C)
w_mv = Cinv @ one / (one @ Cinv @ one)
# non-negative variant: a negative weight means betting against a seed, which does not generalise
w_nn = np.clip(w_mv, 0, None); w_nn = w_nn / w_nn.sum()
print(f"\nequal weights      {np.round(one / K, 4)}")
print(f"min-variance       {np.round(w_mv, 4)}   (sum {w_mv.sum():.3f})")
print(f"min-variance >= 0  {np.round(w_nn, 4)}")

# ---- 3. evaluate on test -------------------------------------------------
print("\nevaluating on WP+EP test ...", flush=True)
PT = np.stack([cum_positions(m, TEST) for m in mods])                  # [K,n,20,2]
res = {}
res["equal mean (v17 as reported)"] = err_km(PT.mean(0), TEST)
res["min-variance weights"] = err_km(np.tensordot(w_mv, PT, axes=1), TEST)
res["min-variance, non-negative"] = err_km(np.tensordot(w_nn, PT, axes=1), TEST)
best_single = min(err_km(PT[i], TEST) for i in range(K))
res["best single seed"] = best_single

# ---- 4. MC-dropout ensemble + RMT-clipped covariance ---------------------
print(f"MC-dropout ensemble, {MC_MEMBERS} passes x {K} seeds ...", flush=True)
MC = np.stack([cum_positions(m, TEST, train_mode=True)
               for m in mods for _ in range(MC_MEMBERS)])              # [K*M,n,20,2]
res[f"MC-dropout mean ({MC.shape[0]} members)"] = err_km(MC.mean(0), TEST)

print("\n" + "=" * 66)
print(f"{'combination':34s} {'120h-avg track err':>20s}")
print("=" * 66)
base = res["equal mean (v17 as reported)"]
for k, v in sorted(res.items(), key=lambda kv: kv[1]):
    print(f"{k:34s} {v:14.2f} km  {v-base:+7.2f}")
print("=" * 66)

# ---- 5. does RMT cleaning improve the UNCERTAINTY? -----------------------
# per-window 40-dim MC covariance: q = 40/members, squarely in the RMT regime
n = len(TEST)
X = MC.reshape(MC.shape[0], n, 40)
T40 = truth(TEST).reshape(n, 40)
M_ = X.shape[0]
qw = 40.0 / M_
cov_raw = np.empty(n); cov_cln = np.empty(n)
for i in range(n):
    Y = X[:, i, :] - X[:, i, :].mean(0)
    S = (Y.T @ Y) / max(1, M_ - 1)
    d = T40[i] - X[:, i, :].mean(0)
    ev_, U = np.linalg.eigh(S)
    ev_ = np.clip(ev_, 0, None)
    s2 = ev_.mean()
    edge = s2 * (1 + math.sqrt(qw)) ** 2
    keep = ev_ > edge
    ev_c = ev_.copy()
    if (~keep).any():                       # clip the noise bulk to its mean, preserving the trace
        ev_c[~keep] = ev_[~keep].mean()
    inv_raw = U @ np.diag(1.0 / np.clip(ev_, 1e-6, None)) @ U.T
    inv_cln = U @ np.diag(1.0 / np.clip(ev_c, 1e-6, None)) @ U.T
    cov_raw[i] = d @ inv_raw @ d
    cov_cln[i] = d @ inv_cln @ d
from math import inf
chi90 = 51.805                               # chi-square 90th percentile, 40 dof
print(f"\n90% coverage of the truth under the ensemble covariance (target 0.90):")
print(f"  raw sample covariance   {(cov_raw <= chi90).mean():.3f}")
print(f"  RMT eigenvalue-clipped  {(cov_cln <= chi90).mean():.3f}   (q = 40/{M_} = {qw:.2f})")
print("  closer to 0.90 is better calibrated; this is where RMT actually does work")

json.dump({k: float(v) for k, v in res.items()}, open("/content/v17_rmt.json", "w"))
print("\nsaved /content/v17_rmt.json")
