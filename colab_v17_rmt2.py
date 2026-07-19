"""Two follow-ups the first RMT pass left open.

    !wget -q -O rmt2.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v17_rmt2.py
    exec(open('rmt2.py').read())

A) SEED-COUNT CONTROL. v17 (462.82 km) used 5 seeds; v16-noSST (466.30 km) used 3. Part of that
   gap is simply a bigger ensemble, not the new loss. v16-noSST's checkpoints have a 5-channel
   conv and cannot be loaded next to 4-channel v17, so the clean control is the other direction:
   ensemble only 3 of v17's seeds and compare THAT to 466.30. Reported over all C(5,3)=10 subsets
   so the answer does not depend on which three happen to be picked.

B) CALIBRATED COVERAGE. The first pass scored coverage on the raw MC-dropout covariance and got
   ~0 for every estimator, because MC-dropout is badly under-dispersed -- its spread is far
   smaller than the true error. That swamps the thing RMT actually fixes, which is the SHAPE of
   the covariance, not its scale. Following rmt_uncertainty.py, we fit a single variance-inflation
   factor lambda per estimator on a FIT half of the storms and report coverage on the held-out
   EVAL half. Same lambda treatment for both estimators, so the comparison isolates shape.
"""
import json, math, os, itertools
import numpy as np, torch

MC = int(os.environ.get("MC_MEMBERS", "24"))
cands = [f"{DATA}/v17_seed{i}.pt" for i in range(5)]
CK = [c for c in cands if os.path.exists(c)]
mods = [load_model(c) for c in CK]
K = len(mods)
full = z["n_leads"].astype(int) == 20
TEST = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
print(f"{K} seeds | {len(TEST)} WP+EP test windows\n", flush=True)


@torch.no_grad()
def cum(model, idx, mc=False):
    was = model.training
    model.train(mc)
    P = []
    for i in range(0, len(idx), 256):
        j = idx[i:i + 256]
        s = model(torch.from_numpy(track[j]).to(DEVICE),
                  torch.from_numpy(vpair[j]).to(DEVICE),
                  torch.from_numpy(SLP[j]).to(DEVICE))[0]
        P.append((s * TARGET_SCALE).float().cpu().numpy()[..., :2])
    model.train(was)
    return np.cumsum(np.concatenate(P), 1)


T = np.cumsum(target[TEST][..., :2], 1)
def err(p): return float(np.sqrt(((p - T) ** 2).sum(-1)).mean())

# ---------------- A. seed-count control ----------------------------------
PT = np.stack([cum(m, TEST) for m in mods])
print("=" * 68)
print("A. SEED-COUNT CONTROL  -- is v17's gain the loss, or just more seeds?")
print("=" * 68)
singles = [err(PT[i]) for i in range(K)]
print(f"single seeds:      {'  '.join(f'{s:.1f}' for s in singles)}   mean {np.mean(singles):.2f}")
sub3 = [err(PT[list(c)].mean(0)) for c in itertools.combinations(range(K), 3)]
sub4 = [err(PT[list(c)].mean(0)) for c in itertools.combinations(range(K), 4)]
all5 = err(PT.mean(0))
print(f"\n3-seed ensembles:  n={len(sub3)}  mean {np.mean(sub3):.2f}  "
      f"range {min(sub3):.2f}-{max(sub3):.2f}")
print(f"4-seed ensembles:  n={len(sub4)}  mean {np.mean(sub4):.2f}  "
      f"range {min(sub4):.2f}-{max(sub4):.2f}")
print(f"5-seed ensemble:                 {all5:.2f}")
print(f"\nv16-noSST (3 seeds, old loss):   466.30")
print(f"v17       (3 seeds, new loss):   {np.mean(sub3):.2f}   "
      f"-> loss change is worth {466.30 - np.mean(sub3):+.2f} km at equal seed count")
print(f"going 3 -> 5 seeds is worth      {np.mean(sub3) - all5:+.2f} km on top")

# ---------------- B. calibrated coverage ---------------------------------
print("\n" + "=" * 68)
print("B. COVERAGE with a fitted variance-inflation lambda (shape isolated)")
print("=" * 68)
print(f"building MC-dropout ensemble, {MC} x {K} = {MC*K} members ...", flush=True)
X = np.stack([cum(m, TEST, mc=True) for m in mods for _ in range(MC)]).reshape(MC * K, len(TEST), 40)
M_ = X.shape[0]
T40 = T.reshape(len(TEST), 40)
mu = X.mean(0)
D = T40 - mu                                        # residual of the ensemble mean
qw = 40.0 / M_

# split storms (not windows) so the fit and eval halves share no storm
sids_t = sids[TEST]
uq = np.unique(sids_t)
rng = np.random.RandomState(0); rng.shuffle(uq)
fit_s = set(uq[: len(uq) // 2])
FIT = np.array([i for i, s in enumerate(sids_t) if s in fit_s])
EVAL = np.array([i for i, s in enumerate(sids_t) if s not in fit_s])
print(f"fit on {len(FIT)} windows / eval on {len(EVAL)} windows, disjoint storms")

def maha(idx_set, clip):
    out = np.empty(len(idx_set))
    for k, i in enumerate(idx_set):
        Y = X[:, i, :] - mu[i]
        S = (Y.T @ Y) / max(1, M_ - 1)
        ev, U = np.linalg.eigh(S); ev = np.clip(ev, 0, None)
        if clip:
            edge = ev.mean() * (1 + math.sqrt(qw)) ** 2
            bad = ev <= edge
            if bad.any(): ev = ev.copy(); ev[bad] = ev[bad].mean()
        inv = U @ np.diag(1.0 / np.clip(ev, 1e-9, None)) @ U.T
        out[k] = D[i] @ inv @ D[i]
    return out

CHI90 = 51.805          # chi-square 90th percentile, 40 dof
for name, clip in [("raw sample covariance", False), ("RMT eigenvalue-clipped", True)]:
    m_fit = maha(FIT, clip)
    # one scalar lambda so that the FIT half hits 90% -- removes the dispersion error,
    # leaving only the covariance SHAPE to be judged on EVAL
    lam = np.percentile(m_fit, 90) / CHI90
    m_ev = maha(EVAL, clip) / lam
    print(f"  {name:24s} lambda {lam:9.1f}   coverage on EVAL {(m_ev <= CHI90).mean():.3f}")
print("  target 0.90; both get the same lambda treatment, so the difference is shape alone")
print("  lambda >> 1 quantifies how badly MC-dropout under-disperses")
