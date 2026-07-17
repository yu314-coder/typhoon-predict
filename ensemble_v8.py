"""DeepMind-style probabilistic ensemble on TrackFormer v8 (IBTrACS-only, no retraining).

Generate coherent "possible routes" by sampling the model's own forecast-error distribution over the
full 20-lead per-step motion (cross-lead correlated -> smooth tracks), then aggregate as a
DISTRIBUTION (ensemble-mean track + cone of uncertainty + strike probability), scored with the proper
multivariate CRPS (energy score) -- not a plain arithmetic mean.

We compare the covariance that generates the routes:
  diagonal | sample | ledoit_wolf | generalized-MP (RMT, the Yau paper)
Fit on train+val storms, scored on 2020+ test storms. The covariance is 40-dim (20 leads x 2 axes);
its cross-lead structure is what makes sampled routes coherent instead of jagged.

Also dumps one WP-2020+ storm's ensemble (routes + truth + cone) to JSON for visualization.
"""
import re, math, json, os
import numpy as np
import torch
import torch.nn as nn

dev = torch.device("cpu")
N_ENS = int(os.environ.get("N_ENS", "50"))
EPS = 1e-6
rng = np.random.RandomState(0)


# ---- inlined RMT helpers (generalized two-point MP edge; pure numpy) ----
def mp_edge(y):
    return (1.0 + math.sqrt(y)) ** 2


def _cubic(z, a, beta, y):
    return np.array([a * z, a * (z - y + 1.0) + z, a + z - y + 1.0 - y * beta * (a - 1.0), 1.0])


def gen_edge(a, beta, y, grid=4000):
    if abs(a - 1.0) < 1e-7:
        return mp_edge(y)
    zmax = max(10.0, 8.0 * max(1.0, a) * (1.0 + math.sqrt(max(1.0, y))) ** 2)
    vals = np.linspace(1e-7, zmax, grid)
    hi = mp_edge(y)
    for zz in vals:
        if np.any(np.imag(np.roots(_cubic(zz, a, beta, y))) > 1e-7):
            hi = zz
    return hi


def ledoit_wolf_cov(X):
    n, p = X.shape
    Xc = X - X.mean(0); S = (Xc.T @ Xc) / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    b2 = sum(np.sum((Xc[i][:, None] @ Xc[i][None, :] - S) ** 2) for i in range(n)) / (n * n * p)
    shr = min(d2, b2) / d2 if d2 > 0 else 0.0
    return (1 - shr) * S + shr * mu * np.eye(p)


def clean(S, edge_scale_fn):
    vals, vecs = np.linalg.eigh((S + S.T) / 2); vals = np.clip(vals, EPS, None)
    scale = float(np.mean(np.diag(S))) + EPS
    edge = scale * edge_scale_fn(vals)
    sig = vals > edge; bulk = vals[~sig]
    floor = float(np.median(bulk)) if bulk.size else float(max(EPS, edge / 2))
    cv = np.where(sig, vals, floor)
    C = (vecs * cv) @ vecs.T
    return (C + C.T) / 2


# ---- v8 model ----
g = {"torch": torch, "nn": nn, "F": __import__("torch.nn.functional", fromlist=["x"]), "math": math, "np": np}
src = open("train_track_v3.py").read()
for blk in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM = len\(KIN_COLS\), len\(THERMO_COLS\)",
            r"def sinusoidal.*?return e", r"def encoder.*?return nn\.TransformerEncoder\(layer, depth\)",
            r"def decoder.*?return nn\.TransformerDecoder\(layer, depth\)",
            r"class TrackFormerV3.*?return state, logscale"]:
    exec(re.search(blk, src, re.S).group(0), g)
model = g["TrackFormerV3"]().eval()
ck = torch.load("models/trackformer_v8_15M_fp16.pt", map_location="cpu", weights_only=False)
model.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["model"].items()})

z = np.load("track_build/track_windows_v8.npz", allow_pickle=True)
tmean, tstd = z["track_mean"].astype("float32"), z["track_std"].astype("float32")
v0 = (z["track"][:, -1, 2:4] * tstd[2:4] + tmean[2:4]).astype("float32")
trk, tgt = z["track"].astype("float32"), z["target"]
nl, yr, bs, sid = z["n_leads"].astype(int), z["year"].astype(int), z["basin"].astype(str), z["storm_id"].astype(str)


@torch.no_grad()
def pred_motion(idx):
    out = np.zeros((len(idx), 20, 2), dtype="float64")
    for s in range(0, len(idx), 256):
        b = idx[s:s + 256]
        st, _ = model(torch.from_numpy(trk[b]), torch.from_numpy(v0[b]))
        out[s:s + len(b)] = (st[..., :2] * 100.0).numpy()
    return out


full = nl == 20
fit_idx = np.where(full & (yr <= 2019))[0]
wp_test = np.where(full & (yr >= 2020) & (bs == "WP"))[0]
ab_test = np.where(full & (yr >= 2020))[0]
print(f"fit windows {len(fit_idx)} | WP test {len(wp_test)} | all-basin test {len(ab_test)} | N_ens {N_ENS}")

# residuals (truth - pred) per-step motion, 40-dim, on fit set
pm_fit = pred_motion(fit_idx)
res_fit = (tgt[fit_idx][..., :2] - pm_fit).reshape(len(fit_idx), 40)
S = np.cov(res_fit.T)                                   # 40x40 sample covariance (cross-lead correlated)
y_ratio = 40 / len(fit_idx)
# fit two-point (a,beta) by max Gaussian log-lik on a held-out slice of fit residuals
half = len(fit_idx) // 2
Sfit = np.cov(res_fit[:half].T)
best = (1e18, 2.0, 0.1)
for a in [2., 4., 8., 16.]:
    for beta in [0.05, 0.1, 0.25]:
        e = gen_edge(a, beta, 40 / half)
        C = clean(Sfit, lambda v: e / (float(np.mean(np.diag(Sfit))) + EPS))
        C = C + EPS * np.eye(40)
        sign, ld = np.linalg.slogdet(C)
        val = np.mean([r @ np.linalg.solve(C, r) for r in res_fit[half:half + 300]]) + ld
        if val < best[0]:
            best = (val, a, beta)
_, A, BETA = best
GE = gen_edge(A, BETA, y_ratio)
COVS = {
    "diagonal": np.diag(np.diag(S)) + EPS * np.eye(40),
    "sample": S + EPS * np.eye(40),
    "ledoit_wolf": ledoit_wolf_cov(res_fit) + EPS * np.eye(40),
    "generalized_mp": clean(S, lambda v: GE / (float(np.mean(np.diag(S))) + EPS)) + EPS * np.eye(40),
}
print(f"fitted RMT two-point: a={A}, beta={BETA} (edge {GE:.2f} vs MP {mp_edge(y_ratio):.2f})")


def energy_score(members, truth):
    # members: [N,40] track (cumulative pos), truth: [40].  ES = mean||X-y|| - 0.5 mean||X-X'||
    d1 = np.mean(np.linalg.norm(members - truth[None], axis=1))
    diff = members[:, None, :] - members[None, :, :]
    d2 = np.mean(np.linalg.norm(diff, axis=2))
    return d1 - 0.5 * d2


def evaluate(test_idx, Sigma, dump_storm=None):
    pm = pred_motion(test_idx)
    L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(40))
    es, det_err, ensmean_err, cover = [], [], [], []
    dump = None
    for k, w in enumerate(range(len(test_idx))):
        mu = pm[w].reshape(40)
        eps = (L @ rng.standard_normal((40, N_ENS))).T            # [N,40] correlated noise
        sm = (mu[None] + eps).reshape(N_ENS, 20, 2)
        tracks = np.cumsum(sm, axis=1).reshape(N_ENS, 40)          # [N,40] coherent routes
        truth = np.cumsum(tgt[test_idx[w]][..., :2], 0).reshape(40)
        det = np.cumsum(mu.reshape(20, 2), 0).reshape(40)
        es.append(energy_score(tracks, truth))
        det_err.append(np.linalg.norm(det - truth) / math.sqrt(20))
        ensmean_err.append(np.linalg.norm(tracks.mean(0) - truth) / math.sqrt(20))
        # 90% coverage: is truth within the per-lead 90% ensemble radius (final lead)?
        fin = tracks.reshape(N_ENS, 20, 2)[:, -1]; tfin = truth.reshape(20, 2)[-1]
        r90 = np.percentile(np.linalg.norm(fin - fin.mean(0), axis=1), 90)
        cover.append(np.linalg.norm(tfin - fin.mean(0)) <= r90)
        if dump_storm is not None and test_idx[w] == dump_storm:
            dump = {"routes": tracks.reshape(N_ENS, 20, 2).tolist(),
                    "truth": truth.reshape(20, 2).tolist(),
                    "mean": tracks.mean(0).reshape(20, 2).tolist()}
    return dict(energy=round(float(np.mean(es)), 2), det_km=round(float(np.mean(det_err)), 1),
                ensmean_km=round(float(np.mean(ensmean_err)), 1), cover90=round(float(np.mean(cover)), 3)), dump


print(f"\n{'covariance':16s} {'energy_score':>13s} {'det_km':>8s} {'ensmean_km':>11s} {'cover90':>8s}")
res = {}
for name, Sig in COVS.items():
    r, _ = evaluate(wp_test, Sig)
    res[name] = r
    print(f"{name:16s} {r['energy']:>13.2f} {r['det_km']:>8.1f} {r['ensmean_km']:>11.1f} {r['cover90']:>8.3f}")
best_cov = min(res, key=lambda k: res[k]["energy"])
print(f"\nbest ensemble (lowest energy score): {best_cov}")

# dump one representative WP storm's ensemble (longest track in test) for visualization
storm_ids_test = sid[wp_test]
uniq, counts = np.unique(storm_ids_test, return_counts=True)
pick_storm = uniq[np.argmax(counts)]
pick_w = wp_test[np.where(storm_ids_test == pick_storm)[0][len(np.where(storm_ids_test == pick_storm)[0]) // 2]]
_, dump = evaluate(np.array([pick_w]), COVS[best_cov], dump_storm=pick_w)
out = {"storm": str(pick_storm), "cov": best_cov, "N": N_ENS, "wp_results": res,
       "rmt": {"a": A, "beta": BETA}, "ensemble": dump}
json.dump(out, open("track_build/ensemble_v8.json", "w"))
print(f"\nsaved track_build/ensemble_v8.json (storm {pick_storm}, {best_cov} cov, {N_ENS} routes)")
