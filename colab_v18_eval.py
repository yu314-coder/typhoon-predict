"""v18 vs v17, measured so the answer is actually resolvable.

    !wget -q -O v18e.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v18_eval.py
    exec(open('v18e.py').read())

The seed-subset control showed 3-seed ensembles of the SAME model spanning 461.7-473.9 km. Any
comparison of two single ensembles is therefore blind to differences under ~10 km, which is
larger than most of the effects claimed in this project so far. So:

  * For each ensemble size k, report mean and full range over ALL C(n,k) subsets.
  * Compare v17 and v18 at MATCHED k, never 5-vs-8.
  * Paired bootstrap over storms (not windows -- windows from one storm are not independent)
    for the headline k, giving a confidence interval on the DIFFERENCE.
"""
import os, json, itertools
import numpy as np, torch

full = z["n_leads"].astype(int) == 20
TEST = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = np.cumsum(target[TEST][..., :2], 1)
sid_t = sids[TEST]
print(f"{len(TEST)} WP+EP test windows, {len(np.unique(sid_t))} storms\n", flush=True)


@torch.no_grad()
def cum(model):
    P = []
    for i in range(0, len(TEST), 256):
        j = TEST[i:i + 256]
        s = model(torch.from_numpy(track[j]).to(DEVICE),
                  torch.from_numpy(vpair[j]).to(DEVICE),
                  torch.from_numpy(SLP[j]).to(DEVICE))[0]
        P.append((s * TARGET_SCALE).float().cpu().numpy()[..., :2])
    return np.cumsum(np.concatenate(P), 1)


def load_all(paths):
    out = []
    for p in paths:
        if os.path.exists(p):
            m = TrackFormerV17().to(DEVICE).eval()
            m.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=False)["model"])
            out.append(cum(m))
    return np.stack(out) if out else None


P17 = load_all([f"{DATA}/v17_seed{i}.pt" for i in range(5)])
P18 = load_all([f"/content/v18_seed{i}.pt" for i in range(12)] +
               [f"{DATA}/v18_seed{i}.pt" for i in range(12)])
print(f"loaded v17: {0 if P17 is None else len(P17)} seeds | "
      f"v18: {0 if P18 is None else len(P18)} seeds\n", flush=True)

# per-window error, so storms can be resampled later
def per_window(p): return np.sqrt(((p - T) ** 2).sum(-1)).mean(1)   # [n] averaged over leads
def score(p): return float(per_window(p).mean())


def subset_table(P, label):
    n = len(P)
    print(f"{label} ({n} seeds)")
    print(f"  {'k':>2s} {'subsets':>8s} {'mean':>9s} {'best':>9s} {'worst':>9s} {'spread':>8s}")
    rows = {}
    for k in range(1, n + 1):
        combos = list(itertools.combinations(range(n), k))
        if len(combos) > 70:
            idx = np.linspace(0, len(combos) - 1, 70).astype(int)
            combos = [combos[i] for i in idx]
        v = [score(P[list(c)].mean(0)) for c in combos]
        rows[k] = v
        print(f"  {k:2d} {len(combos):8d} {np.mean(v):9.2f} {min(v):9.2f} {max(v):9.2f} "
              f"{max(v)-min(v):8.2f}")
    return rows


R17 = subset_table(P17, "v17") if P17 is not None else {}
print()
R18 = subset_table(P18, "v18") if P18 is not None else {}

if R17 and R18:
    print("\n" + "=" * 62)
    print("MATCHED-k COMPARISON  (negative = v18 better)")
    print("=" * 62)
    print(f"  {'k':>2s} {'v17 mean':>10s} {'v18 mean':>10s} {'delta':>9s} {'v17 range':>18s}")
    for k in sorted(set(R17) & set(R18)):
        a, b = np.mean(R17[k]), np.mean(R18[k])
        print(f"  {k:2d} {a:10.2f} {b:10.2f} {b-a:+9.2f}   "
              f"{min(R17[k]):7.1f}-{max(R17[k]):.1f}")

    # paired storm bootstrap at the largest shared k
    k = max(set(R17) & set(R18))
    e17 = per_window(P17[:k].mean(0)); e18 = per_window(P18[:k].mean(0))
    storms = np.unique(sid_t)
    rng = np.random.RandomState(0)
    diffs = []
    for _ in range(2000):
        pick = rng.choice(storms, len(storms), replace=True)
        m = np.concatenate([np.where(sid_t == s)[0] for s in pick])
        diffs.append(e18[m].mean() - e17[m].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\npaired storm bootstrap at k={k}, 2000 resamples:")
    print(f"  v18 - v17 = {diffs.mean():+.2f} km   95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  {'SIGNIFICANT' if (lo < 0) == (hi < 0) else 'NOT SIGNIFICANT'} "
          f"-- CI {'excludes' if (lo < 0) == (hi < 0) else 'straddles'} zero")
    json.dump({"delta": float(diffs.mean()), "ci": [float(lo), float(hi)], "k": int(k)},
              open("/content/v18_vs_v17.json", "w"))
