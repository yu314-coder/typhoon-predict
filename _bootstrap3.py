"""Paired-storm bootstrap over v20 / v21 / v22, all at matched 5 seeds.

Whole STORMS are resampled, not windows: windows from one storm are strongly correlated, and
resampling them independently manufactures significance. Same harness that judged v18
(+3.78 km, CI [-2.56, +9.75]) and v20 (-10.35 km, CI [-22.54, +0.82]).

Predictions come from track_build/errdecomp.npz, written by _errdecomp.py using the forward
validated against Colab (452.47 / 443.62 / 443.41). No re-inference here.

Paired: every resample uses the SAME storms for both arms, so the comparison is not polluted by
which storms happen to be drawn.
"""
import json, os, numpy as np

RNG = np.random.default_rng(0)
NBOOT = int(os.environ.get("NBOOT", "2000"))

d = np.load("track_build/errdecomp.npz")
T, wpep = d["T"], d["wpep"]
z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
sid = z["storm_id"].astype(str)[wpep]

# per-window mean error over the 20 leads -- the same quantity the headline km number averages
E = {k: np.sqrt(((d[k] - T) ** 2).sum(-1)).mean(1) for k in ("v20", "v21", "v22")}
storms = np.unique(sid)
idx = [np.where(sid == s)[0] for s in storms]
n_w = np.array([len(i) for i in idx], float)          # weight each storm by its window count
per = {k: np.array([E[k][i].mean() for i in idx]) for k in E}

print(f"{len(storms)} storms, {len(sid)} windows, {NBOOT} resamples\n")
print("point estimates (window-weighted, = the headline number)")
for k in ("v20", "v21", "v22"):
    print(f"  {k}  {np.average(per[k], weights=n_w):7.2f} km")

PAIRS = [("v21", "v20"), ("v22", "v21"), ("v22", "v20")]
print(f"\n{'comparison':>14s} {'delta':>8s} {'95% CI':>20s} {'p':>7s} {'wins':>10s}")
out = {}
for a, b in PAIRS:
    diffs = np.empty(NBOOT)
    for t in range(NBOOT):
        k = RNG.integers(0, len(storms), len(storms))
        diffs[t] = np.average(per[a][k], weights=n_w[k]) - np.average(per[b][k], weights=n_w[k])
    pt = np.average(per[a], weights=n_w) - np.average(per[b], weights=n_w)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p: how often the sign flips relative to the point estimate
    p = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    wins = int((per[a] < per[b]).sum())
    sig = "significant" if lo * hi > 0 else "NOT significant"
    print(f"{a+' vs '+b:>14s} {pt:+8.2f} [{lo:+7.2f}, {hi:+7.2f}] {p:7.3f} "
          f"{wins:4d}/{len(storms):<5d} {sig}")
    out[f"{a}_vs_{b}"] = {"delta": pt, "ci": [lo, hi], "p": p,
                          "wins": wins, "n_storms": len(storms)}

json.dump(out, open("track_build/bootstrap3.json", "w"), indent=1)
print("\nwrote track_build/bootstrap3.json")
