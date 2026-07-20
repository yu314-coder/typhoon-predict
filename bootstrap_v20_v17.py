"""Is v20's -10.33 km over v17 real, or is it the size of the noise?

The same harness that killed v18: a PAIRED-STORM bootstrap. Both models are scored on identical
windows, the per-storm mean errors are paired, and whole STORMS are resampled with replacement --
not windows, because windows from one storm are strongly correlated and resampling them would
manufacture significance out of that correlation.

v18 - v17 came out +3.78 km with 95% CI [-2.56, +9.75] and was correctly called not significant.
v20's raw gap is -10.33 km, which is larger, but the CI is what decides it.

Both models share the v13 windows and the SLP channels; only the steering channels differ:
v17 reads 500 hPa (steer5_int8.npz), v20 reads the deep-layer mean (dlm4_int8.npz). Both dequantise
the same way, q/31.75 then clip at 4 sigma, which is what training did.
"""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8)
DEVICE = torch.device("cpu")
RNG = np.random.default_rng(0)
NBOOT = int(os.environ.get("NBOOT", "2000"))

nb = json.load(open("colab_train_v17.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
     "DEVICE": DEVICE, "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p, src, re.S).group(0), G)


def load(ck):
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("inner.")} or sd
    m = G["TrackFormerV17"]().to(DEVICE).eval(); m.load_state_dict(sd); return m


z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
sids = z["storm_id"].astype(str); years = z["year"].astype(int)
basins = z["basin"].astype(str); nl = z["n_leads"].astype(int)
tmean = z["track_mean"].astype("float32"); tstd = z["track_std"].astype("float32")
vpair = np.concatenate([track[:, -1, 2:4] * tstd[2:4] + tmean[2:4],
                        track[:, -2, 2:4] * tstd[2:4] + tmean[2:4]], 1).astype("float32")


def steer(path):
    q = np.load(path)["q"][:, :4].astype("float32")
    return np.clip(q / 31.75, -4.0, 4.0)


S17 = steer("track_build/steer5_int8.npz")     # 500 hPa
S20 = steer("track_build/dlm4_int8.npz")       # deep-layer mean
assert S17.shape == S20.shape == (len(track), 4, 17, 17)

fy = {s: int(years[sids == s].min()) for s in np.unique(sids)}
EV = np.array([i for i in range(len(sids))
               if fy[sids[i]] >= 2020 and nl[i] == 20 and basins[i] in ("WP", "EP")])
T = np.cumsum(target[EV][..., :2], 1)
SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12)
ev_storm = sids[EV]
storms = np.unique(ev_storm)
print(f"test: {len(EV)} windows over {len(storms)} storms\n")


@torch.no_grad()
def per_window_err(ckpts, S):
    P = []
    ms = [load(c) for c in ckpts]
    for i in range(0, len(EV), 128):
        j = EV[i:i + 128]
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(S[j])]
        s = torch.stack([m(*a)[0] for m in ms]).mean(0)
        P.append((s * SC).numpy())
    C = np.cumsum(np.concatenate(P)[..., :2], 1)
    return np.sqrt(((C - T) ** 2).sum(-1)).mean(1)      # mean over the 20 leads, per window


e17 = per_window_err([f"downloads/x/v17_seed{i}.pt" for i in range(5)], S17)
e20 = per_window_err([f"downloads/x/v20_seed{i}.pt" for i in range(5)], S20)
print(f"v17 (5 seeds)  {e17.mean():.2f} km      [published 462.8]")
print(f"v20 (5 seeds)  {e20.mean():.2f} km      [Colab said 452.47]")
print(f"difference     {e20.mean() - e17.mean():+.2f} km\n")

# per-storm means, paired
m17 = np.array([e17[ev_storm == s].mean() for s in storms])
m20 = np.array([e20[ev_storm == s].mean() for s in storms])
n_w = np.array([(ev_storm == s).sum() for s in storms])
obs = float(np.average(m20, weights=n_w) - np.average(m17, weights=n_w))

diffs = np.empty(NBOOT)
for b in range(NBOOT):
    k = RNG.integers(0, len(storms), len(storms))     # resample STORMS, not windows
    diffs[b] = np.average(m20[k], weights=n_w[k]) - np.average(m17[k], weights=n_w[k])
lo, hi = np.percentile(diffs, [2.5, 97.5])
p_worse = float((diffs >= 0).mean())

print(f"paired storm bootstrap, {NBOOT} resamples over {len(storms)} storms")
print(f"  v20 - v17 = {obs:+.2f} km   95% CI [{lo:+.2f}, {hi:+.2f}]")
print(f"  fraction of resamples where v20 is NOT better: {p_worse:.3f}")
print()
if hi < 0:
    print("  SIGNIFICANT -- the whole CI is below zero: deep-layer steering is a real gain.")
elif lo > 0:
    print("  SIGNIFICANT THE WRONG WAY -- v20 is genuinely worse.")
else:
    print("  NOT SIGNIFICANT -- the CI straddles zero. The gap is not statistically established.")
print()
n_better = int((m20 < m17).sum())
print(f"storms where v20 beats v17: {n_better}/{len(storms)}  ({100*n_better/len(storms):.0f}%)")
np.savez("track_build/per_storm_v20_v17.npz", storms=storms, m17=m17, m20=m20, n_w=n_w,
         e17=e17, e20=e20, ev_storm=ev_storm)
# where does the mean gain actually come from?
d = m20 - m17
order = np.argsort(d)
print("\nlargest v20 WINS (km better):")
for i in order[:6]:
    print(f"  {storms[i]}  {d[i]:+8.1f}   v17 {m17[i]:6.0f} -> v20 {m20[i]:6.0f}  ({n_w[i]} windows)")
print("largest v20 LOSSES (km worse):")
for i in order[::-1][:6]:
    print(f"  {storms[i]}  {d[i]:+8.1f}   v17 {m17[i]:6.0f} -> v20 {m20[i]:6.0f}  ({n_w[i]} windows)")
print(f"\nper-storm difference: median {np.median(d):+.1f} km, mean {d.mean():+.1f} km")
print(f"  a median near zero with a negative mean = a few big wins carrying it")
big = np.abs(d) > 100
print(f"  storms with |diff| > 100 km: {big.sum()}/{len(d)}; they contribute "
      f"{np.average(d[big], weights=n_w[big])*n_w[big].sum()/n_w.sum():+.2f} km of the {obs:+.2f} total")
json.dump({"v17": float(e17.mean()), "v20": float(e20.mean()), "diff": obs,
           "ci": [float(lo), float(hi)], "p_not_better": p_worse,
           "storms_better": n_better, "n_storms": int(len(storms))},
          open("track_build/bootstrap_v20_v17.json", "w"))
