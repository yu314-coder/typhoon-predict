"""v23 / v23abl: reproduce Colab locally, then bootstrap.

Validates the local forward against Colab (438.51 / 447.05) before trusting anything, exactly as
_v22tracks.py does, then writes per-window errors for the paired-storm bootstrap.

The comparison that matters is v23 vs v23abl: identical code, identical parameter count, identical
seeds -- the ONLY difference is whether the t-24h/t-12h fields reach the steering stem. v23 vs v21
confounds the temporal stack with 189k extra parameters and a different init.
"""
import json, re, math, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

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

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))
CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": 6 * 3600 / 1000.0, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

# temporal index, same construction as the training script
sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

v28 = open("colab_v28_train.py").read()
hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28, re.S).group(0)
tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28, re.S).group(0)


def build(use_hist):
    g = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
         "DSC": DSC, "KM6H": 6 * 3600 / 1000.0, "USE_HIST": use_hist}
    exec(hs, g); exec(tf, g)
    return g["TrackFormerHist"]


full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = np.cumsum(target[wpep][..., :2], 1)


@torch.no_grad()
def predict(cls, tag):
    ms = []
    for s in range(NSEED):
        m = cls().eval()
        m.load_state_dict(torch.load(f"downloads/x/{tag}_seed{s}.pt", map_location="cpu",
                                     weights_only=False)["model"])
        ms.append(m)
    P = []
    for i in range(0, len(wpep), 128):
        j = wpep[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]),
             h, torch.from_numpy(HAVE[j])]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().numpy())
    return np.cumsum(np.concatenate(P)[..., :2], 1)


NSEED = 10
ARMS = {"v23": (build(1), 434.96), "v23abl": (build(0), 444.05)}
out, ok = {}, True
print(f"WP+EP 2020+, {len(wpep)} windows\n")
for tag, (cls, expect) in ARMS.items():
    C = predict(cls, tag)
    out[tag] = C
    e = float(np.sqrt(((C - T) ** 2).sum(-1)).mean())
    d = e - expect
    print(f"  {tag:7s} local {e:8.2f} km | colab {expect:7.2f} | diff {d:+.2f}  "
          f"{'OK' if abs(d) < 0.05 else '*** MISMATCH ***'}")
    ok &= abs(d) < 0.05
if not ok:
    sys.exit("local forward does not reproduce Colab")

print("\nper-seed error, measured directly over %d seeds" % NSEED)
for tag, (cls, _e) in ARMS.items():
    es = []
    for s_ in range(NSEED):
        m = cls().eval()
        m.load_state_dict(torch.load(f"downloads/x/{tag}_seed{s_}.pt", map_location="cpu",
                                     weights_only=False)["model"])
        P = []
        with torch.no_grad():
            for i in range(0, len(wpep), 128):
                j = wpep[i:i + 128]
                h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
                a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]),
                     torch.from_numpy(SLP[j]), h, torch.from_numpy(HAVE[j])]
                P.append((m(*a)[0] * SC).float().numpy())
        C = np.cumsum(np.concatenate(P)[..., :2], 1)
        es.append(float(np.sqrt(((C - T) ** 2).sum(-1)).mean()))
    es = np.array(es)
    print(f"  {tag:7s} mean {es.mean():7.2f}  sd {es.std(ddof=1):6.2f}  "
          f"range {es.min():.1f}-{es.max():.1f}  ensemble gain {es.mean()-ARMS[tag][1]:+.1f} km")

d0 = np.load("track_build/errdecomp.npz")
np.savez("track_build/v23eval.npz", T=T, wpep=wpep, v21=d0["v21"], v20=d0["v20"],
         v22=d0["v22"], **out)
print("\nwrote track_build/v23eval.npz")

# ---- paired-storm bootstrap ----
RNG = np.random.default_rng(0)
NBOOT = 2000
sidw = sid[wpep]
E = {k: np.sqrt(((v - T) ** 2).sum(-1)).mean(1)
     for k, v in {**out, "v21": d0["v21"], "v20": d0["v20"], "v22": d0["v22"]}.items()}
storms = np.unique(sidw)
idx = [np.where(sidw == s)[0] for s in storms]
n_w = np.array([len(i) for i in idx], float)
per = {k: np.array([E[k][i].mean() for i in idx]) for k in E}

print(f"\npaired-storm bootstrap, {len(storms)} storms, {NBOOT} resamples\n")
print(f"{'comparison':>18s} {'delta':>8s} {'95% CI':>20s} {'p':>7s} {'wins':>10s}")
res = {}
for a, b in [("v23", "v23abl"), ("v23", "v21"), ("v23", "v22"), ("v23abl", "v21")]:
    diffs = np.empty(NBOOT)
    for t in range(NBOOT):
        k = RNG.integers(0, len(storms), len(storms))
        diffs[t] = np.average(per[a][k], weights=n_w[k]) - np.average(per[b][k], weights=n_w[k])
    pt = np.average(per[a], weights=n_w) - np.average(per[b], weights=n_w)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    wins = int((per[a] < per[b]).sum())
    print(f"{a+' vs '+b:>18s} {pt:+8.2f} [{lo:+7.2f}, {hi:+7.2f}] {p:7.3f} "
          f"{wins:4d}/{len(storms):<5d} {'SIGNIFICANT' if lo*hi > 0 else 'not significant'}")
    res[f"{a}_vs_{b}"] = {"delta": pt, "ci": [lo, hi], "p": p, "wins": wins}
json.dump(res, open("track_build/bootstrap_v23.json", "w"), indent=1)
print("\nwrote track_build/bootstrap_v23.json")
