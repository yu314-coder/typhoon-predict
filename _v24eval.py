"""v24 vs v24abl: reproduce Colab locally, then paired-storm bootstrap.

The arms differ in ONE thing: whether the decoder sees the basin box (63 tokens, 6-hourly,
100-180E/0-60N) or the storm-centred 17x17 patch (25 tokens, daily mean). Same script, same seeds,
same v21 head, both WITHOUT mirror augmentation. So the delta is attributable to the input.

Scored on basin-COVERED windows only: NCEP R1 ends 2026-03-17, and on uncovered windows v24 sees
exact zeros while v24abl sees its normal patch, which would not be a fair comparison.
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
exec(compile(body, "<v17>", "exec"), G)
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

B = np.load("track_build/basin_all_int8.npz"); BQ = B["q"]
BSC = B["scale"].astype("float32"); BOFF = B["offset"].astype("float32")
NCH, NLAT, NLON = BQ.shape[1], BQ.shape[2], BQ.shape[3]
IX = np.load("track_build/v24_index.npz"); IN_LO = IX["in_lo"]; IN_OK = IX["in_ok"]
BM = np.array([(BQ[::97, c].astype("float32") * BSC[c] + BOFF[c]).mean() for c in range(NCH)], "float32")
BS = np.array([(BQ[::97, c].astype("float32") * BSC[c] + BOFF[c]).std() + 1e-3 for c in range(NCH)], "float32")


def basin_at(r):
    v = BQ[r].astype("float32") * BSC[None, :, None, None] + BOFF[None, :, None, None]
    return (v - BM[None, :, None, None]) / BS[None, :, None, None]


v30 = open("colab_v30_train.py").read()
bs = re.search(r"class BasinStem\(nn\.Module\):.*?return self\.net\(self\.ctx\.reshape\(b, 3 \* NCH, NLAT, NLON\)\)\n", v30, re.S).group(0)
tb = re.search(r"class TrackFormerBasin\(V21\):.*?self\.steer_cnn\.ctx = None\n", v30, re.S).group(0)


def build(use):
    g = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
         "DSC": DSC, "KM6H": 6 * 3600 / 1000.0, "USE_BASIN": use,
         "NCH": NCH, "NLAT": NLAT, "NLON": NLON, "NTOK": 63}
    exec(bs, g); exec(tb, g)
    return g["TrackFormerBasin"]


full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
cov = np.array([i for i in wpep if IN_OK[i].prod() > 0])
T = np.cumsum(target[cov][..., :2], 1)
print(f"basin-covered WP+EP 2020+: {len(cov)} of {len(wpep)} windows\n")


@torch.no_grad()
def predict(cls, tag):
    ms = []
    for s in range(10):
        m = cls().eval()
        m.load_state_dict(torch.load(f"downloads/x/{tag}_seed{s}.pt", map_location="cpu",
                                     weights_only=False)["model"])
        ms.append(m)
    P = []
    for i in range(0, len(cov), 128):
        j = cov[i:i + 128]
        bx = torch.from_numpy(basin_at(IN_LO[j].ravel()).reshape(len(j), 3, NCH, NLAT, NLON)
                              * IN_OK[j][:, :, None, None, None])
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]), bx]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().numpy())
    return np.cumsum(np.concatenate(P)[..., :2], 1)


ARMS = {"v24": (build(1), 529.05), "v24abl": (build(0), 442.54)}
out, ok = {}, True
for tag, (cls, expect) in ARMS.items():
    C = predict(cls, tag); out[tag] = C
    e = float(np.sqrt(((C - T) ** 2).sum(-1)).mean())
    d = e - expect
    print(f"  {tag:7s} local {e:8.2f} km | colab {expect:7.2f} | diff {d:+.2f}  "
          f"{'OK' if abs(d) < 0.05 else '*** MISMATCH ***'}")
    ok &= abs(d) < 0.05
if not ok:
    sys.exit("local forward does not reproduce Colab")

RNG = np.random.default_rng(0); NBOOT = 2000
sidw = z["storm_id"].astype(str)[cov]
E = {k: np.sqrt(((v - T) ** 2).sum(-1)).mean(1) for k, v in out.items()}
storms = np.unique(sidw); idx = [np.where(sidw == s)[0] for s in storms]
n_w = np.array([len(i) for i in idx], float)
per = {k: np.array([E[k][i].mean() for i in idx]) for k in E}
diffs = np.empty(NBOOT)
for t in range(NBOOT):
    k = RNG.integers(0, len(storms), len(storms))
    diffs[t] = np.average(per["v24"][k], weights=n_w[k]) - np.average(per["v24abl"][k], weights=n_w[k])
pt = np.average(per["v24"], weights=n_w) - np.average(per["v24abl"], weights=n_w)
lo, hi = np.percentile(diffs, [2.5, 97.5])
p = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
wins = int((per["v24"] < per["v24abl"]).sum())
print(f"\npaired-storm bootstrap, {len(storms)} storms, {NBOOT} resamples")
print(f"  v24 vs v24abl  {pt:+.2f} km  95% CI [{lo:+.2f}, {hi:+.2f}]  p {p:.3f}  "
      f"wins {wins}/{len(storms)}  {'SIGNIFICANT' if lo*hi > 0 else 'not significant'}")
json.dump({"delta": pt, "ci": [lo, hi], "p": p, "wins": wins}, open("track_build/bootstrap_v24.json", "w"), indent=1)
