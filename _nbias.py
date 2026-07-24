"""GO/NO-GO for v28: is the meridional (N) track error BIASED or merely NOISY?

WHY THIS DECIDES v28. The v28 proposal adds a poleward drift adapter to v23, motivated by the
meridional flow correlation being weak (0.460 vs 0.805 zonal). But those are different failure
modes. A drift adapter adds a SYSTEMATIC OFFSET; it fixes a BIASED error. A low correlation means
the flow PREDICTION is noisy, which an offset cannot fix and may worsen by adding variance.

So the question is not "is N error large" but "is N error's MEAN far from zero relative to its
spread". The statistic is bias-to-noise:

        |mean signed N error| / std(signed N error)     per lead

Large (say > 0.3) and growing with lead  -> a systematic poleward/equatorward miss exists, an
                                            offset term is well targeted, train v28.
Near zero with large spread              -> the N problem is variance, not bias. v28 as proposed
                                            would be chasing something that is not there.

Beta drift is poleward, so if it is the missing physics the signed error should be NEGATIVE in the
northern hemisphere (model not carrying storms far enough north) and POSITIVE in the southern, and
it should be strongest for non-recurving storms in the deep tropics.

Decided on VALIDATION storms (2016-2019). The 2020+ test set is not used to make design decisions.
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

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
nl = z["n_leads"].astype(int); bla = z["base_lat"].astype("float64")
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

MS = []
for p in sorted(glob.glob("downloads/x/v23_seed*.pt")):
    m = V23().eval(); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"]); MS.append(m)
print(f"v23: {len(MS)} seeds")


@torch.no_grad()
def predict(idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]),
             h, torch.from_numpy(HAVE[j])]
        P.append((torch.stack([m(*a)[0] for m in MS]).mean(0) * SC).float().numpy())
    return np.concatenate(P)


full = nl == 20
wpep = np.isin(basins, ["WP", "EP"])


def report(name, idx):
    O = predict(idx); T = target[idx]
    pN, oN = np.cumsum(O[..., 1], 1), np.cumsum(T[..., 1], 1)     # cumulative NORTH, km
    pE, oE = np.cumsum(O[..., 0], 1), np.cumsum(T[..., 0], 1)
    dN, dE = pN - oN, pE - oE                                     # signed error, + = too far north
    lat = bla[idx]
    # in the SOUTHERN hemisphere poleward is negative N, so flip to make "poleward" comparable
    pole = np.where(lat[:, None] >= 0, dN, -dN)
    print(f"\n=== {name}  ({len(idx):,} windows) ===")
    print(f"{'lead':>5s} {'h':>4s} {'meanN':>9s} {'sdN':>8s} {'|b|/sd':>7s} {'meanE':>9s} {'sdE':>8s} {'|b|/sd':>7s}")
    for L in (3, 7, 11, 15, 19):
        mN, sN = dN[:, L].mean(), dN[:, L].std()
        mE, sE = dE[:, L].mean(), dE[:, L].std()
        print(f"{L+1:5d} {6*(L+1):4d} {mN:9.1f} {sN:8.1f} {abs(mN)/sN:7.3f} "
              f"{mE:9.1f} {sE:8.1f} {abs(mE)/sE:7.3f}")
    L = 19
    print(f"  120 h poleward-signed mean error: {pole[:, L].mean():+.1f} km "
          f"(negative = model does NOT carry storms far enough poleward)")
    nh, sh = lat >= 0, lat < 0
    for lab, m in (("NH", nh), ("SH", sh)):
        if m.sum() > 30:
            print(f"    {lab} (n={int(m.sum()):5d}): mean N err {dN[m, L].mean():+8.1f} km "
                  f"| sd {dN[m, L].std():7.1f} | bias/noise {abs(dN[m, L].mean())/dN[m, L].std():.3f}")
    # recurving vs not: does the storm's observed heading swing polewards->eastwards?
    hd0 = np.arctan2(T[:, 0, 1], T[:, 0, 0]); hd9 = np.arctan2(T[:, 9, 1], T[:, 9, 0])
    swing = np.abs(np.arctan2(np.sin(hd9 - hd0), np.cos(hd9 - hd0))) * 180 / np.pi
    rec, non = swing > 45, swing <= 45
    for lab, m in (("recurving  ", rec), ("non-recurv.", non)):
        if m.sum() > 30:
            print(f"    {lab} (n={int(m.sum()):5d}): mean N err {dN[m, L].mean():+8.1f} km "
                  f"| bias/noise {abs(dN[m, L].mean())/dN[m, L].std():.3f}")
    return {"mean_N_120": float(dN[:, 19].mean()), "sd_N_120": float(dN[:, 19].std()),
            "mean_E_120": float(dE[:, 19].mean()), "sd_E_120": float(dE[:, 19].std())}


va = np.array([i for i in va_idx if full[i] and wpep[i]])
te = np.array([i for i in te_idx if full[i] and wpep[i]])
out = {"validation": report("VALIDATION 2016-2019 (the decision basis)", va),
       "test": report("TEST 2020+ (confirmation only, not for design)", te)}
json.dump(out, open("track_build/nbias.json", "w"), indent=1)

b = abs(out["validation"]["mean_N_120"]) / out["validation"]["sd_N_120"]
print("\n" + "=" * 72)
print(f"VERDICT  bias/noise on validation at 120 h = {b:.3f}")
if b > 0.30:
    print("  -> N error is SYSTEMATIC. A drift/offset term is well targeted. v28 worth training.")
elif b > 0.15:
    print("  -> WEAK systematic component. v28 might buy a little; expect a small effect.")
else:
    print("  -> N error is NOISE-DOMINATED, not biased. An offset adapter cannot fix this.")
    print("     v28 as proposed is chasing a bias that is not there -- redirect it.")
print("=" * 72)
