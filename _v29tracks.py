"""Export v29 tracks for the map, gated on reproducing the Colab ensemble first.

v29 = v23 + 0.25-degree ERA5 steering wind (u/v @ 200/550/850 hPa), nested beneath v23's temporal-
history wrap. RESULT (Colab, 5 seeds): full WP+EP 438.70 km vs v23 434.96 km (+3.74, worse), WP-only
468.09 km. Every individual seed landed above v23's baseline -- a small but consistent null/negative,
not scatter. The map is drawn anyway: v23 vs v29 makes the (lack of) difference visible on real
storms, which is itself the finding worth showing.

Nothing is written until the local ensemble reproduces the Colab number (438.70 km, full WP+EP).
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R = 111.2; KM6H = 6 * 3600 / 1000.0
EXPECT = {"full": 438.70, "wp_only": 468.09}

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
mask = z["target_mask"].astype(bool)

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

v28src = open("colab_v28_train.py").read()
hs_src = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28src, re.S).group(0)
tf_src = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28src, re.S).group(0)
g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(hs_src, g23); exec(tf_src, g23)
V23 = g23["TrackFormerHist"]

v29src = open("colab_v35_train.py").read()
era_src = re.search(r"class Era5Stem\(nn\.Module\):.*?\n        return st\n", v29src, re.S).group(0)
tfe_src = re.search(r"class TrackFormerEra5\(V23\):.*?self\.steer_cnn\.base\.ctx = None\n", v29src, re.S).group(0)
ERA5_CLIP = 4.0
USE_ERA5 = 1
g29 = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "ERA5_CLIP": ERA5_CLIP, "USE_ERA5": USE_ERA5}
exec(era_src, g29); exec(tfe_src, g29)
TrackFormerEra5 = g29["TrackFormerEra5"]

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
nl = z["n_leads"].astype(int)
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

_ed = np.load("track_build/era5_steer_int8.npz")
ERA5_Q, ERA5_WIDX, ERA5_SCALE = _ed["q"], _ed["widx"], _ed["scale"].astype("float32")
ERA5_N = int(_ed["N"])
assert ERA5_N == len(sid)
ERA5_POS = np.full(ERA5_N, -1, dtype="int64")
ERA5_POS[ERA5_WIDX] = np.arange(len(ERA5_WIDX))


def era5_of(j):
    p = ERA5_POS[j]
    ok = (p >= 0).astype("float32")
    ps = np.where(p >= 0, p, 0)
    q = ERA5_Q[ps]
    q = np.where(ok[:, None, None, None] > 0, q, 0)
    return q, ok


g29["ERA5_SCALE"] = ERA5_SCALE   # TrackFormerEra5.__init__ closes over this via g29's __globals__


def load(cls, paths):
    ms = []
    for p in paths:
        m = cls().eval()
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        m.load_state_dict(sd)
        ms.append(m)
    return ms


MS = {"v29": load(TrackFormerEra5, sorted(glob.glob("downloads/v29ck/**/v29_seed*.pt", recursive=True))),
      "v23": load(V23, sorted(glob.glob("downloads/x/v23_seed*.pt", recursive=True)))}
print(f"v29: {len(MS['v29'])} seeds | v23: {len(MS['v23'])} seeds")


@torch.no_grad()
def run(tag, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        hv = torch.from_numpy(HAVE[j])
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]), h, hv]
        if tag == "v29":
            eq, eok = era5_of(j)
            a += [torch.from_numpy(eq), torch.from_numpy(eok)]
        sv = torch.stack([m(*a)[0] for m in MS[tag]]).mean(0)
        P.append((sv * SC).float().numpy())
    return np.concatenate(P)


full = nl == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
wp_only = np.array([i for i in wpep if basins[i] == "WP"])
T = target[wpep]; TC = np.cumsum(T[..., :2], 1)

ok = True
for tag, pop, key_ in (("v29", wpep, "full"), ("v29", wp_only, "wp_only")):
    P = run(tag, pop)
    C = np.cumsum(P[..., :2], 1)
    Tp = np.cumsum(target[pop][..., :2], 1)
    agg = float(np.sqrt(((C - Tp) ** 2).sum(-1)).mean())
    d = agg - EXPECT[key_]
    print(f"  {tag:5s} {key_:8s} local {agg:7.2f} km | colab {EXPECT[key_]:7.2f} | diff {d:+.2f}  "
          f"{'OK' if abs(d) < 0.5 else '*** MISMATCH ***'}")
    ok &= abs(d) < 0.5
if not ok:
    sys.exit("local forward does not reproduce Colab -- refusing to export tracks")
print("forward validated\n")

IC = json.load(open("track_build/intensity_compare.json")) if os.path.exists("track_build/intensity_compare.json") else {}
for tag in ("v29", "v23"):
    Pfull = run(tag, wpep)
    Cfull = np.cumsum(Pfull[..., :2], 1)
    agg = float(np.sqrt(((Cfull - TC) ** 2).sum(-1)).mean())
    IC[tag] = {"track": np.sqrt(((Cfull - TC) ** 2).sum(-1)).mean(0).tolist(), "agg_track": agg}
json.dump(IC, open("track_build/intensity_compare.json", "w"))

STORMS = [("1986228N19120", "Wayne"), ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]
for tag in ("v29", "v23"):
    out = {}
    for s, nm in STORMS:
        k = np.where((sid == s) & (nl == 20))[0]; k = k[np.argsort(bt[k])]
        if not len(k):
            continue
        A = run(tag, k)
        cE, cN = np.cumsum(A[..., 0], 1), np.cumsum(A[..., 1], 1)
        tE, tN = np.cumsum(target[k][..., 0], 1), np.cumsum(target[k][..., 1], 1)
        lats, lons = [], []
        for a2 in range(len(k)):
            la = bla[k[a2]] + cN[a2] / R
            lo = blo[k[a2]] + cE[a2] / (R * np.cos(np.radians((bla[k[a2]] + la) / 2)))
            lats.append(np.round(la, 3).tolist()); lons.append(np.round(lo, 3).tolist())
        err = float(np.hypot(cE[:, 19] - tE[:, 19], cN[:, 19] - tN[:, 19]).mean())
        out[nm] = {"lat": lats, "lon": lons, "base_time": bt[k].tolist(),
                   "base_lat": np.round(bla[k], 3).tolist(),
                   "base_lon": np.round(blo[k], 3).tolist(),
                   "err120_mean": err, "n": int(len(k))}
        print(f"  {tag:5s} {nm:11s} {len(k):3d} fc | mean 120h {err:6.0f} km", flush=True)
    json.dump(out, open(f"track_build/{tag}_tracks.json", "w"))
print("\nwrote track_build/{v29,v23}_tracks.json and merged both into intensity_compare.json")
