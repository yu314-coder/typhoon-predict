"""Mean-by-valid-time predicted (lat, lon, vmax) paths for v10, v23, v35 on Tip, Bavi, Co-may, and
Hinnamnor -- for overlaying on the real category-colored track (Dolphin excluded: no real steering
field data exists for it yet, unlike Tip's dedicated tip_dlm4.npy, so v23/v35 cannot run on it).

v10 is track-only (TrackFormerV9, no steering input at all) -- included as the "no environment"
baseline sitting between the real track and the steering-aware v23/v35.

Writes track_build/overlay_predictions.json: {storm: {model: [[lat,lon,vmax_kt],...]}}.
"""
import json, re, math, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8); DEVICE = torch.device("cpu"); R = 111.2; KM6H = 6 * 3600 / 1000.0
SIX = int(6 * 3600 * 1e9)

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
vpair = G["vpair"]; z = G["z"]; SC = G["TARGET_SCALE"]
sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64") % 360
nl = z["n_leads"].astype(int)

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


def build_v10():
    s = open("train_track_v10.py").read()
    g = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
         "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
    for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(pat, s, re.S).group(0), g)
    return g["TrackFormerV9"]


m10 = build_v10()()
m10.load_state_dict(torch.load("track_build/track_v10_best.pt", map_location="cpu", weights_only=False)["model"])
m10.eval()


def load23(ck):
    m = V23().eval(); m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"]); return m


M23 = [load23(f) for f in sorted(glob.glob("downloads/x/v23_seed*.pt"))]
M35 = [load23(f) for f in sorted(glob.glob("downloads/v35ck/v35_seed*.pt"))]
print(f"v10: 1 | v23: {len(M23)} | v35: {len(M35)} seeds")

# ---- temporal history index (for v23/v35's HistStem) over the WHOLE dataset ----
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])


@torch.no_grad()
def predict_indataset(tag, idx):
    P = []
    for i in range(0, len(idx), 64):
        j = idx[i:i + 64]
        if tag == "v10":
            sv, _ = m10(torch.from_numpy(track[j]), torch.from_numpy(vpair[j]))
        else:
            h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
            hv = torch.from_numpy(HAVE[j])
            a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]), h, hv]
            ms = M23 if tag == "v23" else M35
            sv = torch.stack([m(*a)[0] for m in ms]).mean(0)
        P.append((sv * SC).numpy())
    return np.concatenate(P)


def mean_by_valid_time(A, base_ns_arr, base_lat_arr, base_lon_arr, min_members=3):
    cE, cN = np.cumsum(A[..., 0], 1), np.cumsum(A[..., 1], 1)
    vmax = A[..., 2]
    acc = {}
    for w in range(len(base_ns_arr)):
        for L in range(20):
            vt = int(round((int(base_ns_arr[w]) + (L + 1) * SIX) / SIX)) * SIX
            la = base_lat_arr[w] + cN[w, L] / R
            lo = base_lon_arr[w] + cE[w, L] / (R * math.cos(math.radians((base_lat_arr[w] + la) / 2)))
            a = acc.setdefault(vt, [0.0, 0.0, 0.0, 0])
            a[0] += la; a[1] += lo; a[2] += float(vmax[w, L]); a[3] += 1
    out = []
    for vt in sorted(acc):
        la, lo, vm, n = acc[vt]
        if n >= min_members:
            out.append([la / n, lo / n, vm / n, int(vt)])
    return out


res = {}

# ---- in-dataset storms: Bavi, Co-may, Hinnamnor ----
full = nl == 20
for s, nm in [("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor"), ("2026182N09163", "Bavi")]:
    k = np.where((sid == s) & full)[0]
    k = k[np.argsort(bt[k])]
    res[nm] = {}
    for tag in ("v10", "v23", "v35"):
        A = predict_indataset(tag, k)
        res[nm][tag] = mean_by_valid_time(A, bt[k], bla[k], blo[k])
    print(f"{nm}: " + " | ".join(f"{t} {len(res[nm][t])}pts" for t in ("v10", "v23", "v35")))

# ---- Tip: real steering data, separate pipeline ----
tz = np.load("track_build/tip_fixed.npz", allow_pickle=True)
ttr = tz["track"].astype("float32"); ttgt = tz["target"].astype("float32"); tnl = tz["n_leads"].astype(int)
tbt = tz["base_time"].astype("int64"); tbla = tz["base_lat"].astype("float64"); tblo = tz["base_lon"].astype("float64")
o13 = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
otm, ots = o13["track_mean"].astype("float32"), o13["track_std"].astype("float32")
tvp = np.concatenate([ttr[:, -1, 2:4] * ots[2:4] + otm[2:4], ttr[:, -2, 2:4] * ots[2:4] + otm[2:4]], 1).astype("float32")
tS20 = np.load("track_build/tip_dlm4.npy").astype("float32")
tK = np.where(tnl == 20)[0]; tK = tK[np.argsort(tbt[tK])]

t_key = {int(tbt[i]): i for i in range(len(tbt))}
tHIST = np.full((len(tbt), 2), -1, np.int64)
for i in range(len(tbt)):
    for c, back in enumerate((2, 4)):
        tHIST[i, c] = t_key.get(int(tbt[i]) - back * SIX, -1)
tHAVE = (tHIST >= 0).astype("float32")
tHIST_S = np.where(tHIST >= 0, tHIST, np.arange(len(tbt))[:, None])


@torch.no_grad()
def predict_tip(tag):
    P = []
    for i in range(0, len(tK), 64):
        j = tK[i:i + 64]
        if tag == "v10":
            sv, _ = m10(torch.from_numpy(ttr[j]), torch.from_numpy(tvp[j]))
        else:
            h = np.zeros((len(j), 8, 17, 17), "float32")
            for c in range(2):
                h[:, c * 4:(c + 1) * 4] = tS20[tHIST_S[j, c]]
            hv = tHAVE[j]
            a = [torch.from_numpy(ttr[j]), torch.from_numpy(tvp[j]), torch.from_numpy(tS20[j]),
                 torch.from_numpy(h), torch.from_numpy(hv)]
            ms = M23 if tag == "v23" else M35
            sv = torch.stack([m(*a)[0] for m in ms]).mean(0)
        P.append((sv * SC).numpy())
    return np.concatenate(P)


res["Tip"] = {}
for tag in ("v10", "v23", "v35"):
    A = predict_tip(tag)
    res["Tip"][tag] = mean_by_valid_time(A, tbt[tK], tbla[tK], tblo[tK])
print("Tip: " + " | ".join(f"{t} {len(res['Tip'][t])}pts" for t in ("v10", "v23", "v35")))

json.dump(res, open("track_build/overlay_predictions.json", "w"))
print("wrote track_build/overlay_predictions.json")
