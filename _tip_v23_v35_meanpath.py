"""v23 and v35 mean-by-valid-time predicted track + vmax on Typhoon Tip (1979), for drawing
alongside the real category-colored track. Mean-by-valid-time matches make_map_html.py's
convention: at each 6h valid time, average every forecast's predicted position AND predicted vmax
across all issues that predicted that moment, dropping bins with <3 members.

Writes track_build/tip_v23_v35_meanpath.json: {"v23": [[lat,lon,vmax_kt],...], "v35": [...]},
each sorted by valid time.
"""
import json, re, math, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8); DEVICE = torch.device("cpu"); R = 111.2; KM6H = 6 * 3600 / 1000.0

nb = json.load(open("colab_train_v17.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os, "DEVICE": DEVICE,
     "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p, src, re.S).group(0), G)
Base = G["TrackFormerV17"]
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


def load(ck):
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    m = V23().eval(); m.load_state_dict(sd)
    return m


z = np.load("track_build/tip_fixed.npz", allow_pickle=True)
tr = z["track"].astype("float32"); tgt = z["target"].astype("float32"); nl = z["n_leads"].astype(int)
bt = z["base_time"].astype("int64"); bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
o13 = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
tm, ts = o13["track_mean"].astype("float32"), o13["track_std"].astype("float32")
vp = np.concatenate([tr[:, -1, 2:4] * ts[2:4] + tm[2:4], tr[:, -2, 2:4] * ts[2:4] + tm[2:4]], 1).astype("float32")
S20 = np.load("track_build/tip_dlm4.npy").astype("float32")
SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12)
K = np.where(nl == 20)[0]; K = K[np.argsort(bt[K])]

SIX = int(6 * 3600 * 1e9)
_key = {int(bt[i]): i for i in range(len(bt))}
HIST = np.full((len(bt), 2), -1, np.int64)
for i in range(len(bt)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = _key.get(int(bt[i]) - back * SIX, -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(bt))[:, None])


def build_hist(j):
    h = np.zeros((len(j), 8, 17, 17), "float32")
    for c in range(2):
        idx = HIST_S[j, c]
        h[:, c * 4:(c + 1) * 4] = S20[idx]
    return h, HAVE[j]


M23 = [load(f) for f in sorted(glob.glob("downloads/x/v23_seed*.pt"))]
M35 = [load(f) for f in sorted(glob.glob("downloads/v35ck/v35_seed*.pt"))]
print(f"v23: {len(M23)} | v35: {len(M35)} seeds")


@torch.no_grad()
def run(ms):
    P = []
    for i in range(0, len(K), 64):
        j = K[i:i + 64]
        h, hv = build_hist(j)
        a = [torch.from_numpy(tr[j]), torch.from_numpy(vp[j]), torch.from_numpy(S20[j]),
             torch.from_numpy(h), torch.from_numpy(hv)]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).numpy())
    return np.concatenate(P)   # [n_issue, 20, 17]


def mean_by_valid_time(A, min_members=3):
    """A: [n_issue,20,17] predictions. Returns [(lat,lon,vmax_kt), ...] sorted by valid time,
    each the mean across every forecast whose lead lands on that 6h valid-time bin."""
    cE, cN = np.cumsum(A[..., 0], 1), np.cumsum(A[..., 1], 1)
    vmax = A[..., 2]
    acc = {}
    for w in range(len(K)):
        base_ns = int(bt[K[w]])
        for L in range(20):
            vt = int(round((base_ns + (L + 1) * SIX) / SIX)) * SIX
            la = bla[K[w]] + cN[w, L] / R
            lo = blo[K[w]] + cE[w, L] / (R * math.cos(math.radians((bla[K[w]] + la) / 2)))
            a = acc.setdefault(vt, [0.0, 0.0, 0.0, 0])
            a[0] += la; a[1] += lo; a[2] += float(vmax[w, L]); a[3] += 1
    out = []
    for vt in sorted(acc):
        la, lo, vm, n = acc[vt]
        if n >= min_members:
            out.append((la / n, lo / n, vm / n))
    return out


res = {}
for tag, ms in (("v23", M23), ("v35", M35)):
    A = run(ms)
    res[tag] = mean_by_valid_time(A)
    print(f"{tag}: {len(res[tag])} valid-time points, "
          f"vmax {min(p[2] for p in res[tag]):.0f}-{max(p[2] for p in res[tag]):.0f} kt")

json.dump(res, open("track_build/tip_v23_v35_meanpath.json", "w"))
print("wrote track_build/tip_v23_v35_meanpath.json")
