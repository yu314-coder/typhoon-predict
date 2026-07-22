"""Export v10 / v21 / v23 tracks for the map, validating the forward against Colab first.

v23 needs the t-24h/t-12h steering stack threaded through the forward, which the earlier exporters
do not do. Model classes are extracted verbatim from the training scripts rather than retyped.

Nothing is written until the local ensemble error reproduces the Colab number for every arm.
"""
import json, re, math, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
DEVICE = torch.device("cpu"); R = 111.2; KM6H = 6 * 3600 / 1000.0
NSEED = int(os.environ.get("NSEED", "10"))
EXPECT = json.loads(os.environ.get("EXPECT", '{"v23": null, "v21": 443.62}'))

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
       "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

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

v28 = open("colab_v28_train.py").read()
hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28, re.S).group(0)
tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28, re.S).group(0)
gh = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
      "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(hs, gh); exec(tf, gh)
V23 = gh["TrackFormerHist"]


def load(cls, tag, s):
    m = cls().eval()
    m.load_state_dict(torch.load(f"downloads/x/{tag}_seed{s}.pt", map_location="cpu",
                                 weights_only=False)["model"])
    return m


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
m10.load_state_dict(torch.load("track_build/track_v10_best.pt", map_location="cpu",
                               weights_only=False)["model"]); m10.eval()

full = nl == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = np.cumsum(target[wpep][..., :2], 1)
MS = {"v21": [load(V21, "v21", s) for s in range(5)],
      "v23": [load(V23, "v23", s) for s in range(NSEED)]}


@torch.no_grad()
def run(tag, idx):
    P = []
    for i in range(0, len(idx), 64):
        j = idx[i:i + 64]
        if tag == "v10":
            sv, _ = m10(torch.from_numpy(track[j]), torch.from_numpy(vpair[j]))
        elif tag == "v23":
            h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
            a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]),
                 torch.from_numpy(SLP[j]), h, torch.from_numpy(HAVE[j])]
            sv = torch.stack([m(*a)[0] for m in MS["v23"]]).mean(0)
        else:
            a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
            sv = torch.stack([m(*a)[0] for m in MS["v21"]]).mean(0)
        P.append((sv * SC).float().numpy())
    return np.concatenate(P)


print(f"WP+EP 2020+, {len(wpep)} windows | v23 using {NSEED} seeds\n")
ok = True
for tag, exp in EXPECT.items():
    if exp is None:
        continue
    C = np.cumsum(run(tag, wpep)[..., :2], 1)
    e = float(np.sqrt(((C - T) ** 2).sum(-1)).mean())
    d = e - exp
    print(f"  {tag:5s} local {e:8.2f} km | colab {exp:7.2f} | diff {d:+.2f}  "
          f"{'OK' if abs(d) < 0.05 else '*** MISMATCH ***'}")
    ok &= abs(d) < 0.05
if not ok:
    sys.exit("local forward does not reproduce Colab -- refusing to export tracks")
print("forward validated\n")

STORMS = [("2026182N09163", "Bavi"), ("1986228N19120", "Wayne"),
          ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]
os.makedirs("track_build/v23map", exist_ok=True)
for tag in ("v10", "v21", "v23"):
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
        print(f"  {tag:4s} {nm:11s} {len(k):3d} fc | mean 120h {err:6.0f} km", flush=True)
    json.dump(out, open(f"track_build/v23map/{tag}_tracks.json", "w"))
print("\nwrote track_build/v23map/")
