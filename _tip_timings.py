"""Typhoon Tip (1979), one panel per launch time, for v10 / v21 / v22.

Tip is held out of training for every model here, so each line is a genuine forecast of a storm
the weights have never seen.

Each launch is emitted as its own record with n=1, so the map draws a single bold forecast from
the filled dot plus hairlines to where Tip actually was at each valid time. Nothing on the page
uses information from after the launch moment.

Model classes are extracted verbatim from the training scripts (see _v22tracks.py) rather than
retyped, and the v21/v22 forwards are the same objects validated there against Colab.
"""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from make_rmt_tracks import observed_at, SIX_H

torch.set_num_threads(8)
R = 111.2; KM6H = 6 * 3600 / 1000.0


def km(a1, o1, a2, o2):
    return math.hypot((o2 - o1) * R * math.cos(math.radians((a1 + a2) / 2)), (a2 - a1) * R)


nb = json.load(open("colab_train_v17.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
     "DEVICE": torch.device("cpu"), "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p, src, re.S).group(0), G)
Base = G["TrackFormerV17"]

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))
CLS_RE = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"


def build_cls(path, rounds):
    s = re.search(CLS_RE, open(path).read(), re.S).group(0)
    g = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G,
         "ANN": ANN, "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": rounds, "USE_FLOW": 1}
    exec(s, g)
    return g["TrackFormerCoT"]


V21, V22 = build_cls("colab_v26_train.py", 0), build_cls("colab_v27_train.py", 2)


def load(ck, cls):
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("inner.")} or sd
    m = cls().eval(); m.load_state_dict(sd); return m


def build_v10():
    s = open("train_track_v10.py").read()
    g = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
         "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
    for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(pat, s, re.S).group(0), g)
    return g["TrackFormerV9"]


z = np.load("track_build/tip_fixed.npz", allow_pickle=True)
o13 = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
tr = z["track"].astype("float32"); nl = z["n_leads"].astype(int)
bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
tm = o13["track_mean"].astype("float32"); ts = o13["track_std"].astype("float32")
vp = np.concatenate([tr[:, -1, 2:4] * ts[2:4] + tm[2:4],
                     tr[:, -2, 2:4] * ts[2:4] + tm[2:4]], 1).astype("float32")
S20 = np.load("track_build/tip_dlm4.npy").astype("float32")
SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12)
obs = observed_at(bt, bla, blo)

M21 = [load(f"downloads/x/v21_seed{i}.pt", V21) for i in range(5)]
M22 = [load(f"downloads/x/v22_seed{i}.pt", V22) for i in range(5)]
m10 = build_v10()()
m10.load_state_dict(torch.load("track_build/track_v10_best.pt", map_location="cpu",
                               weights_only=False)["model"]); m10.eval()

# 10/17 00Z is deliberately absent: Tip has no best-track fix 120 h later, so its +120 h error is
# undefined and the panel would quote a nan. Every launch here verifies at the full horizon.
LAUNCHES = ["1979-10-05T00:00", "1979-10-07T06:00", "1979-10-09T12:00",
            "1979-10-11T00:00", "1979-10-13T06:00", "1979-10-15T12:00"]

os.makedirs("track_build/tiptime", exist_ok=True)
OUT = {t: {} for t in ("v10", "v21", "v22")}
rows = []


@torch.no_grad()
def forecast(tag, i):
    a_t, a_v = torch.from_numpy(tr[i:i + 1]), torch.from_numpy(vp[i:i + 1])
    if tag == "v10":
        sv, _ = m10(a_t, a_v)
    else:
        a = [a_t, a_v, torch.from_numpy(S20[i:i + 1])]
        sv = torch.stack([m(*a)[0] for m in (M21 if tag == "v21" else M22)]).mean(0)
    A = (sv * SC).float().numpy()[0]
    cE, cN = np.cumsum(A[:, 0]), np.cumsum(A[:, 1])
    la = bla[i] + cN / R
    lo = blo[i] + cE / (R * np.cos(np.radians((bla[i] + la) / 2)))
    return la, lo


for iso in LAUNCHES:
    T0 = int(np.datetime64(iso, "ns").astype("int64"))
    i = int(np.abs(bt - T0).argmin())
    assert abs(int(bt[i]) - T0) < SIX_H and nl[i] == 20, f"no full-horizon launch at {iso}"
    label = iso[5:7] + "/" + iso[8:10] + " " + iso[11:13] + "Z"
    la0, lo0 = bla[i], blo[i]
    row = [label]
    for tag in ("v10", "v21", "v22"):
        lat, lon = forecast(tag, i)
        errs, pairs = [], []
        for L in range(20):
            vt = int(round((T0 + (L + 1) * SIX_H) / SIX_H)) * SIX_H
            if vt in obs:
                errs.append((L + 1, km(obs[vt][0], obs[vt][1], lat[L], lon[L])))
                pairs.append([[float(lat[L]), float(lon[L])], [obs[vt][0], obs[vt][1]]])
        d = dict(errs)
        OUT[tag][label] = {
            "lat": [[float(la0)] + [float(x) for x in lat]],
            "lon": [[float(lo0)] + [float(x) for x in lon]],
            "base_time": [int(bt[i])],
            "base_lat": bla.tolist(), "base_lon": blo.tolist(),
            "launch": [float(la0), float(lo0)], "pairs": pairs, "n": 1,
            "err120_mean": d.get(20, float("nan"))}
        row.append(d.get(20, float("nan")))
    rows.append(row)

for tag in OUT:
    json.dump(OUT[tag], open(f"track_build/tiptime/{tag}_tracks.json", "w"))

print("Typhoon Tip 1979 -- 120 h error by launch time (km), Tip held out of all training\n")
print(f"  {'launch':>10s}  {'v10':>7s} {'v21':>7s} {'v22':>7s}")
for lb, a, b, c in rows:
    print(f"  {lb:>10s}  {a:7.0f} {b:7.0f} {c:7.0f}")
n = len(rows)
print(f"  {'mean':>10s}  " + " ".join(f"{sum(r[k] for r in rows)/n:7.0f}" for k in (1, 2, 3)))
print("\nwrote track_build/tiptime/")
