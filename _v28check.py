"""Local verification of v28 (colab_v34_train.py) before any GPU time.

The v28 claim is only interpretable if the drift adapter touches EXACTLY the North channel and
nothing else, and is exactly antisymmetric across the equator. Five checks, on CPU:
    USE_DRIFT=0 vs v23        max-diff 0 on all 17 channels
    USE_DRIFT=1 North moves   > 0 (small)
    East channel             bit-identical
    intensity ch 2-16        bit-identical
    mirror equivariance      drift(z0,+lat) = -drift(z0,-lat) exactly
    gradient reachability    the adapter receives gradient
plus the state-dict check: loading a v23 checkpoint leaves only 'drift.*' missing.
"""
import json, re, math, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
DEVICE = torch.device("cpu")

nb = json.load(open("colab_train_v17.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
body = body.replace('"/content/d/steer5_int8.npz"', '"track_build/dlm4_int8.npz"')
body = body.replace('"/content/d/track_windows_v13.npz"', '"track_build/track_windows_v13.npz"')
body = body.replace('DEVICE = torch.device("cuda")', 'DEVICE = torch.device("cpu")')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os,
     "json": json, "time": __import__("time"), "math": math}
exec(compile(body, "<v17-notebook>", "exec"), G)
Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; vpair = G["vpair"]; z = G["z"]
tmean = G["tmean"]; tstd = G["tstd"]
TM = torch.tensor(tmean); TS = torch.tensor(tstd)

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))
KM6H = 6 * 3600 / 1000.0
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
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

# extract the two v28 classes verbatim from the training script
src = open("colab_v34_train.py").read()
cd = re.search(r"class MeridionalDrift\(nn\.Module\):.*?return sign\[:, None\] \* self\.a_max \* mag", src, re.S).group(0)
wd = re.search(r"class TrackFormerDrift\(V23\):.*?return s, ls, fp\n", src, re.S).group(0)
gd = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "TM": TM, "TS": TS,
      "A_MAX": 0.65, "USE_DRIFT": 1}
exec(cd, gd); exec(wd, gd)
V28 = gd["TrackFormerDrift"]

j = np.arange(16)
h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]), h, torch.from_numpy(HAVE[j])]

torch.manual_seed(0); m23 = V23().eval()
torch.manual_seed(0); m28 = V28().eval()
print(f"params  v23 {sum(p.numel() for p in m23.parameters()):,} -> "
      f"v28 {sum(p.numel() for p in m28.parameters()):,}  (drift {sum(p.numel() for p in m28.drift.parameters())})")
miss, unexp = m28.load_state_dict(m23.state_dict(), strict=False)
extra = sorted({k.split('.')[0] for k in miss})
print(f"loading v23 into v28: missing modules {extra}, unexpected {unexp[:3]}")
assert not unexp and extra == ["drift"], "state-dict is not exactly v23 + drift"

with torch.no_grad():
    ref = m23(*a)
    gd["USE_DRIFT"] = 0
    off = m28(*a)
    gd["USE_DRIFT"] = 1
    on = m28(*a)
d_off = max(float((x - y).abs().max()) for x, y in zip(ref[:2], off[:2]))
d_N = float((ref[0][..., 1] - on[0][..., 1]).abs().max())
d_E = float((ref[0][..., 0] - on[0][..., 0]).abs().max())
d_int = float((ref[0][..., 2:] - on[0][..., 2:]).abs().max())
with torch.no_grad():
    z0, lat = m28._z0(torch.from_numpy(track[j]))
    d_mirror = float((m28.drift(z0, lat) + m28.drift(z0, -lat)).abs().max())
    n_shift_120 = float((on[0][:, 19, 1] - ref[0][:, 19, 1]).abs().mean()) * 100  # km
m28.train()
s, ls, fp = m28(*a); s[..., 1].sum().backward()
grad_ok = m28.drift.mlp[0].weight.grad is not None and float(m28.drift.mlp[0].weight.grad.abs().sum()) > 0

print(f"\nUSE_DRIFT=0 vs v23          max-diff {d_off:.3e}   (must be 0)")
print(f"USE_DRIFT=1 North moves     max-diff {d_N:.3e}   (must be > 0)  ~{n_shift_120:.2f} km at 120 h")
print(f"East channel untouched      max-diff {d_E:.3e}   (must be 0)")
print(f"intensity 2-16 untouched    max-diff {d_int:.3e}   (must be 0)")
print(f"mirror equivariance         |dpos+dneg| {d_mirror:.3e}   (must be 0)")
print(f"drift gradient reachable     {grad_ok}")

ok = (d_off == 0.0) and (d_N > 0.0) and (d_E == 0.0) and (d_int == 0.0) and (d_mirror == 0.0) and grad_ok
print("\nALL INIT ASSERTIONS PASSED" if ok else "\n*** VERIFICATION FAILED ***")
sys.exit(0 if ok else 1)
