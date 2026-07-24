"""Local verification of v26 (colab_v32_train.py) before it costs a single GPU-hour.

Rebuilds v21 -> v25 -> v26 from the real training sources on CPU and checks the two claims the
whole experiment rests on:
    USE_OCEAN=0  must reproduce v25 EXACTLY  (max-diff 0)   -- else the ablation is meaningless
    USE_OCEAN=1  must MOVE the intensity output              -- else the token is inert
and, because v26 wires the patch into the intensity decoder only, that the TRACK output is
untouched. The classes are extracted verbatim from colab_v32_train.py; nothing is retyped.
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
Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; vpair = G["vpair"]
DEVICE = torch.device("cpu")

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": 6 * 3600 / 1000.0, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

_E = np.load("track_build/env_features.npz", allow_pickle=True)
EFEAT = _E["feat"].astype("float32"); EGOT = _E["got"].astype("float32"); NENV = EFEAT.shape[1]
_p = EGOT > 0
_mu = np.array([EFEAT[_p[:, c], c].mean() if _p[:, c].any() else 0.0 for c in range(NENV)], "float32")
_sd = np.array([EFEAT[_p[:, c], c].std() + 1e-6 if _p[:, c].any() else 1.0 for c in range(NENV)], "float32")
ENORM = ((EFEAT - _mu[None]) / _sd[None]) * EGOT

v31 = open("colab_v31_train.py").read()
ed = re.search(r"class _EnvDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", v31, re.S).group(0)
te = re.search(r"class TrackFormerEnv\(V21\):.*?finally:\n            self\._envn = self\._envg = None\n", v31, re.S).group(0)
g25 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "NENV": NENV, "USE_ENV": 1}
exec(ed, g25); exec(te, g25)
V25 = g25["TrackFormerEnv"]

# ---- ocean patch, exactly as v26 prepares it -------------------------------------------------
OPATCH = _E["ohc_patch"]; OGOT = _E["ohc_got"].astype("float32")
_s = OPATCH[OGOT > 0][::17].astype("float32")
OM = np.array([float(_s[:, c][_s[:, c] != 0].mean()) for c in range(3)], "float32")
OS = np.array([float(_s[:, c][_s[:, c] != 0].std()) + 1e-6 for c in range(3)], "float32")
del _s


def ocean_in(j):
    p = OPATCH[j].astype("float32")
    valid = (p != 0).astype("float32")
    p = (p - OM[None, :, None, None]) / OS[None, :, None, None] * valid
    return np.concatenate([p, valid[:, :1]], 1)


# ---- v26 classes, extracted verbatim from the training script --------------------------------
v32 = open("colab_v32_train.py").read()
src_cnn = re.search(r"class OceanCNN\(nn\.Module\):.*?return self\.net\(x\)\n", v32, re.S).group(0)
src_dec = re.search(r"class _OceanDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", v32, re.S).group(0)
src_mod = re.search(r"class TrackFormerOcean\(V25\):.*?self\._op = self\._og = None\n", v32, re.S).group(0)
g26 = {"V25": V25, "torch": torch, "nn": nn, "F": F, "math": math, "USE_OCEAN": 1}
exec(src_cnn, g26); exec(src_dec, g26); exec(src_mod, g26)
V26 = g26["TrackFormerOcean"]


def remap(sd):
    out = {}
    for k, v in sd.items():
        out[("int_dec.dec." + k[len("int_dec."):]) if k.startswith("int_dec.") else k] = v
    return out


torch.manual_seed(0); m25 = V25().eval()
torch.manual_seed(0); m26 = V26().eval()
miss, unexp = m26.load_state_dict(remap(m25.state_dict()), strict=False)
assert not unexp, f"unexpected keys: {unexp[:5]}"
extra = sorted({k.split(".")[0] for k in miss})
print(f"params  v25 {sum(p.numel() for p in m25.parameters()):,} -> "
      f"v26 {sum(p.numel() for p in m26.parameters()):,}")
print(f"new modules in v26: {extra}")
assert set(extra) <= {"ocean_cnn", "ocean_pos"}, f"v26 added more than the ocean branch: {extra}"

jo = np.where(OGOT > 0)[0][:8]          # windows that REALLY have ocean, so the test is not vacuous
a = [torch.from_numpy(track[jo]), torch.from_numpy(vpair[jo]), torch.from_numpy(SLP[jo]),
     torch.from_numpy(ENORM[jo]), torch.from_numpy(EGOT[jo])]
op = torch.from_numpy(ocean_in(jo)); og = torch.from_numpy(OGOT[jo])
print(f"probe windows: {len(jo)}, ocean present on {int((OGOT[jo]>0).sum())}/{len(jo)}")

with torch.no_grad():
    ref = m25(*a)
    g26["USE_OCEAN"] = 0
    off = m26(*a, op, og)
    g26["USE_OCEAN"] = 1
    on = m26(*a, op, og)

d_off = max(float((x - y).abs().max()) for x, y in zip(ref[:2], off[:2]))
d_int = float((ref[0][..., 2:] - on[0][..., 2:]).abs().max())
d_trk = float((ref[0][..., :2] - on[0][..., :2]).abs().max())
print(f"\nUSE_OCEAN=0 vs v25   max-diff  {d_off:.3e}   (must be exactly 0)")
print(f"USE_OCEAN=1 intensity max-diff  {d_int:.3e}   (must be > 0)")
print(f"USE_OCEAN=1 track     max-diff  {d_trk:.3e}   (intensity-only wiring)")

ok = True
if d_off != 0.0:
    print("FAIL: USE_OCEAN=0 does not reproduce v25 -- the ablation would be meaningless"); ok = False
if d_int <= 0.0:
    print("FAIL: the ocean token does not move the intensity output -- it is inert"); ok = False
# a window with NO ocean must fall back to the learned 'no data' token, not a fabricated sea
jz = np.where(OGOT == 0)[0][:8]
az = [torch.from_numpy(track[jz]), torch.from_numpy(vpair[jz]), torch.from_numpy(SLP[jz]),
      torch.from_numpy(ENORM[jz]), torch.from_numpy(EGOT[jz])]
with torch.no_grad():
    m26._op = torch.from_numpy(ocean_in(jz)); m26._og = torch.from_numpy(OGOT[jz])
    tok = m26._ocean_token()
    m26._op = m26._og = None
    cnn_part = float((tok - m26.ocean_pos).abs().max())
print(f"no-ocean windows: CNN contribution beyond the learned token = {cnn_part:.3e} (must be 0)")
if cnn_part != 0.0:
    print("FAIL: a window without ocean data still receives CNN output"); ok = False

print("\nALL INIT ASSERTIONS PASSED" if ok else "\n*** VERIFICATION FAILED ***")
sys.exit(0 if ok else 1)
