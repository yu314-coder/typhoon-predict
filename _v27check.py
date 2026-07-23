"""Local verification of v27 (colab_v33_train.py) before it costs a single GPU-hour.

v27 claims to be v23 AND v26 at once. That claim is only meaningful if the combination is exactly
additive, so four things are checked on CPU against the real training sources:

    all switches OFF   must reproduce v21 EXACTLY   (max-diff 0)
    HIST only          must reproduce v23 EXACTLY   (max-diff 0)  <- the load-bearing one
    all ON             must MOVE the intensity output
    ocean toggled      must leave the TRACK output bit-identical

The second is what makes the experiment interpretable: if HIST-only does not reproduce v23, then
adding the env/ocean machinery has perturbed the track path and any track number v27 produces
cannot be compared with v23's 435.0 km.
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

# history maps
sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

_E = np.load("track_build/env_features.npz", allow_pickle=True)
EFEAT = _E["feat"].astype("float32"); EGOT = _E["got"].astype("float32"); NENV = EFEAT.shape[1]
_p = EGOT > 0
_mu = np.array([EFEAT[_p[:, c], c].mean() if _p[:, c].any() else 0.0 for c in range(NENV)], "float32")
_sd = np.array([EFEAT[_p[:, c], c].std() + 1e-6 if _p[:, c].any() else 1.0 for c in range(NENV)], "float32")
ENORM = ((EFEAT - _mu[None]) / _sd[None]) * EGOT

_O = np.load("track_build/ocean_patch.npz", allow_pickle=True)
OQ = _O["q"]; OSC = _O["scale"].astype("float32"); OGOT = _O["got"].astype("float32")
_s = (OQ[OGOT > 0][::17].astype("float32") * OSC[None, :, None, None])
OM = np.array([float(_s[:, c][_s[:, c] != 0].mean()) for c in range(3)], "float32")
OS = np.array([float(_s[:, c][_s[:, c] != 0].std()) + 1e-6 for c in range(3)], "float32")
del _s


def ocean_in(j):
    p = OQ[j].astype("float32") * OSC[None, :, None, None]
    v = (p != 0).astype("float32")
    p = (p - OM[None, :, None, None]) / OS[None, :, None, None] * v
    return np.concatenate([p, v[:, :1]], 1)


v31 = open("colab_v31_train.py").read()
ed = re.search(r"class _EnvDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", v31, re.S).group(0)
v32 = open("colab_v32_train.py").read()
oc = re.search(r"class OceanCNN\(nn\.Module\):.*?return self\.net\(x\)\n", v32, re.S).group(0)
od = re.search(r"class _OceanDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", v32, re.S).group(0)
v33 = open("colab_v33_train.py").read()
# anchor on the finally-block, not the bare assignment: the identical line also ends __init__, and a
# non-greedy match stops there -- silently cutting the class off before forward() is defined, which
# then falls through to TrackFormerHist.forward and fails with a positional-argument error.
al = re.search(r"class TrackFormerAll\(V23\):.*?finally:\n            "
               r"self\._envn = self\._envg = self\._op = self\._og = None\n", v33, re.S).group(0)
assert "def forward" in al and "_ocean_token" in al, "TrackFormerAll extraction is truncated"
gA = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "NENV": NENV,
      "USE_ENV": 1, "USE_OCEAN": 1}
exec(ed, gA); exec(oc, gA); exec(od, gA); exec(al, gA)
V27 = gA["TrackFormerAll"]

j = np.where((OGOT > 0) & (HAVE[:, 0] > 0) & (HAVE[:, 1] > 0) & (EGOT.sum(1) > 0))[0][:8]
print(f"probe windows: {len(j)} (history + env + ocean all present)")
a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
hv = torch.from_numpy(HAVE[j])
en = torch.from_numpy(ENORM[j]); eg = torch.from_numpy(EGOT[j])
op = torch.from_numpy(ocean_in(j)); og = torch.from_numpy(OGOT[j])

torch.manual_seed(0); m21 = V21().eval()
torch.manual_seed(0); m23 = V23().eval()
torch.manual_seed(0); mA = V27().eval()
print(f"params  v21 {sum(p.numel() for p in m21.parameters()):,} | "
      f"v23 {sum(p.numel() for p in m23.parameters()):,} | v27 {sum(p.numel() for p in mA.parameters()):,}")


def remap(sd, ntrack, nint):
    out = {}
    for k, v in sd.items():
        if k.startswith("int_dec."):
            out["int_dec." + "dec." * nint + k[len("int_dec."):]] = v
        elif k.startswith("track_dec."):
            out["track_dec." + "dec." * ntrack + k[len("track_dec."):]] = v
        else:
            out[k] = v
    return out


miss, unexp = mA.load_state_dict(remap(m23.state_dict(), 1, 2), strict=False)
assert not unexp, f"unexpected keys mapping v23 -> v27: {unexp[:6]}"
extra = sorted({k.split(".")[0] for k in miss})
print(f"new modules in v27: {extra}")
assert set(extra) <= {"env_mlp", "env_pos", "ocean_cnn", "ocean_pos"}, f"unexpected: {extra}"

ok = True
with torch.no_grad():
    gA["USE_ENV"], gA["USE_OCEAN"], g23["USE_HIST"] = 0, 0, 0
    ref21 = m21(*a); off = mA(*a, None, None, en, eg, op, og)
    d21 = max(float((x - y).abs().max()) for x, y in zip(ref21[:2], off[:2]))

    g23["USE_HIST"] = 1
    ref23 = m23(*a, h, hv); onlyh = mA(*a, h, hv, en, eg, op, og)
    d23 = max(float((x - y).abs().max()) for x, y in zip(ref23[:2], onlyh[:2]))

    gA["USE_ENV"], gA["USE_OCEAN"] = 1, 1
    full = mA(*a, h, hv, en, eg, op, og)
    d_int = float((ref23[0][..., 2:] - full[0][..., 2:]).abs().max())

    gA["USE_OCEAN"] = 0
    noocean = mA(*a, h, hv, en, eg, op, og)
    gA["USE_OCEAN"] = 1
    d_trk = float((noocean[0][..., :2] - full[0][..., :2]).abs().max())

print(f"\nall switches OFF vs v21     max-diff {d21:.3e}   (must be exactly 0)")
print(f"HIST only        vs v23     max-diff {d23:.3e}   (must be exactly 0)")
print(f"all ON: intensity moves     max-diff {d_int:.3e}   (must be > 0)")
print(f"ocean toggled: track        max-diff {d_trk:.3e}   (must be exactly 0)")
for cond, msg in ((d21 != 0.0, "all-off does not reproduce v21"),
                  (d23 != 0.0, "HIST-only does not reproduce v23 -- combination is NOT additive"),
                  (d_int <= 0.0, "env+ocean do not move the intensity output"),
                  (d_trk != 0.0, "the ocean token is leaking into the track output")):
    if cond:
        print("FAIL:", msg); ok = False

print("\nALL INIT ASSERTIONS PASSED" if ok else "\n*** VERIFICATION FAILED ***")
sys.exit(0 if ok else 1)
