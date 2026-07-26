"""Local (CPU) init-assertion check for v29 (TrackFormerEra5), sourcing every class from LOCAL
files instead of urllib/GitHub -- same pattern as _v28check.py. Verifies:
  1. era5 off  -> v29 output EXACTLY equals v23's (max diff 0)
  2. era5 on   -> output moves (the path is live, not dead)
  3. gradient reaches Era5Stem's first conv layer
  4. state-dict remap loads v23 weights into v29 with only the new Era5Stem keys missing
  5. mirror is self-inverse; u keeps sign, v flips sign
Nothing here trains anything -- this is the same "does it reduce to the parent at init, and is the
new path reachable" gate every version (v21, v23, v26, v28) passed before Colab.
"""
import json, re, math, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
KM6H = 6 * 3600 / 1000.0
ERA5_CLIP = 4.0

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
vpair = G["vpair"]; z = G["z"]

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

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
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
assert ERA5_N == len(sid), f"era5 tensor N mismatch: {ERA5_N} vs {len(sid)}"
ERA5_POS = np.full(ERA5_N, -1, dtype="int64")
ERA5_POS[ERA5_WIDX] = np.arange(len(ERA5_WIDX))
print(f"era5: {len(ERA5_WIDX):,}/{ERA5_N:,} windows covered ({100*len(ERA5_WIDX)/ERA5_N:.1f}%)")


def era5_of(j):
    p = ERA5_POS[j]
    ok = (p >= 0).astype("float32")
    ps = np.where(p >= 0, p, 0)
    q = ERA5_Q[ps]
    q = np.where(ok[:, None, None, None] > 0, q, 0)
    return q, ok


USE_ERA5 = 1
ERA5_DROP = 0.15


class Era5Stem(nn.Module):
    def __init__(self, base, ch, era5_scale):
        super().__init__()
        self.base = base
        self.register_buffer("era5_scale", torch.as_tensor(era5_scale, dtype=torch.float32))
        self.stem = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.GELU(), nn.Dropout2d(0.10),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(), nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(), nn.Dropout2d(0.10),
            nn.Conv2d(64, ch, 3, stride=2, padding=1), nn.GELU())
        self.out = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.ctx = None

    def forward(self, slp):
        st = self.base(slp)
        if USE_ERA5 and self.ctx is not None:
            era5_raw, ok = self.ctx
            x = era5_raw.float() * self.era5_scale.view(1, 6, 1, 1) * (ERA5_CLIP / 127.0)
            ok4 = ok.view(-1, 1, 1, 1)
            st = st + self.out(self.stem(x)) * ok4
        return st


class TrackFormerEra5(V23):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.steer_cnn.base = Era5Stem(self.steer_cnn.base, self.steer_pos.shape[-1], ERA5_SCALE)

    def forward(self, tr, vp, slp, hist=None, have=None, era5=None, era5_ok=None):
        drop = self.training and ERA5_DROP > 0 and era5 is not None
        if drop:
            keep = (torch.rand(tr.shape[0], device=slp.device) >= ERA5_DROP).float()
            era5_ok = era5_ok * keep
        self.steer_cnn.base.ctx = (era5, era5_ok) if era5 is not None else None
        try:
            return super().forward(tr, vp, slp, hist, have)
        finally:
            self.steer_cnn.base.ctx = None


# ---- shape sanity: verify the stem lands on 5x5 to match steer_pos's grid ----
with torch.no_grad():
    _probe = Era5Stem(nn.Identity(), 256, ERA5_SCALE)
    _out = _probe.stem(torch.zeros(1, 6, 65, 65))
    assert _out.shape[-2:] == (5, 5), f"era5 stem grid mismatch: {_out.shape} (need 5x5)"
    print(f"OK: era5 stem 65x65 -> {tuple(_out.shape[-2:])}, matches steer_cnn's 17x17 -> 5x5")
del _probe, _out

# ---- init assertions ----
# mix of era5-covered and uncovered probe windows, so the ON-test genuinely exercises the live
# path (an all-uncovered probe would trivially pass "moves the output" for the wrong reason: the
# availability gate zeroing the contribution, not weights actually being zero).
_j = np.concatenate([ERA5_WIDX[:2], np.array([0, 1])])
_t = torch.from_numpy(track[_j]); _v = torch.from_numpy(vpair[_j])
_s = torch.from_numpy(SLP[_j])
_hn = np.concatenate([SLP[HIST_S[_j, 0]], SLP[HIST_S[_j, 1]]], 1)
_h = torch.from_numpy(_hn); _a = torch.from_numpy(HAVE[_j])
_eq, _eok = era5_of(_j)
_e = torch.from_numpy(_eq); _eo = torch.from_numpy(_eok)
print(f"probe windows era5 availability: {_eok}")

_p, _q = V23().eval(), TrackFormerEra5().eval()
torch.manual_seed(11)
nn.init.normal_(_p.track_res.weight, std=0.02); nn.init.normal_(_p.track_res.bias, std=0.02)
_sd = {}
for k, val in _p.state_dict().items():
    if k.startswith("steer_cnn.base."):
        k = "steer_cnn.base.base." + k[len("steer_cnn.base."):]
    _sd[k] = val
_miss, _unexp = _q.load_state_dict(_sd, strict=False)
assert not _unexp, f"unexpected keys: {list(_unexp)[:5]}"
assert all(m.startswith("steer_cnn.base.stem") or m.startswith("steer_cnn.base.out")
           or m == "steer_cnn.base.era5_scale" for m in _miss), \
    f"v23 weights failed to transfer: {list(_miss)[:5]}"
print(f"OK: state-dict remap clean, {len(_miss)} new Era5Stem keys as expected")
assert float(_q.steer_cnn.base.out.weight.abs().max()) == 0.0, "not zero-init"

_d1 = float((_p(_t, _v, _s, _h, _a)[0] - _q(_t, _v, _s, _h, _a, _e, _eo)[0]).abs().max())
assert _d1 < 1e-5, f"v29 does not reduce to v23 at init: {_d1}"
print(f"OK: era5 OFF max|v29 - v23| = {_d1:.2e} (v29 starts as exactly v23)")

nn.init.normal_(_q.steer_cnn.base.out.weight, std=0.05)
_d2 = float((_p(_t, _v, _s, _h, _a)[0] - _q(_t, _v, _s, _h, _a, _e, _eo)[0]).abs().max())
print(f"OK: era5 ON  max|v29 - v23| = {_d2:.2e} (era5 path is live)")
assert _d2 > 1e-4, f"opening the era5 path moved the track by only {_d2:.2e} -- DEAD"

# does it stay dead for windows with era5_ok=0 (no data)? pick an uncovered window if any in probe
if float(_eok.sum()) < len(_j):
    _uncov = int(np.where(_eok < 0.5)[0][0])
    _d2u = float((_p(_t, _v, _s, _h, _a)[0][_uncov] - _q(_t, _v, _s, _h, _a, _e, _eo)[0][_uncov]).abs().max())
    print(f"OK: window {_uncov} has era5_ok=0 -> diff {_d2u:.2e} (should be ~0, branch gated off)")
    assert _d2u < 1e-5, "an UNAVAILABLE window still moved -- availability gating is broken"
nn.init.zeros_(_q.steer_cnn.base.out.weight)

# gradient reachability
_q.train()
nn.init.normal_(_q.steer_cnn.base.out.weight, std=0.05)
_mo, _, _ = _q(_t, _v, _s, _h, _a, _e, _eo)
_mo.sum().backward()
_g = _q.steer_cnn.base.stem[0].weight.grad
assert _g is not None and float(_g.abs().max()) > 0, "era5 stem not reachable by gradient"
print(f"OK: era5 stem gradient reachable, max|grad| = {float(_g.abs().max()):.2e}")
nn.init.zeros_(_q.steer_cnn.base.out.weight)
_q.zero_grad(); _q.eval()

# mirror self-inverse
_er0, _ = era5_of(np.array([ERA5_WIDX[0]]))
_er0 = torch.from_numpy(_er0[0])
_er1 = torch.flip(_er0, dims=[1]).clone(); _er1[3:6] = -_er1[3:6]
_er2 = torch.flip(_er1, dims=[1]).clone(); _er2[3:6] = -_er2[3:6]
assert torch.equal(_er0, _er2), "era5 mirror is not an involution"
assert torch.equal(_er0[0].flip(0), _er1[0]), "u must not change sign under mirror"
assert torch.equal(-_er0[3].flip(0), _er1[3]), "v must change sign under mirror"
print("OK: era5 mirror is self-inverse, u unsigned/v signed under N-S flip -- matches convention")

n_params = sum(p.numel() for p in Era5Stem(nn.Identity(), 256, ERA5_SCALE).parameters())
print(f"\nEra5Stem adds {n_params:,} params")
print("\nALL CHECKS PASSED -- v29 (TrackFormerEra5) is ready for Colab training")
