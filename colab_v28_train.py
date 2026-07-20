"""v23 on Colab — v21's chain-of-thought over a TEMPORAL steering stack.

    !wget -q -O /content/v23.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/<SHA>/colab_v28_train.py
    import os; os.environ["V28_SEEDS"]="5"; exec(open('/content/v23.py').read())

WHY THIS AND NOT MORE ARCHITECTURE.

Measured on WP+EP 2020+ (3763 windows, 5-seed ensembles), feeding v21 the TRUE steering flow
instead of its own prediction drops error 443.62 -> 328.59 km, -115 km / -25.9%. The gain scales
with lead: -1 km at 24 h, -40 at 48, -124 at 72, -220 at 96, -323 at 120. For contrast the whole
architecture chain v17 -> v20 -> v21 -> v22 bought 19.4 km, and a paired-storm bootstrap says none
of it is significant (v21 vs v20 p=0.094, v22 vs v21 p=0.939).

So flow prediction is the bottleneck, and the reason it is hard is visible in the shape of that
curve: the model sees the steering field at ONE instant and must infer 120 h of ridge and trough
evolution from a snapshot. At 24 h the snapshot is enough (-1 km). At 120 h it is not (-323 km).

v23 therefore gives the steering CNN t-24 h, t-12 h and t0 instead of t0 alone, so the MOVEMENT of
the subtropical ridge is visible rather than having to be guessed. v21's flow head, its A
coefficients and the rest of the forward are unchanged -- this changes what the model can see, not
how it reasons.

NO NEW EXTRACTION. The t-24 h field for a window is just the t0 field of the same storm's window
24 h earlier, which is already in dlm4_int8.npz. 82.6% of windows have both predecessors; the rest
fall back to repeating t0, with an availability channel so the model can tell a real history from a
padded one (same convention the project already uses: unavailable == exact zeros, never fabricated).

DEGRADES TO v21 AT INIT. The extra history enters through a zero-initialised 1x1 conv added to the
steer_cnn stem, so at step zero the stack contributes exactly nothing and the forward is v21's --
asserted below, with the same live/dead check that caught v22's vacuous assertion.
"""

import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
N_SEEDS = int(os.environ.get("V28_SEEDS", "5"))
W_FLOW = float(os.environ.get("W_FLOW", "0.3"))
USE_HIST = int(os.environ.get("V28_USE_HIST", "1"))     # 0 = ablation, v21 with the same code path
TAG = os.environ.get("V28_TAG", "v23" if USE_HIST else "v23abl")
KM6H = 6 * 3600 / 1000.0

for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)

nb = json.load(open(urllib.request.urlretrieve(f"{RAW}/colab_train_v17.ipynb",
                                               "/content/_v17.ipynb")[0]))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
assert body.count("steer5_int8.npz") == 1
body = body.replace('"/content/d/steer5_int8.npz"', '"/content/dlm4_int8.npz"')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os, "json": json,
     "time": time, "math": __import__("math")}
exec(compile(body, "<v17-notebook>", "exec"), G)

DEVICE = G["DEVICE"]; TARGET_SCALE = G["TARGET_SCALE"]
Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
mask = G["mask"]; vpair = G["vpair"]; z = G["z"]
tr_idx, va_idx, te_idx = G["tr_idx"], G["va_idx"], G["te_idx"]
basins = G["basins"]; mirror = G["mirror"]
EPOCHS, PATIENCE, BATCH = G["EPOCHS"], G["PATIENCE"], G["BATCH"]
LR, WEIGHT_DECAY, MIRROR_P = G["LR"], G["WEIGHT_DECAY"], G["MIRROR_P"]

# ---- the temporal index: for each window, the row holding the same storm 12 h / 24 h earlier ----
sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
SIX = int(6 * 3600 * 1e9)
_key = {}
for i in range(len(sid)):
    _key[(sid[i], int(bt[i]))] = i
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):                       # 12 h, 24 h
        j = _key.get((sid[i], int(bt[i]) - back * SIX), -1)
        HIST[i, c] = j
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])   # fall back to self (repeat t0)
print(f"temporal stack: t-12h on {100*HAVE[:,0].mean():.1f}% of windows, "
      f"t-24h on {100*HAVE[:,1].mean():.1f}%, both on "
      f"{100*(HAVE.prod(1)).mean():.1f}%", flush=True)

_lf = np.load("/content/lead_flow.npz")
FLOW_T = _lf["flow"].astype("float32"); FLOW_M = _lf["got"].astype("float32")
DSC = np.load("/content/dlm4_int8.npz")["scale"][2:4].astype("float32")
_ii, _jj = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
_d = np.hypot(_ii, _jj) * 2.5
ANN = torch.tensor(((_d >= 3.0) & (_d <= 8.0)).astype("float32"), device=DEVICE)

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
_g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, urllib.request.urlopen(f"{RAW}/colab_v26_train.py").read().decode(),
               re.S).group(0), _g21)
V21 = _g21["TrackFormerCoT"]


class HistStem(nn.Module):
    """v17's steering stem, plus a zero-initialised residual carrying t-12 h and t-24 h.

    The history is added to the stem's OUTPUT rather than widening its input, so every weight in
    the original stem keeps its meaning and the whole addition starts at exactly zero. The extra
    branch mirrors the stem's stride pattern (17 -> 9 -> 5) because it has to land on the same grid.

    Context is stashed on the module instead of threaded through forward() so that v23 can inherit
    v21's forward VERBATIM. Rewriting that forward by hand is how v15 and v21 both broke, and how
    the first draft of this file broke again -- it returned 19 channels where v17 returns 17,
    because the intensity tail was written from memory instead of copied.
    """
    def __init__(self, base, ch):
        super().__init__()
        self.base = base
        self.stem = nn.Sequential(
            nn.Conv2d(10, 24, 3, padding=1), nn.GELU(), nn.Dropout2d(0.10),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(), nn.Dropout2d(0.10),
            nn.Conv2d(48, ch, 3, stride=2, padding=1), nn.GELU())
        self.out = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.ctx = None

    def forward(self, slp):
        st = self.base(slp)
        if USE_HIST and self.ctx is not None:
            hist, have = self.ctx
            hv = have.view(-1, 2, 1, 1).expand(-1, 2, hist.shape[-2], hist.shape[-1])
            st = st + self.out(self.stem(torch.cat([hist, hv], 1)))
        return st


class TrackFormerHist(V21):
    """v21 with the temporal steering stack. forward() is v21's, untouched."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.steer_cnn = HistStem(self.steer_cnn, self.steer_pos.shape[-1])

    def forward(self, tr, vp, slp, hist=None, have=None):
        # STEER_DROP zeroes the present field during training; the history must be dropped with the
        # SAME mask, or the model leans on a past field that is still there when the present one is
        # gone -- an advantage the deployed model never gets.
        #
        # v21's forward applies STEER_DROP itself. Doing it here as well would drop twice with
        # independent masks (0.20 -> 0.36 effective), so the inner one is neutralised for the
        # duration of the call and restored in the finally.
        sd = G["STEER_DROP"]
        drop = self.training and sd > 0 and hist is not None
        if drop:
            keep = (torch.rand(tr.shape[0], 1, 1, 1, device=slp.device) >= sd).float()
            slp = slp * keep
            hist = hist * keep
            have = have * keep.view(-1, 1)
            G["STEER_DROP"] = 0.0
        self.steer_cnn.ctx = (hist, have) if hist is not None else None
        try:
            return super().forward(tr, vp, slp)
        finally:
            self.steer_cnn.ctx = None
            G["STEER_DROP"] = sd


def hist_of(j):
    """[b,8,17,17] stack of the two past steering fields, plus their availability."""
    a = SLP[HIST_S[j, 0]]; b_ = SLP[HIST_S[j, 1]]
    return np.concatenate([a, b_], 1), HAVE[j]


with torch.no_grad():
    _j = np.arange(4)
    _t = torch.from_numpy(track[_j]).to(DEVICE); _v = torch.from_numpy(vpair[_j]).to(DEVICE)
    _s = torch.from_numpy(SLP[_j]).to(DEVICE)
    _hn, _hv = hist_of(_j)
    _h = torch.from_numpy(_hn).to(DEVICE); _a = torch.from_numpy(_hv).to(DEVICE)
    _p, _q = V21().to(DEVICE).eval(), TrackFormerHist().to(DEVICE).eval()
    torch.manual_seed(7)
    nn.init.normal_(_p.track_res.weight, std=0.02); nn.init.normal_(_p.track_res.bias, std=0.02)
    _q.load_state_dict(_p.state_dict(), strict=False)
    assert float(_q.steer_cnn.out.weight.abs().max()) == 0.0, "history path is not zero-init"
    _d1 = float((_p(_t, _v, _s)[0] - _q(_t, _v, _s, _h, _a)[0]).abs().max())
    assert _d1 < 1e-5, f"{TAG} does not reduce to v21 at init: {_d1}"
    nn.init.normal_(_q.steer_cnn.out.weight, std=0.05)
    _d2 = float((_p(_t, _v, _s)[0] - _q(_t, _v, _s, _h, _a)[0]).abs().max())
    nn.init.zeros_(_q.steer_cnn.out.weight)
    assert not USE_HIST or _d2 > 1e-4, (
        f"{TAG}: opening the history path moved the track by {_d2:.2e} -- it is DEAD")
    print(f"init check: history off max|v23 - v21| = {_d1:.2e} (v23 starts as exactly v21)")
    print(f"init check: history on  max|v23 - v21| = {_d2:.2e} (history path is live)", flush=True)
del _p, _q
print(f"\n{TAG} ready. USE_HIST={USE_HIST}, {N_SEEDS} seeds.", flush=True)
