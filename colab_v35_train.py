"""v29 on Colab -- v23 plus 0.25-degree ERA5 steering wind (u/v @ 200/550/850 hPa), on top of v23's
existing 2.5-degree/temporal-history steer_cnn.

    !wget -q -O /content/v29.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/<SHA>/colab_v35_train.py
    import os; os.environ["V29_SEEDS"]="5"; exec(open('/content/v29.py').read())

WHY. v24 showed resolution NEAR THE VORTEX is what a wide, coarse box throws away (443 -> 529 km,
p=0.000, going from a storm-centred 2.5-deg patch to a fixed-box one at the SAME resolution). The
existing steer_cnn reads a 2.5-deg, 17x17, 4-channel field (SLPanom, SLPtend, u500, v500). ERA5 at
0.25 deg over the SAME storm-centred box is 10x finer -- this tests whether that extra resolution
in the wind field itself buys anything beyond what v23 already sees.

THE DATA. era5_steer_int8.npz: u/v @ 200/550/850 hPa, 65x65 patches, WP basin only, 2001-2019
(train) + 2020-2026 (test) -- EP and pre-2001 windows have NO coverage (era5_ok=False there, and
the branch degrades to zero, i.e. pure v23, for them). z (geopotential) was DROPPED: its per-year
source quantization destroyed the within-patch gradient structure (see merge_era5_steer.py); u/v
themselves are unaffected and carry the primary steering signal directly.

    train coverage  20.3% of the full WP+EP population (27.4% of its WP-only rows)
    val   coverage  55.6%                                (88.1% of its WP-only rows)
    test  coverage  65.5%                                (91.4% of its WP-only rows)

Train/val/test population and split are UNCHANGED from v23 -- no windows are added or removed, so
this evaluates as an apples-to-apples ablation, exactly like v26abl/v27abl/v28abl. Windows without
era5 coverage simply see the branch contribute zero, same as v23abl always does.

ARCHITECTURE. Era5Stem wraps v23's steer_cnn a SECOND time (v23 = HistStem wrapping v21's raw
steer_cnn once already), the exact same "zero-init residual added to the wrapped module's output"
idiom used to build v23 from v21. It is nested BENEATH HistStem, not above it -- Era5Stem replaces
`self.steer_cnn.base` (the original conv stack) rather than `self.steer_cnn` itself, so
TrackFormerHist.forward's own `self.steer_cnn.ctx = (hist, have)` line keeps targeting HistStem
directly and is untouched. Rewriting v23's forward by hand is exactly how v15 and v21 broke.

    self.steer_cnn        = HistStem(raw_steer_cnn, ch)       # v23's wrap (already there)
    self.steer_cnn.base   = Era5Stem(raw_steer_cnn, ch)        # v29's wrap, nested underneath

The era5 stem mirrors steer_cnn's own stride pattern (17 -> 9 -> 5) at the finer input size:
65 -> 33 -> 17 -> 9 -> 5, landing on the identical 5x5 grid so the same elementwise add works.

DEGRADES TO v23 AT INIT. Era5Stem's output 1x1 conv is zero-initialised, exactly like HistStem's.
ERA5_DROP applies an INDEPENDENT train-time dropout to the era5 branch (separate mask from
STEER_DROP's own present/history mask): era5 availability correlates strongly with basin (WP-only)
and recency (2001+), and without its own dropout the model could learn "is era5 present" as a
spurious extra feature rather than genuinely using the wind field.
"""

import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
DRIVE = "/content/drive/MyDrive/typhoon"
N_SEEDS = int(os.environ.get("V29_SEEDS", "5"))
W_FLOW = float(os.environ.get("W_FLOW", "0.3"))
USE_ERA5 = int(os.environ.get("V29_USE_ERA5", "1"))     # 0 = ablation, v23 with the same code path
ERA5_DROP = float(os.environ.get("ERA5_DROP", "0.15"))
TAG = os.environ.get("V29_TAG", "v29" if USE_ERA5 else "v29abl")
KM6H = 6 * 3600 / 1000.0
ERA5_CLIP = 4.0

# ---- fetch the datasets. era5_steer_int8.npz is ~300 MB, past GitHub's raw-fetch-friendly size --
# it must already be sitting in Drive (this repo cannot host it) or in /content from a direct
# upload. Checked in that order; nothing is downloaded from GitHub for this one file. ----
for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)

ERA5_PATH = None
for cand in ("/content/era5_steer_int8.npz", f"{DRIVE}/era5_steer_int8.npz"):
    if os.path.exists(cand):
        ERA5_PATH = cand; break
if ERA5_PATH is None:
    raise SystemExit(
        "era5_steer_int8.npz not found in /content or Drive's MyDrive/typhoon/ -- it is ~300 MB, "
        "too big for this repo's raw-fetch pattern. Upload it directly to /content (Colab's file "
        "pane) or copy it into MyDrive/typhoon/ and mount Drive first:\n"
        "  from google.colab import drive; drive.mount('/content/drive')")
print(f"era5 steering tensor: {ERA5_PATH}", flush=True)

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

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
SIX = int(6 * 3600 * 1e9)
_key = {}
for i in range(len(sid)):
    _key[(sid[i], int(bt[i]))] = i
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        j = _key.get((sid[i], int(bt[i]) - back * SIX), -1)
        HIST[i, c] = j
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

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

_v28src = urllib.request.urlopen(f"{RAW}/colab_v28_train.py").read().decode()
_hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", _v28src, re.S).group(0)
_tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", _v28src, re.S).group(0)
_g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(_hs, _g23); exec(_tf, _g23)
V23 = _g23["TrackFormerHist"]

# ---- ERA5 steering tensor: COMPACT (only the M populated rows), reconstruct a full-N lookup ----
_ed = np.load(ERA5_PATH)
ERA5_Q, ERA5_WIDX, ERA5_SCALE = _ed["q"], _ed["widx"], _ed["scale"].astype("float32")
ERA5_N = int(_ed["N"])
assert ERA5_N == len(sid), f"era5 tensor built against a different window space: {ERA5_N} vs {len(sid)}"
ERA5_POS = np.full(ERA5_N, -1, dtype="int64")
ERA5_POS[ERA5_WIDX] = np.arange(len(ERA5_WIDX))
print(f"era5 steering: {len(ERA5_WIDX):,}/{ERA5_N:,} windows covered "
      f"({100*len(ERA5_WIDX)/ERA5_N:.1f}%), {ERA5_Q.nbytes/1e6:.0f} MB in RAM", flush=True)


def era5_of(j):
    """[b,6,65,65] int8 raw patch + [b] float32 availability, for a batch of window indices j."""
    p = ERA5_POS[j]
    ok = (p >= 0).astype("float32")
    ps = np.where(p >= 0, p, 0)
    q = ERA5_Q[ps]
    q = np.where(ok[:, None, None, None] > 0, q, 0)     # unavailable -> exact zeros, never fabricated
    return q, ok


class Era5Stem(nn.Module):
    """Wraps v23's raw steer_cnn stack with a zero-init residual from the 0.25-deg ERA5 u/v
    field -- the SAME idiom HistStem uses for the temporal history, applied one level deeper so
    it does not disturb HistStem's own ctx wiring (see module docstring)."""

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
        self.ctx = None   # (era5_raw [b,6,65,65] int8-as-float, era5_ok [b])

    def forward(self, slp):
        st = self.base(slp)
        if USE_ERA5 and self.ctx is not None:
            era5_raw, ok = self.ctx
            x = era5_raw.float() * self.era5_scale.view(1, 6, 1, 1) * (ERA5_CLIP / 127.0)
            ok4 = ok.view(-1, 1, 1, 1)
            st = st + self.out(self.stem(x)) * ok4
        return st


class TrackFormerEra5(V23):
    """v23 with a second steering source: 0.25-deg ERA5 u/v, nested beneath the temporal-history
    wrap. forward() calls V23's (TrackFormerHist's) forward VERBATIM via super() -- the only new
    code is setting/clearing the era5 ctx on the newly-nested Era5Stem."""

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


# ---- init assertions: same rigor as v23/v26/v28 -- off reduces EXACTLY to v23, on moves the
# output, gradient is reachable, state-dict remaps clean, mirror is self-inverse. ----
with torch.no_grad():
    _j = np.arange(4)
    _t = torch.from_numpy(track[_j]).to(DEVICE); _v = torch.from_numpy(vpair[_j]).to(DEVICE)
    _s = torch.from_numpy(SLP[_j]).to(DEVICE)
    _hn = np.concatenate([SLP[HIST_S[_j, 0]], SLP[HIST_S[_j, 1]]], 1)
    _h = torch.from_numpy(_hn).to(DEVICE); _a = torch.from_numpy(HAVE[_j]).to(DEVICE)
    _eq, _eok = era5_of(_j)
    _e = torch.from_numpy(_eq).to(DEVICE); _eo = torch.from_numpy(_eok).to(DEVICE)

    _p, _q = V23().to(DEVICE).eval(), TrackFormerEra5().to(DEVICE).eval()
    torch.manual_seed(11)
    nn.init.normal_(_p.track_res.weight, std=0.02); nn.init.normal_(_p.track_res.bias, std=0.02)
    # Era5Stem WRAPS steer_cnn.base, so v23's "steer_cnn.base.*" keys become
    # "steer_cnn.base.base.*". Remap explicitly and verify nothing was silently dropped.
    _sd = {}
    for k, v in _p.state_dict().items():
        if k.startswith("steer_cnn.base."):
            k = "steer_cnn.base.base." + k[len("steer_cnn.base."):]
        _sd[k] = v
    _miss, _unexp = _q.load_state_dict(_sd, strict=False)
    assert not _unexp, f"unexpected keys loading v23 into v29: {list(_unexp)[:5]}"
    assert all(m.startswith("steer_cnn.base.stem") or m.startswith("steer_cnn.base.out")
               for m in _miss), f"v23 weights failed to transfer into v29: {list(_miss)[:5]}"
    assert float(_q.steer_cnn.base.out.weight.abs().max()) == 0.0, "era5 path is not zero-init"

    _d1 = float((_p(_t, _v, _s, _h, _a)[0] - _q(_t, _v, _s, _h, _a, _e, _eo)[0]).abs().max())
    assert _d1 < 1e-5, f"{TAG} does not reduce to v23 at init: {_d1}"
    nn.init.normal_(_q.steer_cnn.base.out.weight, std=0.05)
    _d2 = float((_p(_t, _v, _s, _h, _a)[0] - _q(_t, _v, _s, _h, _a, _e, _eo)[0]).abs().max())
    nn.init.zeros_(_q.steer_cnn.base.out.weight)
    assert not USE_ERA5 or _d2 > 1e-4, f"{TAG}: opening the era5 path moved the track by {_d2:.2e} -- DEAD"
    print(f"init check: era5 off max|v29 - v23| = {_d1:.2e} (v29 starts as exactly v23)")
    print(f"init check: era5 on  max|v29 - v23| = {_d2:.2e} (era5 path is live)", flush=True)

    # gradient reachability: does a loss on the track output actually reach Era5Stem's stem?
    _q.train()
    nn.init.normal_(_q.steer_cnn.base.out.weight, std=0.05)
    _mo, _, _ = _q(_t, _v, _s, _h, _a, _e, _eo)
    _mo.sum().backward()
    _g = _q.steer_cnn.base.stem[0].weight.grad
    assert _g is not None and float(_g.abs().max()) > 0, "era5 stem is not reachable by gradient"
    nn.init.zeros_(_q.steer_cnn.base.out.weight)
    _q.zero_grad(); _q.eval()
    print(f"init check: era5 stem gradient reachable, max|grad| = {float(_g.abs().max()):.2e}")
del _p, _q
print(f"\n{TAG} ready. USE_ERA5={USE_ERA5}, ERA5_DROP={ERA5_DROP}, {N_SEEDS} seeds.", flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(self, idx, aug):
        self.idx = np.asarray(idx); self.aug = aug

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        tr = torch.from_numpy(track[j]); tg = torch.from_numpy(target[j])
        mk = torch.from_numpy(mask[j]); sp = torch.from_numpy(SLP[j])
        vp = torch.from_numpy(vpair[j])
        fl = torch.from_numpy(FLOW_T[j].copy()); fm = torch.from_numpy(FLOW_M[j].copy())
        hs = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 0).copy())
        hv = torch.from_numpy(HAVE[j].copy())
        eq, eok = era5_of(np.array([j]))
        er = torch.from_numpy(eq[0]); eo = torch.tensor(float(eok[0]))
        if self.aug and torch.rand(()) < MIRROR_P:
            tr, tg, mk, sp = mirror(tr, tg, mk, sp)
            vp = vp.clone(); vp[1] = -vp[1]; vp[3] = -vp[3]
            fl = fl.clone(); fl[:, 1] = -fl[:, 1]
            hs = torch.flip(hs, dims=[1]).clone(); hs[3] = -hs[3]; hs[7] = -hs[7]
            # era5 mirror: flip the lat axis (dim 1 of [C,H,W]), negate the v-channels (3:6),
            # keep u-channels (0:3) sign unchanged -- same convention as hs/v500 above.
            er = torch.flip(er, dims=[1]).clone(); er[3:6] = -er[3:6]
        return tr, vp, sp, tg, mk, fl, fm, hs, hv, er, eo


def loader(idx, sh, aug=False):
    return torch.utils.data.DataLoader(DS(idx, aug), batch_size=BATCH, shuffle=sh, num_workers=2,
                                       pin_memory=True, persistent_workers=True, drop_last=sh)


def total_loss(s, ls, fp, tgt, m, fl, fm):
    base = G["total_loss"](s, ls, tgt, m)
    fmm = fm.unsqueeze(-1)
    flow = (F.smooth_l1_loss(fp, fl, reduction="none") * fmm).sum() / fmm.sum().clamp(min=1)
    return base + W_FLOW * flow, float(flow.detach())


def train_one(seed, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrackFormerEra5().to(DEVICE)
    print(f"seed {seed} | params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True, aug=True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0.0; fa = 0.0
        for tr, v0, sp, tg, m, fl, fm, hs, hv, er, eo in ld:
            tr, v0, sp, tg, m, fl, fm, hs, hv, er, eo = [x.to(DEVICE, non_blocking=True)
                for x in (tr, v0, sp, tg, m, fl, fm, hs, hv, er, eo)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls, fp = model(tr, v0, sp, hs, hv, er, eo)
                loss, fv = total_loss(s, ls, fp.float(), tg, m, fl, fm)
            if train:
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            tot += float(loss.detach()) * len(tr); fa += fv * len(tr); cnt += len(tr)
        return tot / cnt, fa / cnt

    best, bad, t0 = 1e9, 0, time.time()
    for ep in range(EPOCHS):
        te = time.time(); trl, trf = run(tl, True)
        with torch.no_grad():
            vv, vf = run(vl, False)
        sched.step()
        if vv < best:
            best, bad = vv, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "best_val": best,
                        "track_mean": G["tmean"], "track_std": G["tstd"]}, ckpt)
            if os.path.isdir(DRIVE):
                try:
                    import shutil as _sh; _sh.copy(ckpt, DRIVE)
                except Exception as ex:
                    print("Drive mirror failed:", ex)
        else:
            bad += 1
        with torch.no_grad():
            ew = float(model.steer_cnn.base.out.weight.abs().mean())
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"flow {vf:.3f} | era5W {ew:.5f} | {time.time()-te:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


# mirror must be self-inverse for era5 too
_er0, _ok0 = era5_of(np.array([0]))
_er0 = torch.from_numpy(_er0[0])
_er1 = torch.flip(_er0, dims=[1]).clone(); _er1[3:6] = -_er1[3:6]
_er2 = torch.flip(_er1, dims=[1]).clone(); _er2[3:6] = -_er2[3:6]
assert torch.equal(_er0, _er2), "era5 mirror is not an involution"
assert torch.equal(_er0[0].flip(0), _er1[0]), "u must not change sign under mirror"
assert torch.equal(-_er0[3].flip(0), _er1[3]), "v must change sign under mirror"
print("OK - era5 mirror matches the project convention and is self-inverse", flush=True)
del _er0, _er1, _er2

CK = []
for _s in range(N_SEEDS):
    _c = f"/content/{TAG}_seed{_s}.pt"
    if os.path.exists(_c):
        print(f"seed {_s}: checkpoint already present, reusing", flush=True)
    else:
        train_one(_s, _c)
    CK.append(_c)
print(f"{TAG} trained: {len(CK)} seeds (USE_ERA5={USE_ERA5})", flush=True)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
wp_only = np.array([i for i in wpep if basins[i] == "WP"])
SC = TARGET_SCALE


@torch.no_grad()
def track_err(ms, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        hs = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1)).to(DEVICE)
        hv = torch.from_numpy(HAVE[j]).to(DEVICE)
        eq, eok = era5_of(j)
        er = torch.from_numpy(eq).to(DEVICE); eo = torch.from_numpy(eok).to(DEVICE)
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE), hs, hv, er, eo]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().cpu().numpy())
    C = np.cumsum(np.concatenate(P)[..., :2], 1)
    T = np.cumsum(target[idx][..., :2], 1)
    return float(np.sqrt(((C - T) ** 2).sum(-1)).mean())


def load_m(c):
    m = TrackFormerEra5().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); return m


MS = [load_m(c) for c in CK]
print(f"\nWP+EP 2020+, {len(wpep)} windows ({len(wp_only)} WP-only, era5-covered {100*len(wp_only)/len(wpep):.0f}%)")
print("  BASELINE  v23 434.96 km (the bar -- same architecture, era5 branch off)")
for i, c in enumerate(CK):
    print(f"  {TAG} seed{i}  full {track_err([load_m(c)], wpep):.2f} km  "
          f"WP-only {track_err([load_m(c)], wp_only):.2f} km", flush=True)
e_full = track_err(MS, wpep)
e_wp = track_err(MS, wp_only)
print(f"\n  {TAG} ENSEMBLE ({len(MS)} seeds)  full WP+EP {e_full:.2f} km  |  WP-only {e_wp:.2f} km")
print(f"  vs v23 434.96 (full): {e_full - 434.96:+.2f} km", flush=True)
print("  NOTE: seed spread is ~19 km. WP-only is the number that isolates the effect where era5 "
      "\n  actually has data (91.4% coverage there vs 65.5% on the full WP+EP test set) -- EP "
      "\n  windows always see the branch at zero, diluting the full-population number toward v23.",
      flush=True)
json.dump({TAG: {"full": e_full, "wp_only": e_wp}, "use_era5": USE_ERA5, "n_seeds": len(MS)},
          open(f"/content/{TAG}.json", "w"))
try:
    from google.colab import files
    import subprocess
    subprocess.run(f"tar cf /content/{TAG}_seeds.tar /content/{TAG}_seed*.pt", shell=True)
    if os.path.isdir(DRIVE):
        import shutil as _sh; _sh.copy(f"/content/{TAG}_seeds.tar", DRIVE)
        print(f"checkpoints tarred to Drive: {DRIVE}/{TAG}_seeds.tar", flush=True)
    files.download(f"/content/{TAG}.json"); files.download(f"/content/{TAG}_seeds.tar")
except Exception as ex:
    print("download skipped:", ex)
