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
    # HistStem WRAPS the original stem, so v21's "steer_cnn.*" keys become "steer_cnn.base.*".
    # Loading with strict=False and no remap silently skips every steering-CNN weight, leaving _q
    # with a randomly initialised stem -- which is exactly what the init assertion caught
    # (0.0284 instead of 0). Remap, then verify nothing was silently dropped.
    _sd = {("steer_cnn.base." + k[len("steer_cnn."):]) if k.startswith("steer_cnn.") else k: v
           for k, v in _p.state_dict().items()}
    _miss, _unexp = _q.load_state_dict(_sd, strict=False)
    assert not _unexp, f"unexpected keys when loading v21 into v23: {list(_unexp)[:5]}"
    assert all(m.startswith("steer_cnn.stem") or m.startswith("steer_cnn.out") for m in _miss), \
        f"v21 weights failed to transfer into v23: {[m for m in _miss][:5]}"
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

# ---- dataset: v26's, plus the two past steering fields -----------------------------------------
# Cannot reuse v26's DS verbatim because the history has to ride along with each sample. The mirror
# handling is copied exactly: a N-S flip negates northward flow, and the history fields flip the
# same way the present field does. Getting that wrong would train on physically inconsistent pairs.
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
        if self.aug and torch.rand(()) < MIRROR_P:
            tr, tg, mk, sp = mirror(tr, tg, mk, sp)
            vp = vp.clone(); vp[1] = -vp[1]; vp[3] = -vp[3]
            fl = fl.clone(); fl[:, 1] = -fl[:, 1]
            # mirror each past field the same way mirror() does the present one: flip the lat axis
            # and negate the northward wind channel of each 4-channel block
            hs = torch.flip(hs, dims=[1]).clone()
            hs[3] = -hs[3]; hs[7] = -hs[7]
        return tr, vp, sp, tg, mk, fl, fm, hs, hv


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
    model = TrackFormerHist().to(DEVICE)
    print(f"seed {seed} | params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True, aug=True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0.0; fa = 0.0
        for tr, v0, sp, tg, m, fl, fm, hs, hv in ld:
            tr, v0, sp, tg, m, fl, fm, hs, hv = [x.to(DEVICE, non_blocking=True)
                                                 for x in (tr, v0, sp, tg, m, fl, fm, hs, hv)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls, fp = model(tr, v0, sp, hs, hv)
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
        else:
            bad += 1
        with torch.no_grad():
            # |W| of the history output conv is the collapse diagnostic: if it stays at zero the
            # temporal stack is being ignored and v23 has silently reverted to v21.
            hw = float(model.steer_cnn.out.weight.abs().mean())
            amag = model.A.detach().cpu().numpy()
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"flow {vf:.3f} | A {amag[0]:.2f},{amag[1]:.2f} | histW {hw:.5f} | "
              f"{time.time()-te:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


# the history mirror must be self-inverse, exactly as v17 asserts for the present field
_h0 = torch.from_numpy(np.concatenate([SLP[HIST_S[0, 0]], SLP[HIST_S[0, 1]]], 0).copy())
_h1 = torch.flip(_h0, dims=[1]).clone(); _h1[3] = -_h1[3]; _h1[7] = -_h1[7]
_h2 = torch.flip(_h1, dims=[1]).clone(); _h2[3] = -_h2[3]; _h2[7] = -_h2[7]
assert torch.allclose(_h0, _h2, atol=1e-5), "history mirror is not an involution"
assert torch.allclose(_h0[2].flip(0), _h1[2], atol=1e-5), "u500 must not change sign under mirror"
assert torch.allclose(-_h0[3].flip(0), _h1[3], atol=1e-5), "v500 must change sign under mirror"
print("OK - history mirror matches v17's convention and is self-inverse", flush=True)
del _h0, _h1, _h2

CK = []
for _s in range(N_SEEDS):
    _c = f"/content/{TAG}_seed{_s}.pt"
    if os.path.exists(_c):
        print(f"seed {_s}: checkpoint already present, reusing", flush=True)
    else:
        train_one(_s, _c)
    CK.append(_c)
print(f"{TAG} trained: {len(CK)} seeds (USE_HIST={USE_HIST})", flush=True)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
SC = TARGET_SCALE


@torch.no_grad()
def track_err(ms):
    P = []
    for i in range(0, len(wpep), 128):
        j = wpep[i:i + 128]
        hs = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1)).to(DEVICE)
        hv = torch.from_numpy(HAVE[j]).to(DEVICE)
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE), hs, hv]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().cpu().numpy())
    C = np.cumsum(np.concatenate(P)[..., :2], 1)
    T = np.cumsum(target[wpep][..., :2], 1)
    return float(np.sqrt(((C - T) ** 2).sum(-1)).mean())


def load_m(c):
    m = TrackFormerHist().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); return m


MS = [load_m(c) for c in CK]
print(f"\nWP+EP 2020+, {len(wpep)} windows")
print("  BASELINES   v17 462.8 | v20 452.5 | v21 443.6 | v22 443.4 (the bar)")
for i, c in enumerate(CK):
    print(f"  {TAG} seed{i}  {track_err([load_m(c)]):.2f} km", flush=True)
e = track_err(MS)
print(f"\n  {TAG} ENSEMBLE ({len(MS)} seeds)  {e:.2f} km")
print(f"  vs v21 443.62: {e - 443.62:+.2f} km   vs v22 443.41: {e - 443.41:+.2f} km", flush=True)
print("  NOTE: seed spread is ~19 km and the v21/v22 bootstrap CIs cross zero. Anything inside"
      "\n  ~10 km of the bar is noise until a paired-storm bootstrap says otherwise.", flush=True)
json.dump({TAG: e, "use_hist": USE_HIST, "n_seeds": len(MS)},
          open(f"/content/{TAG}.json", "w"))
try:
    from google.colab import files
    import subprocess
    subprocess.run(f"tar cf /content/{TAG}_seeds.tar /content/{TAG}_seed*.pt", shell=True)
    files.download(f"/content/{TAG}.json"); files.download(f"/content/{TAG}_seeds.tar")
except Exception as ex:
    print("download skipped:", ex)
