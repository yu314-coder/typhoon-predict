"""v28 on Colab -- v23 plus ONE change: an N-only, hemisphere-equivariant meridional drift adapter.

    !wget -q -O /content/v28.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/<REF>/colab_v34_train.py
    import os; os.environ["V34_SEEDS"]="10"; exec(open('/content/v28.py').read())

MOTIVATION AND THE HONEST CAVEAT. v23's meridional (north-south) steering-flow prediction is its
weak axis (correlation 0.46 vs 0.81 zonal), and non-recurving storms are systematically not carried
far enough poleward. v28 adds a small learned poleward-drift correction to the NORTH displacement
channel only -- an analogue of beta drift, the vortex self-propagation that is independent of the
steering flow.

  *** A pre-registered diagnostic (_nbias.py) found the N error is NOISE-dominated, not biased:
      bias/noise 0.105 on validation, and an ORACLE debias is worth <1 km against seed noise of
      3.8-7.3 km. So the expected outcome is NULL. This experiment is run to CONFIRM that cleanly,
      with a proper matched ablation, not because a gain is expected. ***

ONE VARIABLE ONLY. v28 changes nothing else. V34_USE_DRIFT=0 reproduces v23 EXACTLY (max-diff 0 on
all 17 output channels), asserted below before any training. The drift touches ONLY channel 1 (N);
channel 0 (E) and channels 2-16 (intensity/radii) are bit-identical.

HEMISPHERE EQUIVARIANCE. The correction is  DeltaN += sign(lat) * A_max * sigmoid(q_l(z0)),  where
sign(lat)=tanh(lat/5deg) and z0 is built ONLY from mirror-invariant magnitudes (|lat|, vmax, speed,
their trends). Under the equatorial mirror (lat -> -lat) the magnitude q is unchanged and the sign
flips, so the N correction negates and E stays zero -- consistent with the project's augmentation.

CAPACITY. ~350 parameters. Four lead-knots (6,24,72,120 h) linearly interpolated to 20 leads, so
the correction is smooth in lead and cannot fit high-frequency noise. A_max is a per-6h-step cap of
~65 km (=0.65 in channel units), the 1-3 m/s beta-drift scale; the init is deliberately tiny so the
model earns any correction rather than starting with one.
"""
import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
DRIVE = "/content/drive/MyDrive/typhoon"
N_SEEDS = int(os.environ.get("V34_SEEDS", "10"))
USE_DRIFT = int(os.environ.get("V34_USE_DRIFT", "1"))
A_MAX = float(os.environ.get("V34_AMAX", "0.65"))          # per-step cap in channel units (65 km)
W_FLOW = float(os.environ.get("W_FLOW", "0.3"))
TAG = os.environ.get("V34_TAG", "v28" if USE_DRIFT else "v28abl")
KM6H = 6 * 3600 / 1000.0

for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)

nb = json.load(open(urllib.request.urlretrieve(f"{RAW}/colab_train_v17.ipynb", "/content/_v17.ipynb")[0]))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
assert body.count("steer5_int8.npz") == 1
body = body.replace('"/content/d/steer5_int8.npz"', '"/content/dlm4_int8.npz"')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os, "json": json,
     "time": time, "math": __import__("math")}
exec(compile(body, "<v17-notebook>", "exec"), G)

DEVICE = G["DEVICE"]; TARGET_SCALE = G["TARGET_SCALE"]
Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
mask = G["mask"]; vpair = G["vpair"]; z = G["z"]; mirror = G["mirror"]
tr_idx, va_idx, te_idx = G["tr_idx"], G["va_idx"], G["te_idx"]
basins = G["basins"]; tmean = G["tmean"]; tstd = G["tstd"]
EPOCHS, PATIENCE, BATCH = G["EPOCHS"], G["PATIENCE"], G["BATCH"]
LR, WEIGHT_DECAY, MIRROR_P = G["LR"], G["WEIGHT_DECAY"], G["MIRROR_P"]
TM = torch.tensor(tmean, device=DEVICE); TS = torch.tensor(tstd, device=DEVICE)

_lf = np.load("/content/lead_flow.npz")
FLOW_T = _lf["flow"].astype("float32"); FLOW_M = _lf["got"].astype("float32")
DSC = np.load("/content/dlm4_int8.npz")["scale"][2:4].astype("float32")
_ii, _jj = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
_d = np.hypot(_ii, _jj) * 2.5
ANN = torch.tensor(((_d >= 3.0) & (_d <= 8.0)).astype("float32"), device=DEVICE)

# history maps for v23's temporal steering stack
sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
SIX = int(6 * 3600 * 1e9)
_key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = _key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
_v26 = urllib.request.urlopen(f"{RAW}/colab_v26_train.py").read().decode()
_g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, _v26, re.S).group(0), _g21)
V21 = _g21["TrackFormerCoT"]
_v28 = urllib.request.urlopen(f"{RAW}/colab_v28_train.py").read().decode()
_hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", _v28, re.S).group(0)
_tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", _v28, re.S).group(0)
_g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(_hs, _g23); exec(_tf, _g23)
V23 = _g23["TrackFormerHist"]


# ---- the one new component: N-only meridional drift adapter ----------------------------------
class MeridionalDrift(nn.Module):
    """z0 (5 mirror-invariant magnitudes) -> 4 lead-knots -> interp to 20 -> sigmoid -> signed push.

    Only the NORTH channel is ever touched, and only via the odd sign factor tanh(lat/5), so the
    correction is exactly antisymmetric across the equator. Zero-weight, deep-negative-bias init
    makes the correction ~1 km at start: the model must earn any drift.
    """
    def __init__(self, leads=20, knots=4, a_max=A_MAX):
        super().__init__()
        self.a_max = a_max
        self.mlp = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, knots))
        kl = torch.tensor([0., 3., 11., 19.])                 # knots at 6,24,72,120 h (0-indexed)
        Wi = torch.zeros(leads, knots)
        for i in range(leads):
            L = float(i)
            if L <= kl[0]:
                Wi[i, 0] = 1.0
            elif L >= kl[-1]:
                Wi[i, -1] = 1.0
            else:
                for k in range(knots - 1):
                    if kl[k] <= L <= kl[k + 1]:
                        f = (L - kl[k]) / (kl[k + 1] - kl[k])
                        Wi[i, k] = 1 - f; Wi[i, k + 1] = f
                        break
        self.register_buffer("Wi", Wi)                        # [20,4] linear interpolation
        # tiny (not zero) final weight: a hard zero would block the gradient to the first layer
        # (backprop through a zero weight is zero), so the branch could not learn off its own start.
        # A deep-negative bias keeps the init correction ~0.06 km -- the model earns any drift.
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.constant_(self.mlp[-1].bias, -7.0)            # sigmoid(-7) ~ 9e-4 -> ~0.06 km at init

    def forward(self, z0, lat_deg):
        knots = self.mlp(z0)                                  # [b,4]
        mag = torch.sigmoid(knots @ self.Wi.t())             # [b,20] in (0,1)
        sign = torch.tanh(lat_deg / 5.0)                     # [b]  odd across the equator
        return sign[:, None] * self.a_max * mag              # [b,20] channel-unit N correction


class TrackFormerDrift(V23):
    """v23 with a poleward-drift correction added to the North displacement channel only.

    Inherits v23's forward VERBATIM (which inherits v21's): calls super().forward, then adds the
    drift to channel 1. No parent code is rewritten -- the failure that broke v15/v21/v23-first-draft.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.drift = MeridionalDrift()

    def _z0(self, tr):
        last, prev = tr[:, -1], tr[:, -2]
        lat = last[:, 48] * TS[48] + TM[48]                  # signed latitude, deg
        vmax = last[:, 4] * TS[4] + TM[4]
        spd = last[:, 42] * TS[42] + TM[42]
        vmax_p = prev[:, 4] * TS[4] + TM[4]
        spd_p = prev[:, 42] * TS[42] + TM[42]
        z0 = torch.stack([lat.abs() / 30.0, vmax / 60.0, spd / 100.0,
                          (spd - spd_p) / 100.0, (vmax - vmax_p) / 60.0], -1)   # all mirror-invariant
        return z0, lat

    def forward(self, tr, vp, slp, hist=None, have=None):
        s, ls, fp = super().forward(tr, vp, slp, hist, have)
        if USE_DRIFT:
            z0, lat = self._z0(tr)
            s = s.clone()
            s[..., 1] = s[..., 1] + self.drift(z0, lat)      # NORTH channel only
        return s, ls, fp


# ---- init assertions: the experiment is only clean if these pass -----------------------------
def _assert_init():
    global USE_DRIFT
    keep = USE_DRIFT
    torch.manual_seed(0); m23 = V23().to(DEVICE).eval()
    torch.manual_seed(0); m28 = TrackFormerDrift().to(DEVICE).eval()
    miss, unexp = m28.load_state_dict(m23.state_dict(), strict=False)
    assert not unexp, f"unexpected keys loading v23 -> v28: {unexp[:5]}"
    assert {k.split('.')[0] for k in miss} == {"drift"}, f"v28 adds more than the drift: {sorted(set(k.split('.')[0] for k in miss))}"
    j = np.arange(16)
    h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1)).to(DEVICE)
    a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
         torch.from_numpy(SLP[j]).to(DEVICE), h, torch.from_numpy(HAVE[j]).to(DEVICE)]
    with torch.no_grad():
        ref = m23(*a)
        USE_DRIFT = 0
        off = m28(*a)
        USE_DRIFT = 1
        on = m28(*a)
    d_off = max(float((x - y).abs().max()) for x, y in zip(ref[:2], off[:2]))
    d_E = float((ref[0][..., 0] - on[0][..., 0]).abs().max())
    d_int = float((ref[0][..., 2:] - on[0][..., 2:]).abs().max())
    d_N = float((ref[0][..., 1] - on[0][..., 1]).abs().max())
    # mirror equivariance of the adapter: negating latitude negates the correction
    with torch.no_grad():
        z0, lat = m28._z0(torch.from_numpy(track[j]).to(DEVICE))
        dpos = m28.drift(z0, lat); dneg = m28.drift(z0, -lat)
    d_mirror = float((dpos + dneg).abs().max())
    # gradient reachability
    m28.train()
    s, ls, fp = m28(*a)
    s[..., 1].sum().backward()
    g = m28.drift.mlp[0].weight.grad
    grad_ok = g is not None and float(g.abs().sum()) > 0
    print(f"\ninit check | USE_DRIFT=0 vs v23         max-diff {d_off:.3e}  (must be 0)")
    print(f"           | USE_DRIFT=1 North moves     max-diff {d_N:.3e}  (must be > 0, small)")
    print(f"           | East channel untouched      max-diff {d_E:.3e}  (must be 0)")
    print(f"           | intensity ch 2-16 untouched max-diff {d_int:.3e}  (must be 0)")
    print(f"           | mirror equivariance |dpos+dneg| {d_mirror:.3e}  (must be 0)")
    print(f"           | drift gradient reachable        {grad_ok}")
    assert d_off == 0.0 and d_E == 0.0 and d_int == 0.0, "v28 disturbs a channel it must not"
    assert d_N > 0.0, "the drift does not move the North output"
    assert d_mirror == 0.0, "the drift is not hemisphere-antisymmetric"
    assert grad_ok, "the drift receives no gradient"
    USE_DRIFT = keep
    del m23, m28


_assert_init()
print(f"\n{TAG}: USE_DRIFT={USE_DRIFT}, A_max={A_MAX}, {N_SEEDS} seeds", flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(self, idx, aug):
        self.idx = np.asarray(idx); self.aug = aug

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        tr = torch.from_numpy(track[j]); tg = torch.from_numpy(target[j])
        mk = torch.from_numpy(mask[j]); sp = torch.from_numpy(SLP[j]); vp = torch.from_numpy(vpair[j])
        fl = torch.from_numpy(FLOW_T[j].copy()); fm = torch.from_numpy(FLOW_M[j].copy())
        h0 = torch.from_numpy(SLP[HIST_S[j, 0]]); h1 = torch.from_numpy(SLP[HIST_S[j, 1]])
        hv = torch.from_numpy(HAVE[j].copy())
        if self.aug and torch.rand(()) < MIRROR_P:
            tr, tg, mk, sp = mirror(tr, tg, mk, sp)
            vp = vp.clone(); vp[1] = -vp[1]; vp[3] = -vp[3]
            fl = fl.clone(); fl[:, 1] = -fl[:, 1]
            h0 = torch.flip(h0, dims=[1]).clone(); h0[3] = -h0[3]
            h1 = torch.flip(h1, dims=[1]).clone(); h1[3] = -h1[3]
        return tr, vp, sp, tg, mk, fl, fm, torch.cat([h0, h1], 0), hv


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
    model = TrackFormerDrift().to(DEVICE)
    print(f"seed {seed} | params {sum(p.numel() for p in model.parameters()):,} "
          f"(drift {sum(p.numel() for p in model.drift.parameters())})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True, aug=True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0.0
        for tr, v0, sp, tg, m, fl, fm, hh, hv in ld:
            tr, v0, sp, tg, m, fl, fm, hh, hv = [x.to(DEVICE, non_blocking=True)
                for x in (tr, v0, sp, tg, m, fl, fm, hh, hv)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls, fp = model(tr, v0, sp, hh, hv)
                loss, _ = total_loss(s, ls, fp.float(), tg, m, fl, fm)
            if train:
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            tot += float(loss.detach()) * len(tr); cnt += len(tr)
        return tot / cnt

    best, bad, t0 = 1e9, 0, time.time()
    for ep in range(EPOCHS):
        te = time.time(); trl = run(tl, True)
        with torch.no_grad():
            vv = run(vl, False)
        sched.step()
        if vv < best:
            best, bad = vv, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "best_val": best,
                        "track_mean": tmean, "track_std": tstd}, ckpt)
            if os.path.isdir(DRIVE):
                try:
                    import shutil as _sh; _sh.copy(ckpt, DRIVE)
                except Exception as _e:
                    print(f"  (drive mirror failed: {_e})", flush=True)
        else:
            bad += 1
        with torch.no_grad():
            dw = float(model.drift.mlp[-1].weight.abs().mean()) if USE_DRIFT else 0.0
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"driftW {dw:.5f} | {time.time()-te:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


CK = []
for _s in range(N_SEEDS):
    _c = f"/content/{TAG}_seed{_s}.pt"
    if os.path.exists(_c):
        print(f"seed {_s}: present, reusing", flush=True)
    else:
        train_one(_s, _c)
    CK.append(_c)
print(f"{TAG} trained: {len(CK)} seeds", flush=True)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
SC = TARGET_SCALE


@torch.no_grad()
def metrics(ms, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1)).to(DEVICE)
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE), h, torch.from_numpy(HAVE[j]).to(DEVICE)]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().cpu().numpy())
    O = np.concatenate(P); T = target[idx]; K = mask[idx] > 0
    C = np.cumsum(O[..., :2], 1); TC = np.cumsum(T[..., :2], 1)
    out = {"track": float(np.sqrt(((C - TC) ** 2).sum(-1)).mean())}
    # north-error decomposition, the metric v28 is meant to move
    out["meanN"] = float((C[..., 1] - TC[..., 1]).mean())
    out["mN120"] = float((C[:, 19, 1] - TC[:, 19, 1]).mean())
    for ci, nm in ((2, "vmax"), (3, "pres")):
        m_ = K[..., ci]
        out[nm] = float(np.abs(O[..., ci] - T[..., ci])[m_].mean()) if m_.any() else float("nan")
    return out


def load_m(c):
    m = TrackFormerDrift().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); return m


MS = [load_m(c) for c in CK]
A = metrics(MS, wpep)
print(f"\nWP+EP 2020+: {len(wpep)} windows")
print("  BASELINE  v23 track 434.96 | meanN-120h -18.6 (test)")
print(f"\n  {TAG} ENSEMBLE ({len(MS)} seeds)")
print(f"    track {A['track']:.2f} km | vmax {A['vmax']:.2f} kt | pres {A['pres']:.2f} hPa "
      f"| meanN {A['meanN']:+.1f} | N@120h {A['mN120']:+.1f} km")
print("  The claim: v28 should reduce the poleward N under-carry without moving E/intensity.")
print("  Expected per the pre-registered bias check: ~null (the N error is variance, not bias).",
      flush=True)
json.dump({TAG: A, "use_drift": USE_DRIFT, "a_max": A_MAX, "n_seeds": len(MS)},
          open(f"/content/{TAG}.json", "w"))
try:
    from google.colab import files
    import subprocess
    subprocess.run(f"tar cf /content/{TAG}_seeds.tar /content/{TAG}_seed*.pt", shell=True)
    if os.path.isdir(DRIVE):
        import shutil as _sh; _sh.copy(f"/content/{TAG}_seeds.tar", DRIVE)
        print(f"checkpoints tarred to Drive: {DRIVE}/{TAG}_seeds.tar", flush=True)
    files.download(f"/content/{TAG}.json")
except Exception as ex:
    print("download skipped:", ex)
