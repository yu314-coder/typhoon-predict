"""v27 on Colab -- v23's temporal steering stack AND v26's ocean-patch CNN in one model.

    !wget -q -O /content/v27.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/<REF>/colab_v33_train.py
    import os; os.environ["V33_SEEDS"]="10"; exec(open('/content/v27.py').read())

WHY THESE TWO, AND WHY ONLY NOW. Measured separately on the WP+EP 2020+ test set (10-seed
ensembles, each intensity channel masked with its own validity flag):

    metric        v21     v23     v26      winner
    track km    443.6   435.0   451.8      v23   (bootstrap-confirmed, -9.09 km, p=0.011)
    vmax kt     17.09   16.43   16.26      v26
    pressure    13.14   12.35   12.52      v23
    rmw nm      11.04   11.00   10.65      v26

Neither model dominates. The reason they can be combined at all is that they touch DISJOINT halves
of the network: v23 wraps steer_cnn, which feeds the TRACK decoder; v26 appends a token to the
INTENSITY decoder's memory, and an init assertion proved it leaves the track output bit-identical.
So the expectation is v23's track AND v26's wind, and if that is not what comes out, the assumption
that these two parts are independent was wrong -- which is itself worth knowing.

They were deliberately NOT combined earlier. v26 was built on v25 rather than v23 because stacking
two untested changes makes the result uninterpretable; that discipline is what let v23 be confirmed
and v24 be diagnosed. Each has now been measured alone, so the combination is the honest next step.

OCEAN DATA IS NOW GODAS, NOT AOML. v26's ocean CNN trained on 3.9% of windows because AOML starts
2013 while training is storms <= 2015. GODAS reanalysis runs 1980-present and lifts that to ~50%,
at the cost of monthly 1 deg x 1/3 deg resolution instead of daily 0.25 deg. Validated against AOML
on 15,510 shared windows: OHC r=0.80, D26 r=0.85, D20 r=0.81. One instrument on both sides of the
split, deliberately -- feeding AOML to the test set and GODAS to training would be the v24 mistake.

THREE SWITCHES, so every arm of the ablation is one variable:
    V33_USE_HIST=0   drops the temporal steering stack   -> the model is v25/v26 shaped
    V33_USE_ENV=0    drops the 43 environmental scalars
    V33_USE_OCEAN=0  drops the ocean patch token         -> the model is v23 shaped
All three off reproduces v21 EXACTLY, and HIST alone reproduces v23 EXACTLY. Both asserted at
max-diff 0 before a single training step.
"""

import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
REF = os.environ.get("V33_REF", "session/era5-env-v25")   # branch holding v32/v33 + ocean_patch
DRIVE = "/content/drive/MyDrive/typhoon"
N_SEEDS = int(os.environ.get("V33_SEEDS", "10"))
USE_HIST = int(os.environ.get("V33_USE_HIST", "1"))
USE_ENV = int(os.environ.get("V33_USE_ENV", "1"))
USE_OCEAN = int(os.environ.get("V33_USE_OCEAN", "1"))
W_FLOW = float(os.environ.get("W_FLOW", "0.3"))
_dflt = "v27" if (USE_HIST and USE_OCEAN) else ("v27noocean" if USE_HIST else "v27nohist")
TAG = os.environ.get("V33_TAG", _dflt)
KM6H = 6 * 3600 / 1000.0

for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)
for fn, ref in (("env_features.npz", "main"), ("ocean_patch.npz", REF)):
    if not os.path.exists(f"/content/{fn}"):
        src = f"{DRIVE}/{fn}"
        if os.path.exists(src):
            import shutil; shutil.copy(src, f"/content/{fn}"); print(f"copied {fn} from Drive", flush=True)
        else:
            base = RAW if ref == "main" else RAW.replace("/main", f"/{ref}")
            print(f"fetching {fn} from {ref} ...", flush=True)
            urllib.request.urlretrieve(f"{base}/track_build/{fn}", f"/content/{fn}")

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

_lf = np.load("/content/lead_flow.npz")
FLOW_T = _lf["flow"].astype("float32"); FLOW_M = _lf["got"].astype("float32")
DSC = np.load("/content/dlm4_int8.npz")["scale"][2:4].astype("float32")
_ii, _jj = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
_d = np.hypot(_ii, _jj) * 2.5
ANN = torch.tensor(((_d >= 3.0) & (_d <= 8.0)).astype("float32"), device=DEVICE)

# ---- history index maps (v23) -----------------------------------------------------------------
sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
SIX = int(6 * 3600 * 1e9)
_key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):                 # t-12 h and t-24 h
        HIST[i, c] = _key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])
print(f"history stack: t-12 h on {100*HAVE[:,0].mean():.1f}% of windows, "
      f"t-24 h on {100*HAVE[:,1].mean():.1f}%", flush=True)

# ---- environmental scalars (v25) ---------------------------------------------------------------
_E = np.load("/content/env_features.npz", allow_pickle=True)
EFEAT = _E["feat"].astype("float32"); EGOT = _E["got"].astype("float32")
NENV = EFEAT.shape[1]
_p = EGOT > 0
_mu = np.array([EFEAT[_p[:, c], c].mean() if _p[:, c].any() else 0.0 for c in range(NENV)], "float32")
_sd = np.array([EFEAT[_p[:, c], c].std() + 1e-6 if _p[:, c].any() else 1.0 for c in range(NENV)], "float32")
ENORM = ((EFEAT - _mu[None]) / _sd[None]) * EGOT

# ---- ocean patch (v26), now GODAS ---------------------------------------------------------------
_O = np.load("/content/ocean_patch.npz", allow_pickle=True)
OQ = _O["q"]; OSC = _O["scale"].astype("float32"); OGOT = _O["got"].astype("float32")
OSRC = str(_O["source"])
_s = (OQ[OGOT > 0][::17].astype("float32") * OSC[None, :, None, None])
OM = np.array([float(_s[:, c][_s[:, c] != 0].mean()) for c in range(3)], "float32")
OS = np.array([float(_s[:, c][_s[:, c] != 0].std()) + 1e-6 for c in range(3)], "float32")
del _s
_te_wp = np.array([i for i in te_idx if z["n_leads"].astype(int)[i] == 20 and basins[i] in ("WP", "EP")])
print(f"ocean source {OSRC} | train {100*(OGOT[tr_idx]>0).mean():.1f}%  "
      f"TEST WP+EP {100*(OGOT[_te_wp]>0).mean():.1f}%   (v26 trained on 3.9%)", flush=True)


def ocean_in(j):
    p = OQ[j].astype("float32") * OSC[None, :, None, None]
    valid = (p != 0).astype("float32")
    p = (p - OM[None, :, None, None]) / OS[None, :, None, None] * valid
    return np.concatenate([p, valid[:, :1]], 1)


# ---- v21 -> v23 -> +env -> +ocean, every class extracted verbatim -------------------------------
CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
_v26src = urllib.request.urlopen(f"{RAW}/colab_v26_train.py").read().decode()
_g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, _v26src, re.S).group(0), _g21)
V21 = _g21["TrackFormerCoT"]

_v28 = urllib.request.urlopen(f"{RAW}/colab_v28_train.py").read().decode()
_hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", _v28, re.S).group(0)
_tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", _v28, re.S).group(0)
_g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "USE_HIST": USE_HIST}
exec(_hs, _g23); exec(_tf, _g23)
V23 = _g23["TrackFormerHist"]

_v31 = urllib.request.urlopen(f"{RAW}/colab_v31_train.py").read().decode()
_ed = re.search(r"class _EnvDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", _v31, re.S).group(0)
_base = RAW.replace("/main", f"/{REF}")
_v32 = urllib.request.urlopen(f"{_base}/colab_v32_train.py").read().decode()
_oc = re.search(r"class OceanCNN\(nn\.Module\):.*?return self\.net\(x\)\n", _v32, re.S).group(0)
_od = re.search(r"class _OceanDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", _v32, re.S).group(0)
_gA = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": G["math"], "NENV": NENV,
       "USE_ENV": USE_ENV, "USE_OCEAN": USE_OCEAN}
exec(_ed, _gA); exec(_oc, _gA); exec(_od, _gA)
_EnvDec = _gA["_EnvDec"]; OceanCNN = _gA["OceanCNN"]; _OceanDec = _gA["_OceanDec"]


class TrackFormerAll(V23):
    """v23 (steer_cnn wrapped by HistStem) plus v25's env token plus v26's ocean token.

    Composition works because the three additions wrap DIFFERENT submodules and every one of them
    stashes its context on the parent rather than threading it through forward(). v23's forward is
    inherited verbatim, and it in turn inherits v21's verbatim. Nothing is rewritten by hand -- the
    failure that broke v15, v21 and the first draft of v23.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.track_q.shape[-1]
        self.env_mlp = nn.Sequential(nn.Linear(2 * NENV, 256), nn.GELU(), nn.Linear(256, d))
        self.env_pos = nn.Parameter(torch.zeros(1, 1, d))
        self.track_dec = _EnvDec(self.track_dec, self)
        self.int_dec = _EnvDec(self.int_dec, self)
        self.ocean_cnn = OceanCNN(d)
        self.ocean_pos = nn.Parameter(torch.zeros(1, 1, d))
        self.int_dec = _OceanDec(self.int_dec, self)     # intensity only
        self._envn = self._envg = self._op = self._og = None

    def _env_token(self):
        if not USE_ENV or self._envn is None:
            return None
        return self.env_mlp(torch.cat([self._envn, self._envg], -1)).unsqueeze(1) + self.env_pos

    def _ocean_token(self):
        if not USE_OCEAN or self._op is None:
            return None
        h = self.ocean_cnn(self._op) * self._og[:, None]
        return h.unsqueeze(1) + self.ocean_pos

    def forward(self, tr, vp, slp, hist=None, have=None, envn=None, envg=None, op=None, og=None):
        if envn is not None:
            self._envn, self._envg = envn, envg
        if op is not None:
            self._op, self._og = op, og
        try:
            return super().forward(tr, vp, slp, hist, have)
        finally:
            self._envn = self._envg = self._op = self._og = None


# ---- init assertions ---------------------------------------------------------------------------
def _probe(n=8):
    j = np.where((OGOT > 0) & (HAVE[:, 0] > 0) & (HAVE[:, 1] > 0) & (EGOT.sum(1) > 0))[0][:n]
    assert len(j) == n, "not enough windows carrying history AND env AND ocean to test on"
    h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1)).to(DEVICE)
    return (j, [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
                torch.from_numpy(SLP[j]).to(DEVICE)], h,
            torch.from_numpy(HAVE[j]).to(DEVICE),
            torch.from_numpy(ENORM[j]).to(DEVICE), torch.from_numpy(EGOT[j]).to(DEVICE),
            torch.from_numpy(ocean_in(j)).to(DEVICE), torch.from_numpy(OGOT[j]).to(DEVICE))


def _assert_init():
    global USE_HIST, USE_ENV, USE_OCEAN
    keep = (USE_HIST, USE_ENV, USE_OCEAN)
    j, a, h, hv, en, eg, op, og = _probe()
    torch.manual_seed(0); m21 = V21().to(DEVICE).eval()
    torch.manual_seed(0); m23 = V23().to(DEVICE).eval()
    torch.manual_seed(0); mA = TrackFormerAll().to(DEVICE).eval()

    def remap(sd, ndec):
        """each _EnvDec/_OceanDec wrap adds one 'dec.' level to that decoder's keys"""
        out = {}
        for k, v in sd.items():
            if k.startswith("int_dec."):
                out["int_dec." + "dec." * ndec[1] + k[len("int_dec."):]] = v
            elif k.startswith("track_dec."):
                out["track_dec." + "dec." * ndec[0] + k[len("track_dec."):]] = v
            else:
                out[k] = v
        return out

    miss, unexp = mA.load_state_dict(remap(m23.state_dict(), (1, 2)), strict=False)
    assert not unexp, f"unexpected keys mapping v23 -> v27: {unexp[:5]}"
    extra = {k.split(".")[0] for k in miss}
    assert extra <= {"env_mlp", "env_pos", "ocean_cnn", "ocean_pos"}, f"unexpected new modules: {sorted(extra)}"

    with torch.no_grad():
        USE_HIST, USE_ENV, USE_OCEAN = 0, 0, 0
        _g23["USE_HIST"] = 0
        ref21 = m21(*a); off = mA(*a, None, None, en, eg, op, og)
        d21 = max(float((x - y).abs().max()) for x, y in zip(ref21[:2], off[:2]))

        USE_HIST, _g23["USE_HIST"] = 1, 1
        ref23 = m23(*a, h, hv); onlyh = mA(*a, h, hv, en, eg, op, og)
        d23 = max(float((x - y).abs().max()) for x, y in zip(ref23[:2], onlyh[:2]))

        USE_ENV, USE_OCEAN = 1, 1
        full = mA(*a, h, hv, en, eg, op, og)
        d_int = float((ref23[0][..., 2:] - full[0][..., 2:]).abs().max())

        USE_OCEAN = 0
        noocean = mA(*a, h, hv, en, eg, op, og)
        USE_OCEAN = 1
        d_trk_ocean = float((noocean[0][..., :2] - full[0][..., :2]).abs().max())

    print(f"\ninit check | all switches OFF vs v21      max-diff {d21:.3e}  (must be 0)")
    print(f"           | HIST only     vs v23      max-diff {d23:.3e}  (must be 0)")
    print(f"           | all ON: intensity moves   max-diff {d_int:.3e}  (must be > 0)")
    print(f"           | ocean toggled: track      max-diff {d_trk_ocean:.3e}  (must be 0)")
    assert d21 == 0.0, "all-off does not reproduce v21"
    assert d23 == 0.0, "HIST-only does not reproduce v23 -- the combination is not additive"
    assert d_int > 0.0, "env+ocean do not move the intensity output"
    assert d_trk_ocean == 0.0, "the ocean token is leaking into the track output"
    USE_HIST, USE_ENV, USE_OCEAN = keep
    _g23["USE_HIST"] = keep[0]
    del m21, m23, mA


_assert_init()
print(f"\n{TAG}: HIST={USE_HIST} ENV={USE_ENV} OCEAN={USE_OCEAN}, {N_SEEDS} seeds, ocean={OSRC}", flush=True)


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
        en = torch.from_numpy(ENORM[j].copy()); eg = torch.from_numpy(EGOT[j].copy())
        op = torch.from_numpy(ocean_in(np.array([j]))[0]); og = torch.tensor(OGOT[j])
        h0 = torch.from_numpy(SLP[HIST_S[j, 0]]); h1 = torch.from_numpy(SLP[HIST_S[j, 1]])
        hv = torch.from_numpy(HAVE[j].copy())
        if self.aug and torch.rand(()) < MIRROR_P:
            tr, tg, mk, sp = mirror(tr, tg, mk, sp)
            vp = vp.clone(); vp[1] = -vp[1]; vp[3] = -vp[3]
            fl = fl.clone(); fl[:, 1] = -fl[:, 1]
            # the history patches are the same kind of field as sp, so they take the same mirror;
            # v17's mirror() flips dim 1 and negates v500 (channel 3) -- applied per patch here
            h0 = torch.flip(h0, dims=[1]).clone(); h0[3] = -h0[3]
            h1 = torch.flip(h1, dims=[1]).clone(); h1[3] = -h1[3]
            op = torch.flip(op, dims=[-2])
        return tr, vp, sp, tg, mk, fl, fm, en, eg, op, og, torch.cat([h0, h1], 0), hv


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
    model = TrackFormerAll().to(DEVICE)
    print(f"seed {seed} | params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True, aug=True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0.0
        for tr, v0, sp, tg, m, fl, fm, en, eg, op, og, hh, hv in ld:
            tr, v0, sp, tg, m, fl, fm, en, eg, op, og, hh, hv = [
                x.to(DEVICE, non_blocking=True)
                for x in (tr, v0, sp, tg, m, fl, fm, en, eg, op, og, hh, hv)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls, fp = model(tr, v0, sp, hh, hv, en, eg, op, og)
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
            # /content is wiped when the runtime idles out -- that is exactly how v26 lost all 20
            # checkpoints AFTER producing its results. Mirror on every improvement.
            if os.path.isdir(DRIVE):
                try:
                    import shutil as _sh; _sh.copy(ckpt, DRIVE)
                except Exception as _e:
                    print(f"  (drive mirror failed: {_e})", flush=True)
        else:
            bad += 1
        with torch.no_grad():
            ow = float(model.ocean_cnn.net[-1].weight.abs().mean()) if USE_OCEAN else 0.0
            ew = float(model.env_mlp[-1].weight.abs().mean()) if USE_ENV else 0.0
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"oceanW {ow:.5f} | envW {ew:.5f} | {time.time()-te:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


CK = []
for _s in range(N_SEEDS):
    _c = f"/content/{TAG}_seed{_s}.pt"
    if os.path.exists(_c):
        print(f"seed {_s}: checkpoint already present, reusing", flush=True)
    else:
        train_one(_s, _c)
    CK.append(_c)
print(f"{TAG} trained: {len(CK)} seeds", flush=True)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
ocn = np.array([i for i in wpep if OGOT[i] > 0])
SC = TARGET_SCALE


@torch.no_grad()
def metrics(ms, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1)).to(DEVICE)
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE), h, torch.from_numpy(HAVE[j]).to(DEVICE),
             torch.from_numpy(ENORM[j]).to(DEVICE), torch.from_numpy(EGOT[j]).to(DEVICE),
             torch.from_numpy(ocean_in(j)).to(DEVICE), torch.from_numpy(OGOT[j]).to(DEVICE)]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().cpu().numpy())
    O = np.concatenate(P); T = target[idx]; K = mask[idx] > 0
    C = np.cumsum(O[..., :2], 1); TC = np.cumsum(T[..., :2], 1)
    out = {"track": float(np.sqrt(((C - TC) ** 2).sum(-1)).mean())}
    for ci, nm in ((2, "vmax"), (3, "pres"), (4, "rmw")):
        m_ = K[..., ci]
        out[nm] = float(np.abs(O[..., ci] - T[..., ci])[m_].mean()) if m_.any() else float("nan")
    m_ = K[..., 5:17]
    out["radii"] = float(np.abs(O[..., 5:17] - T[..., 5:17])[m_].mean()) if m_.any() else float("nan")
    return out


def load_m(c):
    m = TrackFormerAll().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); return m


MS = [load_m(c) for c in CK]
print(f"\nWP+EP 2020+: {len(wpep)} windows, {len(ocn)} with an ocean patch ({100*len(ocn)/len(wpep):.0f}%)")
print("  BASELINES   track km: v21 443.6 | v23 435.0 | v26 451.8")
print("              vmax kt:  v21 17.09 | v23 16.43 | v26 16.26")
print("              pres hPa: v21 13.14 | v23 12.35 | v26 12.52")
A = metrics(MS, wpep); B = metrics(MS, ocn)
print(f"\n  {TAG} ENSEMBLE ({len(MS)} seeds)")
for lab, r, n in (("all WP+EP", A, len(wpep)), ("ocean-covered", B, len(ocn))):
    print(f"    {lab:14s} ({n:4d}): track {r['track']:7.2f} km | vmax {r['vmax']:5.2f} kt | "
          f"pres {r['pres']:5.2f} hPa | rmw {r['rmw']:5.2f} nm | radii {r['radii']:5.2f} nm")
print("  The claim being tested: v23's TRACK (435.0) together with v26's WIND (16.26). If track")
print("  regresses toward 451 or wind toward 17.1, the two parts are not independent after all.",
      flush=True)
json.dump({TAG: {"all": A, "ocean": B}, "use_hist": USE_HIST, "use_env": USE_ENV,
           "use_ocean": USE_OCEAN, "n_seeds": len(MS), "n_ocean": int(len(ocn)),
           "ocean_source": OSRC}, open(f"/content/{TAG}.json", "w"))
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
