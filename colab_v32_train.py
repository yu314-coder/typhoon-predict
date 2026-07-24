"""v26 on Colab — v25 plus an ocean-heat PATCH CNN wired into the INTENSITY decoder only.

    from google.colab import drive; drive.mount('/content/drive')
    !wget -q -O /content/v26.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/<SHA>/colab_v32_train.py
    import os; os.environ["V32_SEEDS"]="10"; exec(open('/content/v26.py').read())

WHAT THIS ADDS OVER v25. v25 fed 43 environmental SCALARS (including ocean heat reduced to a centre
value and a patch mean) as one token into BOTH decoders. Measured: central pressure -0.46 hPa, vmax
+0.21 kt, track +8.2 km against v21. The pressure gain is real but small, and two summary numbers
throw away the structure of the ocean the storm is about to cross -- a warm eddy ahead of the track
and a cold wake behind it reduce to the same mean.

v26 keeps everything v25 has and ADDS a small CNN over the raw 21x21 ocean-heat patch (OHC, D26,
D20 -- fuel available, and how deep it goes), whose output is appended as one more token to the
INTENSITY decoder's memory ONLY. The track decoder is untouched: ocean heat governs how strong a
storm gets, not where it goes, and v25 already showed that pushing it at the track costs km.

ADDITIVE, NOT SUBSTITUTIVE -- the v24 lesson, again. V32_USE_OCEAN=0 does not append the token,
which reproduces v25 EXACTLY. That is asserted below at max-diff 0 before a single step of training,
together with the converse (USE_OCEAN=1 must MOVE the intensity output). v26 vs v26abl therefore
isolates the ocean patch with no other variable moving.

*** THE HONEST CONSTRAINT: TRAINING COVERAGE. ***
The AOML ocean archive starts in 2013 and was extracted over the WP+EP box only, so:
      test  (WP+EP 2020+)   95.6% of windows carry an ocean patch
      train (all, pre-2020)  8.2%
The CNN therefore gets a real gradient on roughly one window in twelve, while at evaluation it is
asked to speak on nearly every one. This is a data-coverage problem, not an architecture problem,
and it CAPS what v26 can show. It is stated here rather than discovered afterwards. The 43-dim /
per-cell presence masks are fed alongside the values so "no ocean" is a learned state rather than a
lie about a cold ocean; that is the same first-class-mask convention the rest of the project uses.
The real fix is extending the ocean extraction backwards in time, which is a download, not a model.

NO OVERSAMPLING, DELIBERATELY. Upweighting ocean-covered windows would train the branch harder but
would move the baseline too, and v26abl would stop reproducing v25/v21. One variable at a time is
the discipline that made v23 interpretable and v24 diagnosable. V32_OSAMPLE exists for a follow-up
run and is OFF by default; when used it must be applied to BOTH arms.

LEAKAGE. The ocean field is the same-day analysis at t0 and carries no storm identity, no absolute
date and no forward best-track information. Land/ice cells come back as exact zeros behind the
validity channel, never as fabricated temperatures.
"""

import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
DRIVE = "/content/drive/MyDrive/typhoon"
N_SEEDS = int(os.environ.get("V32_SEEDS", "10"))
USE_OCEAN = int(os.environ.get("V32_USE_OCEAN", "1"))
OSAMPLE = float(os.environ.get("V32_OSAMPLE", "0"))     # 0 = off; see docstring
W_FLOW = float(os.environ.get("W_FLOW", "0.3"))
TAG = os.environ.get("V32_TAG", "v26" if USE_OCEAN else "v26abl")
KM6H = 6 * 3600 / 1000.0

for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)
for fn in ("env_features.npz",):
    if not os.path.exists(f"/content/{fn}"):
        src = f"{DRIVE}/{fn}"
        if os.path.exists(src):
            import shutil; shutil.copy(src, f"/content/{fn}"); print(f"copied {fn} from Drive", flush=True)
        else:
            print(f"fetching {fn} from repo ...", flush=True)
            urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", f"/content/{fn}")

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

# ---- v25's environmental scalars (kept verbatim) ---------------------------------------------
_E = np.load("/content/env_features.npz", allow_pickle=True)
EFEAT = _E["feat"].astype("float32"); EGOT = _E["got"].astype("float32")
NENV = EFEAT.shape[1]
_present = EGOT > 0
_mu = np.array([EFEAT[_present[:, c], c].mean() if _present[:, c].any() else 0.0 for c in range(NENV)], "float32")
_sd = np.array([EFEAT[_present[:, c], c].std() + 1e-6 if _present[:, c].any() else 1.0 for c in range(NENV)], "float32")
ENORM = ((EFEAT - _mu[None]) / _sd[None]) * EGOT

# ---- NEW: the ocean-heat patch ---------------------------------------------------------------
# kept float16 in RAM (~510 MB); cast per sample. Channels are OHC / D26 / D20 on a 21x21 grid at
# 0.25 deg, i.e. +-2.5 deg centred on the storm.
OPATCH = _E["ohc_patch"]                      # [N,3,21,21] float16
OGOT = _E["ohc_got"].astype("float32")        # [N]
# per-channel stats over OCEAN cells only -- land is exact zero and must not drag the mean down
_s = OPATCH[OGOT > 0][::17].astype("float32")
_om, _os = [], []
for c in range(3):
    v = _s[:, c]; nz = v[v != 0]
    _om.append(float(nz.mean()) if nz.size else 0.0)
    _os.append(float(nz.std()) + 1e-6 if nz.size else 1.0)
OM = np.array(_om, "float32"); OS = np.array(_os, "float32")
del _s
print(f"env {NENV} scalars on {100*(EGOT.sum(1)>0).mean():.1f}% | ocean patch on {100*(OGOT>0).mean():.1f}% "
      f"of all windows", flush=True)
_te_wp = np.array([i for i in te_idx if z["n_leads"].astype(int)[i] == 20 and basins[i] in ("WP", "EP")])
print(f"  coverage that matters: train {100*(OGOT[tr_idx]>0).mean():.1f}%  |  "
      f"TEST WP+EP {100*(OGOT[_te_wp]>0).mean():.1f}%   <-- the constraint, see docstring", flush=True)


def ocean_in(j):
    """[b,4,21,21]: three normalised fields plus a per-cell validity channel."""
    p = OPATCH[j].astype("float32")
    valid = (p != 0).astype("float32")
    p = (p - OM[None, :, None, None]) / OS[None, :, None, None] * valid
    return np.concatenate([p, valid[:, :1]], 1)      # validity is identical across the 3 fields


# ---- v21, then v25, extracted verbatim -------------------------------------------------------
CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
_v26 = urllib.request.urlopen(f"{RAW}/colab_v26_train.py").read().decode()
_g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, _v26, re.S).group(0), _g21)
V21 = _g21["TrackFormerCoT"]

_v31 = urllib.request.urlopen(f"{RAW}/colab_v31_train.py").read().decode()
_ed = re.search(r"class _EnvDec\(nn\.Module\):.*?return self\.dec\(tgt, memory, \*a, \*\*k\)\n", _v31, re.S).group(0)
_te = re.search(r"class TrackFormerEnv\(V21\):.*?finally:\n            self\._envn = self\._envg = None\n", _v31, re.S).group(0)
_g25 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": G["math"], "NENV": NENV, "USE_ENV": 1}
exec(_ed, _g25); exec(_te, _g25)
V25 = _g25["TrackFormerEnv"]


# ---- the ocean branch ------------------------------------------------------------------------
class OceanCNN(nn.Module):
    """21x21x4 -> one d-dim token. Deliberately small: it trains on ~8% of windows (see docstring),
    so capacity here buys overfitting, not skill."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU(),    # 11x11
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),    # 6x6
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, d))

    def forward(self, x):
        return self.net(x)


class _OceanDec(nn.Module):
    """Appends the ocean token to the INTENSITY decoder's memory, then delegates.

    Same wrap-a-submodule pattern as v23/v25: the parent's forward is inherited verbatim and never
    rewritten. parent is stashed in __dict__ so it is not registered as a submodule (registering it
    would make the module tree infinite -- the parent already owns this wrapper).
    """
    def __init__(self, dec, parent):
        super().__init__()
        self.dec = dec
        self.__dict__["parent"] = parent

    def forward(self, tgt, memory, *a, **k):
        tok = self.parent._ocean_token()
        if tok is not None:
            memory = torch.cat([memory, tok], dim=1)
        return self.dec(tgt, memory, *a, **k)


class TrackFormerOcean(V25):
    """v25 + an ocean-patch CNN token on the intensity decoder only."""
    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.track_q.shape[-1]
        self.ocean_cnn = OceanCNN(d)
        self.ocean_pos = nn.Parameter(torch.zeros(1, 1, d))
        self.int_dec = _OceanDec(self.int_dec, self)      # INTENSITY ONLY -- track_dec untouched
        self._op = None
        self._og = None

    def _ocean_token(self):
        if not USE_OCEAN or self._op is None:
            return None
        # where no ocean patch exists the CNN output is zeroed and only the learned positional
        # token remains, so "no ocean data" is its own state rather than a fabricated cold sea
        h = self.ocean_cnn(self._op) * self._og[:, None]
        return h.unsqueeze(1) + self.ocean_pos

    def forward(self, tr, vp, slp, envn=None, envg=None, op=None, og=None):
        if op is not None:
            self._op, self._og = op, og
        try:
            return super().forward(tr, vp, slp, envn, envg)
        finally:
            self._op = self._og = None


# ---- init assertions: the whole claim rests on these -----------------------------------------
def _remap_env_to_ocean(sd):
    """v26 wraps int_dec once more, so every int_dec.* key gains one 'dec.' level."""
    out = {}
    for k, v in sd.items():
        out[("int_dec.dec." + k[len("int_dec."):]) if k.startswith("int_dec.") else k] = v
    return out


def _assert_init():
    global USE_OCEAN
    keep = USE_OCEAN
    torch.manual_seed(0); m25 = V25().to(DEVICE).eval()
    torch.manual_seed(0); m26 = TrackFormerOcean().to(DEVICE).eval()
    miss, unexp = m26.load_state_dict(_remap_env_to_ocean(m25.state_dict()), strict=False)
    assert not unexp, f"unexpected keys when mapping v25 -> v26: {unexp[:5]}"
    extra = {k.split(".")[0] for k in miss}
    assert extra <= {"ocean_cnn", "ocean_pos"}, f"v26 is missing more than the ocean branch: {sorted(extra)}"
    j = np.arange(8)
    a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
         torch.from_numpy(SLP[j]).to(DEVICE), torch.from_numpy(ENORM[j]).to(DEVICE),
         torch.from_numpy(EGOT[j]).to(DEVICE)]
    # a window that actually HAS ocean, so the "must move" half is not vacuous
    jo = np.where(OGOT > 0)[0][:8]
    ao = [torch.from_numpy(track[jo]).to(DEVICE), torch.from_numpy(vpair[jo]).to(DEVICE),
          torch.from_numpy(SLP[jo]).to(DEVICE), torch.from_numpy(ENORM[jo]).to(DEVICE),
          torch.from_numpy(EGOT[jo]).to(DEVICE)]
    op = torch.from_numpy(ocean_in(jo)).to(DEVICE); og = torch.from_numpy(OGOT[jo]).to(DEVICE)
    with torch.no_grad():
        ref = m25(*ao)
        USE_OCEAN = 0
        off = m26(*ao, op, og)
        USE_OCEAN = 1
        on = m26(*ao, op, og)
    d_off = max(float((x - y).abs().max()) for x, y in zip(ref[:2], off[:2]))
    d_trk = float((ref[0][..., :2] - on[0][..., :2]).abs().max())
    d_int = float((ref[0][..., 2:] - on[0][..., 2:]).abs().max())
    print(f"\ninit check | USE_OCEAN=0 vs v25 max-diff {d_off:.3e}  (must be 0)")
    print(f"           | USE_OCEAN=1 intensity max-diff {d_int:.3e}  (must be > 0)")
    print(f"           | USE_OCEAN=1 track     max-diff {d_trk:.3e}  (intensity-only wiring)")
    assert d_off == 0.0, "USE_OCEAN=0 does NOT reproduce v25 -- the ablation would be meaningless"
    assert d_int > 0.0, "USE_OCEAN=1 does not move the intensity output -- the token is inert"
    USE_OCEAN = keep
    del m25, m26


_assert_init()
print(f"\n{TAG}: USE_OCEAN={USE_OCEAN}, {N_SEEDS} seeds, oversample={OSAMPLE}", flush=True)


# ---- data ------------------------------------------------------------------------------------
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
        if self.aug and torch.rand(()) < MIRROR_P:
            tr, tg, mk, sp = mirror(tr, tg, mk, sp)
            vp = vp.clone(); vp[1] = -vp[1]; vp[3] = -vp[3]
            fl = fl.clone(); fl[:, 1] = -fl[:, 1]
            # the ocean patch is a storm-centred geographic field: the N-S flip is a flip of the
            # latitude axis. The env SCALARS stay unflipped (v25's stated approximation).
            op = torch.flip(op, dims=[-2])
        return tr, vp, sp, tg, mk, fl, fm, en, eg, op, og


def loader(idx, sh, aug=False):
    if sh and OSAMPLE > 0:
        w = np.where(OGOT[idx] > 0, OSAMPLE, 1.0)
        smp = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(idx), True)
        return torch.utils.data.DataLoader(DS(idx, aug), batch_size=BATCH, sampler=smp, num_workers=2,
                                           pin_memory=True, persistent_workers=True, drop_last=True)
    return torch.utils.data.DataLoader(DS(idx, aug), batch_size=BATCH, shuffle=sh, num_workers=2,
                                       pin_memory=True, persistent_workers=True, drop_last=sh)


def total_loss(s, ls, fp, tgt, m, fl, fm):
    base = G["total_loss"](s, ls, tgt, m)
    fmm = fm.unsqueeze(-1)
    flow = (F.smooth_l1_loss(fp, fl, reduction="none") * fmm).sum() / fmm.sum().clamp(min=1)
    return base + W_FLOW * flow, float(flow.detach())


def train_one(seed, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrackFormerOcean().to(DEVICE)
    print(f"seed {seed} | params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True, aug=True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0.0
        for tr, v0, sp, tg, m, fl, fm, en, eg, op, og in ld:
            tr, v0, sp, tg, m, fl, fm, en, eg, op, og = [
                x.to(DEVICE, non_blocking=True) for x in (tr, v0, sp, tg, m, fl, fm, en, eg, op, og)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls, fp = model(tr, v0, sp, en, eg, op, og)
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
            # Mirror to Drive on every improvement. /content is wiped when the runtime idles out,
            # and that is exactly how the first v26 run lost all 20 checkpoints AFTER producing
            # its results -- the numbers survived in the JSON, the models did not, so the tracks
            # could never be drawn without a full retrain. Best-effort: no Drive, no problem.
            if os.path.isdir(DRIVE):
                try:
                    import shutil as _sh; _sh.copy(ckpt, DRIVE)
                except Exception as _e:
                    print(f"  (drive mirror failed: {_e})", flush=True)
        else:
            bad += 1
        with torch.no_grad():
            # |W| of the ocean CNN's output layer. Flat across epochs => the patch is being ignored
            # and v26 has quietly reverted to v25.
            ow = float(model.ocean_cnn.net[-1].weight.abs().mean()) if USE_OCEAN else 0.0
            ew = float(model.env_mlp[-1].weight.abs().mean())
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
print(f"{TAG} trained: {len(CK)} seeds (USE_OCEAN={USE_OCEAN})", flush=True)

# ---- evaluation ------------------------------------------------------------------------------
full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
ocn = np.array([i for i in wpep if OGOT[i] > 0])
SC = TARGET_SCALE


@torch.no_grad()
def metrics(ms, idx):
    """Track km plus vmax / pressure / RMW / wind-radii MAE.

    Each intensity channel is masked with its OWN validity flag. v25 reported these against the
    POSITION mask, which counts windows whose pressure fix does not exist and inflated pressure MAE
    to ~140 hPa. The real figure is ~13. Both arms were wrong the same way so the v25 comparison
    still held, but the absolute number was meaningless; it is fixed here.
    """
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE), torch.from_numpy(ENORM[j]).to(DEVICE),
             torch.from_numpy(EGOT[j]).to(DEVICE),
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
    m = TrackFormerOcean().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); return m


MS = [load_m(c) for c in CK]
print(f"\nWP+EP 2020+: {len(wpep)} windows, {len(ocn)} with an ocean patch ({100*len(ocn)/len(wpep):.0f}%)")
print("  BASELINES  track km: v21 443.6 | v25 451.9    pressure hPa: v21 13.14 | v25 12.68")
A = metrics(MS, wpep); B = metrics(MS, ocn)
print(f"\n  {TAG} ENSEMBLE ({len(MS)} seeds)")
for lab, r, n in (("all WP+EP", A, len(wpep)), ("ocean-covered", B, len(ocn))):
    print(f"    {lab:14s} ({n:4d}): track {r['track']:7.2f} km | vmax {r['vmax']:5.2f} kt | "
          f"pres {r['pres']:5.2f} hPa | rmw {r['rmw']:5.2f} nm | radii {r['radii']:5.2f} nm")
print("  NOTE: the ocean patch feeds the INTENSITY decoder only. Judge it on pres/vmax/rmw/radii;")
print("  track is expected to be flat. Meaningful comparison is v26 vs v26abl.", flush=True)
json.dump({TAG: {"all": A, "ocean": B}, "use_ocean": USE_OCEAN, "n_seeds": len(MS),
           "n_ocean": int(len(ocn)), "oversample": OSAMPLE}, open(f"/content/{TAG}.json", "w"))
try:
    from google.colab import files
    import subprocess
    subprocess.run(f"tar cf /content/{TAG}_seeds.tar /content/{TAG}_seed*.pt", shell=True)
    files.download(f"/content/{TAG}.json")
except Exception as ex:
    print("download skipped:", ex)
