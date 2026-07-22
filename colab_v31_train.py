"""v25 on Colab — v21 plus the environmental feature stream (ocean heat + shear + humidity).

    from google.colab import drive; drive.mount('/content/drive')
    !wget -q -O /content/v25.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/<SHA>/colab_v31_train.py
    import os; os.environ["V31_SEEDS"]="10"; exec(open('/content/v25.py').read())

WHAT THIS ADDS. Every version so far has been blind to the ocean, the vertical shear and the
mid-level humidity -- the environment that governs a storm's intensity. env_features.npz carries
43 of those predictors per window: SHIPS (ocean heat content, 26C isotherm depth, deep-layer
shear, RH at three levels, upper-level flow) plus AOML ocean-heat patches reduced to centre/mean.
A verified ablation (Yuan 2023, held-out 2018-2020) put SHIPS-style environmental predictors at
-55.7% intensity error against -3.0% for physics-as-inductive-bias.

ADDITIVE, NOT SUBSTITUTIVE. This is the lesson v24 paid 86 km to learn: ADDING an environmental
stream (v23's temporal steering stack, -8.66 km, p=0.008) works; REPLACING the vortex view with a
wider one (v24, +86.5 km, p<0.001) fails. So v25 keeps everything v21 has and APPENDS the env
features as one extra token in the decoder memory. V31_USE_ENV=0 does not append it, which
reproduces v21 EXACTLY -- asserted below at max-diff 0. That is the ablation: v25 vs v25abl
isolates the env features against v21's 443.62 km, with no other variable moving.

BASE IS v21, NOT v23. v23's temporal stack is a separate, already-confirmed win. Stacking both at
once would make the result uninterpretable, which is the trap this project keeps hitting. If the
env features earn their place, combining with v23 is the next step, not this one.

HONEST EXPECTATION. Ocean heat, shear and humidity are INTENSITY predictors -- they change how
strong a storm gets, not much where it goes. The gain, if any, should show up in the intensity
metrics (vmax / pressure MAE), not the track error. Both are reported so the claim is testable and
not oversold as a track result.

PATCHY COVERAGE IS A FIRST-CLASS INPUT. The features exist for only ~50% of WP+EP windows (SHIPS
ends 2021, AOML starts 2013). The 43-dim presence mask is fed alongside the values, so the model
learns "features absent" as a state rather than reading zeros as real ocean. Nothing here leaks:
the SHIPS columns are the -12/-6/0 hour values only, never the forward best-track truth.

LEAKAGE / IDENTITY. As with every version, nothing carries storm id or absolute date. The env
features are physical quantities at t0; the mask is presence, not time.
"""

import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
DRIVE = "/content/drive/MyDrive/typhoon"
N_SEEDS = int(os.environ.get("V31_SEEDS", "10"))
USE_ENV = int(os.environ.get("V31_USE_ENV", "1"))
W_FLOW = float(os.environ.get("W_FLOW", "0.3"))
TAG = os.environ.get("V31_TAG", "v25" if USE_ENV else "v25abl")
KM6H = 6 * 3600 / 1000.0

for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)

# env_features.npz is 9.9 MB -- small enough for the repo, but it comes from Drive to match the
# basin/ohc convention and so a re-parse never forces a re-commit
for fn in ("env_features.npz",):
    if not os.path.exists(f"/content/{fn}"):
        src = f"{DRIVE}/{fn}"
        if os.path.exists(src):
            import shutil; shutil.copy(src, f"/content/{fn}")
            print(f"copied {fn} from Drive", flush=True)
        else:
            print(f"fetching {fn} from repo ...", flush=True)
            urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", f"/content/{fn}")

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

# ---- environmental features -----------------------------------------------------------------
_E = np.load("/content/env_features.npz", allow_pickle=True)
EFEAT = _E["feat"].astype("float32"); EGOT = _E["got"].astype("float32")
NENV = EFEAT.shape[1]
# z-normalise each feature over the windows that HAVE it, so the scale is sane and an absent
# feature (already 0) stays at the mean after (x-mean)/std only where got==1
_present = EGOT > 0
_mu = np.array([EFEAT[_present[:, c], c].mean() if _present[:, c].any() else 0.0
                for c in range(NENV)], "float32")
_sd = np.array([EFEAT[_present[:, c], c].std() + 1e-6 if _present[:, c].any() else 1.0
                for c in range(NENV)], "float32")
ENORM = ((EFEAT - _mu[None]) / _sd[None]) * EGOT           # zeroed where absent, mask carries that
print(f"env features: {NENV} predictors | present on "
      f"{100*(EGOT.sum(1) > 0).mean():.1f}% of windows", flush=True)

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
_v26 = urllib.request.urlopen(f"{RAW}/colab_v26_train.py").read().decode()
_g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, _v26, re.S).group(0), _g21)
V21 = _g21["TrackFormerCoT"]


class _EnvDec(nn.Module):
    """Wraps a TransformerDecoder and appends the env token to its memory before delegating.

    This is the v23 pattern: wrap a submodule and stash context on the parent, so v25 inherits
    v21's forward VERBATIM -- no rewriting, no source patching. v21 calls self.track_dec(tq, mem)
    and self.int_dec(iq, mem); both memories gain the same env token. When USE_ENV is off, or no
    env is set, the memory passes through untouched, so the model is v21 exactly.
    """
    def __init__(self, dec, parent):
        super().__init__()
        self.dec = dec
        # store WITHOUT nn.Module registration: self.parent = parent would register the parent as
        # a submodule, and the parent already holds this _EnvDec -> an infinite module tree that
        # blows the recursion limit on .train()/.state_dict(). __dict__ bypasses __setattr__.
        self.__dict__["parent"] = parent

    def forward(self, tgt, memory, *a, **k):
        tok = self.parent._env_token()
        if tok is not None:
            memory = torch.cat([memory, tok], dim=1)
        return self.dec(tgt, memory, *a, **k)


class TrackFormerEnv(V21):
    """v21 with one environmental token appended to both decoders' memory.

    The token is encoded from the 43 features CONCATENATED WITH their presence mask, so the model
    sees the values and which are real. USE_ENV off => token not appended => v21 unchanged.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.track_q.shape[-1]
        self.env_mlp = nn.Sequential(nn.Linear(2 * NENV, 256), nn.GELU(), nn.Linear(256, d))
        self.env_pos = nn.Parameter(torch.zeros(1, 1, d))
        self.track_dec = _EnvDec(self.track_dec, self)
        self.int_dec = _EnvDec(self.int_dec, self)
        self._envn = None
        self._envg = None

    def _env_token(self):
        if not USE_ENV or self._envn is None:
            return None
        return self.env_mlp(torch.cat([self._envn, self._envg], -1)).unsqueeze(1) + self.env_pos

    def set_env(self, envn, envg):
        self._envn, self._envg = envn, envg

    def forward(self, tr, vp, slp, envn=None, envg=None):
        if envn is not None:
            self.set_env(envn, envg)
        try:
            return super().forward(tr, vp, slp)
        finally:
            self._envn = self._envg = None

print(f"\n{TAG}: USE_ENV={USE_ENV}, {N_SEEDS} seeds", flush=True)

# ---- training: v26's loop, with the env features and mask threaded through -------------------
# mirror augmentation is kept (v21's patch is storm-centred, so the N-S flip is valid). The env
# features are storm-relative scalars; only the meridional components flip sign under mirror.
# CSST/COHC/shear MAGNITUDE etc. are invariant, so for safety we leave the env vector unflipped --
# the small number of directional env terms is a second-order effect and mirroring them wrong
# would be worse than not mirroring them at all. This is a deliberate, stated approximation.
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
        if self.aug and torch.rand(()) < MIRROR_P:
            tr, tg, mk, sp = mirror(tr, tg, mk, sp)
            vp = vp.clone(); vp[1] = -vp[1]; vp[3] = -vp[3]
            fl = fl.clone(); fl[:, 1] = -fl[:, 1]
        return tr, vp, sp, tg, mk, fl, fm, en, eg


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
    model = TrackFormerEnv().to(DEVICE)
    print(f"seed {seed} | params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True, aug=True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0.0
        for tr, v0, sp, tg, m, fl, fm, en, eg in ld:
            tr, v0, sp, tg, m, fl, fm, en, eg = [x.to(DEVICE, non_blocking=True)
                for x in (tr, v0, sp, tg, m, fl, fm, en, eg)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls, fp = model(tr, v0, sp, en, eg)
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
        else:
            bad += 1
        with torch.no_grad():
            # |W| of the env MLP's output layer -- if it stays near its init the features are being
            # ignored and v25 has effectively reverted to v21
            ew = float(model.env_mlp[-1].weight.abs().mean()) if USE_ENV else 0.0
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"envW {ew:.5f} | {time.time()-te:.0f}s", flush=True)
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
print(f"{TAG} trained: {len(CK)} seeds (USE_ENV={USE_ENV})", flush=True)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
env_cov = np.array([i for i in wpep if EGOT[i].sum() > 0])
SC = TARGET_SCALE


@torch.no_grad()
def metrics(ms, idx):
    """track error (km) AND intensity MAE (vmax kt, pressure hPa) over idx."""
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE),
             torch.from_numpy(ENORM[j]).to(DEVICE), torch.from_numpy(EGOT[j]).to(DEVICE)]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().cpu().numpy())
    O = np.concatenate(P); T = target[idx]
    C = np.cumsum(O[..., :2], 1); TC = np.cumsum(T[..., :2], 1)
    trk = float(np.sqrt(((C - TC) ** 2).sum(-1)).mean())
    mk = mask[idx][..., 0] > 0
    vmax = float(np.abs(O[..., 2] - T[..., 2])[mk].mean())
    pres = float(np.abs(O[..., 3] - T[..., 3])[mk].mean())
    return trk, vmax, pres


def load_m(c):
    m = TrackFormerEnv().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); return m


MS = [load_m(c) for c in CK]
print(f"\nWP+EP 2020+: {len(wpep)} windows, {len(env_cov)} with env features "
      f"({100*len(env_cov)/len(wpep):.0f}%)")
print("  BASELINES (track km):  v21 443.6 | v22 443.4 | v23 435.0")
tk_all, vx_all, pr_all = metrics(MS, wpep)
tk_cov, vx_cov, pr_cov = metrics(MS, env_cov)
print(f"\n  {TAG} ENSEMBLE ({len(MS)} seeds)")
print(f"    all WP+EP ({len(wpep)}):     track {tk_all:.2f} km | vmax {vx_all:.2f} kt | pres {pr_all:.2f} hPa")
print(f"    env-covered ({len(env_cov)}): track {tk_cov:.2f} km | vmax {vx_cov:.2f} kt | pres {pr_cov:.2f} hPa")
print("  NOTE: env features are INTENSITY predictors (ocean heat, shear, humidity). Expect the")
print("  gain, if any, in vmax/pres -- not track. Meaningful comparison is v25 vs v25abl.", flush=True)
json.dump({TAG: {"track_all": tk_all, "vmax_all": vx_all, "pres_all": pr_all,
                 "track_cov": tk_cov, "vmax_cov": vx_cov, "pres_cov": pr_cov},
           "use_env": USE_ENV, "n_seeds": len(MS), "n_cov": int(len(env_cov))},
          open(f"/content/{TAG}.json", "w"))
try:
    from google.colab import files
    import subprocess
    subprocess.run(f"tar cf /content/{TAG}_seeds.tar /content/{TAG}_seed*.pt", shell=True)
    files.download(f"/content/{TAG}.json"); files.download(f"/content/{TAG}_seeds.tar")
except Exception as ex:
    print("download skipped:", ex)
