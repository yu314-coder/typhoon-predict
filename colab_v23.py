"""v10.3 on Colab — v10.2 made GENERATIVE: sample tracks, then average them.

    !wget -q -O v23.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v23.py
    exec(open('v23.py').read())

WHY THIS AND NOT A CURVATURE PENALTY. v10's forecasts turn 3.6 deg/step against 18.8 observed.
That is not a missing loss term -- it is the double-penalty blur. A deterministic model trained to
minimise squared error cannot do better than predicting the CONDITIONAL MEAN, and the conditional
mean of a bundle of plausible tracks is straighter than any of them. GenCast's authors say the same
thing of gridded fields: forecasts from MSE-trained deterministic models "are blurred and closer to
the ensemble mean". So v10 is not failing to curve; it is correctly reporting an average. Adding a
curvature term fights the primary loss instead of removing the cause, which is why v10.3 is this
rather than that.

WHAT CHANGES. Two things that are really one change -- the model stops emitting a point estimate
and starts emitting samples:

  1. A noise vector z ~ N(0, I_32) modulates the track decoder output (FiLM on h_track). The
     physics baseline is untouched, so z perturbs only the learned residual: samples spread around
     damped curved persistence rather than around nothing.
  2. The track Huber terms are replaced by the FAIR CRPS estimator over 2 samples. CRPS is a proper
     scoring rule, so it is minimised only by the true predictive distribution -- a collapsed
     deterministic predictor is suboptimal unless the truth really is a point mass. Two samples per
     step is what FGN (DeepMind's cyclone model) uses, and its result is that training on marginals
     alone still yields good joint structure.

Then the average is finally legitimate. The 50 samples come from ONE initialisation and are i.i.d.
draws from p(track | history), so they are exchangeable and their mean is a Monte Carlo estimate of
E[track | history] -- the Bayes estimator under squared error. That is exactly the property the
earlier "mean of forecasts launched at different times" lacked: a lagged ensemble mixes draws from
different distributions with different skill and different bias, which is why it needed error
weighting to work at all (DelSole et al. 2017 prove the equally-weighted lagged mean is suboptimal).

THE FAILURE MODE TO WATCH is posterior collapse: if the model learns to ignore z, every sample is
identical, fair CRPS degenerates to MAE, and this is just v10.2 trained to predict a median. The
per-epoch print therefore reports ensemble SPREAD in km. If spread stays near zero the run is dead
and should be killed early rather than after an hour.

The intensity head is deliberately left deterministic and its loss unchanged, so the only thing
being tested is the track distribution.

--- inherited from v10.2 ---
Tip (1979) is held out of training so it stays a genuine out-of-dataset case. Data is the 1950+
rebuild with ONI as feature 54. Valid and test splits are byte-identical to v10 and v10.2, so the
km numbers are directly comparable -- but note the bar is v10's 549.3 km, because v10.2's 536.4
was itself no measurable gain (-12.9 km against a 5.7-24.7 km seed spread).
"""
import os, re, time, json, math, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = int(os.environ.get("V23_SEEDS", "2"))     # 2 fwd passes/step, so ~2x the epoch cost
N_SAMPLES = int(os.environ.get("V23_SAMPLES", "50"))
ZDIM = 32
EPOCHS, PATIENCE = 60, 12
BATCH, LR, WEIGHT_DECAY = 1024, 3e-4, 3e-2
TARGET_SCALE = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12, device=DEVICE)

DATA = "/content/drive/MyDrive/typhoon"
NPZ = "/content/d/track_windows_v20.npz"
if not os.path.exists(NPZ):
    os.makedirs("/content/d", exist_ok=True)
    src = f"{DATA}/track_windows_v20.npz"
    if os.path.exists(src):
        import shutil; shutil.copy(src, NPZ)
    else:
        print("fetching the dataset from GitHub (63 MB) ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/track_windows_v20.npz", NPZ)
    print(f"dataset ready: {os.path.getsize(NPZ)/1e6:.0f} MB", flush=True)

if not os.path.exists("/content/_v20.py"):
    urllib.request.urlretrieve(f"{RAW}/train_track_v20.py", "/content/_v20.py")
src = open("/content/_v20.py").read()
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os, "DEVICE": DEVICE}
for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
            r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
            r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    m = re.search(pat, src, re.S); assert m, pat[:40]
    exec(m.group(0), G)
Base = G["TrackFormerV9"]
KIN_COLS, THERMO_COLS, ENV_COLS = G["KIN_COLS"], G["THERMO_COLS"], G["ENV_COLS"]
print(f"env dims {G['ENV_DIM']} (55 features, ONI included)", flush=True)


class TrackFormerGen(Base):
    """v10.2's network with a noise vector modulating the track decoder output.

    forward() is reimplemented rather than hooked, so it is checked against the parent below:
    with the noise path disabled the two must agree to floating-point equality.
    """

    def __init__(self, d=256, zdim=ZDIM, **kw):
        super().__init__(d=d, **kw)
        self.zdim = zdim
        self.znet = nn.Sequential(nn.Linear(zdim, d), nn.GELU(), nn.Linear(d, 2 * d))
        # NOT zero-initialised, unlike track_res. track_res starts at zero so that an untrained
        # model is exactly the physics baseline, but that also makes d(motion)/d(h_track) zero,
        # which severs the gradient into znet. Zero-initialising znet as well would leave the noise
        # path dead at step 0 and give the model every opportunity to settle into ignoring z --
        # posterior collapse designed in. Small random init instead, so that the moment track_res
        # leaves zero the samples already differ and CRPS can start shaping the spread.
        nn.init.normal_(self.znet[-1].weight, std=0.02); nn.init.zeros_(self.znet[-1].bias)
        self.zgain = nn.Parameter(torch.tensor(0.1))

    def forward(self, track, vpair, z=None):
        b = track.shape[0]
        kin = self.kin_enc(self.kin_proj(track[:, :, KIN_COLS]) + self.kin_time)
        thermo = self.thermo_enc(self.thermo_proj(track[:, :, THERMO_COLS]) + self.thermo_time)
        env = self.env_enc(self.env_proj(track[:, :, ENV_COLS]) + self.env_time)

        tq = (self.track_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_track = self.track_dec(tq, torch.cat([kin, env], dim=1))
        h_track = h_track + self.alpha.view(1, self.leads, 1) * \
            self.adapter(thermo.mean(1).detach()).unsqueeze(1)
        if z is not None:
            g, sh = self.znet(z).chunk(2, -1)
            h_track = h_track * (1 + self.zgain * g.unsqueeze(1)) + self.zgain * sh.unsqueeze(1)

        v0, vp = vpair[:, :2], vpair[:, 2:]
        s0 = torch.linalg.norm(v0, dim=-1)
        phi0 = torch.atan2(v0[:, 1], v0[:, 0])
        phip = torch.atan2(vp[:, 1], vp[:, 0])
        omega = torch.remainder(phi0 - phip + math.pi, 2 * math.pi) - math.pi
        phil = phi0.unsqueeze(1) + self.gturn.view(1, self.leads) * omega.unsqueeze(1)
        speed = self.rho.view(1, self.leads) * s0.unsqueeze(1)
        base = torch.stack([speed * torch.cos(phil), speed * torch.sin(phil)], dim=-1) / 100.0
        motion = base + self.track_res(h_track)

        iq = (self.int_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_int = self.int_dec(iq, torch.cat([thermo, env, kin.detach()], dim=1))
        istate = self.int_state(h_int); ilog = self.int_logscale(h_int).clamp(-5.0, 3.0)
        return (torch.cat([motion, istate], -1),
                torch.cat([torch.zeros_like(motion), ilog], -1))


z = np.load(NPZ, allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
mask = z["target_mask"].astype("float32")
years = z["year"].astype(int); sids = z["storm_id"].astype(str); basins = z["basin"].astype(str)
tmean = z["track_mean"].astype("float32"); tstd = z["track_std"].astype("float32")
v0 = track[:, -1, 2:4] * tstd[2:4] + tmean[2:4]
vp = track[:, -2, 2:4] * tstd[2:4] + tmean[2:4]
vpair = np.concatenate([v0, vp], axis=1).astype("float32")
fy = {s: int(years[sids == s].min()) for s in np.unique(sids)}
tr_idx = np.where(np.isin(sids, [s for s, y in fy.items() if y <= 2015]))[0]
va_idx = np.where(np.isin(sids, [s for s, y in fy.items() if 2016 <= y <= 2019]))[0]
te_idx = np.where(np.isin(sids, [s for s, y in fy.items() if y >= 2020]))[0]

TIP = "1979275N06159"
_n = int((sids[tr_idx] == TIP).sum())
tr_idx = tr_idx[sids[tr_idx] != TIP]
print(f"held out Tip (1979): {_n} windows removed from training", flush=True)
assert _n > 0, "Tip not found in training -- check the SID"
print(f"windows {len(track)} | train {len(tr_idx)} valid {len(va_idx)} test {len(te_idx)}", flush=True)

# ---- the reimplemented forward must match the parent exactly when z is off ----
with torch.no_grad():
    _a, _b = Base().eval(), TrackFormerGen().eval()
    _b.load_state_dict(_a.state_dict(), strict=False)
    _t = torch.from_numpy(track[:4]); _v = torch.from_numpy(vpair[:4])
    _o1 = _a(_t, _v)[0]; _o2 = _b(_t, _v, None)[0]
    _d = float((_o1 - _o2).abs().max())
    assert _d < 1e-6, f"generative forward diverges from the parent by {_d}"
    _z = torch.randn(4, ZDIM)
    _o3 = _b(_t, _v, _z)[0]
    print(f"forward check: max|gen - parent| = {_d:.2e} with z off; z on moves the output by "
          f"{float((_o3-_o2).abs().max()):.2e} (zero at init is expected: track_res is zero, so "
          f"nothing reaches the output until it moves)", flush=True)
del _a, _b


class DS(Dataset):
    def __init__(self, idx): self.idx = np.asarray(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i])
        return (torch.from_numpy(track[j]), torch.from_numpy(vpair[j]),
                torch.from_numpy(target[j]), torch.from_numpy(mask[j]))


def loader(idx, sh):
    return DataLoader(DS(idx), batch_size=BATCH, shuffle=sh, num_workers=2,
                      pin_memory=True, persistent_workers=True, drop_last=sh)


LEADW = torch.sqrt(torch.arange(1, 21, device=DEVICE).float()); LEADW = LEADW / LEADW.mean()


def crps2(x1, x2, y):
    """Fair CRPS from 2 samples: mean|xi-y| - (1/2)|x1-x2|. Zero iff both samples equal y."""
    return 0.5 * ((x1 - y).abs() + (x2 - y).abs()) - 0.5 * (x1 - x2).abs()


def track_loss(s1, s2, tn, m):
    p1, p2, t_, mm = s1[..., :2], s2[..., :2], tn[..., :2], m[..., :2]
    step = (crps2(p1, p2, t_) * mm * LEADW.view(1, 20, 1)).sum() / mm.sum().clamp(min=1)
    c1, c2, ct = torch.cumsum(p1, 1), torch.cumsum(p2, 1), torch.cumsum(t_, 1)
    pos = (crps2(c1, c2, ct) * mm).sum() / mm.sum().clamp(min=1)
    return step + pos


def int_loss(s, logs, tn, m):
    ps, ts_, ms = s[..., 2:], tn[..., 2:], m[..., 2:]
    huber = (F.smooth_l1_loss(ps, ts_, reduction="none") * ms).sum() / ms.sum().clamp(min=1)
    nll = ((0.5 * ((ts_ - ps) * torch.exp(-logs[..., 2:])) ** 2 + logs[..., 2:]) * ms).sum() / ms.sum().clamp(min=1)
    r34, r50, r64 = ps[..., 3:7], ps[..., 7:11], ps[..., 11:15]
    return 0.7 * huber + 0.3 * nll + 0.01 * (F.relu(r50 - r34).mean() + F.relu(r64 - r50).mean())


def total_loss(s1, s2, logs, tgt, m):
    tn = tgt / TARGET_SCALE
    return track_loss(s1, s2, tn, m) + int_loss(s1, logs, tn, m)


def spread_km(s1, s2):
    """Mean |sample1 - sample2| at +120 h, in km -- the collapse alarm."""
    d = (torch.cumsum(s1[..., :2], 1) - torch.cumsum(s2[..., :2], 1))[:, -1, :] * 100.0
    return float(torch.linalg.norm(d, dim=-1).mean())


def train_one(seed, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrackFormerGen().to(DEVICE)
    if seed == 0:
        print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0.0; sp = 0.0
        for tr, v_, tg, m in ld:
            tr, v_, tg, m = [x.to(DEVICE, non_blocking=True) for x in (tr, v_, tg, m)]
            zz = torch.randn(2 * tr.shape[0], ZDIM, device=DEVICE)
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s1, ls = model(tr, v_, zz[:tr.shape[0]])
                s2, _ = model(tr, v_, zz[tr.shape[0]:])
                loss = total_loss(s1, s2, ls, tg, m)
            if train:
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            tot += float(loss.detach()) * len(tr); cnt += len(tr)
            sp += spread_km(s1.detach().float(), s2.detach().float()) * len(tr)
        return tot / cnt, sp / cnt

    best, bad, t0 = float("inf"), 0, time.time()
    for ep in range(EPOCHS):
        te = time.time(); trl, trsp = run(tl, True)
        with torch.no_grad(): vv, vsp = run(vl, False)
        sched.step()
        if vv < best:
            best, bad = vv, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "best_val": best,
                        "track_mean": tmean, "track_std": tstd}, ckpt)
            if os.path.isdir(DATA):
                try:
                    import shutil; shutil.copy(ckpt, DATA)
                except Exception as e: print("  (drive copy failed:", e, ")")
        else:
            bad += 1
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"spread {vsp:6.1f} km | {time.time()-te:.0f}s", flush=True)
        if ep == 4 and vsp < 1.0:
            print("  !! spread has not left zero by epoch 4 -- posterior collapse, "
                  "the noise is being ignored. Killing this seed.", flush=True)
            break
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


CK = [train_one(s, f"/content/v23_seed{s}.pt") for s in range(N_SEEDS)]
print(f"v10.3 trained: {len(CK)} seeds, {N_SAMPLES} samples at inference", flush=True)

full = z["n_leads"].astype(int) == 20
EV = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = np.cumsum(target[EV][..., :2], 1)


@torch.no_grad()
def predict(models, n_samples, seed=0):
    """Mean over n_samples draws per model -- exchangeable samples from one initialisation."""
    torch.manual_seed(1234 + seed)
    P = []
    for i in range(0, len(EV), 256):
        j = EV[i:i + 256]
        tt = torch.from_numpy(track[j]).to(DEVICE); vv_ = torch.from_numpy(vpair[j]).to(DEVICE)
        acc = 0.0
        for m in models:
            for _ in range(n_samples):
                zz = torch.randn(len(j), ZDIM, device=DEVICE)
                acc = acc + m(tt, vv_, zz)[0]
        P.append((acc / (len(models) * n_samples) * TARGET_SCALE).float().cpu().numpy())
    return np.cumsum(np.concatenate(P)[..., :2], 1)


MS = []
for c in CK:
    mm_ = TrackFormerGen().to(DEVICE).eval()
    mm_.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); MS.append(mm_)
print(f"\nWP+EP 2020+, {len(EV)} full-horizon windows — the SAME set v10 and v10.2 were scored on")
print(f"  BASELINES   v10 549.3 | v10.2 536.4 | v17 462.8 | v18 466.2 | v19 466.7")
for i, mdl in enumerate(MS):
    e1 = float(np.sqrt(((predict([mdl], 1) - T) ** 2).sum(-1)).mean())
    eN = float(np.sqrt(((predict([mdl], N_SAMPLES) - T) ** 2).sum(-1)).mean())
    print(f"  v10.3 seed{i}   1 sample {e1:7.2f} km | mean of {N_SAMPLES} {eN:7.2f} km", flush=True)
e = float(np.sqrt(((predict(MS, N_SAMPLES) - T) ** 2).sum(-1)).mean())
print(f"  v10.3 ENSEMBLE ({len(MS)} seeds x {N_SAMPLES} samples)  {e:.2f} km")
print(f"  vs v10: {e - 549.3:+.2f} km   vs v10.2: {e - 536.38:+.2f} km   vs v17: {e - 462.8:+.2f} km")
json.dump({"v10_3": e, "n_samples": N_SAMPLES}, open("/content/v23.json", "w"))


def turn_rate(lat, lon):
    h = [math.atan2(lat[i + 1] - lat[i],
                    (lon[i + 1] - lon[i]) * math.cos(math.radians(lat[i])))
         for i in range(len(lat) - 1)]
    d = [abs(((h[i + 1] - h[i] + math.pi) % (2 * math.pi)) - math.pi) for i in range(len(h) - 1)]
    return math.degrees(sum(d) / len(d)) if d else float("nan")


R = 111.2
NAMES = {"1979275N06159": "Tip", "1949317N09158": "Allyn",
         "1946222N15152": "Lilly", "1948011N07147": "Karen"}


@torch.no_grad()
def tracks_for(tr, vp, bt_, bla_, blo_, tgt_, K, n_samples):
    """Export the sample mean AND one raw sample, so blur can be told from realism."""
    out = {}
    for tag_, ns in (("mean", n_samples), ("sample", 1)):
        torch.manual_seed(99)
        P = []
        for i in range(0, len(K), 64):
            j = K[i:i + 64]
            tt = torch.from_numpy(tr[j]).to(DEVICE); vv_ = torch.from_numpy(vp[j]).to(DEVICE)
            acc = 0.0
            for m in MS:
                for _ in range(ns):
                    acc = acc + m(tt, vv_, torch.randn(len(j), ZDIM, device=DEVICE))[0]
            P.append((acc / (len(MS) * ns) * TARGET_SCALE).float().cpu().numpy())
        A = np.concatenate(P)
        cE, cN = np.cumsum(A[..., 0], 1), np.cumsum(A[..., 1], 1)
        T_ = tgt_[K]; tE, tN = np.cumsum(T_[..., 0], 1), np.cumsum(T_[..., 1], 1)
        lats, lons = [], []
        for a in range(len(K)):
            la = bla_[K[a]] + cN[a] / R
            lo = blo_[K[a]] + cE[a] / (R * np.cos(np.radians((bla_[K[a]] + la) / 2)))
            lats.append(np.round(la, 3).tolist()); lons.append(np.round(lo, 3).tolist())
        err = float(np.hypot(cE[:, 19] - tE[:, 19], cN[:, 19] - tN[:, 19]).mean())
        tr_ = sum(turn_rate(a, o) for a, o in zip(lats, lons)) / max(len(lats), 1)
        out[tag_] = {"lat": lats, "lon": lons, "base_time": bt_[K].tolist(),
                     "base_lat": np.round(bla_[K], 3).tolist(),
                     "base_lon": np.round(blo_[K], 3).tolist(),
                     "err120_mean": err, "turn_pred": tr_, "n": int(len(K))}
    return out


EXPORT = {}
print("\nout-of-dataset tracks (v10.3):", flush=True)
print("  v10/v10.2 individual forecasts turn 3.5-3.6 deg/step; the observed track turns 18.8.")
for tag, fn, prenorm in [("tip", "tip_v20_fixed.npz", True),
                         ("pre1950", "pre1950_windows.npz", False)]:
    p = f"/content/{fn}"
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", p)
    w = np.load(p, allow_pickle=True)
    tr = w["track"].astype("float32")
    assert tr.shape[2] == 55, f"{fn} has {tr.shape[2]} features, expected 55"
    if not prenorm:
        raw = tr * w["track_std"].astype("float32") + w["track_mean"].astype("float32")
        tr = (raw - tmean) / tstd
    vp_ = np.concatenate([tr[:, -1, 2:4] * tstd[2:4] + tmean[2:4],
                          tr[:, -2, 2:4] * tstd[2:4] + tmean[2:4]], 1).astype("float32")
    nl_ = w["n_leads"].astype(int); bt_ = w["base_time"].astype("int64")
    sid_ = w["storm_id"].astype(str)
    bla_ = w["base_lat"].astype("float64"); blo_ = w["base_lon"].astype("float64")
    tgt_ = w["target"].astype("float32")
    out = {}
    for s_, nm in NAMES.items():
        K = np.where((sid_ == s_) & (nl_ == 20))[0]
        if not len(K):
            continue
        K = K[np.argsort(bt_[K])]
        out[nm] = tracks_for(tr, vp_, bt_, bla_, blo_, tgt_, K, N_SAMPLES)
        a, b_ = out[nm]["mean"], out[nm]["sample"]
        print(f"  {nm:6s} {a['n']:3d} fc | mean of {N_SAMPLES}: {a['err120_mean']:6.0f} km, "
              f"turn {a['turn_pred']:5.1f} | one sample: {b_['err120_mean']:6.0f} km, "
              f"turn {b_['turn_pred']:5.1f}", flush=True)
    EXPORT[tag] = out

J = "/content/v23_tracks.json"
json.dump(EXPORT, open(J, "w"))
print(f"\nwrote {J} ({os.path.getsize(J)/1000:.0f} KB)", flush=True)
if os.path.isdir(DATA):
    try:
        import shutil; shutil.copy(J, DATA); print("  mirrored to Drive", flush=True)
    except Exception as ex: print("  (drive copy failed:", ex, ")", flush=True)

import gzip, base64
_b = base64.b64encode(gzip.compress(open(J, "rb").read())).decode()
print(f"\nBASE64 gzip payload, {len(_b)} chars", flush=True)
print("<<<V23TRACKS", flush=True)
for i in range(0, len(_b), 200):
    print(_b[i:i + 200], flush=True)
print("V23TRACKS>>>", flush=True)

try:
    from google.colab import files
    files.download(J)
    import subprocess
    subprocess.run("tar cf /content/v23_seeds.tar /content/v23_seed*.pt", shell=True)
    files.download("/content/v23_seeds.tar")
except Exception as ex:
    print("(auto-download unavailable:", ex, ")", flush=True)
