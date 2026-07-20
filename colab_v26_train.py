"""v21 on Colab — predict the steering flow, then DERIVE the track from it (chain-of-thought).

    !pip install -q netCDF4
    !wget -q -O v26t.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v26_train.py
    exec(open('v26t.py').read())

THE STRUCTURE. v20 regresses displacement directly. v21 predicts an intermediate physical quantity
-- the deep-layer steering flow at each of 20 leads -- and computes the track from it. That is the
chain-of-thought pattern: an intermediate the answer is derived from. Unlike a language model's
reasoning chain, this one is supervised against physical truth, so a wrong forecast can be traced to
either a wrong flow or a wrong integration.

WHY IT SHOULD WORK, measured rather than assumed. Regressing observed motion on the extracted flow
over 2,857,098 lead-points gives corr +0.872 east / +0.782 north, slope 0.76, and -- without being
fitted for -- intercepts of -2.03 m/s east and +0.40 m/s north, i.e. the west-northwest beta drift
of the right sign and size. The correlation is FLAT across leads (0.863 at +6 h, 0.873 at +120 h),
so the intermediate is equally informative at every horizon.

DEGRADES TO v20 AT INIT, BY CONSTRUCTION. The flow head predicts a DELTA from the current flow and
is zero-initialised:

    flow_now   = annulus mean of the input steering patch          (the storm's present steering)
    flow_pred  = flow_now + flow_delta(h_track)                    (delta zero-init)
    motion     = curved_persistence_base + A * flow_delta * KM6H + track_res(h_track)

At step zero flow_delta = 0, so motion is exactly v20's. Any change is something the flow path
learned, and if it learns nothing v21 cannot score worse than v20 for architectural reasons. A is a
learned scalar initialised to the measured 0.76.

WHAT IS BEING TESTED. Not "is the decomposition real" -- the extraction already showed it is. It is
"does forcing the model to explain its track through a supervised physical intermediate improve the
track". Those are different claims and only the second is in doubt.

The auxiliary flow loss is also a regulariser: the encoder must produce a representation that
explains the real atmosphere, not merely fit 20 displacement numbers, which is harder to satisfy by
memorising storm-specific quirks. Relevant because v17 overfits 1.29x train->test.
"""
import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
N_SEEDS = int(os.environ.get("V26_SEEDS", "3"))
W_FLOW = float(os.environ.get("W_FLOW", "0.3"))     # weight on the auxiliary flow loss
KM6H = 6 * 3600 / 1000.0                            # m/s -> km per 6 h step (21.6)

for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)

# ---- v17 machinery verbatim, with the deep-layer steering tensor (this is v20's setup) ----
nb = json.load(open(urllib.request.urlretrieve(f"{RAW}/colab_train_v17.ipynb",
                                               "/content/_v17.ipynb")[0]))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
assert body.count("steer5_int8.npz") == 1
body = body.replace('"/content/d/steer5_int8.npz"', '"/content/dlm4_int8.npz"')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os, "json": json,
     "time": time, "math": __import__("math")}
exec(compile(body, "<v17-notebook>", "exec"), G)
print(f"\nsteering: deep-layer mean | availability {G['AVAIL'][:,1].mean():.3f}", flush=True)

DEVICE = G["DEVICE"]; TARGET_SCALE = G["TARGET_SCALE"]
Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
mask = G["mask"]; vpair = G["vpair"]; tmean = G["tmean"]; tstd = G["tstd"]
tr_idx, va_idx, te_idx = G["tr_idx"], G["va_idx"], G["te_idx"]
basins = G["basins"]; z = G["z"]; mirror = G["mirror"]
EPOCHS, PATIENCE, BATCH = G["EPOCHS"], G["PATIENCE"], G["BATCH"]
LR, WEIGHT_DECAY, MIRROR_P = G["LR"], G["WEIGHT_DECAY"], G["MIRROR_P"]

# ---- flow targets, in m/s, and the annulus mask that reads present flow off the patch ----
_lf = np.load("/content/lead_flow.npz")
FLOW_T = _lf["flow"].astype("float32")          # [N,20,2]
FLOW_M = _lf["got"].astype("float32")           # [N,20]
DSC = np.load("/content/dlm4_int8.npz")["scale"][2:4].astype("float32")   # m/s per unit
print(f"flow targets: {FLOW_M.sum():,.0f} supervised lead-points "
      f"({100*FLOW_M.mean():.1f}%) | scale u {DSC[0]:.2f} v {DSC[1]:.2f} m/s", flush=True)

_ii, _jj = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
_d = np.hypot(_ii, _jj) * 2.5                                   # degrees from centre
ANN = torch.tensor(((_d >= 3.0) & (_d <= 8.0)).astype("float32"), device=DEVICE)
print(f"annulus mask: {int(ANN.sum())} of 289 patch cells, 3-8 deg ring "
      f"(same definition the targets were extracted with)", flush=True)


class TrackFormerCoT(Base):
    """v20's network, with the track derived from a predicted steering flow."""

    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.track_q.shape[-1]
        # predicts the CHANGE in steering from now; zero-init so v21 starts as exactly v20
        self.flow_delta = nn.Linear(d, 2)
        nn.init.zeros_(self.flow_delta.weight); nn.init.zeros_(self.flow_delta.bias)
        # measured on 2.86M lead-points: motion = 0.76 x flow_east, 0.91 x flow_north
        self.A = nn.Parameter(torch.tensor([0.76, 0.91]))

    def forward(self, track, vpair, slp):
        b = track.shape[0]
        kin = self.kin_enc(self.kin_proj(track[:, :, G["KIN_COLS"]]) + self.kin_time)
        thermo = self.thermo_enc(self.thermo_proj(track[:, :, G["THERMO_COLS"]]) + self.thermo_time)
        env = self.env_enc(self.env_proj(track[:, :, G["ENV_COLS"]]) + self.env_time)
        if self.training and G["STEER_DROP"] > 0:
            keep = (torch.rand(b, 1, 1, 1, device=slp.device) >= G["STEER_DROP"]).float()
            slp = slp * keep
        st = self.steer_cnn(slp).flatten(2).transpose(1, 2) + self.steer_pos
        tq = (self.track_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_track = self.track_dec(tq, torch.cat([kin, env, st], dim=1))
        h_track = h_track + self.alpha.view(1, self.leads, 1) * \
            self.adapter(thermo.mean(1).detach()).unsqueeze(1)

        # present steering, read off the input patch over the same 3-8 deg ring as the targets
        w = ANN / ANN.sum()
        sc = torch.tensor(DSC, device=slp.device)
        flow_now = (slp[:, 2:4] * w).sum((-2, -1)) * sc                    # [b,2] m/s

        fd = self.flow_delta(h_track)                                      # [b,20,2] m/s, zero at init
        flow_pred = flow_now.unsqueeze(1) + fd                             # supervised

        v0, vp = vpair[:, :2], vpair[:, 2:]
        s0 = torch.linalg.norm(v0, dim=-1)
        phi0 = torch.atan2(v0[:, 1], v0[:, 0]); phip = torch.atan2(vp[:, 1], vp[:, 0])
        om = torch.remainder(phi0 - phip + __import__("math").pi, 2 * __import__("math").pi) \
            - __import__("math").pi
        phil = phi0.unsqueeze(1) + self.gturn.view(1, self.leads) * om.unsqueeze(1)
        speed = self.rho.view(1, self.leads) * s0.unsqueeze(1)
        base = torch.stack([speed * torch.cos(phil), speed * torch.sin(phil)], -1) / 100.0
        # the flow DELTA moves the storm off persistence; at init this term is exactly zero
        motion = base + (self.A.view(1, 1, 2) * fd) * KM6H / 100.0 + self.track_res(h_track)

        iq = (self.int_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_int = self.int_dec(iq, torch.cat([thermo, env, kin.detach()], dim=1))
        istate = self.int_state(h_int); ilog = self.int_logscale(h_int).clamp(-5.0, 3.0)
        return (torch.cat([motion, istate], -1),
                torch.cat([torch.zeros_like(motion), ilog], -1), flow_pred)


# ---- the generative forward must reduce to v20's when the flow path is silent ----
with torch.no_grad():
    _a, _b = Base().to(DEVICE).eval(), TrackFormerCoT().to(DEVICE).eval()
    _b.load_state_dict(_a.state_dict(), strict=False)
    _t = torch.from_numpy(track[:4]).to(DEVICE); _v = torch.from_numpy(vpair[:4]).to(DEVICE)
    _s = torch.from_numpy(SLP[:4]).to(DEVICE)
    _d1 = float((_a(_t, _v, _s)[0] - _b(_t, _v, _s)[0]).abs().max())
    assert _d1 < 1e-5, f"v21 does not reduce to v20 at init: max diff {_d1}"
    print(f"init check: max|v21 - v20| = {_d1:.2e} -- v21 starts as exactly v20", flush=True)
del _a, _b


class DS(torch.utils.data.Dataset):
    def __init__(self, idx, aug): self.idx = np.asarray(idx); self.aug = aug
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i])
        tr = torch.from_numpy(track[j]); tg = torch.from_numpy(target[j])
        mk = torch.from_numpy(mask[j]); sp = torch.from_numpy(SLP[j])
        vp = torch.from_numpy(vpair[j])
        fl = torch.from_numpy(FLOW_T[j].copy()); fm = torch.from_numpy(FLOW_M[j].copy())
        if self.aug and torch.rand(()) < MIRROR_P:
            tr, tg, mk, sp = mirror(tr, tg, mk, sp)
            vp = vp.clone(); vp[1] = -vp[1]; vp[3] = -vp[3]
            fl = fl.clone(); fl[:, 1] = -fl[:, 1]      # northward flow negates under the N-S mirror
        return tr, vp, sp, tg, mk, fl, fm


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
    model = TrackFormerCoT().to(DEVICE)
    print(f"seed {seed} | params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True, aug=True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0.0; fa = 0.0
        for tr, v0, sp, tg, m, fl, fm in ld:
            tr, v0, sp, tg, m, fl, fm = [x.to(DEVICE, non_blocking=True)
                                         for x in (tr, v0, sp, tg, m, fl, fm)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls, fp = model(tr, v0, sp)
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
        with torch.no_grad(): vv, vf = run(vl, False)
        sched.step()
        if vv < best:
            best, bad = vv, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "best_val": best,
                        "track_mean": tmean, "track_std": tstd}, ckpt)
        else:
            bad += 1
        # |A| and the flow-delta magnitude say whether the CoT path is actually being used
        with torch.no_grad():
            amag = model.A.detach().cpu().numpy()
            fdw = float(model.flow_delta.weight.abs().mean())
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"flow {vf:.3f} | A {amag[0]:.2f},{amag[1]:.2f} | |dW| {fdw:.4f} | "
              f"{time.time()-te:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


CK = [train_one(s, f"/content/v21_seed{s}.pt") for s in range(N_SEEDS)]
print(f"v21 trained: {len(CK)} seeds", flush=True)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
SC = TARGET_SCALE


@torch.no_grad()
def track_err(ms):
    P = []
    for i in range(0, len(wpep), 128):
        j = wpep[i:i + 128]
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE)]
        s = torch.stack([m(*a)[0] for m in ms]).mean(0)
        P.append((s * SC).float().cpu().numpy())
    C = np.cumsum(np.concatenate(P)[..., :2], 1)
    T = np.cumsum(target[wpep][..., :2], 1)
    return float(np.sqrt(((C - T) ** 2).sum(-1)).mean())


def load_m(c):
    m = TrackFormerCoT().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); return m


MS = [load_m(c) for c in CK]
print(f"\nWP+EP 2020+, {len(wpep)} windows")
print("  BASELINES   v10 549.3 | v17 462.8 | v20 452.5 (the bar)")
for i, c in enumerate(CK):
    print(f"  v21 seed{i}  {track_err([load_m(c)]):.2f} km", flush=True)
e = track_err(MS)
print(f"\n  v21 ENSEMBLE ({len(MS)} seeds)  {e:.2f} km")
print(f"  vs v20 452.5: {e - 452.47:+.2f} km   vs v17 462.8: {e - 462.8:+.2f} km", flush=True)
json.dump({"v21": e}, open("/content/v21.json", "w"))

try:
    from google.colab import files
    import subprocess
    subprocess.run("tar cf /content/v21_seeds.tar /content/v21_seed*.pt", shell=True)
    files.download("/content/v21.json"); files.download("/content/v21_seeds.tar")
except Exception as ex:
    print("(auto-download unavailable:", ex, ")", flush=True)
