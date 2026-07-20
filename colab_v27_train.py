"""v22 on Colab — v21 plus LATENT chain-of-thought (recurrent feedback tokens).

    !pip install -q netCDF4
    !wget -q -O v27t.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v27_train.py
    exec(open('v27t.py').read())

TWO KINDS OF CHAIN-OF-THOUGHT, STACKED.

v21 added an EXPLICIT intermediate: predict the steering flow, derive the track from it. That works
-- the control arm which supervised the same flow head but did not feed it into the motion scored
452.91 km, identical to v20's 452.47, while v21 scored 443.62. So the -9.29 km belongs to the
STRUCTURE, not to multi-task regularisation.

v22 adds a LATENT intermediate on top, following Dudley & Oymak (arXiv:2605.11262). After a forward
pass the query-position hidden states are compressed by a two-layer MLP into feedback tokens, which
are appended to the decoder memory and the decoder is re-run. Weights are tied across rounds; loss
is applied to the final round only. R=2 rounds, which is that paper's optimum for time series.

WHY IT SUITS THIS PROBLEM SPECIFICALLY. Their Table 1 reports that on time-series data a
depth-matched deeper transformer (2x layers, ~2x parameters) makes things WORSE by 5.45%, which the
authors attribute to overfitting on small datasets, while latent CoT improves by 12.63% -- because
weight reuse adds effective depth without adding parameters, and the reuse itself regularises. v17
overfits 1.29x train->test, so that is our regime exactly. It also beats a weight-tied looped
baseline by 7.74%, which is the evidence that the feedback TOKENS matter and not merely recurrence.

Caveat carried from the paper: the authors explicitly do NOT characterise how latent CoT interacts
with distribution shift, and training on pre-2020 storms while testing on 2020+ is precisely that.

DEGRADES TO v21 AT INIT, BY CONSTRUCTION. The refined states are blended through a gate,
    h_out = h_0 + g * (h_R - h_0),   g = tanh(g0),   g0 initialised to 0
so at step zero g = 0, h_out = h_0, and the forward pass is v21's exactly -- asserted below. The
paper used g0 = 2.0; starting at 0 instead means any movement is something the recurrence earned.
|g| is the diagnostic: if it stays near zero the latent rounds are being ignored, which is a real
negative result and should be reported as one.

Parameters added: the feedback MLP, about 131k on 12.9M (~1%). Compute: the track decoder runs
R+1 = 3 times instead of once, so expect roughly 1.5-2x the epoch time.
"""

import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
N_SEEDS = int(os.environ.get("V27_SEEDS", "3"))
R_ROUNDS = int(os.environ.get("V27_ROUNDS", "2"))   # the paper's optimum for time series
W_FLOW = float(os.environ.get("W_FLOW", "0.3"))     # weight on the auxiliary flow loss
# ABLATION SWITCH. v21 changes two things at once: it supervises a flow head (multi-task
# regularisation) AND derives the track from that flow (the chain-of-thought structure). A km gain
# could come from either. USE_FLOW=0 keeps the supervision but cuts the flow out of the motion, so
# the two effects can be separated:
#   v20            no flow head at all                      452.47 km
#   USE_FLOW=0     flow supervised, NOT used for motion      <- isolates the auxiliary regulariser
#   USE_FLOW=1     flow supervised AND drives motion         <- adds the CoT structure
# CoT only earns credit for the gap between the last two.
USE_FLOW = int(os.environ.get("V27_USE_FLOW", "1"))
TAG = os.environ.get("V27_TAG", "v22" if USE_FLOW else "v22aux")
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
        # LATENT CoT: compress query-position states into feedback tokens, append, re-run.
        # Two-layer MLP with GELU, per Dudley & Oymak. Weight-tied across rounds by construction --
        # there is one phi and one track_dec, used R+1 times.
        self.phi = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, d))
        # gate init 0 => tanh(0) = 0 => v22 is exactly v21 at step zero
        self.g0 = nn.Parameter(torch.zeros(1))

    def forward(self, track, vpair, slp):
        # v17's forward VERBATIM, with only the flow lines inserted. Written from the notebook
        # rather than from memory: the first attempt silently omitted st.detach() from the
        # intensity decoder, dropped a clamp on s0 and added one on ilog, and the init assertion
        # caught it at 0.65 max diff.
        b = track.shape[0]
        KIN_COLS, THERMO_COLS, ENV_COLS = G["KIN_COLS"], G["THERMO_COLS"], G["ENV_COLS"]
        STEER_DROP = G["STEER_DROP"]
        kin = self.kin_enc(self.kin_proj(track[:, :, KIN_COLS]) + self.kin_time)
        thermo = self.thermo_enc(self.thermo_proj(track[:, :, THERMO_COLS]) + self.thermo_time)
        env = self.env_enc(self.env_proj(track[:, :, ENV_COLS]) + self.env_time)
        if self.training and STEER_DROP > 0:
            keep = (torch.rand(b, 1, 1, 1, device=slp.device) >= STEER_DROP).float()
            slp = slp * keep
        st = self.steer_cnn(slp).flatten(2).transpose(1, 2) + self.steer_pos
        tq = (self.track_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        mem = torch.cat([kin, env, st], dim=1)
        h_track = self.track_dec(tq, mem)
        # ---- LATENT CoT: R rounds of feedback tokens, weight-tied, loss on the final round only ----
        if R_ROUNDS > 0:
            h0 = h_track
            h = h_track
            fb = []
            for _ in range(R_ROUNDS):
                fb.append(self.phi(h))                       # one feedback token per query position
                h = self.track_dec(tq, torch.cat([mem] + fb, dim=1))
            # gated residual: at init g = 0 so this is a no-op and the model is exactly v21
            g = torch.tanh(self.g0)
            h_track = h0 + g * (h - h0)
        h_track = h_track + self.alpha.view(1, self.leads, 1) * self.adapter(thermo.mean(1).detach()).unsqueeze(1)

        # ---- the only addition: predicted steering, and the track derived from it ----
        # present steering read off the input patch over the same 3-8 deg ring as the targets.
        # NOTE this uses the possibly steer-dropped slp, which is correct: if the field is
        # unavailable the model should see zero present flow, exactly as the target is zeroed.
        w = ANN / ANN.sum()
        sc = torch.as_tensor(DSC, device=slp.device, dtype=slp.dtype)
        flow_now = (slp[:, 2:4] * w).sum((-2, -1)) * sc                    # [b,2] m/s
        fd = self.flow_delta(h_track)                                      # [b,20,2], zero at init
        flow_pred = flow_now.unsqueeze(1) + fd                             # supervised

        v0, vp = vpair[:, :2], vpair[:, 2:]
        s0 = v0.norm(dim=1, keepdim=True).clamp(min=1e-3)
        phi0 = torch.atan2(v0[:, 1], v0[:, 0])
        dphi = phi0 - torch.atan2(vp[:, 1], vp[:, 0])
        omega = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        phil = phi0.unsqueeze(1) + self.gturn.view(1, self.leads) * omega.unsqueeze(1)
        speed = self.rho.view(1, self.leads) * s0
        base = torch.stack([speed * torch.cos(phil), speed * torch.sin(phil)], dim=-1) / 100.0
        motion = base + self.track_res(h_track)
        if USE_FLOW:
            motion = motion + (self.A.view(1, 1, 2) * fd) * KM6H / 100.0
        iq = (self.int_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_int = self.int_dec(iq, torch.cat([thermo, env, kin.detach(), st.detach()], dim=1))
        istate = self.int_state(h_int); ilog = self.int_logscale(h_int)
        return (torch.cat([motion, istate], -1),
                torch.cat([torch.zeros_like(motion), ilog], -1), flow_pred)


# ---- init checks ----
# v17 zero-inits track_res, so at true init the track output does not depend on h_track AT ALL.
# That makes a bare "reduces to v20" assertion vacuous: it passes even if the CoT recurrence is
# broken or dead. v21 shipped with exactly that vacuous check. So: make track_res live first,
# then assert BOTH that the gate closed is a no-op AND that the gate open actually moves the
# output. The second half is what proves the feedback path is wired to the track.
with torch.no_grad():
    _t = torch.from_numpy(track[:4]).to(DEVICE); _v = torch.from_numpy(vpair[:4]).to(DEVICE)
    _s = torch.from_numpy(SLP[:4]).to(DEVICE)
    _a, _b = Base().to(DEVICE).eval(), TrackFormerCoT().to(DEVICE).eval()
    torch.manual_seed(7)
    nn.init.normal_(_a.track_res.weight, std=0.02); nn.init.normal_(_a.track_res.bias, std=0.02)
    _b.load_state_dict(_a.state_dict(), strict=False)   # flow_delta/phi/g0 keep their own init
    assert float(_b.flow_delta.weight.abs().max()) == 0.0, "flow head is not zero-init"
    assert float(_b.g0.abs().max()) == 0.0, "CoT gate is not zero-init"

    _d1 = float((_a(_t, _v, _s)[0] - _b(_t, _v, _s)[0]).abs().max())
    assert _d1 < 1e-5, f"{TAG} does not reduce to v20 at init: max diff {_d1}"

    _b.g0.data.fill_(2.0)                              # g = tanh(2) = 0.964
    _d2 = float((_a(_t, _v, _s)[0] - _b(_t, _v, _s)[0]).abs().max())
    _b.g0.data.zero_()
    assert R_ROUNDS == 0 or _d2 > 1e-4, (
        f"{TAG}: opening the CoT gate changed the track by {_d2:.2e} -- the latent path is DEAD, "
        "the feedback tokens are not reaching the track output")
    print(f"init check: gate closed max|v22 - v20| = {_d1:.2e} (v22 starts as exactly v20)", flush=True)
    print(f"init check: gate open   max|v22 - v20| = {_d2:.2e} (latent CoT path is live)", flush=True)
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
            gate = float(torch.tanh(model.g0).item())
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"flow {vf:.3f} | A {amag[0]:.2f},{amag[1]:.2f} | |dW| {fdw:.4f} | "
              f"g {gate:+.4f} | {time.time()-te:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


# reuse any checkpoint already on this VM, so extending 3 seeds -> 5 costs two runs, not five
CK = []
for _s in range(N_SEEDS):
    _c = f"/content/{TAG}_seed{_s}.pt"
    if os.path.exists(_c):
        print(f"seed {_s}: checkpoint already present, reusing", flush=True)
    else:
        train_one(_s, _c)
    CK.append(_c)
print(f"{TAG} trained: {len(CK)} seeds (USE_FLOW={USE_FLOW})", flush=True)

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
print("  BASELINES   v10 549.3 | v17 462.8 | v20 452.5 | v21 443.6 (the bar)")
for i, c in enumerate(CK):
    print(f"  {TAG} seed{i}  {track_err([load_m(c)]):.2f} km", flush=True)
e = track_err(MS)
print(f"\n  {TAG} ENSEMBLE ({len(MS)} seeds)  {e:.2f} km")
print(f"  vs v21 443.6: {e - 443.62:+.2f} km   vs v20 452.5: {e - 452.47:+.2f} km", flush=True)
json.dump({TAG: e, "use_flow": USE_FLOW}, open(f"/content/{TAG}.json", "w"))

try:
    from google.colab import files
    import subprocess
    subprocess.run(f"tar cf /content/{TAG}_seeds.tar /content/{TAG}_seed*.pt", shell=True)
    files.download(f"/content/{TAG}.json"); files.download(f"/content/{TAG}_seeds.tar")
except Exception as ex:
    print("(auto-download unavailable:", ex, ")", flush=True)
