"""TrackFormer v9 — protected TRIPLE-stream (kinematic + thermodynamic + ENVIRONMENT), MPS, IBTrACS-only.

Adds an environmental stream from IBTrACS-derived features (absolute lat, |lat|, lon sin/cos,
distance-to-land, and a lat+month climatological SST proxy) -- the stand-in for an ERA5 steering
stream without any download. v8 was position-blind (translation-invariant); absolute latitude and
distance-to-land are the biggest missing signals (recurvature/Coriolis/land interaction), and the
SST proxy feeds the intensity head.

Routing (protected):
  * kinematic encoder  -> track decoder            [track gradients only; intensity sees stopgrad(kin)]
  * thermodynamic enc  -> intensity decoder        [intensity gradients; thermo->track via detached gated adapter]
  * environmental enc  -> BOTH decoders            [position informs track steering-climatology; SST informs intensity]
Persistence-residual track head + zero-init gated thermo->track adapter kept from v8.
"""
import math, json, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NPZ = os.environ.get("TRACK_NPZ", "track_build/track_windows_v9.npz")
CKPT = os.environ.get("TRACK_CKPT", "track_build/track_v9_best.pt")
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
TARGET_SCALE = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12, device=DEVICE)
EPOCHS = int(os.environ.get("EPOCHS", "60")); PATIENCE = int(os.environ.get("PATIENCE", "12"))
BATCH = int(os.environ.get("BATCH", "1024")); LR = 3e-4; WEIGHT_DECAY = 3e-2
K_MODES = int(os.environ.get("K_MODES", "5"))   # candidate trajectories (winner-take-all)

KIN_COLS = [0, 1, 2, 3, 21, 22, 23, 40, 41, 42, 43]
THERMO_COLS = [4, 5, 6, 7] + list(range(8, 20)) + list(range(24, 40)) + [44, 45, 46, 47]
ENV_COLS = [48, 49, 50, 51, 52, 53]
KIN_DIM, THERMO_DIM, ENV_DIM = len(KIN_COLS), len(THERMO_COLS), len(ENV_COLS)

print("device:", DEVICE, "| loading", NPZ)
z = np.load(NPZ, allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
mask = z["target_mask"].astype("float32"); years = z["year"].astype(int)
sids = z["storm_id"].astype(str); basins = z["basin"].astype(str)
tmean = z["track_mean"].astype("float32"); tstd = z["track_std"].astype("float32")
v0_raw = track[:, -1, 2:4] * tstd[2:4] + tmean[2:4]                 # current 6h velocity (km)
vp_raw = track[:, -2, 2:4] * tstd[2:4] + tmean[2:4]                 # previous 6h velocity (km)
vpair = np.concatenate([v0_raw, vp_raw], axis=1).astype("float32")  # [N,4] -> curved baseline needs the turn
print(f"windows {len(track)} | dims kin {KIN_DIM} thermo {THERMO_DIM} env {ENV_DIM} | curved baseline (v0,vprev)")

fy = {s: int(years[sids == s].min()) for s in np.unique(sids)}
tr_idx = np.where(np.isin(sids, [s for s, y in fy.items() if y <= 2015]))[0]
va_idx = np.where(np.isin(sids, [s for s, y in fy.items() if 2016 <= y <= 2019]))[0]
te_idx = np.where(np.isin(sids, [s for s, y in fy.items() if y >= 2020]))[0]
print(f"split train={len(tr_idx)} valid={len(va_idx)} test={len(te_idx)}")


class DS(Dataset):
    def __init__(self, idx): self.idx = np.asarray(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i])
        return (torch.from_numpy(track[j]), torch.from_numpy(vpair[j]),
                torch.from_numpy(target[j]), torch.from_numpy(mask[j]))


def loader(idx, sh): return DataLoader(DS(idx), batch_size=BATCH, shuffle=sh, num_workers=0, pin_memory=False)


def sinusoidal(length, d):
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    e = torch.zeros(length, d); e[:, 0::2] = torch.sin(pos * div); e[:, 1::2] = torch.cos(pos * div)
    return e


def enc(d, h, ffn, depth, dr):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, ffn, dr, batch_first=True, norm_first=True, activation="gelu"), depth)
def dec(d, h, ffn, depth, dr):
    return nn.TransformerDecoder(nn.TransformerDecoderLayer(d, h, ffn, dr, batch_first=True, norm_first=True, activation="gelu"), depth)


class TrackFormerV9(nn.Module):
    def __init__(self, d=256, heads=8, ffn=1024, leads=20, dropout=0.2):
        super().__init__(); self.leads = leads
        self.kin_proj = nn.Linear(KIN_DIM, d); self.thermo_proj = nn.Linear(THERMO_DIM, d); self.env_proj = nn.Linear(ENV_DIM, d)
        self.kin_time = nn.Parameter(torch.zeros(1, 9, d)); self.thermo_time = nn.Parameter(torch.zeros(1, 9, d)); self.env_time = nn.Parameter(torch.zeros(1, 9, d))
        self.kin_enc = enc(d, heads, ffn, 4, dropout); self.thermo_enc = enc(d, heads, ffn, 3, dropout); self.env_enc = enc(d, heads, ffn, 2, dropout)
        self.track_dec = dec(d, heads, ffn, 4, dropout); self.int_dec = dec(d, heads, ffn, 5, dropout)
        self.track_q = nn.Parameter(torch.randn(1, leads, d) * 0.02); self.int_q = nn.Parameter(torch.randn(1, leads, d) * 0.02)
        self.register_buffer("qpos", sinusoidal(leads, d))
        self.adapter = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d))
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(leads)); self.rho = nn.Parameter(torch.ones(leads))
        self.gturn = nn.Parameter(torch.zeros(leads))   # per-lead turn-rate extrapolation; 0 == v9 straight baseline
        # MULTI-MODAL track head: K candidate trajectories + a mode-probability head.
        # A single head must predict the MEAN of possible futures (too slow, never turns);
        # K heads trained winner-take-all let each mode specialise on a plausible future.
        self.K = K_MODES
        self.track_res = nn.Linear(d, 2 * self.K); nn.init.zeros_(self.track_res.weight); nn.init.zeros_(self.track_res.bias)
        self.mode_head = nn.Linear(d, self.K)
        self.int_state = nn.Linear(d, 15); self.int_logscale = nn.Linear(d, 15)

    def forward(self, track, vpair):
        b = track.shape[0]
        kin = self.kin_enc(self.kin_proj(track[:, :, KIN_COLS]) + self.kin_time)
        thermo = self.thermo_enc(self.thermo_proj(track[:, :, THERMO_COLS]) + self.thermo_time)
        env = self.env_enc(self.env_proj(track[:, :, ENV_COLS]) + self.env_time)
        # track path: kinematic + environmental memory
        tq = (self.track_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_track = self.track_dec(tq, torch.cat([kin, env], dim=1))
        h_track = h_track + self.alpha.view(1, self.leads, 1) * self.adapter(thermo.mean(1).detach()).unsqueeze(1)
        # CURVED baseline: extrapolate the current (damped) turn rate instead of a straight line.
        v0, vp = vpair[:, :2], vpair[:, 2:]
        s0 = v0.norm(dim=1, keepdim=True).clamp(min=1e-3)                 # current speed [b,1]
        phi0 = torch.atan2(v0[:, 1], v0[:, 0])                            # current heading [b]
        dphi = phi0 - torch.atan2(vp[:, 1], vp[:, 0])
        omega = torch.atan2(torch.sin(dphi), torch.cos(dphi))            # wrapped turn rate / step [b]
        phil = phi0.unsqueeze(1) + self.gturn.view(1, self.leads) * omega.unsqueeze(1)   # heading per lead [b,L]
        speed = self.rho.view(1, self.leads) * s0                        # damped speed [b,L]
        base = torch.stack([speed * torch.cos(phil), speed * torch.sin(phil)], dim=-1) / 100.0  # [b,L,2]
        res = self.track_res(h_track).view(b, self.leads, self.K, 2)      # [b,L,K,2]
        motion = base.unsqueeze(2) + res                                   # K candidate motion paths
        mode_logits = self.mode_head(h_track.mean(1))                      # [b,K]
        # intensity path: thermo + env + stopgrad(kinematic)
        iq = (self.int_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_int = self.int_dec(iq, torch.cat([thermo, env, kin.detach()], dim=1))
        istate = self.int_state(h_int); ilog = self.int_logscale(h_int).clamp(-5.0, 3.0)
        return motion, mode_logits, istate, ilog        # motion: [b,L,K,2]


LEADW = torch.sqrt(torch.arange(1, 21, device=DEVICE).float()); LEADW = LEADW / LEADW.mean()
def wta_track_loss(motion, mode_logits, tn, m):
    """Winner-take-all: score each of the K candidate paths, train only the closest to truth,
    and teach the mode head to rank them. This lets modes specialise instead of averaging."""
    tm, mm = tn[..., :2], m[..., 0]                                   # [b,L,2], [b,L]
    pos_k = torch.cumsum(motion, dim=1)                                # [b,L,K,2]
    pos_t = torch.cumsum(tm, dim=1).unsqueeze(2)                       # [b,L,1,2]
    d_k = torch.sqrt(((pos_k - pos_t) ** 2).sum(-1) + 1e-8)            # [b,L,K] displacement error
    ade_k = (d_k * mm.unsqueeze(-1)).sum(1) / mm.sum(1, keepdim=True).clamp(min=1)   # [b,K]
    best = ade_k.argmin(dim=1)                                         # [b] oracle-best mode
    idx = torch.arange(motion.shape[0], device=motion.device)
    mb = motion[idx, :, best]                                          # [b,L,2] winner path
    mm2 = m[..., :2]
    step = (F.smooth_l1_loss(mb, tm, reduction="none") * mm2 * LEADW.view(1, 20, 1)).sum() / mm2.sum().clamp(min=1)
    pos = (F.smooth_l1_loss(torch.cumsum(mb, 1), torch.cumsum(tm, 1), reduction="none") * mm2).sum() / mm2.sum().clamp(min=1)
    cls = F.cross_entropy(mode_logits, best.detach())                  # rank the modes
    # diversity: discourage mode collapse (push candidate endpoints apart)
    ends = pos_k[:, -1]                                                # [b,K,2]
    pd = torch.cdist(ends, ends).mean()
    return step + pos + 0.2 * cls - 0.02 * torch.log1p(pd)
def int_loss(s, logs, tn, m):
    ps, ts, ms = s, tn[..., 2:], m[..., 2:]
    huber = (F.smooth_l1_loss(ps, ts, reduction="none") * ms).sum() / ms.sum().clamp(min=1)
    nll = ((0.5 * ((ts - ps) * torch.exp(-logs)) ** 2 + logs) * ms).sum() / ms.sum().clamp(min=1)
    r34, r50, r64 = ps[..., 3:7], ps[..., 7:11], ps[..., 11:15]
    return 0.7 * huber + 0.3 * nll + 0.01 * (F.relu(r50 - r34).mean() + F.relu(r64 - r50).mean())
def total_loss(motion, mode_logits, istate, ilog, tgt, m):
    tn = tgt / TARGET_SCALE
    return wta_track_loss(motion, mode_logits, tn, m) + int_loss(istate, ilog, tn, m)


model = TrackFormerV9().to(DEVICE)
print(f"TrackFormerV9 params: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
tl, vl = loader(tr_idx, True), loader(va_idx, False)


def run(ld, train):
    model.train(train); tot, cnt = 0.0, 0
    for tr, v0, tg, m in ld:
        tr, v0, tg, m = tr.to(DEVICE), v0.to(DEVICE), tg.to(DEVICE), m.to(DEVICE)
        with torch.set_grad_enabled(train):
            mo, ml, ist, ilg = model(tr, v0); loss = total_loss(mo, ml, ist, ilg, tg, m)
        if train:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += float(loss) * len(tr); cnt += len(tr)
    return tot / max(1, cnt)


os.makedirs(os.path.dirname(CKPT), exist_ok=True)
best, bad, t0 = float("inf"), 0, time.time()
for ep in range(EPOCHS):
    te = time.time(); trl = run(tl, True); val = run(vl, False); sched.step()
    if val < best:
        best, bad = val, 0
        torch.save({"model": model.state_dict(), "epoch": ep, "best_val": best,
                    "kin_cols": KIN_COLS, "thermo_cols": THERMO_COLS, "env_cols": ENV_COLS,
                    "track_mean": tmean, "track_std": tstd}, CKPT)
    else:
        bad += 1
    if ep % 2 == 0 or bad == 0:
        print(f"epoch {ep:03d} | train {trl:.5f} | val {val:.5f} | best {best:.5f} | "
              f"alpha|max| {model.alpha.abs().max().item():.3f} | {time.time()-te:.1f}s", flush=True)
    if bad >= PATIENCE:
        print("early stopping at epoch", ep); break
print(f"trained in {(time.time()-t0)/60:.1f} min; best_val {best:.5f}")


@torch.no_grad()
def metrics(idx):
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False)["model"]); model.eval()
    MO, PR, IS, T, M = [], [], [], [], []
    for tr, v0, tg, m in loader(idx, False):
        mo, ml, ist, _ = model(tr.to(DEVICE), v0.to(DEVICE))
        MO.append((mo * 100.0).float().cpu().numpy())                       # motion km [b,L,K,2]
        PR.append(torch.softmax(ml, -1).float().cpu().numpy())              # mode probs [b,K]
        IS.append((ist * TARGET_SCALE[2:]).float().cpu().numpy())
        T.append(tg.numpy()); M.append(m.numpy())
    MO, PR, IS, T, M = (np.concatenate(x) for x in (MO, PR, IS, T, M))
    out = {}
    pos_k = np.cumsum(MO, 1)                                                # [N,L,K,2]
    pos_t = np.cumsum(T[..., :2], 1)[:, :, None, :]
    d_k = np.sqrt(((pos_k - pos_t) ** 2).sum(-1))                           # [N,L,K]
    top = PR.argmax(1); ar = np.arange(len(MO))
    out["track_top1_km"] = round(float(d_k[ar, :, top].mean()), 2)          # most-likely mode
    ade = d_k.mean(1); out["track_bestofK_km"] = round(float(d_k[ar, :, ade.argmin(1)].mean()), 2)
    mmean = (MO * PR[:, None, :, None]).sum(2)                              # prob-weighted mean path
    out["track_mean_km"] = round(float(np.sqrt(((np.cumsum(mmean, 1) - np.cumsum(T[..., :2], 1)) ** 2).sum(-1)).mean()), 2)
    # speed / direction of the TOP-1 mode (is the winner physically realistic?)
    mt = MO[ar, :, top]; ps = np.hypot(mt[..., 0], mt[..., 1]); ts = np.hypot(T[..., 0], T[..., 1]); sm = M[..., 0] > 0.5
    out["speed_bias_km6h"] = round(float((ps - ts)[sm].mean()), 2)
    out["speed_mae_km6h"] = round(float(np.abs(ps - ts)[sm].mean()), 2)
    cos = (mt[..., 0] * T[..., 0] + mt[..., 1] * T[..., 1]) / (ps * ts + 1e-6)
    out["dir_mae_deg"] = round(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))[sm].mean()), 2)
    for j, i, nm in [(0, 2, "vmax_mae_kt"), (1, 3, "pressure_mae_hpa"), (2, 4, "rmw_mae_km")]:
        v = M[..., i] > 0.5; out[nm] = round(float(np.abs(IS[..., j][v] - T[..., i][v]).mean()), 2)
    rm = M[..., 5:9] > 0.5; out["R34_lvl7_mae_km"] = round(float(np.abs(IS[..., 3:7] - T[..., 5:9])[rm].mean()), 2)
    return out


full = z["n_leads"].astype(int) == 20
print("\nTrackFormer v9 (triple-stream + env). v8 baseline WP: track 649/vmax 20.7/pres 15.9/rmw 11.3/radius 27.8; "
      "all-basin track 580")
wp = np.array([i for i in te_idx if full[i] and basins[i] == "WP"]); ab = np.array([i for i in te_idx if full[i]])
print("Test WP-2020+ (full):", json.dumps(metrics(wp)))
print("Test all-basin (full):", json.dumps(metrics(ab)))
