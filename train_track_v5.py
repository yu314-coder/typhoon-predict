"""TrackFormer v5 — v3 protected dual-stream + EMA weight-averaging (MPS, IBTrACS-only).

This is the deployable presentation model: it needs ONLY IBTrACS best-track (a fast CSV), no ERA5.
Architecture is identical to v3 (protected dual-stream: separate kinematic/thermodynamic encoders,
gradient routing, zero-init gated thermo->track adapter, persistence-residual track head). The only
change is EMA (exponential moving average, decay 0.999) of the weights: validation + the saved
checkpoint use the EMA parameters, a near-free generalization boost that reduces the overfitting v3
showed (it early-stopped at epoch 26 with train loss far below val).
"""
import math, json, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NPZ = os.environ.get("TRACK_NPZ", "track_build/track_windows_v2.npz")
CKPT = os.environ.get("TRACK_CKPT", "track_build/track_v5_best.pt")
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
TARGET_SCALE = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12, device=DEVICE)

EPOCHS = int(os.environ.get("EPOCHS", "160"))
PATIENCE = int(os.environ.get("PATIENCE", "26"))
BATCH = int(os.environ.get("BATCH", "512"))
LR = 3e-4
WEIGHT_DECAY = 3e-2
K_BASIS = int(os.environ.get("K_BASIS", "8"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.999"))

KIN_COLS = [0, 1, 2, 3, 21, 22, 23, 40, 41, 42, 43]
THERMO_COLS = [4, 5, 6, 7] + list(range(8, 20)) + list(range(24, 40)) + [44, 45, 46, 47]
KIN_DIM, THERMO_DIM = len(KIN_COLS), len(THERMO_COLS)

print("device:", DEVICE, "| loading", NPZ)
z = np.load(NPZ, allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
mask = z["target_mask"].astype("float32"); years = z["year"].astype(int)
sids = z["storm_id"].astype(str); basins = z["basin"].astype(str)
tmean = z["track_mean"].astype("float32"); tstd = z["track_std"].astype("float32")
v0_raw = track[:, -1, 2:4] * tstd[2:4] + tmean[2:4]
print(f"windows {len(track)} | kin {KIN_DIM} thermo {THERMO_DIM} | K_basis {K_BASIS} ema {EMA_DECAY}")

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
        return (torch.from_numpy(track[j]), torch.from_numpy(v0_raw[j]),
                torch.from_numpy(target[j]), torch.from_numpy(mask[j]))


def loader(idx, shuffle):
    return DataLoader(DS(idx), batch_size=BATCH, shuffle=shuffle, num_workers=0, pin_memory=False)


def sinusoidal(length, d):
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    e = torch.zeros(length, d); e[:, 0::2] = torch.sin(pos * div); e[:, 1::2] = torch.cos(pos * div)
    return e


def dct_basis(L, K):
    l = torch.arange(L).float().unsqueeze(1); k = torch.arange(K).float().unsqueeze(0)
    B = torch.cos(math.pi * (l + 0.5) * k / L)     # [L,K] DCT-II basis
    return B


def encoder(d, h, ffn, depth, drop):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, ffn, drop, batch_first=True, norm_first=True, activation="gelu"), depth)


def decoder(d, h, ffn, depth, drop):
    return nn.TransformerDecoder(nn.TransformerDecoderLayer(d, h, ffn, drop, batch_first=True, norm_first=True, activation="gelu"), depth)


class TrackFormerV4(nn.Module):  # name kept for the EMA/eval code below; this is the v3 architecture
    def __init__(self, d=256, heads=8, ffn=1024, leads=20, dropout=0.2, K=K_BASIS):
        super().__init__()
        self.leads = leads
        self.kin_proj = nn.Linear(KIN_DIM, d); self.thermo_proj = nn.Linear(THERMO_DIM, d)
        self.kin_time = nn.Parameter(torch.zeros(1, 9, d)); self.thermo_time = nn.Parameter(torch.zeros(1, 9, d))
        self.kin_enc = encoder(d, heads, ffn, 4, dropout); self.thermo_enc = encoder(d, heads, ffn, 3, dropout)
        self.track_dec = decoder(d, heads, ffn, 4, dropout); self.int_dec = decoder(d, heads, ffn, 5, dropout)
        self.track_q = nn.Parameter(torch.randn(1, leads, d) * 0.02); self.int_q = nn.Parameter(torch.randn(1, leads, d) * 0.02)
        self.register_buffer("qpos", sinusoidal(leads, d))
        self.adapter = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d))
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(leads))
        self.rho = nn.Parameter(torch.ones(leads))
        # v3 direct per-step residual head (zero-init -> starts as pure damped persistence)
        self.track_res = nn.Linear(d, 2); nn.init.zeros_(self.track_res.weight); nn.init.zeros_(self.track_res.bias)
        self.int_state = nn.Linear(d, 15); self.int_logscale = nn.Linear(d, 15)

    def forward(self, track, v0_raw):
        b = track.shape[0]
        kin = self.kin_enc(self.kin_proj(track[:, :, KIN_COLS]) + self.kin_time)
        thermo = self.thermo_enc(self.thermo_proj(track[:, :, THERMO_COLS]) + self.thermo_time)
        thermo_ctx = thermo.mean(1)
        tq = (self.track_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_track = self.track_dec(tq, kin)
        h_track = h_track + self.alpha.view(1, self.leads, 1) * self.adapter(thermo_ctx.detach()).unsqueeze(1)
        res = self.track_res(h_track)                                     # [b,L,2] normalized residual
        base = (self.rho.view(1, self.leads, 1) * v0_raw.unsqueeze(1)) / 100.0
        motion = base + res
        iq = (self.int_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_int = self.int_dec(iq, torch.cat([thermo, kin.detach()], dim=1))
        istate = self.int_state(h_int); ilog = self.int_logscale(h_int).clamp(-5.0, 3.0)
        state = torch.cat([motion, istate], dim=-1)
        logscale = torch.cat([torch.zeros_like(motion), ilog], dim=-1)
        return state, logscale


LEADW = torch.sqrt(torch.arange(1, 21, device=DEVICE).float()); LEADW = LEADW / LEADW.mean()


def track_loss(state, tn, m):
    pm, tm, mm = state[..., :2], tn[..., :2], m[..., :2]
    step = (F.smooth_l1_loss(pm, tm, reduction="none") * mm * LEADW.view(1, 20, 1)).sum() / mm.sum().clamp(min=1)
    pos = (F.smooth_l1_loss(torch.cumsum(pm, 1), torch.cumsum(tm, 1), reduction="none") * mm).sum() / mm.sum().clamp(min=1)
    return step + pos


def intensity_loss(state, logs, tn, m):
    ps, ts, ms = state[..., 2:], tn[..., 2:], m[..., 2:]
    huber = (F.smooth_l1_loss(ps, ts, reduction="none") * ms).sum() / ms.sum().clamp(min=1)
    nll = (( 0.5 * ((ts - ps) * torch.exp(-logs[..., 2:])) ** 2 + logs[..., 2:]) * ms).sum() / ms.sum().clamp(min=1)
    r34, r50, r64 = ps[..., 3:7], ps[..., 7:11], ps[..., 11:15]
    phys = F.relu(r50 - r34).mean() + F.relu(r64 - r50).mean()
    return 0.7 * huber + 0.3 * nll + 0.01 * phys


def total_loss(state, logs, tgt, m):
    tn = tgt / TARGET_SCALE
    return track_loss(state, tn, m) + intensity_loss(state, logs, tn, m)


model = TrackFormerV4().to(DEVICE)
print(f"TrackFormerV4 params: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
tl, vl = loader(tr_idx, True), loader(va_idx, False)

# ---- EMA ----
ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
def ema_update():
    msd = model.state_dict()
    for k, v in ema.items():
        if torch.is_floating_point(v):
            v.mul_(EMA_DECAY).add_(msd[k].detach(), alpha=1 - EMA_DECAY)
        else:
            v.copy_(msd[k])


def run_train():
    model.train(); tot, cnt = 0.0, 0
    for tr, v0, tg, m in tl:
        tr, v0, tg, m = tr.to(DEVICE), v0.to(DEVICE), tg.to(DEVICE), m.to(DEVICE)
        s, ls = model(tr, v0); loss = total_loss(s, ls, tg, m)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema_update()
        tot += float(loss) * len(tr); cnt += len(tr)
    return tot / max(1, cnt)


@torch.no_grad()
def run_val_ema():
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema); model.eval()
    tot, cnt = 0.0, 0
    for tr, v0, tg, m in vl:
        tr, v0, tg, m = tr.to(DEVICE), v0.to(DEVICE), tg.to(DEVICE), m.to(DEVICE)
        s, ls = model(tr, v0); tot += float(total_loss(s, ls, tg, m)) * len(tr); cnt += len(tr)
    model.load_state_dict(backup)
    return tot / max(1, cnt)


os.makedirs(os.path.dirname(CKPT), exist_ok=True)
best, bad, t0 = float("inf"), 0, time.time()
for ep in range(EPOCHS):
    te = time.time(); trl = run_train(); val = run_val_ema(); sched.step()
    if val < best:
        best, bad = val, 0
        torch.save({"model": {k: v.clone() for k, v in ema.items()}, "epoch": ep, "best_val": best,
                    "kin_cols": KIN_COLS, "thermo_cols": THERMO_COLS, "K_basis": K_BASIS,
                    "track_mean": tmean, "track_std": tstd}, CKPT)
    else:
        bad += 1
    if ep % 2 == 0 or bad == 0:
        print(f"epoch {ep:03d} | train {trl:.5f} | val(ema) {val:.5f} | best {best:.5f} "
              f"| alpha|max| {model.alpha.abs().max().item():.3f} | {time.time()-te:.1f}s", flush=True)
    if bad >= PATIENCE:
        print("early stopping at epoch", ep); break
print(f"trained in {(time.time()-t0)/60:.1f} min; best_val {best:.5f}")


@torch.no_grad()
def metrics(idx):
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False)["model"]); model.eval()
    P, T, M = [], [], []
    for tr, v0, tg, m in loader(idx, False):
        s, _ = model(tr.to(DEVICE), v0.to(DEVICE))
        P.append((s * TARGET_SCALE).float().cpu().numpy()); T.append(tg.numpy()); M.append(m.numpy())
    P, T, M = np.concatenate(P), np.concatenate(T), np.concatenate(M); out = {}
    pt, tt = np.cumsum(P[..., :2], 1), np.cumsum(T[..., :2], 1)
    out["track_error_km"] = round(float(np.sqrt(((pt - tt) ** 2).sum(-1)).mean()), 2)
    for i, nm in [(2, "vmax_mae_kt"), (3, "pressure_mae_hpa"), (4, "rmw_mae_km")]:
        v = M[..., i] > 0.5; out[nm] = round(float(np.abs(P[..., i][v] - T[..., i][v]).mean()), 2) if v.any() else None
    rm = M[..., 5:17] > 0.5; out["radius_mae_km"] = round(float(np.abs(P[..., 5:17] - T[..., 5:17])[rm].mean()), 2)
    return out


print("\nTrackFormer v5 (v3 protected dual-stream + EMA, IBTrACS-only). v3 baseline WP: track 659.0/"
      "pres 18.06/radius 28.83; all-basin: track 592.1")
print("Validation:", json.dumps(metrics(va_idx)))
print("Test all-basin:", json.dumps(metrics(te_idx)))
wp = np.array([i for i in te_idx if basins[i] == "WP"])
if len(wp): print(f"Test WP-only ({len(wp)}):", json.dumps(metrics(wp)))
