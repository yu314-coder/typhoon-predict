"""TrackFormer v3 — protected dual-stream architecture (Apple GPU / MPS).

Design (from the negative-transfer analysis + GPT-5 consult, grounded in the RMT covariance work):
  * KINEMATIC encoder sees only motion/heading/speed/turn/season -> track decoder. Track gradients only.
  * THERMODYNAMIC encoder sees vmax/pressure/rmw/radii/trends/validity -> intensity+radii decoder.
  * Cross-task transfer is ASYMMETRIC and PROTECTED:
      - intensity decoder consumes stopgrad(kinematic memory)  (motion helps intensity, no back-corruption)
      - thermo->track flows ONLY through a zero-init gated adapter on detached thermo features:
            H_track = H_kin_dec + alpha_l * Adapter(stopgrad(H_thermo_ctx))
        alpha_l is per-lead, zero-initialized -> at init the model == a protected track-only model,
        so intensity tasks can only help track, never hurt it.
  * PERSISTENCE-RESIDUAL track head: per-step motion = rho_l * v0 (damped constant-velocity baseline)
        + learned residual. The net only predicts the deviation from persistence (recurvature/accel).
  * Track loss = lead-weighted (sqrt) Huber on per-step motion + position-space (cumulative) Huber.
    Intensity loss = masked Huber + Gaussian NLL as before. Deterministic track loss is NOT downweighted
    by any predicted variance.

Runs fp32 on MPS. Column split is by feature index in the 48-dim history vector.
"""
import math, json, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NPZ = os.environ.get("TRACK_NPZ", "track_build/track_windows_v2.npz")
CKPT = os.environ.get("TRACK_CKPT", "track_build/track_v3_best.pt")
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
TARGET_SCALE = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12, device=DEVICE)

EPOCHS = int(os.environ.get("EPOCHS", "150"))
PATIENCE = int(os.environ.get("PATIENCE", "22"))
BATCH = int(os.environ.get("BATCH", "512"))
LR = 3e-4
WEIGHT_DECAY = 3e-2

# ---- feature column split (indices into the 48-dim history vector) ----
KIN_COLS = [0, 1, 2, 3, 21, 22, 23, 40, 41, 42, 43]           # motion, season, dt, heading, speed, turn
THERMO_COLS = [4, 5, 6, 7] + list(range(8, 20)) + list(range(24, 40)) + [44, 45, 46, 47]
KIN_DIM, THERMO_DIM = len(KIN_COLS), len(THERMO_COLS)

print("device:", DEVICE, "| loading", NPZ)
z = np.load(NPZ, allow_pickle=True)
track = z["track"].astype("float32")
target = z["target"].astype("float32")
mask = z["target_mask"].astype("float32")
years = z["year"].astype(int)
sids = z["storm_id"].astype(str)
basins = z["basin"].astype(str)
tmean = z["track_mean"].astype("float32")
tstd = z["track_std"].astype("float32")
# raw current velocity v0 (km/6h) = de-standardized step-motion (channels 2,3) of the last history step
v0_raw = track[:, -1, 2:4] * tstd[2:4] + tmean[2:4]           # [N,2] east/north km per 6h
print(f"windows {len(track)} | kin_dim {KIN_DIM} thermo_dim {THERMO_DIM} | v0 mean {v0_raw.mean(0)}")

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
    e = torch.zeros(length, d)
    e[:, 0::2] = torch.sin(pos * div); e[:, 1::2] = torch.cos(pos * div)
    return e


def encoder(d, heads, ffn, depth, dropout):
    layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True,
                                       norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, depth)


def decoder(d, heads, ffn, depth, dropout):
    layer = nn.TransformerDecoderLayer(d, heads, ffn, dropout, batch_first=True,
                                       norm_first=True, activation="gelu")
    return nn.TransformerDecoder(layer, depth)


class TrackFormerV3(nn.Module):
    def __init__(self, d=256, heads=8, ffn=1024, leads=20, dropout=0.2):
        super().__init__()
        self.leads = leads
        # dual encoders
        self.kin_proj = nn.Linear(KIN_DIM, d)
        self.thermo_proj = nn.Linear(THERMO_DIM, d)
        self.kin_time = nn.Parameter(torch.zeros(1, 9, d))
        self.thermo_time = nn.Parameter(torch.zeros(1, 9, d))
        self.kin_enc = encoder(d, heads, ffn, 4, dropout)
        self.thermo_enc = encoder(d, heads, ffn, 3, dropout)
        # decoders (separate query embeddings per task)
        self.track_dec = decoder(d, heads, ffn, 4, dropout)
        self.int_dec = decoder(d, heads, ffn, 5, dropout)
        self.track_q = nn.Parameter(torch.randn(1, leads, d) * 0.02)
        self.int_q = nn.Parameter(torch.randn(1, leads, d) * 0.02)
        self.register_buffer("qpos", sinusoidal(leads, d))
        # zero-init gated thermo->track adapter (bottleneck 64), alpha per-lead zero-init
        self.adapter = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d))
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(leads))
        # persistence damping schedule rho_l (init ~1, gently decaying)
        self.rho = nn.Parameter(torch.ones(leads))
        # heads
        self.track_res = nn.Linear(d, 2)          # residual per-step motion (normalized units)
        nn.init.zeros_(self.track_res.weight); nn.init.zeros_(self.track_res.bias)
        self.int_state = nn.Linear(d, 15)         # vmax,pres,rmw,12 radii
        self.int_logscale = nn.Linear(d, 15)

    def forward(self, track, v0_raw):
        b = track.shape[0]
        kin = self.kin_enc(self.kin_proj(track[:, :, KIN_COLS]) + self.kin_time)
        thermo = self.thermo_enc(self.thermo_proj(track[:, :, THERMO_COLS]) + self.thermo_time)
        # thermo context token (mean-pooled), detached for the track path
        thermo_ctx = thermo.mean(1)                                  # [b,d]
        # ---- track path (kinematic memory only) ----
        tq = (self.track_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_track = self.track_dec(tq, kin)                            # [b,L,d]
        gate = self.alpha.view(1, self.leads, 1)
        h_track = h_track + gate * self.adapter(thermo_ctx.detach()).unsqueeze(1)
        res = self.track_res(h_track)                                # [b,L,2] normalized residual
        # persistence baseline: rho_l * v0 (km/6h) normalized by motion scale (100)
        base = (self.rho.view(1, self.leads, 1) * v0_raw.unsqueeze(1)) / 100.0
        motion = base + res                                          # normalized per-step motion
        # ---- intensity path (detached kinematic memory + own thermo memory) ----
        iq = (self.int_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h_int = self.int_dec(iq, torch.cat([thermo, kin.detach()], dim=1))
        istate = self.int_state(h_int)
        ilog = self.int_logscale(h_int).clamp(-5.0, 3.0)
        # assemble full 17-dim state: [motion(2), intensity(15)]
        state = torch.cat([motion, istate], dim=-1)
        logscale = torch.cat([torch.zeros_like(motion), ilog], dim=-1)
        return state, logscale


LEADW = torch.sqrt(torch.arange(1, 21, device=DEVICE).float()); LEADW = LEADW / LEADW.mean()  # ~sqrt(l), mean 1


def track_loss(state, tgt_norm, m):
    # per-step motion Huber (lead-weighted) + cumulative position Huber
    pm, tm, mm = state[..., :2], tgt_norm[..., :2], m[..., :2]
    step = (F.smooth_l1_loss(pm, tm, reduction="none") * mm * LEADW.view(1, 20, 1)).sum() / mm.sum().clamp(min=1)
    pos = F.smooth_l1_loss(torch.cumsum(pm, 1), torch.cumsum(tm, 1), reduction="none")
    pos = (pos * mm).sum() / mm.sum().clamp(min=1)
    return step + pos


def intensity_loss(state, logs, tgt_norm, m):
    ps, ts, ms = state[..., 2:], tgt_norm[..., 2:], m[..., 2:]
    huber = (F.smooth_l1_loss(ps, ts, reduction="none") * ms).sum() / ms.sum().clamp(min=1)
    nll = 0.5 * ((ts - ps) * torch.exp(-logs[..., 2:])) ** 2 + logs[..., 2:]
    nll = (nll * ms).sum() / ms.sum().clamp(min=1)
    # physical monotone penalty on radii
    r34, r50, r64 = ps[..., 3:7], ps[..., 7:11], ps[..., 11:15]
    phys = F.relu(r50 - r34).mean() + F.relu(r64 - r50).mean()
    return 0.7 * huber + 0.3 * nll + 0.01 * phys


def total_loss(state, logs, tgt, m):
    nt = tgt / TARGET_SCALE
    return track_loss(state, nt, m) + intensity_loss(state, logs, nt, m)


model = TrackFormerV3().to(DEVICE)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"TrackFormerV3 params: {n:,}")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
tl, vl = loader(tr_idx, True), loader(va_idx, False)


def run(ld, train):
    model.train(train)
    tot, cnt = 0.0, 0
    for tr, v0, tg, m in ld:
        tr, v0, tg, m = tr.to(DEVICE), v0.to(DEVICE), tg.to(DEVICE), m.to(DEVICE)
        with torch.set_grad_enabled(train):
            s, ls = model(tr, v0)
            loss = total_loss(s, ls, tg, m)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += float(loss) * len(tr); cnt += len(tr)
    return tot / max(1, cnt)


os.makedirs(os.path.dirname(CKPT), exist_ok=True)
best, bad, t0 = float("inf"), 0, time.time()
for ep in range(EPOCHS):
    te = time.time()
    trl = run(tl, True); val = run(vl, False); sched.step()
    if val < best:
        best, bad = val, 0
        torch.save({"model": model.state_dict(), "epoch": ep, "best_val": best,
                    "kin_cols": KIN_COLS, "thermo_cols": THERMO_COLS,
                    "track_mean": tmean, "track_std": tstd}, CKPT)
    else:
        bad += 1
    if ep % 2 == 0 or bad == 0:
        print(f"epoch {ep:03d} | train {trl:.5f} | val {val:.5f} | lr {sched.get_last_lr()[0]:.2e} "
              f"| best {best:.5f} | alpha|max| {model.alpha.abs().max().item():.3f} | {time.time()-te:.1f}s", flush=True)
    if bad >= PATIENCE:
        print("early stopping at epoch", ep); break
print(f"trained in {(time.time()-t0)/60:.1f} min; best_val {best:.5f}")


@torch.no_grad()
def metrics(idx):
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False)["model"])
    model.eval()
    P, T, M = [], [], []
    for tr, v0, tg, m in loader(idx, False):
        s, _ = model(tr.to(DEVICE), v0.to(DEVICE))
        P.append((s * TARGET_SCALE).float().cpu().numpy()); T.append(tg.numpy()); M.append(m.numpy())
    P, T, M = np.concatenate(P), np.concatenate(T), np.concatenate(M)
    out = {}
    pt, tt = np.cumsum(P[..., :2], 1), np.cumsum(T[..., :2], 1)
    out["track_error_km"] = round(float(np.sqrt(((pt - tt) ** 2).sum(-1)).mean()), 2)
    for i, nm in [(2, "vmax_mae_kt"), (3, "pressure_mae_hpa"), (4, "rmw_mae_km")]:
        v = M[..., i] > 0.5
        out[nm] = round(float(np.abs(P[..., i][v] - T[..., i][v]).mean()), 2) if v.any() else None
    rm = M[..., 5:17] > 0.5
    out["radius_mae_km"] = round(float(np.abs(P[..., 5:17] - T[..., 5:17])[rm].mean()), 2) if rm.any() else None
    return out


print("\nTrackFormer v3 (protected dual-stream + persistence). Baselines WP-2020+: old 720 / v2 737 (tied); "
      "pressure old 21.2 / v2 17.7")
print("Validation:", json.dumps(metrics(va_idx)))
print("Test all-basin:", json.dumps(metrics(te_idx)))
wp = np.array([i for i in te_idx if basins[i] == "WP"])
if len(wp):
    print(f"Test WP-only ({len(wp)}):", json.dumps(metrics(wp)))
