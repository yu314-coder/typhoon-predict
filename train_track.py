"""Right-sized TRACK-ONLY model + training, self-contained for Colab.

Run in one Colab cell after mounting Drive:
    exec(open('/content/drive/MyDrive/typhoon_predict_stormfusion_mt/train_track.py').read())

Trains on the ~79k track-only windows (all-basin IBTrACS, no ERA5). Model ~15-20M params
-- larger than v2 (3.3M) but matched to the data. Same multi-task loss and metrics so the
track error is directly comparable to the ERA5 model.
"""
import math, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DRIVE = "/content/drive/MyDrive/typhoon_predict_stormfusion_mt"
NPZ = f"{DRIVE}/track_windows.npz"
CKPT = f"{DRIVE}/checkpoints/track_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SCALE = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12, device=DEVICE)
RADIUS_NAMES = [f"r{t}_{q}" for t in (34, 50, 64) for q in ("ne", "se", "sw", "nw")]
STATE_NAMES = ["east_km", "north_km", "vmax_kt", "pressure_hpa", "rmw_km"] + RADIUS_NAMES

EPOCHS = 200
PATIENCE = 25
BATCH = 256
LR = 3e-4
WEIGHT_DECAY = 3e-2

print("loading", NPZ)
z = np.load(NPZ, allow_pickle=True)
track = z["track"].astype("float32")
target = z["target"].astype("float32")
mask = z["target_mask"].astype("float32")
years = z["year"].astype(int)
sids = z["storm_id"].astype(str)
print("windows:", len(track), "track", track.shape, "target", target.shape)

fy = {s: int(years[sids == s].min()) for s in np.unique(sids)}
tr_s = {s for s, y in fy.items() if y <= 2015}
va_s = {s for s, y in fy.items() if 2016 <= y <= 2019}
te_s = {s for s, y in fy.items() if y >= 2020}
tr_idx = np.where(np.isin(sids, list(tr_s)))[0]
va_idx = np.where(np.isin(sids, list(va_s)))[0]
te_idx = np.where(np.isin(sids, list(te_s)))[0]
print(f"split windows train={len(tr_idx)} valid={len(va_idx)} test={len(te_idx)} | storms {len(tr_s)}/{len(va_s)}/{len(te_s)}")


class DS(Dataset):
    def __init__(self, idx):
        self.idx = np.asarray(idx)
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i])
        return (torch.from_numpy(track[j]), torch.from_numpy(target[j]), torch.from_numpy(mask[j]))


def loader(idx, shuffle):
    return DataLoader(DS(idx), batch_size=BATCH, shuffle=shuffle, num_workers=0, pin_memory=True)


def sinusoidal(length, d):
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    e = torch.zeros(length, d)
    e[:, 0::2] = torch.sin(pos * div); e[:, 1::2] = torch.cos(pos * div)
    return e


class TrackModel(nn.Module):
    def __init__(self, d_model=384, heads=8, ffn=1536, ctx_depth=4, dec_depth=6, leads=20, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(40, d_model)
        self.track_time = nn.Parameter(torch.zeros(1, 9, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, ffn, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.ctx = nn.TransformerEncoder(enc, ctx_depth)
        dec = nn.TransformerDecoderLayer(d_model, heads, ffn, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(dec, dec_depth)
        self.q = nn.Parameter(torch.randn(1, leads, d_model) * 0.02)
        self.register_buffer("qpos", sinusoidal(leads, d_model))
        self.state = nn.Linear(d_model, 17)
        self.logscale = nn.Linear(d_model, 17)
    def forward(self, track):
        b = track.shape[0]
        mem = self.ctx(self.proj(track) + self.track_time)
        q = (self.q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        d = self.dec(q, mem)
        return self.state(d), self.logscale(d).clamp(-5.0, 3.0)


def masked_huber(pred, tgt, m):
    e = F.smooth_l1_loss(pred, tgt, reduction="none")
    return (e * m).sum() / m.sum().clamp(min=1)


def masked_nll(pred, logs, tgt, m):
    nll = 0.5 * ((tgt - pred) * torch.exp(-logs)) ** 2 + logs
    return (nll * m).sum() / m.sum().clamp(min=1)


def physical(pred):
    vmax, rmw = pred[..., 2], pred[..., 4]
    r34, r50, r64 = pred[..., 5:9], pred[..., 9:13], pred[..., 13:17]
    nn_ = F.relu(-vmax).mean() + F.relu(-rmw).mean() + F.relu(-r34).mean() + F.relu(-r50).mean() + F.relu(-r64).mean()
    nest = F.relu(r50 - r34).mean() + F.relu(r64 - r50).mean()
    return 0.01 * (nn_ + nest)


def total_loss(pred, logs, tgt, m):
    nt = tgt / TARGET_SCALE
    return 0.7 * masked_huber(pred, nt, m) + 0.3 * masked_nll(pred, logs, nt, m) + physical(pred)


model = TrackModel().to(DEVICE)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"TrackModel params: {n:,}")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
amp = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

tl, vl = loader(tr_idx, True), loader(va_idx, False)


def run(loader, train):
    model.train(train)
    tot, cnt = 0.0, 0
    for tr, tg, m in loader:
        tr, tg, m = tr.to(DEVICE), tg.to(DEVICE), m.to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=amp, enabled=DEVICE.type == "cuda"):
            with torch.set_grad_enabled(train):
                p, ls = model(tr)
                loss = total_loss(p, ls, tg, m)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += float(loss) * len(tr); cnt += len(tr)
    return tot / max(1, cnt)


import os
os.makedirs(f"{DRIVE}/checkpoints", exist_ok=True)
best, bad = float("inf"), 0
t0 = time.time()
for ep in range(EPOCHS):
    trl = run(tl, True)
    val = run(vl, False)
    sched.step()
    if val < best:
        best, bad = val, 0
        torch.save({"model": model.state_dict(), "epoch": ep, "best_val": best,
                    "track_mean": z["track_mean"], "track_std": z["track_std"]}, CKPT)
    else:
        bad += 1
    if ep % 5 == 0 or bad == 0:
        print(f"epoch {ep:03d} | train {trl:.5f} | val {val:.5f} | lr {sched.get_last_lr()[0]:.2e} | best {best:.5f}")
    if bad >= PATIENCE:
        print("early stopping at epoch", ep); break
print(f"trained in {(time.time()-t0)/60:.1f} min; best_val {best:.5f}")


@torch.no_grad()
def metrics(idx):
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False)["model"])
    model.eval()
    P, T, M = [], [], []
    for tr, tg, m in loader(idx, False):
        with torch.autocast(DEVICE.type, dtype=amp, enabled=DEVICE.type == "cuda"):
            p, _ = model(tr.to(DEVICE))
        P.append((p * TARGET_SCALE).float().cpu().numpy()); T.append(tg.numpy()); M.append(m.numpy())
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


print("\nTRACK-ONLY model metrics (compare vs ERA5 v2 test: track 731 / vmax 24.2 / pres 21.6 / rmw 16.2 / radius 31.8):")
print("Validation:", json.dumps(metrics(va_idx)))
if len(te_idx):
    print("Test:      ", json.dumps(metrics(te_idx)))
