"""v10.1 on Colab — v10's architecture, 72% more training data (1950+), plus ONI.

    !wget -q -O v20.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v20.py
    exec(open('v20.py').read())

Self-contained: loads its own data and defines its own model, so it does not depend on the
notebook's globals (which are set up for the 4-channel steering models and the 1980+ dataset).

WHY v10 IS THE RIGHT BASE FOR THIS. v10 consumes no reanalysis fields at all, so extending the
record from 1980 back to 1950 costs a single IBTrACS rebuild rather than ~10 GB of SLP and wind
downloads for thirty extra years. Training windows go 153,378 -> 264,454 while valid and test stay
byte-identical, so the result is directly comparable to v10 (549.3 km) and v17 (462.8 km).

ONI is feature 54: the Nino 3.4 SST anomaly, a basin-scale ENSO signal that shifts where WP
typhoons form and how far they recurve. No storm-centred patch can see it, which makes it a
different KIND of input rather than more of the same -- and the SST ablation already showed that
more of the same does not help.

DELIBERATELY NOT CHANGED: no mirror augmentation, no EMA, no rarity weighting. v10.1 differs from
v10 in exactly two ways -- the data and ONI -- so if the number moves, the cause is unambiguous.
Every confounded comparison in this project has had to be redone.

CAVEAT: pre-satellite best-track data is less accurate. The WP had aircraft reconnaissance from
1945-1987 so intensities hold up better than one might fear, but 1950s-60s positions carry more
error than modern fixes. If this does not beat v10, try 1970+ before concluding the extension
failed.
"""
import os, re, time, json, math, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = int(os.environ.get("V20_SEEDS", "3"))
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
        # No Drive copy: fetch straight from the repo. 63 MB, so this needs no manual upload
        # and no Drive mount at all -- the whole run can proceed without touching the account.
        print("fetching the dataset from GitHub (63 MB) ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/track_windows_v20.npz", NPZ)
    print(f"dataset ready: {os.path.getsize(NPZ)/1e6:.0f} MB", flush=True)

# ---- model definition lifted from the training script that defines v10.1 ----
if not os.path.exists("/content/_v20.py"):
    urllib.request.urlretrieve(f"{RAW}/train_track_v20.py", "/content/_v20.py")
src = open("/content/_v20.py").read()
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os, "DEVICE": DEVICE}
for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
            r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
            r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    m = re.search(pat, src, re.S); assert m, pat[:40]
    exec(m.group(0), G)
Net = G["TrackFormerV9"]
print(f"env dims {G['ENV_DIM']} (54 = 6 -> 55 = 7 with ONI)", flush=True)

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
print(f"windows {len(track)} | train {len(tr_idx)} valid {len(va_idx)} test {len(te_idx)}", flush=True)


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

def track_loss(s, tn, m):
    pm, tm_, mm = s[..., :2], tn[..., :2], m[..., :2]
    step = (F.smooth_l1_loss(pm, tm_, reduction="none") * mm * LEADW.view(1, 20, 1)).sum() / mm.sum().clamp(min=1)
    pos = (F.smooth_l1_loss(torch.cumsum(pm, 1), torch.cumsum(tm_, 1), reduction="none") * mm).sum() / mm.sum().clamp(min=1)
    return step + pos

def int_loss(s, logs, tn, m):
    ps, ts_, ms = s[..., 2:], tn[..., 2:], m[..., 2:]
    huber = (F.smooth_l1_loss(ps, ts_, reduction="none") * ms).sum() / ms.sum().clamp(min=1)
    nll = ((0.5 * ((ts_ - ps) * torch.exp(-logs[..., 2:])) ** 2 + logs[..., 2:]) * ms).sum() / ms.sum().clamp(min=1)
    r34, r50, r64 = ps[..., 3:7], ps[..., 7:11], ps[..., 11:15]
    return 0.7 * huber + 0.3 * nll + 0.01 * (F.relu(r50 - r34).mean() + F.relu(r64 - r50).mean())

def total_loss(s, logs, tgt, m):
    tn = tgt / TARGET_SCALE
    return track_loss(s, tn, m) + int_loss(s, logs, tn, m)


def train_one(seed, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net().to(DEVICE)
    if seed == 0:
        print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0
        for tr, v_, tg, m in ld:
            tr, v_, tg, m = [x.to(DEVICE, non_blocking=True) for x in (tr, v_, tg, m)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls = model(tr, v_); loss = total_loss(s, ls, tg, m)
            if train:
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            tot += float(loss.detach()) * len(tr); cnt += len(tr)
        return tot / cnt

    best, bad, t0 = float("inf"), 0, time.time()
    for ep in range(EPOCHS):
        te = time.time(); trl = run(tl, True)
        with torch.no_grad(): vv = run(vl, False)
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
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | {time.time()-te:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


CK = [train_one(s, f"/content/v20_seed{s}.pt") for s in range(N_SEEDS)]
print("v10.1 trained:", len(CK), flush=True)

# ---- evaluation on the SAME test set as v10 / v17 ----
full = z["n_leads"].astype(int) == 20
EV = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = np.cumsum(target[EV][..., :2], 1)

@torch.no_grad()
def predict(models):
    P = []
    for i in range(0, len(EV), 256):
        j = EV[i:i + 256]
        s = torch.stack([m(torch.from_numpy(track[j]).to(DEVICE),
                           torch.from_numpy(vpair[j]).to(DEVICE))[0] for m in models]).mean(0)
        P.append((s * TARGET_SCALE).float().cpu().numpy())
    return np.cumsum(np.concatenate(P)[..., :2], 1)

MS = []
for c in CK:
    m = Net().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); MS.append(m)
print(f"\nWP+EP 2020+, {len(EV)} full-horizon windows — the SAME set v10 and v17 were scored on")
print(f"  BASELINES   v10 549.3 | v17 462.8 | v18 466.2 | v19 466.7")
for i, m in enumerate(MS):
    print(f"  v10.1 seed{i}  {np.sqrt(((predict([m]) - T) ** 2).sum(-1)).mean():.2f} km")
e = float(np.sqrt(((predict(MS) - T) ** 2).sum(-1)).mean())
print(f"  v10.1 ENSEMBLE ({len(MS)} seeds)  {e:.2f} km")
print(f"  vs v10: {e - 549.3:+.2f} km   vs v17: {e - 462.8:+.2f} km")
json.dump({"v10_1": e}, open("/content/v20.json", "w"))
