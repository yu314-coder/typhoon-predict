"""v19 -- give the intensity head the persistence baseline the track head has always had.

    !wget -q -O v19.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v19.py
    exec(open('v19.py').read())

Run AFTER v18 finishes (it needs the GPU).

THE DIAGNOSIS. Measured on WP+EP 2020+, peak-wind bias by observed strength:

    TD <34      +12 to +14 kt      (over-predicts weak storms)
    TS 34-63     +3 to  +4
    Cat 1-2     -10 to -11
    Cat 3       -24 to -27
    Cat 4-5     -37 to -41 kt      (a 115 kt Cat-4 comes out near 78 kt, a Cat-1)

For Cat 4-5 the MAE is 37.40 and the bias is -36.76: 98% of the error is systematic, not
scatter. Pressure mirrors it (+12 to +25 hPa too high) and forward speed does the same thing
(-21 km/h on fast movers). Every model in the lineage, identically.

THE CAUSE. The track head predicts a RESIDUAL on top of curved persistence:

    base   = curved persistence from the current velocity
    motion = base + track_res(h_track)

The intensity head predicts the absolute value from scratch:

    istate = int_state(h_int)

Under a symmetric Huber loss on an imbalanced distribution (37,682 TD+TS windows against 7,665
Cat 4-5) the safest absolute prediction is climatology -- so the model predicts climatology.
Nothing anchors it to the storm's CURRENT intensity, even though that is sitting in the input.

THE FIX. Two changes, both small:
  1. Intensity persistence baseline: predict the DELTA from the current observed vmax, pressure,
     RMW and 12 radii, exactly as track predicts a delta from current motion. Input columns were
     verified empirically against the lead-0 targets (corr 0.94-1.00), not assumed.
  2. Rarity weighting in the intensity loss so Cat 3-5 windows are not outvoted 5:1.

EXPECT MAE ON WEAK STORMS TO GET SLIGHTLY WORSE. Removing a mean-seeking bias necessarily costs
accuracy on the common cases it was helping. The point is the strong storms, which are the ones
anyone actually cares about; the report below breaks it out by bin so the trade is visible rather
than hidden inside an average.
"""
import os, time, json, math, itertools
import numpy as np, torch, torch.nn as nn

N_SEEDS = int(os.environ.get("V19_SEEDS", "5"))
W_CAT = [float(x) for x in os.environ.get("V19_W", "1.0,1.6,2.6,3.6").split(",")]
# forward speed shows the same mean-collapse as intensity: fast movers (>=35 km/h) are
# under-predicted by -20.7 km/h against an MAE of 21.3, so ~97% of that error is bias.
# Up-weight fast windows so the model cannot win by predicting the average storm's speed.
W_SPEED = [float(x) for x in os.environ.get("V19_WS", "1.0,1.0,1.8,2.8").split(",")]
# NOTE: heading is deliberately NOT given a bias correction. Measured, its fast-mover bias is
# -4.5 deg against an MAE of 23.2 -- only ~19% systematic. Heading error is genuine scatter, and
# a bias-shaped fix would be treating it as a problem it does not have.

# verified empirically: col4 vmax, col5 pressure, col7 rmw, cols 8-19 the twelve radii,
# aligning with target indices 2,3,4 and 5..16
INT_SRC = [4, 5, 7] + list(range(8, 20))
assert len(INT_SRC) == 15
_tm = torch.tensor(tmean, device=DEVICE); _ts = torch.tensor(tstd, device=DEVICE)
_SRC = torch.tensor(INT_SRC, device=DEVICE)
_SCALE_INT = TARGET_SCALE[2:]


def intensity_baseline(tr):
    """Current observed intensity, broadcast across leads, in the model's scaled units."""
    cur = tr[:, -1, :][:, _SRC] * _ts[_SRC] + _tm[_SRC]      # de-normalise to physical units
    cur = torch.where(cur > 0, cur, torch.zeros_like(cur))    # 0 marks 'missing' in this dataset
    return (cur / _SCALE_INT).unsqueeze(1).expand(-1, 20, -1)


class V19(nn.Module):
    """v17 with the intensity head re-expressed as persistence + residual."""
    def __init__(self, dr=0.15):
        super().__init__()
        self.inner = TrackFormerV17(dr=dr)

    def forward(self, tr, vp, sp):
        s, ls = self.inner(tr, vp, sp)
        base = intensity_baseline(tr)
        return torch.cat([s[..., :2], s[..., 2:] + base], -1), ls


def cat_weights(tn, m):
    """Up-weight rare strong storms. tn is the SCALED target, so undo it for vmax."""
    vmax = tn[..., 2] * _SCALE_INT[0]
    w = torch.full_like(vmax, W_CAT[0])
    w = torch.where(vmax >= 64, torch.full_like(w, W_CAT[1]), w)
    w = torch.where(vmax >= 96, torch.full_like(w, W_CAT[2]), w)
    w = torch.where(vmax >= 113, torch.full_like(w, W_CAT[3]), w)
    return w.unsqueeze(-1)


def speed_weights(tn, m):
    """Per (window, lead) weight from the OBSERVED step speed, in km/h."""
    # tn is scaled by TARGET_SCALE, whose first two entries are 100 -- so *100 recovers km
    # per 6 h step, and /6 makes it km/h.
    ts_ = torch.sqrt(((tn[..., :2] * 100.0) ** 2).sum(-1) + 1e-8) / 6.0
    w = torch.full_like(ts_, W_SPEED[0])
    w = torch.where(ts_ >= 15, torch.full_like(w, W_SPEED[1]), w)
    w = torch.where(ts_ >= 25, torch.full_like(w, W_SPEED[2]), w)
    w = torch.where(ts_ >= 35, torch.full_like(w, W_SPEED[3]), w)
    return w


def track_loss_v19(s, tn, m):
    """v17's four terms, with the per-step/speed/heading terms weighted by observed speed.

    The cumulative-position term is left unweighted: it is about where the storm ends up, and
    re-weighting it by instantaneous speed would distort the thing it exists to measure.
    """
    pm, tm_, mm = s[..., :2], tn[..., :2], m[..., :2]
    sw = speed_weights(tn, m).unsqueeze(-1)
    w = mm * LEADW.view(1, 20, 1) * sw
    step = (F.smooth_l1_loss(pm, tm_, reduction="none") * w).sum() / w.sum().clamp(min=1)
    pos = (F.smooth_l1_loss(torch.cumsum(pm, 1), torch.cumsum(tm_, 1), reduction="none")
           * mm).sum() / mm.sum().clamp(min=1)
    ps = torch.sqrt((pm ** 2).sum(-1) + 1e-8)
    ts_ = torch.sqrt((tm_ ** 2).sum(-1) + 1e-8)
    mv = mm[..., 0] * sw[..., 0]
    spd = (F.smooth_l1_loss(ps, ts_, reduction="none") * mv).sum() / mv.sum().clamp(min=1)
    cos = (pm * tm_).sum(-1) / (ps.clamp(min=1e-3) * ts_.clamp(min=1e-3))
    hv = mv * (ts_ > 1e-3).float()
    dirl = ((1.0 - cos) * hv).sum() / hv.sum().clamp(min=1)
    return step + pos + W_SPD * spd + W_DIR * dirl


def int_loss_v19(s, logs, tn, m):
    ps, ts_, ms = s[..., 2:], tn[..., 2:], m[..., 2:]
    w = cat_weights(tn, m) * ms
    d = (ps - ts_).abs()
    lg = logs[..., 2:].clamp(-4, 4)
    nll = (d * torch.exp(-lg) + lg) * w
    return nll.sum() / w.sum().clamp(min=1)


def total_loss_v19(s, logs, tgt, m):
    tn = tgt / TARGET_SCALE
    return track_loss_v19(s, tn, m) + int_loss_v19(s, logs, tn, m)


def train_v19(seed, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    model = V19().to(DEVICE)
    if seed == 0:
        print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl = loader(tr_idx, True, aug=True); vl = loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0
        for tr, v0, sp, tg, m in ld:
            tr, v0, sp, tg, m = [x.to(DEVICE, non_blocking=True) for x in (tr, v0, sp, tg, m)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls = model(tr, v0, sp); loss = total_loss_v19(s, ls, tg, m)
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
            try: shutil.copy(ckpt, DRIVE_OUT)
            except Exception as e: print("  (drive copy failed:", e, ")")
        else:
            bad += 1
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"{time.time()-te:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


CK19 = [train_v19(s, f"/content/v19_seed{s}.pt") for s in range(N_SEEDS)]
print("v19 trained:", len(CK19), flush=True)

# ---------------- evaluation, broken out by intensity bin ------------------
full = z["n_leads"].astype(int) == 20
TEST = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = target[TEST]; K = mask[TEST].astype(bool)


@torch.no_grad()
def predict(models):
    P = []
    for i in range(0, len(TEST), 256):
        j = TEST[i:i + 256]
        s = torch.stack([mm(torch.from_numpy(track[j]).to(DEVICE),
                            torch.from_numpy(vpair[j]).to(DEVICE),
                            torch.from_numpy(SLP[j]).to(DEVICE))[0] for mm in models]).mean(0)
        P.append((s * TARGET_SCALE).float().cpu().numpy())
    return np.concatenate(P)


m19 = []
for c in CK19:
    mm = V19().to(DEVICE).eval()
    mm.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"])
    m19.append(mm)
P19 = predict(m19)
m17 = [load_model(f"{DATA}/v17_seed{i}.pt") for i in range(5)
       if os.path.exists(f"{DATA}/v17_seed{i}.pt")]
P17 = predict(m17)

BINS = [(0, 34, "TD <34"), (34, 64, "TS 34-63"), (64, 96, "Cat1-2"), (96, 113, "Cat3"), (113, 300, "Cat4-5")]
print("\n" + "=" * 78)
print("PEAK WIND by observed strength   (bias = predicted - observed; negative = too weak)")
print("=" * 78)
print(f"{'bin':12s} {'n':>7s} | {'v17 MAE':>8s} {'v17 bias':>9s} | {'v19 MAE':>8s} {'v19 bias':>9s} | {'bias fixed':>11s}")
for lo, hi, lab in BINS:
    sel = K[..., 2] & (T[..., 2] >= lo) & (T[..., 2] < hi)
    if sel.sum() < 50: continue
    d17 = (P17[..., 2] - T[..., 2])[sel]; d19 = (P19[..., 2] - T[..., 2])[sel]
    print(f"{lab:12s} {int(sel.sum()):7d} | {np.abs(d17).mean():8.2f} {d17.mean():+9.2f} | "
          f"{np.abs(d19).mean():8.2f} {d19.mean():+9.2f} | "
          f"{abs(d17.mean()) - abs(d19.mean()):+11.2f}")

print("\nPRESSURE (hPa) by observed strength")
for lo, hi, lab in BINS:
    sel = K[..., 3] & (T[..., 2] >= lo) & (T[..., 2] < hi)
    if sel.sum() < 50: continue
    d17 = (P17[..., 3] - T[..., 3])[sel]; d19 = (P19[..., 3] - T[..., 3])[sel]
    print(f"{lab:12s} {int(sel.sum()):7d} | {np.abs(d17).mean():8.2f} {d17.mean():+9.2f} | "
          f"{np.abs(d19).mean():8.2f} {d19.mean():+9.2f} | "
          f"{abs(d17.mean()) - abs(d19.mean()):+11.2f}")

pt17 = np.cumsum(P17[..., :2], 1); pt19 = np.cumsum(P19[..., :2], 1); tt = np.cumsum(T[..., :2], 1)
e17 = np.sqrt(((pt17 - tt) ** 2).sum(-1)).mean(); e19 = np.sqrt(((pt19 - tt) ** 2).sum(-1)).mean()
print(f"\nTRACK (should be roughly unchanged -- the track head was not touched):")
print(f"  v17 {e17:.2f} km   v19 {e19:.2f} km   delta {e19-e17:+.2f}")
print(f"\nseeds: v19 {len(m19)} vs v17 {len(m17)} -- compare at matched count before concluding")
json.dump({"v19_track": float(e19), "v17_track": float(e17)}, open("/content/v19.json", "w"))
