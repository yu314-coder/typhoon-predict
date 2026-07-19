"""v18 -- attack the overfitting, and measure the result properly.

    !wget -q -O v18.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v18.py
    exec(open('v18.py').read())

Every version so far ends training with train loss ~0.09 against val ~1.19 -- a 13x gap, 12.9M
parameters on 153k windows. Adding inputs has not touched that (SST bought nothing) and neither
did the loss rework (-0.6 km at equal seed count). So v18 spends its budget on the gap itself:

  1. EMA of the weights (decay 0.999). Averaging the trajectory instead of taking the last point
     is close to free and is the single most reliable win in this situation.
  2. Higher dropout, 0.15 -> 0.22, and weight decay 5e-2 -> 8e-2.
  3. Input jitter: small Gaussian noise on the normalised track features during training only.
     The inputs are best-track estimates with real observational error; pretending they are exact
     is part of why the model can memorise them.
  4. Steering dropout 0.20 -> 0.25, since the repaired data now legitimately contains missing
     fields and the model should be comfortable without them.

MEASUREMENT is the other half. Every previous comparison was one ensemble against one ensemble,
and the seed-subset control showed 3-seed ensembles range over 461.7-473.9 km -- so differences
under ~10 km were never resolvable. v18 trains 8 seeds and reports, for each ensemble size, the
mean and spread over ALL subsets of that size, plus a paired comparison against v17 on identical
windows. A number without a spread is not a result.
"""
import os, math, time, json, itertools, copy
import numpy as np, torch

N_SEEDS = int(os.environ.get("N_SEEDS", "8"))
DROPOUT = float(os.environ.get("V18_DROPOUT", "0.22"))
WD18 = float(os.environ.get("V18_WD", "8e-2"))
JITTER = float(os.environ.get("V18_JITTER", "0.02"))     # sigma, in normalised feature units
EMA_DECAY = float(os.environ.get("V18_EMA", "0.999"))
STEER_DROP_18 = float(os.environ.get("V18_STEER_DROP", "0.25"))

print(f"v18: dropout {DROPOUT} | wd {WD18} | jitter {JITTER} | EMA {EMA_DECAY} | "
      f"steer-drop {STEER_DROP_18} | {N_SEEDS} seeds\n", flush=True)

import __main__ as G
G.STEER_DROP = STEER_DROP_18            # the model reads this global inside forward()


class EMA:
    """Exponential moving average of the weights, with the usual warmup correction."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        self.n = 0

    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))     # ramp in, else early junk dominates
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone().float()

    def copy_to(self, model):
        sd = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(sd[k].dtype) for k in sd})


def train_one_v18(seed, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrackFormerV17(dr=DROPOUT).to(DEVICE)
    if seed == 0:
        print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD18)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, EMA_DECAY)
    tl = loader(tr_idx, True, aug=True); vl = loader(va_idx, False)
    probe = TrackFormerV17(dr=DROPOUT).to(DEVICE)          # scratch module for scoring the EMA

    def run(ld, train):
        model.train(train); tot = cnt = 0
        for tr, v0, sp, tg, m in ld:
            tr, v0, sp, tg, m = [x.to(DEVICE, non_blocking=True) for x in (tr, v0, sp, tg, m)]
            if train and JITTER > 0:
                tr = tr + torch.randn_like(tr) * JITTER
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls = model(tr, v0, sp); loss = total_loss(s, ls, tg, m)
            if train:
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); ema.update(model)
            tot += float(loss.detach()) * len(tr); cnt += len(tr)
        return tot / cnt

    @torch.no_grad()
    def val_ema():
        ema.copy_to(probe); probe.eval()
        tot = cnt = 0
        for tr, v0, sp, tg, m in vl:
            tr, v0, sp, tg, m = [x.to(DEVICE, non_blocking=True) for x in (tr, v0, sp, tg, m)]
            with torch.cuda.amp.autocast():
                s, ls = probe(tr, v0, sp); loss = total_loss(s, ls, tg, m)
            tot += float(loss) * len(tr); cnt += len(tr)
        return tot / cnt

    best, bad, t0 = float("inf"), 0, time.time()
    for ep in range(EPOCHS):
        te = time.time(); trl = run(tl, True)
        with torch.no_grad(): raw = run(vl, False)
        ev = val_ema(); sched.step()
        use = min(raw, ev)
        if use < best:
            best, bad = use, 0
            ema.copy_to(probe)
            sd = probe.state_dict() if ev <= raw else model.state_dict()
            torch.save({"model": sd, "epoch": ep, "best_val": best,
                        "track_mean": tmean, "track_std": tstd, "ema": bool(ev <= raw)}, ckpt)
            try: shutil.copy(ckpt, DRIVE_OUT)
            except Exception as e: print("  (drive copy failed:", e, ")")
        else:
            bad += 1
        print(f"ep {ep:03d} | train {trl:.5f} | val {raw:.5f} | ema {ev:.5f} | "
              f"best {best:.5f} | {time.time()-te:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


CK18 = []
for s in range(N_SEEDS):
    CK18.append(train_one_v18(s, f"/content/v18_seed{s}.pt"))
print("v18 training complete:", len(CK18), "checkpoints", flush=True)
json.dump(CK18, open("/content/v18_ckpts.json", "w"))
