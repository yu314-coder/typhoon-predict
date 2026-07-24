"""Stronger 'mean' inference for TrackFormer v3: average many stochastic forward passes.

Strategies (all use the SINGLE trained v3 checkpoint -- no retraining):
  * det        : one deterministic pass (baseline)
  * mcdrop-N    : N Monte-Carlo dropout passes (dropout left ON), predictions averaged
  * jitter-N    : N passes with small Gaussian input jitter (test-time augmentation), averaged
  * combo-N     : MC-dropout + input jitter together

We report held-out metrics vs N so the benefit (if any) is measured, not assumed. Averaging the
MEAN of many subnetworks reduces prediction variance; it helps most where the model is uncertain.
"""
import re, math, json, os, sys
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TS = np.array([100., 100., 35., 20., 50.] + [50.] * 12, dtype="float32")

# ---- rebuild TrackFormerV3 from the training script ----
g = {"torch": torch, "nn": nn, "F": __import__("torch.nn.functional", fromlist=["x"]), "math": math, "np": np}
src = open("train_track_v3.py").read()
for blk in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM = len\(KIN_COLS\), len\(THERMO_COLS\)",
            r"def sinusoidal.*?return e", r"def encoder.*?return nn\.TransformerEncoder\(layer, depth\)",
            r"def decoder.*?return nn\.TransformerDecoder\(layer, depth\)",
            r"class TrackFormerV3.*?return state, logscale"]:
    exec(re.search(blk, src, re.S).group(0), g)

ck = torch.load("models/trackformer_v3_15M_fp16.pt", map_location="cpu", weights_only=False)
model = g["TrackFormerV3"]().to(DEV)
model.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["model"].items()})
model.eval()

z = np.load("track_build/track_windows_v2.npz", allow_pickle=True)
tmean, tstd = z["track_mean"].astype("float32"), z["track_std"].astype("float32")
v0 = (z["track"][:, -1, 2:4] * tstd[2:4] + tmean[2:4]).astype("float32")
trk, tgt, msk = z["track"].astype("float32"), z["target"], z["target_mask"].astype("float32")
yr, bs = z["year"].astype(int), z["basin"].astype(str)


def enable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()


def metrics(P, T, M):
    o = {}
    pt, tt = np.cumsum(P[..., :2], 1), np.cumsum(T[..., :2], 1)
    o["track"] = round(float(np.sqrt(((pt - tt) ** 2).sum(-1)).mean()), 1)
    for i, nm in [(2, "vmax"), (3, "pres"), (4, "rmw")]:
        v = M[..., i] > 0.5
        o[nm] = round(float(np.abs(P[..., i][v] - T[..., i][v]).mean()), 2)
    rm = M[..., 5:17] > 0.5
    o["radius"] = round(float(np.abs(P[..., 5:17] - T[..., 5:17])[rm].mean()), 2)
    return o


@torch.no_grad()
def predict(idx, n_samples=1, mc_dropout=False, jitter=0.0, seed=0):
    """Return averaged prediction over n_samples stochastic passes."""
    model.eval()
    if mc_dropout:
        enable_dropout(model)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    acc = np.zeros((len(idx), 20, 17), dtype="float64")
    for s in range(n_samples):
        P = []
        for b0 in range(0, len(idx), 512):
            b = idx[b0:b0 + 512]
            x = torch.from_numpy(trk[b]).to(DEV)
            if jitter > 0:
                x = x + jitter * torch.randn(x.shape, generator=gen, device=DEV)
            st, _ = model(x, torch.from_numpy(v0[b]).to(DEV))
            P.append((st * torch.tensor(TS, device=DEV)).float().cpu().numpy())
        acc += np.concatenate(P)
    model.eval()  # restore
    return acc / n_samples


def run_split(name, idx):
    T, M = tgt[idx], msk[idx]
    print(f"\n### {name} ({len(idx)} windows)")
    base = metrics(predict(idx, 1), T, M)
    print(f"  det (baseline)      : {json.dumps(base)}")
    for N in SWEEP:
        mc = metrics(predict(idx, N, mc_dropout=True, seed=1), T, M)
        print(f"  mcdrop-{N:<4d}         : {json.dumps(mc)}")
    for N in [SWEEP[-1]]:
        jt = metrics(predict(idx, N, jitter=0.05, seed=2), T, M)
        print(f"  jitter-{N:<4d} (s=.05) : {json.dumps(jt)}")
        cb = metrics(predict(idx, N, mc_dropout=True, jitter=0.05, seed=3), T, M)
        print(f"  combo-{N:<4d}          : {json.dumps(cb)}")
    return base


SWEEP = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["10", "50", "100", "500"])]
print(f"device {DEV} | MC-dropout N sweep {SWEEP}")
wp = np.where((yr >= 2020) & (bs == "WP"))[0]
ab = np.where(yr >= 2020)[0]
run_split("WP-2020+", wp)
run_split("all-basin 2020+", ab)
