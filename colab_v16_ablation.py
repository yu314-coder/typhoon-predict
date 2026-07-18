"""v16 SST ablation + per-lead diagnostics. Run INSIDE the finished colab_train_v16 session:

    !wget -q -O abl.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v16_ablation.py
    exec(open('abl.py').read())

It reuses the notebook's already-loaded globals (SLP, track, target, mask, loader, train_one,
TrackFormerV16, CKPTS, te_idx, basins, ...), so nothing is re-downloaded or re-parsed.

The ablation ZEROES the SST channel rather than building a 4-channel model. That keeps the
architecture, parameter count and seeds byte-identical, so any difference is the *information*
SST carries and not a change in capacity. Zero is also exactly how the pipeline represents
"this field is unavailable", so the ablated model sees a state it was already trained to handle.
"""
import numpy as np, torch, json, time

LEADS = np.arange(1, 21) * 6                     # +6h .. +120h
full = z["n_leads"].astype(int) == 20
EVAL = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
print(f"ablation eval set: WP+EP 2020+, {len(EVAL)} full-horizon windows\n", flush=True)


@torch.no_grad()
def per_lead(models, idx):
    """Per-lead-time metrics. Track error is on the CUMULATIVE position, as everywhere else."""
    P, T, M = [], [], []
    for tr, v0, sp, tg, m in loader(idx, False):
        tr, v0, sp = tr.to(DEVICE), v0.to(DEVICE), sp.to(DEVICE)
        s = torch.stack([mm(tr, v0, sp)[0] for mm in models]).mean(0)
        P.append((s * TARGET_SCALE).float().cpu().numpy()); T.append(tg.numpy()); M.append(m.numpy())
    P, T, M = np.concatenate(P), np.concatenate(T), np.concatenate(M)
    pt, tt = np.cumsum(P[..., :2], 1), np.cumsum(T[..., :2], 1)
    out = {"track": np.sqrt(((pt - tt) ** 2).sum(-1)).mean(0)}
    for i, nm in [(2, "vmax"), (3, "pressure"), (4, "rmw")]:
        v = M[..., i] > 0.5
        out[nm] = np.array([np.abs(P[:, L, i] - T[:, L, i])[v[:, L]].mean() if v[:, L].any() else np.nan
                            for L in range(20)])
    rm = M[..., 5:17] > 0.5
    out["radii"] = np.array([np.abs(P[:, L, 5:17] - T[:, L, 5:17])[rm[:, L]].mean() if rm[:, L].any() else np.nan
                             for L in range(20)])
    return out


# ---- 1. v16 WITH SST, before we touch anything ----------------------------
print("evaluating v16 (SST on) ...", flush=True)
m5 = [load_model(c) for c in CKPTS]
R5 = per_lead(m5, EVAL)

# ---- 2. zero the SST channel and retrain the same three seeds -------------
print("\nzeroing SST channel (ch4) and retraining 3 seeds ...\n", flush=True)
assert SLP.shape[1] == 5, f"expected 5 channels, got {SLP.shape[1]}"
sst_rms_before = float(np.sqrt((SLP[:, 4] ** 2).mean()))
SLP[:, 4] = 0.0
print(f"  SST channel rms {sst_rms_before:.4f} -> {float(np.sqrt((SLP[:,4]**2).mean())):.4f}", flush=True)

ABL = []
t0 = time.time()
for seed in (0, 1, 2):
    c = f"/content/v16nosst_seed{seed}.pt"
    train_one(seed, c); ABL.append(c)
print(f"\nablation trained in {(time.time()-t0)/60:.1f} min", flush=True)

print("evaluating v16-noSST ...", flush=True)
m4 = [load_model(c) for c in ABL]
R4 = per_lead(m4, EVAL)

# ---- 3. restore SST so the session is left usable ------------------------
_q = np.load("/content/d/steer5_int8.npz")
SLP[:, 4] = np.clip(_q["q"][:, 4].astype("float32") / 31.75, -STEER_CLIP, STEER_CLIP)
del _q
print(f"SST channel restored (rms {float(np.sqrt((SLP[:,4]**2).mean())):.4f})\n", flush=True)

# ---- 4. verdict -----------------------------------------------------------
print("=" * 74)
print("SST ABLATION — WP+EP 2020+, ensemble of 3 seeds, mean over all 20 leads")
print("=" * 74)
print(f"{'metric':16s} {'with SST':>12s} {'no SST':>12s} {'delta':>10s} {'':>8s}")
UNIT = {"track": "km", "vmax": "kt", "pressure": "hPa", "rmw": "km", "radii": "km"}
verdict = {}
for k in ["track", "vmax", "pressure", "rmw", "radii"]:
    a, b = float(np.nanmean(R5[k])), float(np.nanmean(R4[k]))
    verdict[k] = (a, b)
    tag = "SST helps" if a < b else "SST hurts" if a > b else ""
    print(f"{k:16s} {a:12.2f} {b:12.2f} {a-b:+10.2f} {UNIT[k]:>5s}  {tag}")
print("=" * 74)
print("negative delta = WITH SST is better\n")

# ---- 5. plots -------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
PANELS = [("track", "Track error (km)"), ("vmax", "Vmax MAE (kt)"),
          ("pressure", "Pressure MAE (hPa)"), ("rmw", "RMW MAE (km)"),
          ("radii", "Wind-radii MAE (km)")]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (k, lab) in zip(axes.ravel(), PANELS):
    ax.plot(LEADS, R5[k], "-o", ms=3, lw=1.8, color="#0f6f80", label="v16 (with SST)")
    ax.plot(LEADS, R4[k], "--s", ms=3, lw=1.6, color="#b04a18", label="v16 (SST removed)")
    ax.set_title(lab, fontsize=10, loc="left")
    ax.set_xlabel("lead time (h)"); ax.set_xticks([24, 48, 72, 96, 120])
axes.ravel()[0].legend(frameon=False, fontsize=8)
ax = axes.ravel()[5]
d = [100 * (np.nanmean(R4[k]) - np.nanmean(R5[k])) / np.nanmean(R4[k]) for k, _ in PANELS]
ax.barh([p[1].split(" ")[0] for p in PANELS], d,
        color=["#0f6f80" if x > 0 else "#b04a18" for x in d])
ax.axvline(0, color="#444", lw=.8); ax.set_title("SST benefit (%)", fontsize=10, loc="left")
ax.set_xlabel("% improvement from adding SST"); ax.grid(axis="y", alpha=0)
fig.suptitle("v16 SST ablation — WP+EP 2020+, identical architecture and seeds, SST channel zeroed",
             fontsize=11, x=.01, ha="left")
fig.tight_layout()
fig.savefig("/content/v16_sst_ablation.png", bbox_inches="tight")
try: fig.savefig(f"{DATA}/v16_sst_ablation.png", bbox_inches="tight")
except Exception as e: print("(drive copy failed:", e, ")")
plt.show()

json.dump({k: {"with_sst": list(map(float, R5[k])), "no_sst": list(map(float, R4[k]))}
           for k in R5}, open("/content/v16_ablation.json", "w"), indent=1)
print("saved /content/v16_sst_ablation.png and v16_ablation.json")
