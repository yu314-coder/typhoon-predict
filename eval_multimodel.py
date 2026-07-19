"""Does pooling v17 + v18 + v19 across architectures beat the best single one?

No training. Each of the three is already a seed-ensemble; the question is whether a DEEP ensemble
across the three -- independently trained models, same initialisation, so their outputs are
exchangeable in the epistemic sense FGN relies on -- reduces error below v17's 462.8 km.

The models are rebuilt exactly as _tip_tracks.py does: v17/v18/v19 share one architecture defined
in colab_train_v17.ipynb, loaded with the 'inner.' prefix stripped. Steering is decoded with the
notebook's own recipe, clip(q[:,:4]/31.75, -4, 4), so the input matches what they trained on.
"""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8)
DEVICE = torch.device("cpu")

nb = json.load(open("colab_train_v17.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
     "DEVICE": DEVICE, "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p, src, re.S).group(0), G)


def load(ck):
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("inner.")} or sd
    m = G["TrackFormerV17"]().to(DEVICE).eval()
    m.load_state_dict(sd); return m


z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
sids = z["storm_id"].astype(str); years = z["year"].astype(int)
basins = z["basin"].astype(str); nl = z["n_leads"].astype(int)
tmean = z["track_mean"].astype("float32"); tstd = z["track_std"].astype("float32")
v0 = track[:, -1, 2:4] * tstd[2:4] + tmean[2:4]
vp = track[:, -2, 2:4] * tstd[2:4] + tmean[2:4]
vpair = np.concatenate([v0, vp], 1).astype("float32")
SLP = np.clip(np.load("track_build/steer5_int8.npz")["q"][:, :4].astype("float32") / 31.75, -4, 4)
assert len(SLP) == len(track), f"steer {len(SLP)} != windows {len(track)}"

fy = {s: int(years[sids == s].min()) for s in np.unique(sids)}
EV = np.array([i for i in range(len(sids))
               if fy[sids[i]] >= 2020 and nl[i] == 20 and basins[i] in ("WP", "EP")])
T = np.cumsum(target[EV][..., :2], 1)
SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12)
print(f"test set: {len(EV)} WP+EP 2020+ full-horizon windows\n")

MODELS = {"v17": [f"downloads/x/v17_seed{i}.pt" for i in range(5)],
          "v18": [f"downloads/x/v18_seed{i}.pt" for i in range(8)],
          "v19": [f"downloads/x/v19_seed{i}.pt" for i in range(5)]}


@torch.no_grad()
def preds(models):
    """Per-window mean displacement over a list of models -> cumulative track. [len(EV),20,2]."""
    P = []
    for i in range(0, len(EV), 128):
        j = EV[i:i + 128]
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
        s = torch.stack([m(*a)[0] for m in models]).mean(0)
        P.append((s * SC).numpy())
    return np.cumsum(np.concatenate(P)[..., :2], 1)


def err(P):
    return float(np.sqrt(((P - T) ** 2).sum(-1)).mean())


# ---- each architecture on its own (reproduce the published numbers as a sanity check) ----
loaded = {k: [load(c) for c in v] for k, v in MODELS.items()}
single = {}
for k, ms in loaded.items():
    single[k] = err(preds(ms))
    print(f"{k:14s} {len(ms)} seeds   {single[k]:.2f} km   (published: "
          f"{ {'v17':462.8,'v18':466.2,'v19':466.7}[k] })")
best_single = min(single.values())

# ---- the deep ensemble: pool every seed of every architecture, equal weight ----
allm = [m for ms in loaded.values() for m in ms]
e_all = err(preds(allm))
print(f"\n{'v17+v18+v19':14s} {len(allm)} models  {e_all:.2f} km   "
      f"vs best single {best_single:.2f}: {e_all - best_single:+.2f} km")

# ---- average the three architecture-means equally, so 8 v18 seeds don't dominate ----
@torch.no_grad()
def arch_mean_preds():
    P = []
    for i in range(0, len(EV), 128):
        j = EV[i:i + 128]
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
        per = [torch.stack([m(*a)[0] for m in loaded[k]]).mean(0) for k in loaded]
        P.append((torch.stack(per).mean(0) * SC).numpy())
    return np.cumsum(np.concatenate(P)[..., :2], 1)


e_bal = err(arch_mean_preds())
print(f"{'balanced 3-arch':14s} {'':8s} {e_bal:.2f} km   "
      f"vs best single {best_single:.2f}: {e_bal - best_single:+.2f} km")
json.dump({"single": single, "pooled": e_all, "balanced": e_bal},
          open("track_build/multimodel.json", "w"))
