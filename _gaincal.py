"""Does correcting the systematic slow bias help? Per-lead gain, fit on VALIDATION only.

The decomposition showed every arm running behind the storm: -297 km along-track at 120 h on a
2088 km mean displacement. A bias that large and that monotone in lead is not scatter, it is
shrinkage -- the signature of an MSE-trained model hedging under uncertainty, made worse by
averaging five seeds whose tracks diverge (the mean of diverging paths is shorter than any of
them).

If that is what it is, one scalar per lead should recover part of it:  C' = g_L * C.

THE GAIN IS FIT ON THE VALIDATION SPLIT AND APPLIED UNCHANGED TO TEST. Fitting it on test would
be guaranteed to "work" and would mean nothing. Validation is 2016-2019, test is 2020+, so this
also asks whether the bias is stable across a distribution shift -- if it is not, the gain fitted
on validation will not transfer and that is the honest answer.

Closed form: minimising sum ||g*C - T||^2 over g gives g = <C,T> / <C,C>, per lead.
"""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
nb = json.load(open("colab_train_v17.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
body = body.replace('"/content/d/steer5_int8.npz"', '"track_build/dlm4_int8.npz"')
body = body.replace('"/content/d/track_windows_v13.npz"', '"track_build/track_windows_v13.npz"')
body = body.replace('DEVICE = torch.device("cuda")', 'DEVICE = torch.device("cpu")')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os,
     "json": json, "time": __import__("time"), "math": math}
exec(compile(body, "<v17-notebook>", "exec"), G)

Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
vpair = G["vpair"]; basins = G["basins"]; z = G["z"]; SC = G["TARGET_SCALE"]
va_idx, te_idx = G["va_idx"], G["te_idx"]

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))
CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"


def build_cls(path, rounds):
    g = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
         "DSC": DSC, "KM6H": 6 * 3600 / 1000.0, "R_ROUNDS": rounds, "USE_FLOW": 1}
    exec(re.search(CLS, open(path).read(), re.S).group(0), g)
    return g["TrackFormerCoT"]


ARMS = {"v20": (Base, "v20"), "v21": (build_cls("colab_v26_train.py", 0), "v21"),
        "v22": (build_cls("colab_v27_train.py", 2), "v22")}
full = z["n_leads"].astype(int) == 20
VA = np.array([i for i in va_idx if full[i] and basins[i] in ("WP", "EP")])
TE = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
print(f"validation {len(VA)} windows (fit)   test {len(TE)} windows (apply)\n")


@torch.no_grad()
def cum(cls, tag, idx, single=False):
    ms = []
    for s in range(1 if single else 5):
        m = cls().eval()
        m.load_state_dict(torch.load(f"downloads/x/{tag}_seed{s}.pt", map_location="cpu",
                                     weights_only=False)["model"])
        ms.append(m)
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().numpy())
    return np.cumsum(np.concatenate(P)[..., :2], 1)


def err(C, T):
    return float(np.sqrt(((C - T) ** 2).sum(-1)).mean())


print(f"{'arm':5s} {'plain':>8s} {'gain-cal':>9s} {'delta':>7s}   per-lead gain (24/72/120 h)")
res = {}
for tag, (cls, ck) in ARMS.items():
    Cv, Tv = cum(cls, ck, VA), np.cumsum(target[VA][..., :2], 1)
    Ct, Tt = cum(cls, ck, TE), np.cumsum(target[TE][..., :2], 1)
    # closed-form least-squares gain per lead, from VALIDATION
    g = np.array([float((Cv[:, L] * Tv[:, L]).sum() / (Cv[:, L] * Cv[:, L]).sum())
                  for L in range(20)])
    e0, e1 = err(Ct, Tt), err(Ct * g[None, :, None], Tt)
    res[tag] = (e0, e1, g)
    print(f"{tag:5s} {e0:8.2f} {e1:9.2f} {e1-e0:+7.2f}   "
          f"{g[3]:.3f} / {g[11]:.3f} / {g[19]:.3f}")

# Is the shrinkage caused by ENSEMBLING? Compare one seed against the five-seed mean.
print("\nis the slow bias an ensembling artifact? 120 h displacement ratio (pred/obs)")
Tt = np.cumsum(target[TE][..., :2], 1)
tl = float(np.linalg.norm(Tt[:, 19], axis=-1).mean())
for tag, (cls, ck) in ARMS.items():
    c1 = cum(cls, ck, TE, single=True); c5 = cum(cls, ck, TE)
    r1 = float(np.linalg.norm(c1[:, 19], axis=-1).mean()) / tl
    r5 = float(np.linalg.norm(c5[:, 19], axis=-1).mean()) / tl
    print(f"  {tag}  1 seed {r1:.3f}   5 seeds {r5:.3f}   shrink from ensembling {r5-r1:+.3f}")

json.dump({k: {"plain": v[0], "cal": v[1], "gain": v[2].tolist()} for k, v in res.items()},
          open("track_build/gaincal.json", "w"), indent=1)
print("\nwrote track_build/gaincal.json")
