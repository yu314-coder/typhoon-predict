"""Where does the track error actually come from? Along-track vs cross-track, per lead.

Total error is one number and hides the mechanism. Operational TC verification splits the miss
into ALONG-track (too fast / too slow along the predicted direction) and CROSS-track (steered the
wrong way). They call for completely different fixes: along-track is a speed/timing problem,
cross-track is a steering problem. Two models with identical mean error can be wrong in ways that
need opposite work.

Signed means are reported alongside RMS because a signed bias is fixable by calibration while
symmetric scatter is not.

Reuses the forward validated in _v22tracks.py against Colab (452.47 / 443.62 / 443.41).
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
vpair = G["vpair"]; te_idx = G["te_idx"]; basins = G["basins"]; z = G["z"]
SC = G["TARGET_SCALE"]

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
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = np.cumsum(target[wpep][..., :2], 1)                       # truth displacement, km


@torch.no_grad()
def predict(cls, tag):
    ms = []
    for s in range(5):
        m = cls().eval()
        m.load_state_dict(torch.load(f"downloads/x/{tag}_seed{s}.pt", map_location="cpu",
                                     weights_only=False)["model"])
        ms.append(m)
    P = []
    for i in range(0, len(wpep), 128):
        j = wpep[i:i + 128]
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().numpy())
    return np.cumsum(np.concatenate(P)[..., :2], 1)


# Decompose in the frame of the OBSERVED motion at each lead: unit vector along the truth
# displacement, and its left-hand normal. Using the observed rather than the forecast direction
# keeps the frame identical across models, so the numbers are comparable between arms.
norm = np.linalg.norm(T, axis=-1, keepdims=True)
u = np.divide(T, np.maximum(norm, 1e-6))                       # along-track unit
nvec = np.stack([-u[..., 1], u[..., 0]], -1)                   # cross-track unit (left of motion)

LEADS = [(3, "24 h"), (7, "48 h"), (11, "72 h"), (15, "96 h"), (19, "120 h")]
NHC = {"24 h": 52, "48 h": 84, "72 h": 124, "96 h": 170, "120 h": 214}

print("Track error decomposed in the observed-motion frame, WP+EP 2020+, "
      f"{len(wpep)} windows, 5-seed ensembles\n")
print(f"{'':6s} {'lead':>6s} {'total':>7s} {'along':>8s} {'cross':>8s} "
      f"{'along bias':>11s} {'cross bias':>11s} {'NHC':>5s}")

store = {}
for tag, (cls, ck) in ARMS.items():
    C = predict(cls, ck)
    store[tag] = C
    E = C - T
    al = (E * u).sum(-1)                                       # + = forecast ran ahead
    cr = (E * nvec).sum(-1)                                     # + = forecast went left
    for L, nm in LEADS:
        tot = float(np.hypot(E[:, L, 0], E[:, L, 1]).mean())
        print(f"{tag:6s} {nm:>6s} {tot:7.0f} {float(np.abs(al[:, L]).mean()):8.0f} "
              f"{float(np.abs(cr[:, L]).mean()):8.0f} {float(al[:, L].mean()):+11.0f} "
              f"{float(cr[:, L].mean()):+11.0f} {NHC[nm]:5d}")
    print()

# How much of the squared error is along vs cross at the full horizon?
print("share of squared error at 120 h")
for tag in ARMS:
    E = store[tag] - T
    al = (E * u).sum(-1)[:, 19]; cr = (E * nvec).sum(-1)[:, 19]
    a2, c2 = float((al ** 2).mean()), float((cr ** 2).mean())
    print(f"  {tag}  along {100*a2/(a2+c2):4.1f}%   cross {100*c2/(a2+c2):4.1f}%")

# Speed bias: is the model systematically slow? Compare predicted vs observed path length.
print("\npredicted vs observed 120 h displacement (km)")
for tag in ARMS:
    pl = float(np.linalg.norm(store[tag][:, 19], axis=-1).mean())
    tl = float(np.linalg.norm(T[:, 19], axis=-1).mean())
    print(f"  {tag}  predicted {pl:6.0f}   observed {tl:6.0f}   ratio {pl/tl:.3f}")

np.savez("track_build/errdecomp.npz", T=T, wpep=wpep, **{k: v for k, v in store.items()})
print("\nwrote track_build/errdecomp.npz")
