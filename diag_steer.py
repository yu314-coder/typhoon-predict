"""Does v17 use the steering patch's spatial DETAIL, or mostly its mean flow?

If replacing every 17x17 patch with its spatial mean (a spatially-flat patch carrying only the
average flow) barely changes v17's error, then the CNN's spatial resolution is not doing much work
-- most of what it extracts is the bulk steering vector, and a low-dimensional bottleneck would
keep the signal while removing the capacity that overfits. If the error jumps, the detail is real
and a bottleneck would cost accuracy.

Run on a random 1200-window slice of the test set with v17's 5 seeds; CPU, a couple of minutes.
"""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8); np.random.seed(0)
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
    m = G["TrackFormerV17"]().to(DEVICE).eval(); m.load_state_dict(sd); return m


z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
sids = z["storm_id"].astype(str); years = z["year"].astype(int)
basins = z["basin"].astype(str); nl = z["n_leads"].astype(int)
tmean = z["track_mean"].astype("float32"); tstd = z["track_std"].astype("float32")
vpair = np.concatenate([track[:, -1, 2:4] * tstd[2:4] + tmean[2:4],
                        track[:, -2, 2:4] * tstd[2:4] + tmean[2:4]], 1).astype("float32")
SLP = np.clip(np.load("track_build/steer5_int8.npz")["q"][:, :4].astype("float32") / 31.75, -4, 4)

fy = {s: int(years[sids == s].min()) for s in np.unique(sids)}
EVall = np.array([i for i in range(len(sids))
                  if fy[sids[i]] >= 2020 and nl[i] == 20 and basins[i] in ("WP", "EP")])
EV = np.sort(np.random.choice(EVall, 1200, replace=False))
T = np.cumsum(target[EV][..., :2], 1)
SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12)
MS = [load(f"downloads/x/v17_seed{i}.pt") for i in range(5)]
print(f"{len(EV)} test windows, v17 x5\n")


@torch.no_grad()
def run(slp_of):
    P = []
    for i in range(0, len(EV), 128):
        j = EV[i:i + 128]
        sp = slp_of(SLP[j])
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(sp)]
        s = torch.stack([m(*a)[0] for m in MS]).mean(0)
        P.append((s * SC).numpy())
    return np.cumsum(np.concatenate(P)[..., :2], 1)


def e(P):
    return float(np.sqrt(((P - T) ** 2).sum(-1)).mean())


real = run(lambda s: s)                                             # patches as trained
flat = run(lambda s: np.broadcast_to(s.mean((2, 3), keepdims=True), s.shape).astype("float32"))
zero = run(lambda s: np.zeros_like(s))                              # no steering at all

print(f"  real patches        {e(real):.2f} km")
print(f"  spatial mean only   {e(flat):.2f} km   ({e(flat)-e(real):+.2f} vs real)")
print(f"  steering zeroed     {e(zero):.2f} km   ({e(zero)-e(real):+.2f} vs real)")
print()
print("read: if 'spatial mean only' is close to 'real', the CNN barely uses spatial detail and a")
print("low-dim steering bottleneck should keep the signal. If it is close to 'zeroed' instead, the")
print("detail is what's carrying the steering and a bottleneck would throw the signal away.")
