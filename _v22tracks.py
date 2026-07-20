"""Export v10 / v21 / v22 tracks for the map, and VALIDATE the local forward first.

The model classes are extracted VERBATIM from the training scripts by regex rather than
rewritten from memory. Rewriting the forward from memory has silently broken this twice
(v15, and v21 at 0.655 max diff), so the only trustworthy source is the file that trained
the weights.

Nothing is exported until the local ensemble errors reproduce the Colab numbers:
    v20 452.47   v21 443.62   v22 443.40
If a local number disagrees the forward is wrong and the maps would be wrong too.
"""
import json, re, math, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
DEVICE = torch.device("cpu"); R = 111.2; KM6H = 6 * 3600 / 1000.0

# ---- v17 machinery + data, exactly as the training scripts build it ----
nb = json.load(open("colab_train_v17.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
assert body.count("steer5_int8.npz") == 1
body = body.replace('"/content/d/steer5_int8.npz"', '"track_build/dlm4_int8.npz"')
body = body.replace('"/content/d/track_windows_v13.npz"', '"track_build/track_windows_v13.npz"')
assert body.count('DEVICE = torch.device("cuda")') == 1
body = body.replace('DEVICE = torch.device("cuda")', 'DEVICE = torch.device("cpu")')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os,
     "json": json, "time": __import__("time"), "math": math}
exec(compile(body, "<v17-notebook>", "exec"), G)

Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
vpair = G["vpair"]; te_idx = G["te_idx"]; basins = G["basins"]; z = G["z"]
TARGET_SCALE = G["TARGET_SCALE"]

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))

CLS_RE = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"


def build_cls(path, rounds):
    """Compile the CoT class straight out of the training script it was trained by."""
    src = re.search(CLS_RE, open(path).read(), re.S).group(0)
    g = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G,
         "ANN": ANN, "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": rounds, "USE_FLOW": 1}
    exec(src, g)
    return g["TrackFormerCoT"]


V21 = build_cls("colab_v26_train.py", 0)     # explicit CoT only
V22 = build_cls("colab_v27_train.py", 2)     # + latent CoT, R=2


def load(ck, cls):
    m = cls().eval()
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
    return m                                  # strict=True: key mismatch is a hard error


full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
SC = TARGET_SCALE


@torch.no_grad()
def track_err(ms):
    P = []
    for i in range(0, len(wpep), 128):
        j = wpep[i:i + 128]
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().numpy())
    C = np.cumsum(np.concatenate(P)[..., :2], 1)
    T = np.cumsum(target[wpep][..., :2], 1)
    return float(np.sqrt(((C - T) ** 2).sum(-1)).mean())


print(f"WP+EP 2020+, {len(wpep)} windows (Colab reported 3763)", flush=True)
ARMS = {"v20": ([f"downloads/x/v20_seed{i}.pt" for i in range(5)], Base,  452.47),
        "v21": ([f"downloads/x/v21_seed{i}.pt" for i in range(5)], V21,   443.62),
        "v22": ([f"downloads/x/v22_seed{i}.pt" for i in range(5)], V22,   443.40)}

MODELS, ok = {}, True
for tag, (cks, cls, expect) in ARMS.items():
    ms = [load(c, cls) for c in cks]
    MODELS[tag] = ms
    e = track_err(ms)
    d = e - expect
    flag = "OK" if abs(d) < 0.05 else "*** MISMATCH ***"
    print(f"  {tag}  local {e:8.2f} km | colab {expect:7.2f} | diff {d:+.2f}  {flag}", flush=True)
    ok &= abs(d) < 0.05

if not ok:
    sys.exit("local forward does not reproduce Colab -- refusing to export tracks")
print("forward validated against all three Colab numbers\n", flush=True)

# ---- v10, a different architecture entirely ----
def build_v10():
    src = open("train_track_v10.py").read()
    g = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
         "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
    for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(pat, src, re.S).group(0), g)
    return g["TrackFormerV9"]

m10 = build_v10()()
m10.load_state_dict(torch.load("track_build/track_v10_best.pt", map_location="cpu",
                               weights_only=False)["model"]); m10.eval()

sid = z["storm_id"].astype(str); nl = z["n_leads"].astype(int); bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
STORMS = [("2026182N09163", "Bavi"), ("1986228N19120", "Wayne"),
          ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]

os.makedirs("track_build/v22map", exist_ok=True)
for tag in ("v10", "v21", "v22"):
    out = {}
    for s, nm in STORMS:
        k = np.where((sid == s) & (nl == 20))[0]; k = k[np.argsort(bt[k])]
        if not len(k):
            continue
        P = []
        with torch.no_grad():
            for i in range(0, len(k), 64):
                j = k[i:i + 64]
                if tag == "v10":
                    sv, _ = m10(torch.from_numpy(track[j]), torch.from_numpy(vpair[j]))
                else:
                    a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]),
                         torch.from_numpy(SLP[j])]
                    sv = torch.stack([m(*a)[0] for m in MODELS[tag]]).mean(0)
                P.append((sv * SC).float().numpy())
        A = np.concatenate(P)
        cE, cN = np.cumsum(A[..., 0], 1), np.cumsum(A[..., 1], 1)
        T = target[k]; tE, tN = np.cumsum(T[..., 0], 1), np.cumsum(T[..., 1], 1)
        lats, lons = [], []
        for a2 in range(len(k)):
            la = bla[k[a2]] + cN[a2] / R
            lo = blo[k[a2]] + cE[a2] / (R * np.cos(np.radians((bla[k[a2]] + la) / 2)))
            lats.append(np.round(la, 3).tolist()); lons.append(np.round(lo, 3).tolist())
        err = float(np.hypot(cE[:, 19] - tE[:, 19], cN[:, 19] - tN[:, 19]).mean())
        out[nm] = {"lat": lats, "lon": lons, "base_time": bt[k].tolist(),
                   "base_lat": np.round(bla[k], 3).tolist(),
                   "base_lon": np.round(blo[k], 3).tolist(),
                   "err120_mean": err, "n": int(len(k))}
        print(f"  {tag:4s} {nm:11s} {len(k):3d} fc | mean 120h {err:6.0f} km", flush=True)
    json.dump(out, open(f"track_build/v22map/{tag}_tracks.json", "w"))
print("\nwrote track_build/v22map/")
