"""Is TrackFormer data-limited? A learning curve over TRAINING STORMS.

    !wget -q -O /content/curve.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/<SHA>/colab_v29_curve.py
    import os; exec(open('/content/curve.py').read())

THE QUESTION THIS SETTLES. Proposal on the table: generate ~10x training data by adding
covariance-shaped noise to existing samples. Augmentation is a REGULARISER -- it adds no
information. It can only help if the model is actually short of data. So measure that first,
before building anything.

  curve still falling steeply at 100%  -> more data helps; augmentation is worth building
  curve flat by 50%                    -> the model has all the data it can use; 10x perturbed
                                          copies of the same storms buy nothing

Prior that makes this worth checking rather than assuming: v18 was "v17 + EMA + stronger dropout
+ input jitter" and scored +3.78 km WORSE (CI [-2.56, +9.75]). Input jitter has already lost once
here, though with isotropic noise rather than covariance-shaped.

SUBSAMPLE BY STORM, NOT BY WINDOW. Windows from one storm are strongly correlated -- that is why
the paired bootstrap resamples storms. Dropping random windows would leave the effective sample
size almost unchanged and flatten the curve artificially, which would answer the question wrong in
the direction of "not data limited". Whole storms are removed, so the effective sample size falls
with the fraction.

Validation and test are untouched at every point, so the numbers are comparable down the column.
One seed per point: this is a shape question, not a 9-km question, and per-seed spread here is
~19 km. Read the SHAPE, not the individual values.
"""

import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
FRACS = [float(x) for x in os.environ.get("V29_FRACS", "0.25,0.5,1.0").split(",")]
SEED = int(os.environ.get("V29_SEED", "0"))
W_FLOW = float(os.environ.get("W_FLOW", "0.3"))
KM6H = 6 * 3600 / 1000.0

for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)

nb = json.load(open(urllib.request.urlretrieve(f"{RAW}/colab_train_v17.ipynb",
                                               "/content/_v17.ipynb")[0]))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
assert body.count("steer5_int8.npz") == 1
body = body.replace('"/content/d/steer5_int8.npz"', '"/content/dlm4_int8.npz"')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os, "json": json,
     "time": time, "math": __import__("math")}
exec(compile(body, "<v17-notebook>", "exec"), G)

DEVICE = G["DEVICE"]; SC = G["TARGET_SCALE"]
Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
mask = G["mask"]; vpair = G["vpair"]; z = G["z"]; mirror = G["mirror"]
tr_idx, va_idx, te_idx = G["tr_idx"], G["va_idx"], G["te_idx"]
basins = G["basins"]
EPOCHS, PATIENCE, BATCH = G["EPOCHS"], G["PATIENCE"], G["BATCH"]
LR, WEIGHT_DECAY, MIRROR_P = G["LR"], G["WEIGHT_DECAY"], G["MIRROR_P"]

_lf = np.load("/content/lead_flow.npz")
FLOW_T = _lf["flow"].astype("float32"); FLOW_M = _lf["got"].astype("float32")
DSC = np.load("/content/dlm4_int8.npz")["scale"][2:4].astype("float32")
_ii, _jj = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
_d = np.hypot(_ii, _jj) * 2.5
ANN = torch.tensor(((_d >= 3.0) & (_d <= 8.0)).astype("float32"), device=DEVICE)

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
_g = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
      "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, urllib.request.urlopen(f"{RAW}/colab_v26_train.py").read().decode(),
               re.S).group(0), _g)
V21 = _g["TrackFormerCoT"]

sid = z["storm_id"].astype(str)
tr_storms = np.unique(sid[tr_idx])
print(f"training pool: {len(tr_idx):,} windows over {len(tr_storms):,} storms", flush=True)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])


# ---- reuse v26's dataset, loss and training loop VERBATIM ------------------------------------
# The model returns THREE values (s, ls, flow_pred) and the loss is v17's 4-term total_loss with a
# smooth-l1 flow term on top. A hand-rolled loop here would train a different objective and the
# curve would not be comparable to v21's 443.62 km. So the whole block is extracted from the file
# that produced those weights, and only `tr_idx` is swapped for the subsample.
_v26 = urllib.request.urlopen(f"{RAW}/colab_v26_train.py").read().decode()
_blk = re.search(r"class DS\(torch\.utils\.data\.Dataset\):.*?\n    return ckpt\n", _v26, re.S)
assert _blk, "could not extract v26's training block -- refusing to improvise one"
_SRC = _blk.group(0)
for _need in ("def total_loss", "def train_one", "s, ls, fp = model(tr, v0, sp)"):
    assert _need in _SRC, f"extracted block is missing {_need!r}"

NS = dict(torch=torch, nn=nn, F=F, np=np, os=os, time=time, math=G["math"], G=G,
          DEVICE=DEVICE, BATCH=BATCH, EPOCHS=EPOCHS, PATIENCE=PATIENCE, LR=LR,
          WEIGHT_DECAY=WEIGHT_DECAY, MIRROR_P=MIRROR_P, W_FLOW=W_FLOW,
          track=track, target=target, mask=mask, vpair=vpair, SLP=SLP, mirror=mirror,
          FLOW_T=FLOW_T, FLOW_M=FLOW_M, va_idx=va_idx, TrackFormerCoT=V21,
          tmean=G["tmean"], tstd=G["tstd"])
exec(_SRC, NS)


@torch.no_grad()
def track_err(m):
    P = []
    for i in range(0, len(wpep), 128):
        j = wpep[i:i + 128]
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE)]
        P.append((m(*a)[0] * SC).float().cpu().numpy())
    C = np.cumsum(np.concatenate(P)[..., :2], 1)
    T = np.cumsum(target[wpep][..., :2], 1)
    return float(np.sqrt(((C - T) ** 2).sum(-1)).mean())


def train_on(idx, tag):
    NS["tr_idx"] = idx                      # the ONLY thing that differs between points
    t0 = time.time()
    ck = NS["train_one"](SEED, f"/content/curve_{tag}.pt")
    m = V21().to(DEVICE)
    m.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=False)["model"])
    m.eval()
    return track_err(m), (time.time() - t0) / 60


print(f"\nlearning curve, seed {SEED}, 1 seed per point (read the SHAPE, per-seed spread ~19 km)\n")
print(f"{'frac':>6s} {'storms':>7s} {'windows':>9s} {'test km':>9s} {'min':>6s}")
rows = []
rng = np.random.default_rng(0)
order = rng.permutation(len(tr_storms))           # one nested sequence: 25% is inside 50% is
for f in sorted(FRACS):                            # inside 100%, so the curve is not confounded
    keep = set(tr_storms[order[:max(1, int(round(f * len(tr_storms))))]])
    idx = np.array([i for i in tr_idx if sid[i] in keep])
    e, mins = train_on(idx, f"{int(f*100)}")
    rows.append((f, len(keep), len(idx), e, mins))
    print(f"{f:6.2f} {len(keep):7d} {len(idx):9,d} {e:9.2f} {mins:6.1f}", flush=True)

print("\ninterpretation")
if len(rows) >= 2:
    d = rows[-1][3] - rows[-2][3]
    print(f"  last step ({rows[-2][0]:.0%} -> {rows[-1][0]:.0%}) moved the error {d:+.2f} km")
    print("  a step much smaller than the ~19 km seed spread means the curve has flattened")
    print("  and more data -- real OR augmented -- is not the binding constraint")
json.dump([{"frac": r[0], "storms": r[1], "windows": r[2], "km": r[3]} for r in rows],
          open("/content/curve.json", "w"), indent=1)
try:
    from google.colab import files
    files.download("/content/curve.json")
except Exception as ex:
    print("download skipped:", ex)
