"""Local CPU verification for v35 (v23's exact architecture, intensity-loss reweighted) before
pushing to Colab. Architecture is UNCHANGED from v23 (already exhaustively verified by v23's own
init checks in colab_v28_train.py), so this only needs to verify the NEW total_loss logic:

1. W_INT=1.0 reproduces v17's original total_loss EXACTLY (byte-for-byte on the non-flow part) --
   the built-in "does this collapse to v23" check for a loss-only change.
2. W_INT>1 actually scales the intensity term's contribution as expected (not a no-op).
3. Gradients reach both a track-facing parameter (track_res) and the intensity/log-scale head from
   a single backward() call -- the reweighted loss doesn't accidentally disconnect either one.
"""
import re, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)

nb = json.load(open("colab_train_v17.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
body = body.replace('"/content/d/steer5_int8.npz"', '"track_build/dlm4_int8.npz"')
body = body.replace('"/content/d/track_windows_v13.npz"', '"track_build/track_windows_v13.npz"')
body = body.replace('DEVICE = torch.device("cuda")', 'DEVICE = torch.device("cpu")')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": __import__("os"),
     "json": json, "time": __import__("time"), "math": math}
exec(compile(body, "<v17-notebook>", "exec"), G)
Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
mask = G["mask"]; vpair = G["vpair"]; z = G["z"]; TARGET_SCALE = G["TARGET_SCALE"]

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))
KM6H = 6 * 3600 / 1000.0

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

v28src = open("colab_v28_train.py").read()
hs_src = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28src, re.S).group(0)
tf_src = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28src, re.S).group(0)
g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(hs_src, g23); exec(tf_src, g23)
TrackFormerHist = g23["TrackFormerHist"]

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

_lf = np.load("track_build/lead_flow.npz")
FLOW_T = _lf["flow"].astype("float32"); FLOW_M = _lf["got"].astype("float32")

# ---- the v35 total_loss under test, copied verbatim from colab_v40_train.py ----
W_FLOW = 0.3
W_INT = 1.0  # mutated below per check


def total_loss_v35(s, ls, fp, tgt, m, fl, fm):
    tn = tgt / TARGET_SCALE
    base = G["track_loss"](s, tn, m) + W_INT * G["int_loss"](s, ls, tn, m)
    fmm = fm.unsqueeze(-1)
    flow = (F.smooth_l1_loss(fp, fl, reduction="none") * fmm).sum() / fmm.sum().clamp(min=1)
    return base + W_FLOW * flow, float(flow.detach())


j = np.arange(8)
tr = torch.from_numpy(track[j]); v0 = torch.from_numpy(vpair[j]); sp = torch.from_numpy(SLP[j])
tg = torch.from_numpy(target[j]); m = torch.from_numpy(mask[j])
fl = torch.from_numpy(FLOW_T[j].copy()); fm = torch.from_numpy(FLOW_M[j].copy())
hs = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1).copy())
hv = torch.from_numpy(HAVE[j].copy())

torch.manual_seed(0)
model = TrackFormerHist().eval()
with torch.no_grad():
    s, ls, fp = model(tr, v0, sp, hs, hv)

# ---- check 1: W_INT=1.0 reproduces v17's original total_loss exactly (non-flow part too) ----
W_INT = 1.0
loss1, _ = total_loss_v35(s, ls, fp.float(), tg, m, fl, fm)
tn = tg / TARGET_SCALE
ref_total = G["total_loss"](s, ls, tn * TARGET_SCALE, m)
fmm = fm.unsqueeze(-1)
ref_flow = (F.smooth_l1_loss(fp.float(), fl, reduction="none") * fmm).sum() / fmm.sum().clamp(min=1)
ref_total = ref_total + W_FLOW * ref_flow
d1 = float((loss1 - ref_total).abs())
assert d1 < 1e-5, f"W_INT=1.0 does not reproduce v23's original total_loss: diff {d1:.2e}"
print(f"check 1 OK: W_INT=1.0 reproduces v23's original total_loss exactly (diff {d1:.2e})")

# ---- check 2: raising W_INT actually scales the intensity term (not a no-op) ----
int_term = G["int_loss"](s, ls, tn, m)
W_INT = 3.0
loss3, _ = total_loss_v35(s, ls, fp.float(), tg, m, fl, fm)
expected_delta = 2.0 * float(int_term)   # (3-1) * int_loss
actual_delta = float(loss3 - loss1)
assert abs(actual_delta - expected_delta) < 1e-4, (
    f"W_INT scaling is wrong: expected +{expected_delta:.5f}, got +{actual_delta:.5f}")
print(f"check 2 OK: raising W_INT 1->3 adds {actual_delta:.5f} to the loss "
      f"(expected {expected_delta:.5f}, int_loss={float(int_term):.5f})")

# ---- check 3: gradient reaches both a track-facing and the intensity/log-scale-facing params ----
model.train()
torch.manual_seed(0)
s, ls, fp = model(tr, v0, sp, hs, hv)
W_INT = 3.0
loss, _ = total_loss_v35(s, ls, fp.float(), tg, m, fl, fm)
model.zero_grad(set_to_none=True)
loss.backward()
track_g = model.track_res.weight.grad
assert track_g is not None and float(track_g.abs().max()) > 0, "no gradient reached track_res"
print(f"check 3a OK: gradient reaches track_res (max|grad|={float(track_g.abs().max()):.2e})")

int_named = [(n, p) for n, p in model.named_parameters()
             if "logscale" in n.lower() or "log_scale" in n.lower() or n.endswith("state.weight")]
assert int_named, "could not find an intensity/log-scale head parameter by name -- inspect model.named_parameters()"
int_n, int_p = int_named[0]
assert int_p.grad is not None and float(int_p.grad.abs().max()) > 0, f"no gradient reached {int_n}"
print(f"check 3b OK: gradient reaches {int_n} (max|grad|={float(int_p.grad.abs().max()):.2e})")

print("\nALL CHECKS PASSED -- v35 (intensity-reweighted loss, same architecture as v23) is ready "
      "for Colab")
