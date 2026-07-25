"""GO/NO-GO for v30: is v23's SHORT-LEAD (6-24 h) track error a PREDICTABLE signal or noise?

The flow oracle showed the 24 h error is ~75 km and perfect steering removes only ~1 km of it, so
the short-lead miss is NOT a steering problem. v30 proposes a learned kinematic correction gated to
zero by 36-48 h. But v28 just taught the lesson: only build a correction if the thing it targets is
predictable enough to clear seed noise (3.8-7.3 km).

THE TEST. v23 already SEES the full kinematic history. If a simple regression on those same inputs
can predict v23's own short-lead error, then v23 is leaving PREDICTABLE STRUCTURE on the table and a
dedicated correction head (v30) could capture it. If the error is unpredictable from the available
state, nothing -- v30 included -- can fix it.

    features (current vortex state, what v30 would use):
        current motion v0 (E,N), |v0| speed, heading sin/cos, turn rate,
        previous motion, vmax, lat, |lat|, speed trend, vmax trend
    targets: v23's cumulative E/N error at 6,12,18,24 h, and its ALONG-track and CROSS-track parts
    fit Ridge on VALIDATION storms, report R^2 on held-out TEST storms.

Along-track error = component along the observed motion direction (a speed error).
Cross-track error = perpendicular (a direction error). They have different fixability.

VERDICT: test R^2 at 24 h, converted to km of removable error. If the predictable part clears seed
noise, v30 is well-aimed. If R^2 ~ 0 or the removable km < seed noise, skip it -- like v28.
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

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
va_idx, te_idx = G["va_idx"], G["te_idx"]; nl = z["n_leads"].astype(int)
tmean = G["tmean"]; tstd = G["tstd"]
KM6H = 6 * 3600 / 1000.0

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))
CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]
v28 = open("colab_v28_train.py").read()
hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28, re.S).group(0)
tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28, re.S).group(0)
g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(hs, g23); exec(tf, g23)
V23 = g23["TrackFormerHist"]

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

MS = []
for p in sorted(glob.glob("downloads/x/v23_seed*.pt")):
    m = V23().eval(); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"]); MS.append(m)
print(f"v23: {len(MS)} seeds")

TM = tmean; TS = tstd


@torch.no_grad()
def errors(idx):
    """v23 cumulative E/N error [n,20,2] km."""
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]), h, torch.from_numpy(HAVE[j])]
        P.append((torch.stack([m(*a)[0] for m in MS]).mean(0)[..., :2] * SC[:2]).float().numpy())
    return np.cumsum(np.concatenate(P), 1) - np.cumsum(target[idx][..., :2], 1)


def features(idx):
    """current vortex state, physical units."""
    v0 = vpair[idx, :2]; vp = vpair[idx, 2:]          # current, previous motion (E,N)
    s0 = np.hypot(v0[:, 0], v0[:, 1]) + 1e-6
    phi0 = np.arctan2(v0[:, 1], v0[:, 0])
    dphi = phi0 - np.arctan2(vp[:, 1], vp[:, 0])
    omega = np.arctan2(np.sin(dphi), np.cos(dphi))    # turn rate
    lat = track[idx, -1, 48] * TS[48] + TM[48]
    vmax = track[idx, -1, 4] * TS[4] + TM[4]
    vmax_p = track[idx, -2, 4] * TS[4] + TM[4]
    spd_p = np.hypot(vp[:, 0], vp[:, 1])
    X = np.stack([v0[:, 0], v0[:, 1], s0, np.sin(phi0), np.cos(phi0), omega,
                  vp[:, 0], vp[:, 1], vmax, lat, np.abs(lat), s0 - spd_p, vmax - vmax_p], 1)
    return X.astype("float64")


full = nl == 20; wpep = np.isin(basins, ["WP", "EP"])
VA = np.array([i for i in va_idx if full[i] and wpep[i]])
TE = np.array([i for i in te_idx if full[i] and wpep[i]])
print("computing v23 short-lead errors ...", flush=True)
Ev, Et = errors(VA), errors(TE)
Xv, Xt = features(VA), features(TE)
# standardise features on validation
mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
Xv = (Xv - mu) / sd; Xt = (Xt - mu) / sd
Xv = np.hstack([Xv, np.ones((len(Xv), 1))]); Xt = np.hstack([Xt, np.ones((len(Xt), 1))])


def ridge_r2(y_tr, X_tr, y_te, X_te, lam=10.0):
    """fit ridge on train, return held-out R^2 and RMS of the removable (predicted) part."""
    A = X_tr.T @ X_tr + lam * np.eye(X_tr.shape[1])
    w = np.linalg.solve(A, X_tr.T @ y_tr)
    pred = X_te @ w
    ss_res = ((y_te - pred) ** 2).sum(); ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, float(np.sqrt((pred ** 2).mean()))


def along_cross(idx, E):
    """decompose error into along-track / cross-track using the observed step direction per lead."""
    obs = np.cumsum(target[idx][..., :2], 1)
    prev = np.concatenate([np.zeros((len(idx), 1, 2)), obs[:, :-1]], 1)
    step = obs - prev
    hd = np.arctan2(step[..., 1], step[..., 0])
    uE, uN = np.cos(hd), np.sin(hd)
    along = E[..., 0] * uE + E[..., 1] * uN
    cross = -E[..., 0] * uN + E[..., 1] * uE
    return along, cross


av_al, av_cr = along_cross(VA, Ev)
at_al, at_cr = along_cross(TE, Et)

print(f"\n{'lead':>5s} {'h':>4s} | {'RMS err':>8s} | {'dE R2':>7s} {'dN R2':>7s} | "
      f"{'along R2':>8s} {'cross R2':>8s} | {'along bias':>10s} {'cross bias':>10s}")
seed_floor = 5.5
for L in (0, 1, 2, 3):
    rmse = float(np.sqrt((Et[:, L] ** 2).sum(-1).mean()))
    r2E, _ = ridge_r2(Ev[:, L, 0], Xv, Et[:, L, 0], Xt)
    r2N, _ = ridge_r2(Ev[:, L, 1], Xv, Et[:, L, 1], Xt)
    r2al, rms_al = ridge_r2(av_al[:, L], Xv, at_al[:, L], Xt)
    r2cr, rms_cr = ridge_r2(av_cr[:, L], Xv, at_cr[:, L], Xt)
    print(f"{L+1:5d} {6*(L+1):4d} | {rmse:8.1f} | {r2E:7.3f} {r2N:7.3f} | "
          f"{r2al:8.3f} {r2cr:8.3f} | {at_al[:, L].mean():+10.1f} {at_cr[:, L].mean():+10.1f}")

# the decision number: at 24 h, how much error is PREDICTABLE (removable) vs seed noise?
L = 3
r2al, rms_al = ridge_r2(av_al[:, L], Xv, at_al[:, L], Xt)
r2cr, rms_cr = ridge_r2(av_cr[:, L], Xv, at_cr[:, L], Xt)
std_al, std_cr = at_al[:, L].std(), at_cr[:, L].std()
rem_al = std_al * math.sqrt(max(r2al, 0)); rem_cr = std_cr * math.sqrt(max(r2cr, 0))
rem_total = math.hypot(rem_al, rem_cr)
print("\n" + "=" * 74)
print(f"24 h error: along-track SD {std_al:.1f} km (R2 {r2al:.3f}), cross-track SD {std_cr:.1f} km (R2 {r2cr:.3f})")
print(f"PREDICTABLE (removable) short-lead error at 24 h: ~{rem_total:.1f} km  "
      f"(along {rem_al:.1f}, cross {rem_cr:.1f})")
print(f"seed-noise floor on the all-lead track metric: ~{seed_floor:.1f} km")
if rem_total > seed_floor:
    print("  -> a state-based short-lead correction (v30) has MORE than seed noise to capture. Worth building.")
elif rem_total > 0.4 * seed_floor:
    print("  -> modest predictable signal; v30 might buy a little at short leads. Marginal.")
else:
    print("  -> short-lead error is essentially UNPREDICTABLE from current state. v30 cannot fix it -- skip,")
    print("     like v28. The 24 h gap is irreducible given the inputs the model already has.")
print("=" * 74)
json.dump({"leads_h": [6, 12, 18, 24],
           "removable_km_24h": rem_total, "r2_along_24h": r2al, "r2_cross_24h": r2cr,
           "along_bias_24h": float(at_al[3].mean()), "cross_bias_24h": float(at_cr[3].mean())},
          open("track_build/shortlead.json", "w"), indent=1, default=float)
print("wrote track_build/shortlead.json")
