"""Do v20's intensity outputs obey the physics that ties them together?

Three constraints a real cyclone cannot violate, none of which appears anywhere in the loss:

  1. WIND-PRESSURE. vmax and central pressure are two views of one thing. In the training data they
     correlate at -0.954, close to deterministic. The model emits them as independent regression
     outputs, so nothing stops it predicting 140 kt at 1000 hPa.
  2. RADII ORDERING. The 34 kt wind field must enclose the 50 kt field, which must enclose 64 kt.
     The loss has a weak 0.01-weighted hinge on this, so it is nudged, not enforced.
  3. SST CEILING. Potential intensity is bounded by the sea surface temperature beneath the storm.
     v17 and v20 do not read SST at all -- the v16 ablation dropped it -- so they cannot respect it
     even in principle.

This measures 1 and 2 on the real test set, against what the observations do, so the size of any
violation is a number rather than an opinion.
"""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8); DEVICE = torch.device("cpu")

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
    m = G["TrackFormerV17"]().eval(); m.load_state_dict(sd); return m


z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
mask = z["target_mask"].astype("float32")
sids = z["storm_id"].astype(str); years = z["year"].astype(int)
basins = z["basin"].astype(str); nl = z["n_leads"].astype(int)
tmean = z["track_mean"].astype("float32"); tstd = z["track_std"].astype("float32")
vpair = np.concatenate([track[:, -1, 2:4] * tstd[2:4] + tmean[2:4],
                        track[:, -2, 2:4] * tstd[2:4] + tmean[2:4]], 1).astype("float32")
S20 = np.clip(np.load("track_build/dlm4_int8.npz")["q"][:, :4].astype("float32") / 31.75, -4, 4)

fy = {s: int(years[sids == s].min()) for s in np.unique(sids)}
EV = np.array([i for i in range(len(sids))
               if fy[sids[i]] >= 2020 and nl[i] == 20 and basins[i] in ("WP", "EP")])
SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12)
MS = [load(f"downloads/x/v20_seed{i}.pt") for i in range(5)]


@torch.no_grad()
def predict():
    P = []
    for i in range(0, len(EV), 128):
        j = EV[i:i + 128]
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(S20[j])]
        s = torch.stack([m(*a)[0] for m in MS]).mean(0)
        P.append((s * SC).numpy())
    return np.concatenate(P)


P = predict()
T = target[EV]; M = mask[EV]
pv, pp = P[..., 2].ravel(), P[..., 3].ravel()
tv, tp = T[..., 2].ravel(), T[..., 3].ravel()
ok = (M[..., 2].ravel() > 0.5) & (M[..., 3].ravel() > 0.5) & (tv > 0) & (tp > 0)

print(f"{ok.sum():,} test points where both vmax and pressure are observed\n")
print("--- 1. WIND-PRESSURE COHERENCE ---")
print(f"  observed   corr(vmax, pressure) = {np.corrcoef(tv[ok], tp[ok])[0,1]:+.3f}")
print(f"  v20 pred   corr(vmax, pressure) = {np.corrcoef(pv[ok], pp[ok])[0,1]:+.3f}")

# fit the observed relationship, then measure how far the model's pairs sit off it
c = np.polyfit(tv[ok], tp[ok], 2)
res_obs = tp[ok] - np.polyval(c, tv[ok])
res_mod = pp[ok] - np.polyval(c, pv[ok])
print(f"\n  scatter about the observed wind-pressure curve (hPa):")
print(f"    observations         {res_obs.std():6.2f}   <- real storms vary this much")
print(f"    v20 predictions      {res_mod.std():6.2f}")
if res_mod.std() > res_obs.std():
    print(f"    -> the model is {res_mod.std()/res_obs.std():.2f}x LESS coherent than reality:"
          f" it emits wind and pressure pairs real storms do not produce")
else:
    print(f"    -> the model is at least as coherent as reality")
bad = np.abs(res_mod) > 3 * res_obs.std()
print(f"    pairs off the curve by >3 sigma: {bad.sum():,} of {ok.sum():,} ({100*bad.mean():.1f}%)")

print("\n--- 2. WIND-RADII ORDERING (r34 >= r50 >= r64) ---")
r34, r50, r64 = P[..., 5:9], P[..., 9:13], P[..., 13:17]
m34, m50, m64 = M[..., 5:9] > .5, M[..., 9:13] > .5, M[..., 13:17] > .5
a = m34 & m50; b = m50 & m64
v1 = (r50 > r34 + 1e-3) & a; v2 = (r64 > r50 + 1e-3) & b
print(f"  r50 > r34 violations: {v1.sum():,} of {a.sum():,} ({100*v1.sum()/max(a.sum(),1):.2f}%)"
      f"   worst {(r50-r34)[v1].max() if v1.any() else 0:.0f} km")
print(f"  r64 > r50 violations: {v2.sum():,} of {b.sum():,} ({100*v2.sum()/max(b.sum(),1):.2f}%)"
      f"   worst {(r64-r50)[v2].max() if v2.any() else 0:.0f} km")
to = ((target[EV][..., 9:13] > target[EV][..., 5:9] + 1e-3) & a).sum()
print(f"  same check on the OBSERVATIONS: {to:,} ({100*to/max(a.sum(),1):.2f}%)  <- the floor")

print("\n--- 3. SST ---")
q = np.load("track_build/steer5_int8.npz")
print(f"  steer5 carries an SST channel, availability {q['ok'][:,2].mean():.3f}")
print("  v17/v20 read channels 0:4 only -- SST (ch4) is DROPPED, so neither model can")
print("  respect a potential-intensity ceiling even in principle.")
json.dump({"corr_obs": float(np.corrcoef(tv[ok], tp[ok])[0,1]),
           "corr_pred": float(np.corrcoef(pv[ok], pp[ok])[0,1]),
           "scatter_obs": float(res_obs.std()), "scatter_pred": float(res_mod.std()),
           "r50_gt_r34_pct": float(100*v1.sum()/max(a.sum(),1)),
           "r64_gt_r50_pct": float(100*v2.sum()/max(b.sum(),1))},
          open("track_build/physics_diag.json", "w"))
