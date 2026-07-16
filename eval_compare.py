"""Run the saved models locally (Apple MPS) and compare on WP-2020+. No training."""
import json, math, re
import numpy as np
import torch

# ERA5 conv model uses AdaptiveAvgPool2d(3) from 17x17 -> non-divisible, unsupported on MPS.
# Eval sets are small, so run on CPU for correctness.
DEV = torch.device("cpu")
print("device:", DEV)
TS = np.array([100., 100., 35., 20., 50.] + [50.] * 12, dtype="float32")


def metrics(P, T, M):
    o = {}
    pt, tt = np.cumsum(P[..., :2], 1), np.cumsum(T[..., :2], 1)
    o["track_km"] = round(float(np.sqrt(((pt - tt) ** 2).sum(-1)).mean()), 1)
    for i, nm in [(2, "vmax_kt"), (3, "pres_hpa"), (4, "rmw_km")]:
        v = M[..., i] > 0.5
        o[nm] = round(float(np.abs(P[..., i][v] - T[..., i][v]).mean()), 2) if v.any() else None
    rm = M[..., 5:17] > 0.5
    o["radius_km"] = round(float(np.abs(P[..., 5:17] - T[..., 5:17])[rm].mean()), 2)
    return o


def exec_cls(path, cls):
    ns = {}
    exec(open(path).read(), ns)
    return ns[cls]


# ---------- ERA5 v2 (3.3M) on its WP-2020+ test ----------
print("\n=== ERA5 v2 (spatial tokens, 3.3M) ===")
StormFusionMT = exec_cls("model_v2.py", "StormFusionMT")
zc = np.load("typhoon_build/data/windows/stormfusion_windows.npz", allow_pickle=True)
yr = zc["year"].astype(int)
te = np.where(yr >= 2020)[0]                       # ERA5 data is WP-only
m2 = StormFusionMT("recommended", lead_count=20).to(DEV).eval()
m2.load_state_dict(torch.load("track_build/era5_v2_best.pt", map_location=DEV, weights_only=False)["model"])
inner, outer, trk, env = zc["inner"], zc["outer"], zc["track"], zc["env"]
tgt, msk = zc["target"], zc["target_mask"].astype("float32")
P = []
with torch.no_grad():
    for s in range(0, len(te), 64):
        b = te[s:s + 64]
        p, _ = m2(torch.from_numpy(inner[b]).to(DEV), torch.from_numpy(outer[b]).to(DEV),
                  torch.from_numpy(trk[b]).to(DEV), torch.from_numpy(env[b]).to(DEV))
        P.append((p.float().cpu().numpy() * TS))
era5_v2 = metrics(np.concatenate(P), tgt[te], msk[te])
print(f"WP-2020+ test: {len(te)} windows, {len(np.unique(zc['storm_id'][te]))} storms")
print("  ERA5 v2:", json.dumps(era5_v2))

# ---------- Track-only (21M) on WP-2020+ test ----------
print("\n=== Track-only (no ERA5, 21M) ===")
TrackModel = exec_cls("train_track.py_MODELONLY", "TrackModel") if False else None
src = open("train_track.py").read()
mm = re.search(r"def sinusoidal.*?return self\.state\(d\), self\.logscale\(d\)\.clamp\(-5\.0, 3\.0\)", src, re.S)
ns = {"math": math, "torch": torch, "nn": __import__("torch.nn", fromlist=["x"])}
exec(mm.group(0), ns)
mt = ns["TrackModel"]().to(DEV).eval()
mt.load_state_dict(torch.load("track_build/track_best.pt", map_location=DEV, weights_only=False)["model"])
zt = np.load("track_build/track_windows.npz", allow_pickle=True)
yt, bt = zt["year"].astype(int), zt["basin"].astype(str)
twp = np.where((yt >= 2020) & (bt == "WP"))[0]
trk2, tgt2, msk2 = zt["track"], zt["target"], zt["target_mask"].astype("float32")
P = []
with torch.no_grad():
    for s in range(0, len(twp), 256):
        b = twp[s:s + 256]
        p, _ = mt(torch.from_numpy(trk2[b]).to(DEV))
        P.append((p.float().cpu().numpy() * TS))
track_only = metrics(np.concatenate(P), tgt2[twp], msk2[twp])
print(f"WP-2020+ test: {len(twp)} windows, {len(np.unique(zt['storm_id'][twp]))} storms")
print("  Track-only:", json.dumps(track_only))

# ---------- comparison table ----------
rows = [
    ("v1 ERA5 (3.3M, recorded)", dict(track_km=747, vmax_kt=24.37, pres_hpa=23.15, rmw_km=17.01, radius_km=32.64)),
    ("v3 ERA5 (17.7M, recorded)", dict(track_km=793.2, vmax_kt=24.35, pres_hpa=20.5, rmw_km=17.73, radius_km=27.72)),
    ("v2 ERA5 (3.3M, re-run)", era5_v2),
    ("Track-only (21M, re-run)", track_only),
]
cols = ["track_km", "vmax_kt", "pres_hpa", "rmw_km", "radius_km"]
lines = ["", "=== COMPARISON (test set; v2/track-only WP-2020+ re-run on Mac MPS; v1/v3 recorded) ==="]
lines.append(f"{'model':30s} " + " ".join(f"{c:>10s}" for c in cols))
for name, d in rows:
    lines.append(f"{name:30s} " + " ".join(f"{str(d.get(c)):>10s}" for c in cols))
report = "\n".join(lines)
print(report)
open("track_build/model_comparison.txt", "w").write(report + "\n")
print("\nsaved track_build/model_comparison.txt")
