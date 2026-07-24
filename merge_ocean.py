"""One consistent ocean-patch file for v27, from GODAS -- with AOML kept only as a cross-check.

WHY GODAS AND NOT AOML. AOML is the better product (daily, 0.25 deg) but it starts in 2013, and the
training split is storms whose first year is <= 2015. That three-year overlap is why v26's ocean CNN
saw real data on 3.9% of training windows while being scored on 95.6% of test windows, and it is the
measured ceiling on v26's 0.86 kt. GODAS runs 1980-present, the earliest training storm here is 1980,
and it lifts training coverage to ~50%.

WHY NOT BOTH. Feeding AOML where it exists and GODAS elsewhere would put a different instrument on
the training set than on the test set -- the exact distribution shift that cost v24 86 km. The two
products agree well but not identically (measured on 15,510 shared windows: OHC r=0.80 with GODAS
biased 11.3 kJ/cm2 low, D26 r=0.85, D20 r=0.81), and a systematic offset that lands on one side of
the split becomes a feature the model can exploit and then be wrong about. One instrument, both
sides. The bias itself is harmless because it is constant across the split -- the model simply
learns the GODAS scale.

AOML is written into the file anyway, unused by training, so the cross-check stays reproducible.

Output: track_build/ocean_patch.npz, aligned 1:1 with track_windows_v13.npz.
"""
import glob, os
import numpy as np

z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
N = len(z["base_time"])
yr = z["year"].astype(int); bas = z["basin"].astype(str); nl = z["n_leads"].astype(int)
sids = z["storm_id"].astype(str)
fy = {s: int(yr[sids == s].min()) for s in np.unique(sids)}
first = np.array([fy[s] for s in sids])


def load(pat):
    P = np.zeros((N, 3, 21, 21), "float32"); g = np.zeros(N, "float32")
    fs = sorted(glob.glob(pat))
    for f in fs:
        d = np.load(f)
        i = d["widx"]
        P[i] = d["q"].astype("float32") * d["scale"][None, :, None, None]
        g[i] = d["got"]
    return P, g, len(fs)


GP, GG, ng = load("track_build/godas/godas_*.npz")
AP, AG, na = load("track_build/ohc/ohc_*.npz")
print(f"GODAS {ng} year-files, {int((GG>0).sum()):,} windows | AOML {na} files, {int((AG>0).sum()):,} windows")

both = (GG > 0) & (AG > 0)
if both.sum() > 200:
    print(f"\ncross-check on {int(both.sum()):,} shared windows (centre cell):")
    for c, nm in enumerate(["OHC", "D26", "D20"]):
        a = AP[both, c, 10, 10]; g = GP[both, c, 10, 10]
        k = (a != 0) & (g != 0)
        r = float(np.corrcoef(a[k], g[k])[0, 1])
        print(f"  {nm:4s} r={r:+.3f}  AOML {a[k].mean():7.2f}  GODAS {g[k].mean():7.2f}  "
              f"bias {float((g[k]-a[k]).mean()):+.2f}")
        if r < 0.6:
            print("  *** WARNING: poor agreement -- one derivation is probably wrong ***")

tr = first <= 2015; va = (first >= 2016) & (first <= 2019)
te = (first >= 2020) & np.isin(bas, ["WP", "EP"]) & (nl == 20)
print(f"\n{'split':8s} {'windows':>9s} {'AOML':>8s} {'GODAS':>8s}")
for nm, m in (("train", tr), ("valid", va), ("test", te)):
    n = int(m.sum())
    print(f"{nm:8s} {n:9,} {100*(AG[m]>0).sum()/n:7.1f}% {100*(GG[m]>0).sum()/n:7.1f}%")

sc = np.array([max(np.abs(GP[:, v]).max(), 1e-6) / 127.0 for v in range(3)], "float32")
q = np.clip(np.round(GP / sc[None, :, None, None]), -127, 127).astype("int8")
np.savez_compressed("track_build/ocean_patch.npz",
                    q=q, scale=sc, got=GG, source=np.array("GODAS"),
                    aoml_got=AG, vars=np.array(["OHC", "D26", "D20"]))
print(f"\nwrote track_build/ocean_patch.npz "
      f"({os.path.getsize('track_build/ocean_patch.npz')/1e6:.1f} MB) | training source: GODAS")
