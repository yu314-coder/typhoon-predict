"""Merge SHIPS predictors and AOML ocean heat into one environmental feature set.

WHY BOTH. SHIPS has 37 predictors but stops in 2021, covering 7% of our 2020+ test set -- feeding
it alone would mean training on real values and evaluating on zeros, the distribution shift that
cost v24 86 km. AOML ocean runs 2013-2026 and covers ~98% of WP+EP test windows, but only carries
ocean terms. Together they span more than either does alone, and the mask records exactly which
source supplied each value so nothing is silently fabricated.

OHC PATCHES -> SCALARS. AOML gives a 21x21 patch (+-2.5 deg) per window. For a feature vector we
reduce each of the three fields to two numbers: the value at the storm centre, and the mean over
the patch. Centre is what the storm sits on now; the patch mean is what it is about to move over.
The full patch is kept in the file too, so a CNN stream can use it later.

THE CROSS-CHECK. SHIPS carries COHC and AOML carries Ocean_Heat_Content -- the same physical
quantity from independent pipelines. Where both exist they must agree. That is checked and
reported rather than assumed; a poor correlation would mean one of the two joins is wrong.

Output: track_build/env_features.npz, aligned 1:1 with track_windows_v13.npz.
"""
import os, glob, sys
import numpy as np

z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
N = len(z["base_time"])
bt = z["base_time"].astype("int64")
bas = z["basin"].astype(str)
yr = np.array([int(str(np.datetime64(int(t), "ns").astype("datetime64[Y]"))) for t in bt])

# ---- SHIPS -------------------------------------------------------------------------------
S = np.load("track_build/ships_features.npz", allow_pickle=True)
sf, sg = S["feat"], S["got"]
snames = [str(x) for x in S["names"]]
print(f"SHIPS  {sf.shape[1]} predictors, {int((sg.sum(1) > 0).sum()):,} windows")

# ---- AOML ocean --------------------------------------------------------------------------
files = sorted(glob.glob("track_build/ohc/ohc_*.npz"))
if not files:
    sys.exit("no ocean files in track_build/ohc/ -- run extract_ohc.py first")
OV = ["OHC", "D26", "D20"]
opatch = np.zeros((N, 3, 21, 21), "float32")
ogot = np.zeros(N, "float32")
for f in files:
    d = np.load(f)
    idx = d["widx"]; q = d["q"].astype("float32") * d["scale"][None, :, None, None]
    opatch[idx] = q
    ogot[idx] = d["got"]
print(f"AOML   {len(files)} years, {int(ogot.sum()):,} windows with an ocean patch")

# centre value and patch mean, computed only over cells that are real ocean (nonzero)
oc = np.zeros((N, 6), "float32")
for v in range(3):
    p = opatch[:, v]
    oc[:, 2 * v] = p[:, 10, 10]
    valid = p != 0
    cnt = valid.sum((1, 2))
    oc[:, 2 * v + 1] = np.where(cnt > 0, p.sum((1, 2)) / np.maximum(cnt, 1), 0.0)
onames = [f"AOML_{v}_{k}" for v in OV for k in ("ctr", "mean")]

# ---- the cross-check ---------------------------------------------------------------------
ci = snames.index("COHC")
both = (sg[:, ci] > 0) & (ogot > 0)
print(f"\ncross-check: {int(both.sum()):,} windows have BOTH SHIPS COHC and AOML OHC")
if both.sum() > 50:
    a = sf[both, ci]; b = oc[both, 0]
    keep = (a > 0) & (b > 0)
    r = float(np.corrcoef(a[keep], b[keep])[0, 1])
    print(f"  correlation r = {r:.3f}   SHIPS mean {a[keep].mean():.1f}, AOML mean {b[keep].mean():.1f}")
    if r < 0.6:
        print("  *** WARNING: poor agreement -- one of the two joins is probably wrong ***")
    else:
        print("  agreement is good; the two independent joins corroborate each other")

# ---- merge ---------------------------------------------------------------------------------
feat = np.concatenate([sf, oc], 1)
got = np.concatenate([sg, np.repeat(ogot[:, None], 6, 1)], 1)
names = snames + onames

anyenv = got.sum(1) > 0
wpep = np.isin(bas, ["WP", "EP"])
test = yr >= 2020
print(f"\n{'':22s} {'all':>10s} {'WP+EP':>10s} {'TEST 2020+':>12s} {'TEST WP+EP':>12s}")
def row(lab, m):
    print(f"  {lab:20s} {int(m.sum()):>10,} {int((m&wpep).sum()):>10,} "
          f"{int((m&test).sum()):>12,} {int((m&test&wpep).sum()):>12,}")
row("windows", np.ones(N, bool))
row("SHIPS only", (sg.sum(1) > 0) & (ogot == 0))
row("AOML only", (sg.sum(1) == 0) & (ogot > 0))
row("both", (sg.sum(1) > 0) & (ogot > 0))
row("ANY env feature", anyenv)
row("NO env feature", ~anyenv)

np.savez_compressed("track_build/env_features.npz",
                    feat=feat, got=got, names=np.array(names),
                    ohc_patch=opatch.astype("float16"), ohc_got=ogot,
                    era_gfs=S["era_gfs"])
print(f"\nwrote track_build/env_features.npz "
      f"({os.path.getsize('track_build/env_features.npz')/1e6:.1f} MB) "
      f"| {feat.shape[1]} features")
