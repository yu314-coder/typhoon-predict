"""Merge per-year 500 hPa steering patches into one array aligned to the window index, and combine
with the sea-level-pressure patches into a single 4-channel steering tensor:
  ch0 SLP anomaly (surface ridge/trough pattern)
  ch1 SLP 24h tendency (is the ridge building or collapsing?)
  ch2 u500  (mid-level steering wind, east)
  ch3 v500  (mid-level steering wind, north)
"""
import numpy as np, glob, os

z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
N = len(z["year"])
steer = np.zeros((N, 2, 17, 17), dtype="float16")
seen = np.zeros(N, dtype=bool)
for f in sorted(glob.glob("track_build/geo/steer_*.npz")):
    d = np.load(f)
    idx = d["idx"]
    if len(idx):
        steer[idx] = d["patch"]; seen[idx] = True
print(f"steering coverage: {seen.sum()}/{N} windows ({100*seen.mean():.1f}%)")

slp = np.load("track_build/slp_patches.npy")            # [N,2,17,17] already built
comb = np.concatenate([slp, steer], axis=1)             # [N,4,17,17]
# per-channel scaling to ~unit variance (computed on covered windows)
f32 = comb.astype("float32")
sc = np.array([f32[:, c][seen].std() if seen.any() else 1.0 for c in range(4)], dtype="float32")
sc = np.where(sc < 1e-3, 1.0, sc)
print("channel std (SLPanom, SLPtend, u500, v500):", np.round(sc, 2))
np.save("track_build/steer4_patches.npy", comb)
np.save("track_build/steer4_scale.npy", sc)
print(f"saved track_build/steer4_patches.npy ({os.path.getsize('track_build/steer4_patches.npy')/1e6:.0f} MB)")
