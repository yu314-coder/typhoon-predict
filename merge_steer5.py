"""Merge every environmental field into one 5-channel storm-centered tensor for v16.

  ch0 SLP anomaly        surface ridge/trough pattern
  ch1 SLP 24h tendency   is the steering ridge building or collapsing?
  ch2 u500               mid-level steering wind, east
  ch3 v500               mid-level steering wind, north
  ch4 SST - 28 degC      observed ocean fuel (v16; through v15 this was a lat+month climatology)

All five are on the same storm-centered 2.5 deg 17x17 grid, so a cell means the same place in
every channel.

Availability is tracked per field GROUP and saved alongside, because "no data" and "zero" are
different statements and conflating them is what sent v14's Bavi forecast 3442 km the wrong way:
the 2026 reanalysis stops on 2026-03-17, and an unbounded nearest-time match silently handed
every later storm a March wind field. Unavailable groups are written as exact zeros AND recorded
in the mask, so training can drop them deliberately instead of learning from fiction.

  ok[:,0] SLP pair    ok[:,1] steering pair    ok[:,2] SST
"""
import numpy as np, glob, os

z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
N = len(z["year"])

# ---- steering (per-year files, each carrying its own ok mask) ----
steer = np.zeros((N, 2, 17, 17), dtype="float16")
steer_ok = np.zeros(N, dtype=bool)
legacy = []
for f in sorted(glob.glob("track_build/geo/steer_*.npz")):
    d = np.load(f); idx = d["idx"]
    if not len(idx):
        continue
    steer[idx] = d["patch"]
    if "ok" in d.files:
        steer_ok[idx] = d["ok"]
    else:                                  # built before the tolerance fix -- trust it, but say so
        steer_ok[idx] = True
        legacy.append(os.path.basename(f))
if legacy:
    print(f"NOTE: {len(legacy)} steering file(s) predate the time-tolerance fix and are assumed "
          f"fully valid: {', '.join(legacy[:6])}{' ...' if len(legacy) > 6 else ''}")

# ---- SLP ----
slp = np.load("track_build/slp_patches.npy")
slp_ok = (np.load("track_build/slp_ok.npy") if os.path.exists("track_build/slp_ok.npy")
          else np.ones(N, dtype=bool))

# ---- SST ----
sst = np.load("track_build/sst_patches.npy")
sst_ok = np.load("track_build/sst_ok.npy")

# ---- enforce the contract: unavailable => exact zeros ----
slp[~slp_ok] = 0
steer[~steer_ok] = 0
sst[~sst_ok] = 0

comb = np.concatenate([slp, steer, sst], axis=1)          # [N,5,17,17]
ok = np.stack([slp_ok, steer_ok, sst_ok], axis=1)          # [N,3]

# ---- per-channel scaling, computed only over windows where that channel is real ----
grp = [0, 0, 1, 1, 2]
f32 = comb.astype("float32")
sc = np.ones(5, dtype="float32")
for c in range(5):
    m = ok[:, grp[c]]
    s = f32[m, c].std() if m.any() else 1.0
    sc[c] = s if s > 1e-3 else 1.0

names = ["SLPanom", "SLPtend", "u500", "v500", "SST"]
print(f"windows {N}")
for c in range(5):
    m = ok[:, grp[c]]
    print(f"  ch{c} {names[c]:8s} available {m.sum():6d} ({100*m.mean():5.1f}%)  scale {sc[c]:6.3f}")

np.save("track_build/steer5_patches.npy", comb)
np.save("track_build/steer5_scale.npy", sc)
np.save("track_build/steer5_ok.npy", ok)
print(f"saved track_build/steer5_patches.npy "
      f"({os.path.getsize('track_build/steer5_patches.npy')/1e6:.0f} MB)")

# ---- int8 pack for Colab upload: /scale then clip at 4 sigma, which training does anyway ----
CLIP = 4.0
q = np.clip(f32 / sc[None, :, None, None], -CLIP, CLIP)
q = np.round(q * (127.0 / CLIP)).astype("int8")
np.savez_compressed("track_build/steer5_int8.npz", q=q, ok=ok, scale=sc)
err = np.abs(q.astype("float32") * (CLIP / 127.0) - np.clip(f32 / sc[None, :, None, None], -CLIP, CLIP)).max()
print(f"saved track_build/steer5_int8.npz "
      f"({os.path.getsize('track_build/steer5_int8.npz')/1e6:.0f} MB)  max quantization error {err:.4f} sigma")
