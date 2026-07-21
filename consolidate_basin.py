"""Merge the per-year basin files into ONE tensor to upload to Drive.

extract_basin.py writes one .npz per year so an interrupted pull only costs the year in progress.
For Colab we want the opposite: a single self-describing file that mounts and loads in one line,
so the basin fields never have to be re-extracted on a fresh VM. That has happened three times
with the steering tensor and cost ~45 min of idle GPU each time.

RE-QUANTISES GLOBALLY. Each year currently carries its own scale/offset, which would force the
loader to track a year index per timestep. One global scale per channel removes that entirely.
The cost is small -- the global range is barely wider than any single year's -- and it is measured
and asserted below rather than assumed.

Writes track_build/basin_all_int8.npz with:
    q        [T,7,25,33] int8      the fields, time-sorted, duplicates removed
    scale    [7] float32           value = q * scale + offset
    offset   [7] float32
    time     [T] int64             ns since epoch, joins straight onto base_time
    lat/lon  [25]/[33] float32
    channels [7] str               hgt500, uwnd850, vwnd850, uwnd500, vwnd500, uwnd200, vwnd200
"""
import glob, os, sys
import numpy as np

FILES = sorted(glob.glob("track_build/basin/basin_*.npz"))
if not FILES:
    sys.exit("no per-year files in track_build/basin/")
print(f"{len(FILES)} year files")

# pass 1: global per-channel range, from the dequantised years
lo = np.full(7, np.inf, "float64"); hi = np.full(7, -np.inf, "float64")
n_t = 0
for f in FILES:
    d = np.load(f)
    v = d["q"].astype("float32") * d["scale"][None, :, None, None] + d["offset"][None, :, None, None]
    lo = np.minimum(lo, v.min((0, 2, 3)))
    hi = np.maximum(hi, v.max((0, 2, 3)))
    n_t += d["q"].shape[0]
print(f"{n_t:,} timesteps before dedup")

off = ((hi + lo) / 2).astype("float32")
sca = np.maximum((hi - lo) / 254.0, 1e-6).astype("float32")

# pass 2: requantise and stack
Q = np.empty((n_t, 7, 25, 33), "int8"); T = np.empty(n_t, "int64")
worst = 0.0
i = 0
meta = None
for f in FILES:
    d = np.load(f)
    v = d["q"].astype("float32") * d["scale"][None, :, None, None] + d["offset"][None, :, None, None]
    q = np.clip(np.round((v - off[None, :, None, None]) / sca[None, :, None, None]),
                -127, 127).astype("int8")
    back = q.astype("float32") * sca[None, :, None, None] + off[None, :, None, None]
    worst = max(worst, float(np.abs(back - v).max()))
    n = q.shape[0]
    Q[i:i + n] = q; T[i:i + n] = d["time"].astype("int64"); i += n
    if meta is None:
        meta = (d["lat"], d["lon"], d["channels"])

assert worst <= float(sca.max()) * 1.01, \
    f"global requantisation error {worst:.3f} exceeds one step {sca.max():.3f}"
print(f"global requantisation worst error {worst:.3f} (one step {sca.max():.3f})")

# sort by time and drop any duplicate timestamps (year files can overlap at boundaries)
o = np.argsort(T, kind="stable")
Q, T = Q[o], T[o]
keep = np.concatenate([[True], np.diff(T) > 0])
if (~keep).sum():
    print(f"dropped {(~keep).sum()} duplicate timestamps")
Q, T = Q[keep], T[keep]

step = np.unique(np.diff(T))
print(f"{len(T):,} unique timesteps | spacing (h): {(step // int(1e9 * 3600)).tolist()[:5]}")
print(f"span {np.datetime64(int(T[0]), 'ns')} .. {np.datetime64(int(T[-1]), 'ns')}")

# how well do the storm windows join onto this axis?
z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
bt = z["base_time"].astype("int64")
inside = bt[(bt >= T[0]) & (bt <= T[-1])]
exact = np.isin(inside, T).sum()
print(f"\nstorm windows inside span : {len(inside):,}")
print(f"  landing exactly on a field: {exact:,} ({100*exact/len(inside):.1f}%)")
print(f"  the rest sit BETWEEN two 6-h fields (3-hourly best-track fixes) and must be")
print(f"  linearly interpolated, not snapped -- the v24 loader builds that index.")

out = "track_build/basin_all_int8.npz"
np.savez_compressed(out, q=Q, scale=sca, offset=off, time=T,
                    lat=meta[0], lon=meta[1], channels=meta[2])
print(f"\nwrote {out}  ({os.path.getsize(out)/1e6:.0f} MB)")
print("upload this one file to Drive; Colab reads it directly, no extraction ever again.")
