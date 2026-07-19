"""Extract DEEP-LAYER-MEAN steering patches on Colab, then build v20's 4-channel tensor.

    !wget -q -O v24e.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v24_extract.py
    exec(open('v24e.py').read())

v17 reads a single steering level, 500 hPa. Operational TC forecasting steers with the DEEP-LAYER
MEAN flow -- a mass-weighted vertical average, because a storm's depth sets which levels push it.
This builds the deep-layer mean from 850 + 500 + 200 hPa and drops it into v17's existing steering
channels, keeping the input at 4 channels so the architecture and its capacity are UNCHANGED. If
the number moves, it is the physics, not more parameters.

Weights are pressure-thickness over the 850-200 layer split at level midpoints (675, 350 hPa):
    850 -> 175 hPa, 500 -> 325 hPa, 200 -> 150 hPa   ->   0.269 / 0.500 / 0.231
500 keeps half the weight, so v20 is a gentle generalisation of v17 (which is 0/1/0) and should
degrade toward it rather than away if the extra levels carry nothing.

Reuses the SLP channels from steer5_int8.npz unchanged and the SAME time-tolerance guard that the
500 hPa extractor got after the 2026 fabrication bug: a window with no daily mean inside MAX_DT is
left as exact zeros and flagged not-ok, never snapped to a file edge.

Writes /content/dlm4_int8.npz -- the training cell reads it, so a bad training run never forces a
re-download of the 14 GB of reanalysis.
"""
import os, time, subprocess, urllib.request
import numpy as np, netCDF4

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
HALF = 8
MAX_DT_H = float(os.environ.get("STEER_MAX_DT_H", "18"))
MAX_DT = int(MAX_DT_H * 3600 * 1e9)
LEVELS = [850, 500, 200]
WEIGHTS = np.array([0.269, 0.500, 0.231], dtype="float32")   # thickness-weighted deep-layer mean
BASE = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure"

for fn in ("track_windows_v13.npz", "steer5_int8.npz"):
    if not os.path.exists(f"/content/{fn}"):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", f"/content/{fn}")

z = np.load("/content/track_windows_v13.npz", allow_pickle=True)
years = z["year"].astype(int)
bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
N = len(years)
DLM = np.zeros((N, 2, 2 * HALF + 1, 2 * HALF + 1), dtype="float16")
OKD = np.zeros(N, dtype=bool)


def get(var, year):
    """Return (times_ns, lat, lon, [u850,u500,u200]-style stack) for one variable-year."""
    f = f"/content/tmp_{var}_{year}.nc"
    if not os.path.exists(f):
        for a in range(4):
            r = subprocess.run(["curl", "-sSL", "--max-time", "600", "-o", f, f"{BASE}/{var}.{year}.nc"])
            if r.returncode == 0 and os.path.exists(f) and os.path.getsize(f) > 10_000_000:
                break
            time.sleep(5)
    d = netCDF4.Dataset(f)
    lev = d.variables["level"][:]
    idx = [int(np.where(lev == L)[0][0]) for L in LEVELS]
    tv = d.variables["time"]
    dts = netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False,
                           only_use_python_datetimes=True)
    tns = np.array([np.datetime64(x).astype("datetime64[ns]").astype("int64") for x in dts])
    lat = d.variables["lat"][:].astype("float64"); lon = d.variables["lon"][:].astype("float64")
    arr = np.stack([np.asarray(d.variables[var][:, i, :, :], dtype="float32") for i in idx], 0)
    d.close(); os.remove(f)
    assert np.abs(arr).max() > 1.0, f"{var} {year} looks empty"
    return tns, lat, lon, arr        # arr: [3 levels, T, lat, lon]


t0 = time.time()
for year in range(int(years.min()), int(years.max()) + 1):
    sel = np.where(years == year)[0]
    if len(sel) == 0:
        continue
    tns, lat, lon, U = get("uwnd", year)
    _, _, _, V = get("vwnd", year)
    # deep-layer mean over the three levels, mass-weighted
    Ud = (WEIGHTS[:, None, None, None] * U).sum(0)     # [T, lat, lon]
    Vd = (WEIGHTS[:, None, None, None] * V).sum(0)
    matched = 0
    for k in sel:
        ti = int(np.abs(tns - bt[k]).argmin())
        if abs(int(tns[ti]) - int(bt[k])) > MAX_DT:
            continue                                   # outside coverage -> zeros, ok stays False
        OKD[k] = True; matched += 1
        li = int(np.abs(lat - bla[k]).argmin())
        lj = int(np.abs(lon - (blo[k] % 360.0)).argmin())
        rows = np.clip(np.arange(li - HALF, li + HALF + 1), 0, len(lat) - 1)
        cols = np.mod(np.arange(lj - HALF, lj + HALF + 1), len(lon))
        DLM[k, 0] = Ud[ti][np.ix_(rows, cols)].astype("float16")
        DLM[k, 1] = Vd[ti][np.ix_(rows, cols)].astype("float16")
    print(f"{year}: {matched}/{len(sel)} matched | uDLM {DLM[sel][:, 0].astype('float32').mean():+.1f} "
          f"vDLM {DLM[sel][:, 1].astype('float32').mean():+.1f} m/s | {time.time()-t0:.0f}s", flush=True)

print(f"\ntotal matched {OKD.sum()}/{N}  ({100*OKD.mean():.1f}%)", flush=True)

# ---- assemble the 4-channel tensor: SLP pair (kept) + deep-layer mean (new) ----
q = np.load("/content/steer5_int8.npz")
sc5 = q["scale"]; ok5 = q["ok"]
# training reads q/31.75 to recover the normalised-clipped value, so the int8 IS the input and the
# SLP channels just carry through verbatim -- no dequant/requant round-trip that could drift them
slp_int8 = q["q"][:, :2]                                    # [N,2,17,17] int8, already scaled+clipped
slp_scale = sc5[:2]
slp_ok = ok5[:, 0]

# scale the DLM channels the same way: per-channel std over available windows, then int8 at 4 sigma
CLIP = 4.0
d32 = DLM.astype("float32")
dsc = np.array([d32[OKD, c].std() if OKD.any() else 1.0 for c in range(2)], dtype="float32")
dsc[dsc == 0] = 1.0
dq = np.clip(d32 / dsc[None, :, None, None], -CLIP, CLIP)
dq = np.round(dq * (127.0 / CLIP)).astype("int8")
dq[~OKD] = 0

q4 = np.concatenate([slp_int8, dq], axis=1)                # [N,4,17,17] int8
scale4 = np.concatenate([slp_scale, dsc])                  # [4]
# 3 columns to match v17's AVAIL shape [SLP pair, steering pair, SST]; SST is unused here (all False)
ok4 = np.stack([slp_ok, OKD, np.zeros(N, bool)], axis=1)   # [N,3]
np.savez_compressed("/content/dlm4_int8.npz", q=q4, ok=ok4, scale=scale4)
sz = os.path.getsize("/content/dlm4_int8.npz") / 1e6
print(f"\nwrote /content/dlm4_int8.npz ({sz:.0f} MB)  channels [SLPanom, SLPtend, uDLM, vDLM]", flush=True)
print(f"  DLM scale (m/s): u {dsc[0]:.2f}  v {dsc[1]:.2f}   steering availability {ok4[:,1].mean():.3f}", flush=True)
if os.path.isdir("/content/drive/MyDrive/typhoon"):
    import shutil; shutil.copy("/content/dlm4_int8.npz", "/content/drive/MyDrive/typhoon")
    print("  mirrored to Drive (survives a disconnect)", flush=True)
