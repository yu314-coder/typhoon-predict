"""Deep-layer-mean steering patches for Typhoon Tip (1979), so v20 can forecast it.

Tip predates the v13 window set, so the training-time steering tensor does not cover it. Same
recipe as colab_v24_extract.py: 850/500/200 hPa winds, thickness-weighted 0.269/0.500/0.231, the
same 18 h time-tolerance guard, and -- critically -- normalised with the SCALE THE MODEL WAS
TRAINED WITH, not Tip's own standard deviation. Rescaling to Tip's own statistics would hand the
model an input distribution it never saw and quietly invalidate the comparison.

SLP channels (0,1) are carried across from tip_steer4.npy unchanged, exactly as the training
tensor carried them from steer5.

CAVEAT worth stating before any number appears: 1979 is pre-satellite. The reanalysis Tip is
embedded in is far less constrained than the 1980+ fields v20 trained on, and 200 hPa over the
1979 Pacific is the least constrained level of the three. v17 already fell apart on Tip for this
reason (1776 km at +120 h). v20 inherits that exposure and adds a level to it.
"""
import os, subprocess, time
import numpy as np, netCDF4

HALF = 8
MAX_DT = int(18 * 3600 * 1e9)
LEVELS = [850, 500, 200]
W = np.array([0.269, 0.500, 0.231], dtype="float32")
BASE = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure"
CACHE = "track_build/geo"          # on disk D, never the main disk
os.makedirs(CACHE, exist_ok=True)

z = np.load("track_build/tip_fixed.npz", allow_pickle=True)
bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
N = len(bt)


def get(var):
    f = f"{CACHE}/tmp_{var}_1979.nc"
    if not os.path.exists(f):
        print(f"downloading {var}.1979.nc (~149 MB) ...", flush=True)
        for a in range(4):
            r = subprocess.run(["curl", "-sSL", "--max-time", "900", "-o", f, f"{BASE}/{var}.1979.nc"])
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
    d.close()
    assert np.abs(arr).max() > 1.0, f"{var} 1979 looks empty"
    return tns, lat, lon, arr


tns, lat, lon, U = get("uwnd")
_, _, _, V = get("vwnd")
Ud = (W[:, None, None, None] * U).sum(0)
Vd = (W[:, None, None, None] * V).sum(0)

DLM = np.zeros((N, 2, 17, 17), dtype="float32")
ok = np.zeros(N, dtype=bool)
for k in range(N):
    ti = int(np.abs(tns - bt[k]).argmin())
    if abs(int(tns[ti]) - int(bt[k])) > MAX_DT:
        continue
    ok[k] = True
    li = int(np.abs(lat - bla[k]).argmin())
    lj = int(np.abs(lon - (blo[k] % 360.0)).argmin())
    rows = np.clip(np.arange(li - HALF, li + HALF + 1), 0, len(lat) - 1)
    cols = np.mod(np.arange(lj - HALF, lj + HALF + 1), len(lon))
    DLM[k, 0] = Ud[ti][np.ix_(rows, cols)]
    DLM[k, 1] = Vd[ti][np.ix_(rows, cols)]

print(f"\nTip: {ok.sum()}/{N} windows matched  "
      f"| uDLM {DLM[ok][:,0].mean():+.1f}  vDLM {DLM[ok][:,1].mean():+.1f} m/s")

# normalise with the TRAINING scale, then pack the same way training's int8 did
scale = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
print(f"training DLM scale (m/s): u {scale[0]:.2f}  v {scale[1]:.2f}   (Tip's own would be "
      f"u {DLM[ok][:,0].std():.2f}  v {DLM[ok][:,1].std():.2f})")
norm = np.clip(DLM / scale[None, :, None, None], -4.0, 4.0)
norm[~ok] = 0.0

# tip_steer4.npy holds RAW PHYSICAL values, not normalised ones -- the range check below caught
# the wrong assumption. Normalise by the same steer5 scale the models were trained against, which
# is exactly what _tip_tracks.py does for v17/v18/v19.
_sc5 = np.load("track_build/steer5_scale.npy")[:2].astype("float32")
slp = np.clip(np.load("track_build/tip_steer4.npy")[:, :2].astype("float32")
              / _sc5[None, :, None, None], -4.0, 4.0)
out = np.concatenate([slp, norm], axis=1).astype("float32")
np.save("track_build/tip_dlm4.npy", out)
print(f"\nwrote track_build/tip_dlm4.npy {out.shape}  [SLPanom, SLPtend, uDLM, vDLM]")
assert np.isfinite(out).all() and np.abs(out).max() <= 4.0 + 1e-6, \
    f"normalised range {out.min():.2f}..{out.max():.2f} is outside [-4,4]"
print(f"  normalised range {out.min():.2f} .. {out.max():.2f}  (inside [-4, 4], as training saw)")
for f in ("uwnd", "vwnd"):
    p = f"{CACHE}/tmp_{f}_1979.nc"
    if os.path.exists(p):
        os.remove(p)
print("  removed the raw NetCDF files")
