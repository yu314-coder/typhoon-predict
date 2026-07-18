"""Extract storm-centered SEA SURFACE TEMPERATURE patches -- the ocean fuel supply.

Through v15 the model's only ocean input was a lat+month climatological proxy: a smooth function
of latitude and calendar month with ZERO interannual variability. It cannot tell an El Nino year
from a La Nina one, cannot see a warm eddy, and cannot see the cold wake a storm leaves behind.
This replaces it with observed SST.

Sources (both are absolute SST in degC, land masked):
  * NOAA OISST v2 weekly 1 deg   -- 1981-10-29 .. 2023-01-29   (primary)
  * NOAA ERSST v5 monthly 2 deg  -- 1854-01 .. present          (fills 1980-81 and 2023+)
The two agree to +0.19 degC over the WP in 2010-2020; ERSST is offset onto the OISST calibration
so a window's source does not leak into the value.

Output is on the SAME storm-centered 2.5 deg 17x17 grid as the SLP and steering patches, so all
channels of the stacked tensor describe the same piece of ocean/atmosphere.

Values are stored as (SST - 28 degC) so the channel is centered near zero; land cells are filled
with the patch's own ocean mean before differencing, so a coastline does not inject a cliff.

Windows with no source inside the tolerance are left as zeros and flagged not-ok -- same contract
as extract_steer.py, so the model treats them as MISSING rather than as a fabricated ocean.
"""
import os
import numpy as np, netCDF4

NPZ = "track_build/track_windows_v13.npz"
OUT = "track_build/sst_patches.npy"
OUT_OK = "track_build/sst_ok.npy"
OUT_SRC = "track_build/sst_src.npy"          # 0 = none, 1 = OISST weekly, 2 = ERSST monthly
HALF, STEP = 8, 2.5                          # +/-20 deg at 2.5 deg -> 17x17, matches the other channels
REF_C = 28.0                                 # centering constant, degC
OISST_TOL = int(7 * 24 * 3600 * 1e9)         # weekly product: a fix is <= 3.5 d from a week centre
ERSST_TOL = int(31 * 24 * 3600 * 1e9)        # monthly product

z = np.load(NPZ, allow_pickle=True)
bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64") % 360.0
N = len(bt)

def load(path):
    d = netCDF4.Dataset(path); d.set_auto_mask(True)
    tv = d.variables["time"]
    dts = netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    tns = np.array([np.datetime64(x).astype("datetime64[ns]").astype("int64") for x in dts])
    lat = np.asarray(d.variables["lat"][:], dtype="float64")
    lon = np.asarray(d.variables["lon"][:], dtype="float64") % 360.0
    sst = np.ma.filled(d.variables["sst"][:].astype("float32"), np.nan)
    d.close()
    return tns, lat, lon, sst

print("loading SST sources ...", flush=True)
o1t, olat, olon, o1 = load("track_build/geo/sst/sst.wkmean.1981-1989.nc")
o2t, _, _, o2 = load("track_build/geo/sst/sst.wkmean.1990-present.nc")
ot = np.concatenate([o1t, o2t]); osst = np.concatenate([o1, o2], axis=0)
order = np.argsort(ot); ot, osst = ot[order], osst[order]
et, elat, elon, esst = load("track_build/geo/sst/sst.mnmean.nc")
print(f"  OISST weekly {osst.shape}  {ot.min().astype('datetime64[ns]')} .. {ot.max().astype('datetime64[ns]')}")
print(f"  ERSST monthly {esst.shape}  {et.min().astype('datetime64[ns]')} .. {et.max().astype('datetime64[ns]')}")

# --- put ERSST on the OISST calibration, measured over their tropical overlap ---
def tropics_mean(tns, lat, lon, arr, y0=2010, y1=2020):
    yrs = tns.astype("datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
    k = np.where((yrs >= y0) & (yrs <= y1))[0]
    la = np.where(np.abs(lat) <= 30)[0]
    return float(np.nanmean(arr[np.ix_(k, la, np.arange(len(lon)))]))
OFF = tropics_mean(ot, olat, olon, osst) - tropics_mean(et, elat, elon, esst)
print(f"  ERSST -> OISST offset over 30S-30N, 2010-2020: {OFF:+.3f} degC", flush=True)

# storm-centered sample offsets, degrees
doff = (np.arange(-HALF, HALF + 1) * STEP)

def sample(lat_g, lon_g, field, clat, clon):
    """Nearest-neighbour sample of `field` on the storm-centered 2.5deg grid."""
    rows = np.abs(lat_g[None, :] - np.clip(clat + doff, -89.9, 89.9)[:, None]).argmin(1)
    cols = np.abs(lon_g[None, :] - ((clon + doff) % 360.0)[:, None]).argmin(1)
    return field[np.ix_(rows, cols)]

P = np.zeros((N, 1, 2 * HALF + 1, 2 * HALF + 1), dtype="float16")
ok = np.zeros(N, dtype=bool)
src = np.zeros(N, dtype="uint8")
for i in range(N):
    patch = None
    ti = int(np.abs(ot - bt[i]).argmin())
    if abs(int(ot[ti]) - int(bt[i])) <= OISST_TOL:
        patch = sample(olat, olon, osst[ti], bla[i], blo[i]); src[i] = 1
    else:
        tj = int(np.abs(et - bt[i]).argmin())
        if abs(int(et[tj]) - int(bt[i])) <= ERSST_TOL:
            patch = sample(elat, elon, esst[tj], bla[i], blo[i]) + OFF; src[i] = 2
    if patch is None:
        continue
    ocean = np.isfinite(patch)
    if not ocean.any():
        src[i] = 0; continue                     # entirely land -- refuse to invent an ocean
    patch = np.where(ocean, patch, np.nanmean(patch[ocean]))
    P[i, 0] = (patch - REF_C).astype("float16")
    ok[i] = True
    if i % 25000 == 0:
        print(f"  {i}/{N}", flush=True)

np.save(OUT, P); np.save(OUT_OK, ok); np.save(OUT_SRC, src)
f = P[ok].astype("float32")
print(f"\nsaved {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB) | {OUT_OK} | {OUT_SRC}")
print(f"  matched {ok.sum()}/{N} ({100*ok.mean():.1f}%)  "
      f"OISST {int((src==1).sum())}  ERSST {int((src==2).sum())}  none {int((src==0).sum())}")
print(f"  SST-{REF_C:.0f}C: mean {f.mean():+.2f} std {f.std():.2f}  "
      f"=> absolute mean {f.mean()+REF_C:.2f} degC, range {f.min()+REF_C:.1f}..{f.max()+REF_C:.1f}")
