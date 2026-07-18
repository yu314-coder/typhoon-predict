"""Extract storm-centered 500 hPa STEERING WIND (u,v) patches from NCEP daily-mean reanalysis.

500 hPa is the classic steering level: a TC is advected by the deep-layer environmental flow, and
u500/v500 IS that current. v13 used sea-level pressure -- the SURFACE pattern -- which fixed the
speed bias but not direction, because steering lives in the mid-troposphere.

Per year: download uwnd/vwnd daily means (149 MB each), keep only the 500 hPa level, extract
storm-centered 17x17 patches, delete the raw files.
Usage: python extract_steer.py <year>

Time matching is nearest-neighbour WITH A TOLERANCE. The reanalysis file for the current year is
partial, and an unbounded argmin silently snaps every window past its end onto the last available
day -- which is how 2026 ended up with a +8 sigma mean u500 and sent v14's Bavi forecast 3442 km
the wrong way. Windows with no timestep inside MAX_DT are left as zeros and flagged not-ok, so the
model's steering-dropout path treats them as MISSING rather than as fabricated data.
"""
import sys, os, time, subprocess
import numpy as np, netCDF4

YEAR = int(sys.argv[1]); HALF = 8
# daily means: a 6-hourly storm fix is at most 12 h from the nearest day centre. 18 h leaves
# margin for leap/edge cases while still catching a snap of days-to-months.
MAX_DT_H = float(os.environ.get("STEER_MAX_DT_H", "18"))
MAX_DT = int(MAX_DT_H * 3600 * 1e9)
OUT = f"track_build/geo/steer_{YEAR}.npz"
if os.path.exists(OUT):
    print(f"{YEAR}: exists, skip"); sys.exit(0)

z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
sel = np.where(z["year"].astype(int) == YEAR)[0]
if len(sel) == 0:
    np.savez_compressed(OUT, idx=np.array([], "int64"), patch=np.zeros((0, 2, 17, 17), "float16"),
                        ok=np.zeros(0, bool))
    print(f"{YEAR}: none"); sys.exit(0)
bt = z["base_time"].astype("int64")[sel]
bla = z["base_lat"].astype("float64")[sel]; blo = z["base_lon"].astype("float64")[sel]

BASE = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure"
def get(var):
    f = f"track_build/geo/tmp_{var}_{YEAR}.nc"
    if not os.path.exists(f):
        for a in range(4):
            r = subprocess.run(["curl", "-sSL", "--max-time", "600", "-o", f, f"{BASE}/{var}.{YEAR}.nc"])
            if r.returncode == 0 and os.path.exists(f) and os.path.getsize(f) > 10_000_000:
                break
            time.sleep(5)
    d = netCDF4.Dataset(f)
    lev = d.variables["level"][:]; i5 = int(np.where(lev == 500)[0][0])
    tv = d.variables["time"]
    dts = netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    tns = np.array([np.datetime64(x).astype("datetime64[ns]").astype("int64") for x in dts])
    lat = d.variables["lat"][:].astype("float64"); lon = d.variables["lon"][:].astype("float64")
    arr = np.asarray(d.variables[var][:, i5, :, :], dtype="float32")
    d.close(); os.remove(f)
    assert np.abs(arr).max() > 1.0, f"{var} {YEAR} looks empty"
    return tns, lat, lon, arr

t0 = time.time()
tns, lat, lon, U = get("uwnd")
_, _, _, V = get("vwnd")
cov0 = tns.min().astype("datetime64[ns]"); cov1 = tns.max().astype("datetime64[ns]")
P = np.zeros((len(sel), 2, 2 * HALF + 1, 2 * HALF + 1), dtype="float16")
ok = np.zeros(len(sel), dtype=bool)
for k in range(len(sel)):
    ti = int(np.abs(tns - bt[k]).argmin())
    if abs(int(tns[ti]) - int(bt[k])) > MAX_DT:
        continue                                   # outside file coverage -> leave zeros, ok stays False
    ok[k] = True
    li = int(np.abs(lat - bla[k]).argmin()); lj = int(np.abs(lon - (blo[k] % 360.0)).argmin())
    rows = np.clip(np.arange(li - HALF, li + HALF + 1), 0, len(lat) - 1)
    cols = np.mod(np.arange(lj - HALF, lj + HALF + 1), len(lon))
    P[k, 0] = U[ti][np.ix_(rows, cols)].astype("float16")
    P[k, 1] = V[ti][np.ix_(rows, cols)].astype("float16")
np.savez_compressed(OUT, idx=sel.astype("int64"), patch=P, ok=ok)
f = P[ok].astype("float32") if ok.any() else np.zeros((1, 2, 1, 1), "float32")
print(f"{YEAR}: {ok.sum()}/{len(sel)} win matched in {time.time()-t0:.0f}s | file covers {cov0} .. {cov1} | "
      f"u500 mean {f[:,0].mean():+.1f} std {f[:,0].std():.1f} | v500 mean {f[:,1].mean():+.1f} "
      f"std {f[:,1].std():.1f} m/s", flush=True)
if not ok.all():
    print(f"  !! {(~ok).sum()} window(s) outside +/-{MAX_DT_H:.0f}h of any timestep -> marked "
          f"steering-unavailable (zeros), NOT snapped to the nearest edge", flush=True)
