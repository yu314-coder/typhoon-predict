"""Extract storm-centered sea-level-pressure patches (the SURROUNDING pressure field that steers the
storm) from NCEP/NCAR reanalysis, for every window in the dataset.

Two channels per window:
  ch0 = SLP anomaly now      (regional pattern: ridge vs trough around the storm)
  ch1 = 24-h SLP tendency    (is the steering ridge building or collapsing?)
Patch is +/-20 deg at 2.5 deg = 17x17, i.e. for a South China Sea storm it spans China, Indochina,
Taiwan and the Philippine Sea -- the region that actually sets the steering flow.

Time matching is nearest-neighbour WITH A TOLERANCE -- see extract_steer.py for why. An unbounded
argmin snaps windows past the end of a partial file onto its last timestep; the 2026 file stops at
2026-03-17, so every later storm silently got a March field. Such windows are left as zeros and
flagged not-ok so downstream code can treat them as missing instead of real.
"""
import numpy as np, netCDF4, os, glob

NPZ = "track_build/track_windows_v13.npz"
OUT = "track_build/slp_patches.npy"
OUT_OK = "track_build/slp_ok.npy"
HALF = 8                      # +/-8 cells at 2.5 deg = +/-20 deg -> 17x17
# 6-hourly data: a fix is at most 3 h from the nearest timestep. 6 h leaves margin.
MAX_DT_H = float(os.environ.get("SLP_MAX_DT_H", "6"))
MAX_DT = int(MAX_DT_H * 3600 * 1e9)
z = np.load(NPZ, allow_pickle=True)
bt = z["base_time"].astype("int64")          # ns since epoch
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
N = len(bt)
years = z["year"].astype(int)
P = np.zeros((N, 2, 2 * HALF + 1, 2 * HALF + 1), dtype="float16")
ok = np.zeros(N, dtype=bool)
print(f"{N} windows | patch {2*HALF+1}x{2*HALF+1} (+/-{HALF*2.5:.0f} deg) | 2 channels")

done = 0
for yr in sorted(set(years.tolist())):
    f = f"track_build/geo/slp/slp.{yr}.nc"
    if not os.path.exists(f):
        continue
    d = netCDF4.Dataset(f)
    tv = d.variables["time"]
    # file times -> int64 ns since epoch
    dts = netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    tns = np.array([np.datetime64(x).astype("datetime64[ns]").astype("int64") for x in dts])
    lat = d.variables["lat"][:].astype("float64")      # 90 -> -90
    lon = d.variables["lon"][:].astype("float64")      # 0 -> 357.5
    slp = d.variables["slp"][:].astype("float32") / 100.0   # Pa -> hPa  [T,73,144]
    sel = np.where(years == yr)[0]
    for i in sel:
        ti = int(np.abs(tns - bt[i]).argmin())
        if abs(int(tns[ti]) - int(bt[i])) > MAX_DT:
            continue                                   # past the end of a partial file -> leave zeros
        if ti < 4:
            continue                                   # no true 24 h lookback; clamping would fake a tendency
        ok[i] = True
        ti24 = ti - 4                                  # 24 h earlier (4 x 6h steps)
        li = int(np.abs(lat - bla[i]).argmin())
        lj = int(np.abs(lon - (blo[i] % 360.0)).argmin())
        rows = np.clip(np.arange(li - HALF, li + HALF + 1), 0, len(lat) - 1)
        cols = np.mod(np.arange(lj - HALF, lj + HALF + 1), len(lon))   # wrap longitude
        now = slp[ti][np.ix_(rows, cols)]
        prev = slp[ti24][np.ix_(rows, cols)]
        P[i, 0] = (now - now.mean()).astype("float16")   # regional anomaly pattern
        P[i, 1] = (now - prev).astype("float16")         # 24 h tendency
    d.close()
    done += len(sel)
    print(f"  {yr}: {len(sel):>6} windows  (total {done}/{N})", flush=True)

np.save(OUT, P)
np.save(OUT_OK, ok)
print(f"\nsaved {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB) | {OUT_OK}")
print(f"  matched {ok.sum()}/{N} windows ({100*ok.mean():.1f}%); {(~ok).sum()} left as zeros "
      f"(outside +/-{MAX_DT_H:.0f}h of a timestep, or no 24h lookback)")
print(f"  anomaly ch: mean {P[:,0].astype('float32').mean():+.2f} std {P[:,0].astype('float32').std():.2f} hPa")
print(f"  tendency ch: mean {P[:,1].astype('float32').mean():+.2f} std {P[:,1].astype('float32').std():.2f} hPa/24h")
