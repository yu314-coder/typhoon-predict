"""Basin-scale 6-hourly fields for v24 — the map the forecaster actually reads.

WHAT THIS FIXES. Everything so far has fed the model a 17x17 patch at 2.5 deg centred on the storm:
+-21 deg. For a typhoon at 20N that reaches 41N. The mid-latitude trough that decides whether it
recurves sits at 40-50N and is OUTSIDE the input entirely. The subtropical ridge edge -- the 5880
gpm contour that the storm runs along -- is only partly visible. The model has been asked to
predict recurvature while blind to its cause.

It has also been reading DAILY MEANS. colab_v24_extract.py pulls from
ncep.reanalysis.dailyavgs, so every 6-hourly window was matched to a field averaged over up to
+-12 h. This pulls the 6-hourly product instead: 4x the temporal resolution, same source, same
2.5 deg grid, same proven access path.

DOMAIN. 100-180E, 0-60N at 2.5 deg = 33 x 25 = 825 points. That is the box on an operational
WP steering chart: the ridge, the monsoon trough, and the mid-latitude westerlies all inside it.

CHANNELS (7). Z500 is the steering map itself. The winds are kept as separate levels rather than
pre-averaged into a deep-layer mean, because the 200-850 difference IS the vertical shear -- a
first-order intensity predictor the model has never seen. Pre-averaging would throw it away.
    0  hgt  500          the ridge/trough field
    1  uwnd 850   2  vwnd 850
    3  uwnd 500   4  vwnd 500
    5  uwnd 200   6  vwnd 200

STORAGE. int8 with a per-channel scale and offset, the same convention as dlm4_int8.npz. Fields
are stored once per TIMESTEP, not per window: 193,609 windows collapse to ~119k unique 6-hourly
times, so the whole basin dataset is well under 1 GB.

RESUMABLE. One .npz per year under track_build/basin/. A year already on disk is skipped, so an
interrupted run costs only the year it was in the middle of. This exists because the steering
tensor has been rebuilt from scratch on three separate VMs.
"""
import os, sys, time
import numpy as np

try:
    import netCDF4
except ImportError:
    sys.exit("pip install netCDF4")

DODS = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/pressure"
OUT = "track_build/basin"
LAT0, LAT1, LON0, LON1 = 0.0, 60.0, 100.0, 180.0
CHANNELS = [("hgt", 500), ("uwnd", 850), ("vwnd", 850),
            ("uwnd", 500), ("vwnd", 500), ("uwnd", 200), ("vwnd", 200)]
YEARS = range(int(os.environ.get("Y0", "1980")), int(os.environ.get("Y1", "2027")))

os.makedirs(OUT, exist_ok=True)


def grid_index(d):
    lat = d.variables["lat"][:]; lon = d.variables["lon"][:]; lev = d.variables["level"][:]
    ila = np.where((lat >= LAT0) & (lat <= LAT1))[0]
    ilo = np.where((lon >= LON0) & (lon <= LON1))[0]
    return ila, ilo, lev, lat[ila], lon[ilo]


def fetch_year(year):
    """[T,7,nlat,nlon] float32 plus the time axis, or None if the year is unavailable."""
    per, tvals, meta = [], None, None
    for var, level in CHANNELS:
        url = f"{DODS}/{var}.{year}.nc"
        for attempt in range(4):
            try:
                d = netCDF4.Dataset(url)
                break
            except Exception as ex:
                if attempt == 3:
                    print(f"  {year} {var}: unreachable after 4 tries ({ex})", flush=True)
                    return None, None, None
                time.sleep(6 * (attempt + 1))
        ila, ilo, lev, la, lo = grid_index(d)
        k = int(np.where(lev == level)[0][0])
        assert int(lev[k]) == level, f"level lookup failed for {level}"
        a = np.asarray(d.variables[var][:, k, ila[0]:ila[-1] + 1, ilo[0]:ilo[-1] + 1],
                       dtype="float32")
        if tvals is None:
            tv = d.variables["time"]
            # NCEP time is hours since 1800-01-01; convert to ns since epoch so it joins
            # straight onto track_windows_v13's base_time without a second convention
            tvals = ((np.asarray(tv[:], dtype="float64") * 3600.0
                      - 5364662400.0) * 1e9).astype("int64")
            meta = (la.astype("float32"), lo.astype("float32"))
        d.close()
        per.append(a)
    return np.stack(per, 1), tvals, meta


def quantise(x):
    """int8 with per-channel scale/offset. Z500 is ~5000 gpm with a small spread, so a shared
    scale would waste almost the whole range on the offset.

    FULL min/max, not a percentile clip. A percentile clip saves a little resolution but throws
    away exactly the extremes that matter here -- the deepest trough and the strongest ridge are
    the features that decide recurvature, and they live in the tails. With full range every value
    round-trips to within one quantisation step, which the caller asserts.
    """
    C = x.shape[1]
    off = np.empty(C, "float32"); sca = np.empty(C, "float32")
    q = np.empty(x.shape, "int8")
    for c in range(C):
        v = x[:, c]
        lo, hi = float(v.min()), float(v.max())
        off[c] = (hi + lo) / 2.0
        sca[c] = max((hi - lo) / 254.0, 1e-6)
        q[:, c] = np.clip(np.round((v - off[c]) / sca[c]), -127, 127).astype("int8")
    return q, sca, off


t_start = time.time()
done, skipped, failed = 0, 0, []
for year in YEARS:
    f = f"{OUT}/basin_{year}.npz"
    if os.path.exists(f):
        skipped += 1
        continue
    t0 = time.time()
    x, tv, meta = fetch_year(year)
    if x is None:
        failed.append(year)
        continue
    q, sca, off = quantise(x)
    # round-trip check: a silent quantisation bug would poison every downstream model
    err = float(np.abs((q.astype("float32") * sca[None, :, None, None]
                        + off[None, :, None, None]) - x).max(initial=0.0))
    tol = float(sca.max())
    assert err <= tol * 1.01, f"{year}: quantisation error {err:.3f} exceeds one step {tol:.3f}"
    np.savez_compressed(f, q=q, scale=sca, offset=off, time=tv,
                        lat=meta[0], lon=meta[1],
                        channels=np.array([f"{v}{l}" for v, l in CHANNELS]))
    done += 1
    print(f"  {year}  {x.shape[0]:4d} steps  {os.path.getsize(f)/1e6:6.1f} MB  "
          f"maxerr {err:.3f} (step {tol:.3f})  {time.time()-t0:5.1f}s", flush=True)

print(f"\n{done} years written, {skipped} already present, {len(failed)} failed "
      f"in {(time.time()-t_start)/60:.1f} min")
if failed:
    print("failed:", failed)
