"""ERA5 0.25 deg storm-centred patches, from NCAR RDA OPeNDAP. CPU only, runs on disk D.

WHY 0.25 AND WHY STORM-CENTRED. v24 replaced a 17x17 storm-centred patch (2.5 deg) with a fixed
basin box at the SAME 2.5 deg and error went 443 -> 529 km, p=0.000, bootstrap CI [+64, +111].
At 2.5 deg a typhoon is about one grid cell, so a wide box at coarse resolution throws away the
vortex-scale steering flow that actually moves the storm. The lesson is not "domain too small" --
it is RESOLUTION NEAR THE VORTEX. ERA5 at 0.25 deg is 10x finer than NCEP R1.

    65 x 65 at 0.25 deg = +-8 deg around the storm, at 10x the resolution of the old +-21 deg patch

ACCESS. RDA needs NO authentication (verified: .das returns 200). Dataset id is d633000; the old
ds633.0 form 404s. DAP rejects an arbitrary index list for a dimension -- levels must be a
CONTIGUOUS slice, which is also free: measured 17 levels = 21.0 MB in 15.0 s against 1 level =
1.2 MB in 19.5 s. Cost is per REQUEST, not per byte, so take the whole 200-850 hPa stack.

STRATEGY. One request per (day, variable) for the whole basin box, then crop every storm patch in
that day locally and discard the rest. Requesting each window separately would be 193,609 requests
instead of ~25,000.

    box   100-180E, 0-60N  ->  lat idx 120:361, lon idx 400:721  (241 x 321)   [verified]
    levels 850/500/200     ->  indices 30/21/14, taken as the slice 14:31      [verified]

OUTPUT. int8 per year under track_build/era5/, one file per year, resumable. Final size is about
7 GB for all 193,609 windows -- small enough to upload to Drive, unlike the ~500 GB of raw
transfer it is cropped from.

PROBE MODE (default): does N days, reports real throughput, writes nothing. Run that first and
decide before committing to the full pull.
"""
import os, sys, time, math, socket
import numpy as np

# ERA5 last stalled with every RDA socket in CLOSE_WAIT and all 10 workers blocked on reads that
# never returned -- netCDF4/HDF5 does no request timeout of its own. A default socket timeout makes
# a dead connection raise (caught by the per-day retry loop) instead of hanging the whole pull.
socket.setdefaulttimeout(90)

# macOS defaults to a 256 open-file-descriptor limit. OPeNDAP connections in CLOSE_WAIT linger,
# so under any real concurrency the cap is reached and new connections fail. Raise it.
import resource
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, _hard), _hard))

try:
    import netCDF4
except ImportError:
    sys.exit("pip install netCDF4")

RDA = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/e5.oper.an.pl"
# GRID SUFFIX DIFFERS BY VARIABLE. Scalars are ll025sc; the WIND fields are ll025uv. Hardcoding
# ll025sc for all three made every u/v request a 404 -- 3,444 "No such file" errors across a
# 4.3-hour run that still reported an ETA instead of stopping. Verified against the 199001 and
# 202001 catalogs; it is not year-dependent.
VARS = [("128_129_z", "Z", "ll025sc"), ("128_131_u", "U", "ll025uv"), ("128_132_v", "V", "ll025uv")]
LAT0, LAT1 = 120, 361          # 60N .. 0N   (ERA5 latitude runs north -> south)
# 100E..180E. Widening east to 260E to pick up the EP costs MORE than linearly -- measured
# 7.4 MB / 38.0 s against 14.8 MB / 97.4 s, so WP+EP 1990+ would be ~40 h at 12 workers rather
# than 13. EP is a second pass on the same machinery; nothing here has to be redone for it.
LON0, LON1 = 400, 721          # 100E .. 180E
LEV0, LEV1 = 14, 31            # 200 .. 850 hPa inclusive
LEV_HPA = [200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
KEEP = [LEV_HPA.index(p) for p in (850, 500, 200)]      # what we store
HALF = 32                       # 65 x 65 patch
PROBE = int(os.environ.get("PROBE", "20"))
OUT = "track_build/era5"

z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
N = len(bt)

# ERA5 is hourly; our windows are 3- or 6-hourly. Snap each window to the nearest ERA5 hour that is
# AT OR BEFORE it -- never after, or the patch would carry information from after the forecast
# launched. ERA5 being hourly makes this at most 59 minutes stale, against up to 5.85 h on the
# 6-hourly NCEP fields.
HOUR = int(3600 * 1e9)
snap = (bt // HOUR) * HOUR
age_h = (bt - snap) / 3.6e12
assert (age_h >= 0).all() and (age_h < 1.0).all(), "snap is not causal or is over an hour stale"

days = np.unique(snap // (24 * HOUR))
print(f"{N:,} windows -> {len(days):,} distinct ERA5 days "
      f"({np.datetime64(int(days[0]*24*HOUR),'ns').astype('datetime64[D]')} .. "
      f"{np.datetime64(int(days[-1]*24*HOUR),'ns').astype('datetime64[D]')})")
print(f"field age after snapping to the hour: mean {age_h.mean()*60:.1f} min, "
      f"max {age_h.max()*60:.1f} min\n")

lat_g = 90.0 - 0.25 * np.arange(721)      # ERA5 grid, north -> south
lon_g = 0.25 * np.arange(1440)


def fetch_day(day_idx, want_levels=True):
    """[nt, nvar, nlev, 241, 321] float32 for one day, plus the hour stamps."""
    d0 = int(day_idx) * 24 * HOUR
    ds = np.datetime64(d0, "ns").astype("datetime64[D]")
    y, m, dd = str(ds)[:4], str(ds)[5:7], str(ds)[8:10]
    out, times = [], None
    for code, dapname, grid in VARS:
        url = (f"{RDA}/{y}{m}/e5.oper.an.pl.{code}.{grid}."
               f"{y}{m}{dd}00_{y}{m}{dd}23.nc")
        for attempt in range(3):
            try:
                nc = netCDF4.Dataset(url)
                sl = slice(LEV0, LEV1) if want_levels else slice(21, 22)
                a = np.asarray(nc.variables[dapname][:, sl, LAT0:LAT1, LON0:LON1], "float32")
                if times is None:
                    times = d0 + np.arange(a.shape[0]) * HOUR
                nc.close()
                out.append(a)
                break
            except Exception as ex:
                if attempt == 2:
                    return None, None, f"{ds} {dapname}: {ex}"
                time.sleep(4 * (attempt + 1))
    return np.stack(out, 1), times, None


# ---- concurrency probe ---------------------------------------------------------------------
# 16-way vs 32-way is only a 2x win if BOTH the link and the server scale. Neither is verified,
# so measure it rather than assume: same work at each concurrency, report effective speedup.
if os.environ.get("MODE", "probe") == "conc":
    from concurrent.futures import ThreadPoolExecutor
    NDAY = int(os.environ.get("NDAY", "16"))
    pick = days[::max(1, len(days) // NDAY)][:NDAY]
    print(f"CONCURRENCY PROBE: {NDAY} days x 3 vars = {NDAY*3} requests at each level\n")
    print(f"{'workers':>8} {'wall':>8} {'MB/s':>7} {'s/day':>7} {'speedup':>8} {'fails':>6}")
    base = None
    for w in (4, 8, 16, 32):
        t0 = time.time(); nb = 0; nf = 0
        with ThreadPoolExecutor(max_workers=w) as ex:
            for a, tt, err in ex.map(lambda d: fetch_day(d), pick):
                if err:
                    nf += 1
                else:
                    nb += a.nbytes
        el = time.time() - t0
        if base is None:
            base = el * 4          # normalise to per-worker-equivalent
        sp = base / el if el > 0 else 0
        print(f"{w:>8} {el:>7.1f}s {nb/1e6/el:>6.1f} {el/NDAY:>6.1f} {sp:>7.1f}x {nf:>6}")
    print("\nspeedup near linear -> use 32. flattening -> the link or RDA is the limit, use the knee.")
    sys.exit(0)

if os.environ.get("MODE", "probe") == "probe":
    print(f"PROBE: {PROBE} days, 3 variables, 17 levels each\n")
    t0 = time.time(); nbytes = 0; nreq = 0; fails = []
    for k, di in enumerate(days[::max(1, len(days) // PROBE)][:PROBE]):
        t1 = time.time()
        a, tt, err = fetch_day(di)
        if err:
            fails.append(err); print(f"  FAIL {err}"); continue
        nbytes += a.nbytes; nreq += 3
        if k < 3 or k == PROBE - 1:
            ds = np.datetime64(int(di) * 24 * HOUR, "ns").astype("datetime64[D]")
            print(f"  {ds}  {a.shape}  {a.nbytes/1e6:5.1f} MB  {time.time()-t1:5.1f}s")
    el = time.time() - t0
    if nreq == 0:
        sys.exit("every probe request failed")
    per_day = el / (nreq / 3)
    print(f"\n{nreq} requests, {nbytes/1e6:.0f} MB, {el:.0f}s  "
          f"-> {per_day:.1f}s per day, {nbytes/1e6/el:.1f} MB/s")
    tot_gb = nbytes / (nreq / 3) * len(days) / 1e9
    print(f"\nFULL PULL ESTIMATE ({len(days):,} days, 3 vars, 17 levels)")
    print(f"  transfer   {tot_gb:.0f} GB")
    for par in (1, 8, 16, 32):
        print(f"  wall clock at {par:2d}-way parallel: {per_day*len(days)/par/3600:6.1f} h")
    print(f"\n  stored (int8, 65x65 patches, 3 levels, 3 vars, {N:,} windows): "
          f"{N*65*65*3*3/1e9:.1f} GB")
    if fails:
        print(f"\n{len(fails)} failures during probe:")
        for f in fails[:5]:
            print("  ", f)
    print("\nrerun with MODE=full to do the real pull")
    sys.exit(0)

# ---- full pull -----------------------------------------------------------------------------
# PARALLEL over days. The first draft of this loop was serial: 3 variables x ~15 s x 10,392 days
# is 130 hours, which would never have finished. Days are independent, so a thread pool over them
# is the whole fix.
#
# STRIDED LEVELS. slice(14, 31, 8) -> global levels 14/22/30 = 200/550/850 hPa. Per-request cost
# is fixed, so this does not change wall clock, but it cuts transfer 5.7x (655 -> 116 GB). The
# only sacrifice is 550 hPa in place of 500 for the mid-level steering, and in that layer the two
# are strongly correlated.
from concurrent.futures import ThreadPoolExecutor

W = int(os.environ.get("WORKERS", "24"))
Y0 = int(os.environ.get("Y0", "1990"))
STRIDE = int(os.environ.get("STRIDE", "8"))
TSTRIDE = int(os.environ.get("TSTRIDE", "3"))   # 98.8% of windows sit on the 8 synoptic hours
YMAX = int(os.environ.get("YMAX", "9999"))      # upper year bound, so two instances can split
REVERSE = int(os.environ.get("REVERSE", "1"))   # 1 = newest-first (recent half), 0 = oldest-first
LEV_SL = slice(LEV0, LEV1, STRIDE)
KEEP3 = list(range(len(range(LEV0, LEV1, STRIDE))))
LEV_LABEL = [LEV_HPA[k] for k in range(0, len(LEV_HPA), STRIDE)]
os.makedirs(OUT, exist_ok=True)


def fetch_day_sl(day_idx):
    """One day, all 3 variables, strided levels. Returns (day_idx, array, err)."""
    d0 = int(day_idx) * 24 * HOUR
    ds = np.datetime64(d0, "ns").astype("datetime64[D]")
    y, m, dd = str(ds)[:4], str(ds)[5:7], str(ds)[8:10]
    out = []
    for code, dapname, grid in VARS:
        url = (f"{RDA}/{y}{m}/e5.oper.an.pl.{code}.{grid}.{y}{m}{dd}00_{y}{m}{dd}23.nc")
        ok = False
        for attempt in range(8):
            nc = None
            try:
                nc = netCDF4.Dataset(url)
                # stride the HOUR axis too: time is NOT free per request (24 steps = 22.3 MB /
                # 64.5 s against 4 steps = 3.7 MB / 7.2 s), and 98.8% of windows land on the
                # 3-hourly synoptic times.
                a = np.asarray(nc.variables[dapname][::TSTRIDE, LEV_SL, LAT0:LAT1, LON0:LON1],
                               "float32")
                out.append(a); ok = True; break
            except Exception as ex:
                time.sleep(min(4 * (attempt + 1), 20))
                last = str(ex)[:90]
            finally:
                # close on EVERY path. The leak that piled 262 CLOSE_WAIT sockets to the FD cap
                # was a Dataset opened, the read timing out, and close() never reached. THIS is
                # the real fix; the socket timeout only bounds how long each hang lasts.
                if nc is not None:
                    try:
                        nc.close()
                    except Exception:
                        pass
        if not ok:
            return day_idx, None, f"{ds} {dapname}: {last}"
    return day_idx, np.stack(out, 1), None


# WP+EP only -- those are the basins we score, and fetching SI/SP/NI would be pure waste.
# The patch must also fit inside the box with its 8 deg half-width.
_bas = z["basin"].astype(str)
_lo360 = blo % 360
BASIN = os.environ.get("BASIN", "WP")
_fits = (bla >= 8.0) & (bla <= 52.0) & (_lo360 >= 108.0) & (_lo360 <= 172.0)
by_year = {}
for i in range(N):
    y = int(str(np.datetime64(int(snap[i]), "ns").astype("datetime64[Y]")))
    if Y0 <= y <= YMAX and _bas[i] == BASIN and _fits[i]:
        by_year.setdefault(y, []).append(i)

print(f"levels {LEV_LABEL} hPa | {W} workers | years {Y0}+ "
      f"| {sum(len(v) for v in by_year.values()):,} windows", flush=True)
print(f"TMPDIR={os.environ.get('TMPDIR','(default)')}\n", flush=True)

t_all = time.time()
# NEWEST FIRST. The test period is 2020+, so processing in reverse means the years we evaluate
# on land within the first couple of hours and the pilot experiment can run while the rest of the
# archive is still downloading.
for year in sorted(by_year, reverse=bool(REVERSE)):
    f = f"{OUT}/era5_{year}.npz"
    if os.path.exists(f):
        print(f"  {year}: present, skipping", flush=True); continue
    idx = np.array(by_year[year]); t0 = time.time()
    nlev = len(LEV_LABEL)
    P = np.zeros((len(idx), 3, nlev, 2 * HALF + 1, 2 * HALF + 1), "float32")
    got = np.zeros(len(idx), "float32")
    dmap = {}
    for s_, i in enumerate(idx):
        dmap.setdefault(int(snap[i] // (24 * HOUR)), []).append(s_)
    fails = 0
    with ThreadPoolExecutor(max_workers=W) as ex:
        for di, a, err in ex.map(fetch_day_sl, sorted(dmap)):
            if err:
                fails += 1; continue
            for s_ in dmap[di]:
                i = idx[s_]
                # index into the strided hour axis; windows off the 3-hourly grid snap back to
                # the previous synoptic hour, which stays causal (at most 2 h stale)
                ti = int((snap[i] - int(di) * 24 * HOUR) // HOUR) // TSTRIDE
                if ti >= a.shape[0]:
                    continue
                r = int(np.abs(lat_g[LAT0:LAT1] - bla[i]).argmin())
                c = int(np.abs(lon_g[LON0:LON1] - (blo[i] % 360)).argmin())
                r0, r1, c0, c1 = r - HALF, r + HALF + 1, c - HALF, c + HALF + 1
                if r0 < 0 or c0 < 0 or r1 > (LAT1 - LAT0) or c1 > (LON1 - LON0):
                    continue
                P[s_] = a[ti][:, :, r0:r1, c0:c1]
                got[s_] = 1.0
    sc = np.array([max(np.abs(P[:, v]).max(), 1e-6) / 127.0 for v in range(3)], "float32")
    q = np.clip(np.round(P / sc[None, :, None, None, None]), -127, 127).astype("int8")
    np.savez_compressed(f, q=q, scale=sc, got=got, widx=idx,
                        levels=np.array(LEV_LABEL), vars=np.array(["z", "u", "v"]))
    el = (time.time() - t0) / 60
    done = sorted(by_year, reverse=bool(REVERSE)).index(year) + 1
    eta = (time.time() - t_all) / done * (len(by_year) - done) / 3600
    print(f"  {year}: {len(idx):6,} win {100*got.mean():5.1f}% got  "
          f"{os.path.getsize(f)/1e6:6.1f} MB  {fails:3d} fail  {el:5.1f} min   ETA {eta:4.1f} h",
          flush=True)
    # A year where nothing landed is a broken run, not slow progress. The first version caught
    # every exception, wrote a file of zeros, printed an ETA and carried on for 4.3 hours.
    # A year where nothing landed is usually a broken run. But it can also mean the source
    # simply stops there: ERA5's d633000 ends after 2026-03 (202605 is a 404), so 2026 windows
    # from April on have no file to fetch. Distinguish the two -- halt on a systematic failure,
    # but only warn and continue when a SHORT year is partially covered, which is the signature
    # of hitting the end of the archive rather than a bug.
    if got.mean() < 0.5:
        if len(idx) < 1000 and got.mean() > 0.05:
            print(f"    NOTE: {year} is partial ({100*got.mean():.0f}%) and short "
                  f"({len(idx)} windows) -- looks like the end of the archive, continuing",
                  flush=True)
        else:
            os.remove(f)
            raise SystemExit(f"ABORT: {year} got only {100*got.mean():.1f}% of windows "
                             f"({fails} failed days). Not continuing -- fix the cause first.")
print(f"\ndone in {(time.time()-t_all)/3600:.1f} h -> {OUT}/")
