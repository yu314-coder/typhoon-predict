"""AOML ocean heat content patches at storm positions. CPU only, disk D.

WHY. TrackFormer has never had ANY ocean data. Verified effect sizes from the literature put
SST at Cohen's d=0.60, OHC at 0.62, MPI at 0.76 -- and SST is the weakest of the three. SHIPS
carries COHC/CD26 but stops in 2021, covering only 7% of our 2020+ test set, so it cannot be the
source. This can: 2012-08-27 through 2026-07-19, which spans the whole test period.

    Ocean_Heat_Content   kJ/cm2, integrated above the 26C isotherm -- the fuel available
    D26 / D20            depth of the 26C / 20C isotherms, m -- how deep that fuel goes

SOURCE. ERDDAP griddap, no authentication:
    https://upwell.pfeg.noaa.gov/erddap/griddap/noaa_aoml_6b09_4e6f_46dd
Variable is Ocean_Heat_Content, NOT Ocean_Heat -- the short name 500s. Verified: one day of all
three variables over 100-180E/0-60N is 940 KB in 4.2 s.

DOMAIN. WP+EP, 100E to 260E, 0-60N. The first attempt used 100-180E only and got 19.3% of 2013
windows, which aborted -- correctly. The cause was the domain, not the data: our window file is
GLOBAL (SI 944, EP 627, SP 283, NI 148 windows in 2013 alone, longitudes spanning 38E to 334E).
We only ever SCORE on WP+EP, so fetching the other basins would be waste; widening east to 260E
captures 97.7% of WP+EP windows and 98% of the WP+EP test set.

The dataset's longitude axis runs -180..180, so 100-260E WRAPS. Each day therefore needs two
requests -- 100..180 and -180..-100 -- stitched into one 0-360 frame.

STRATEGY. Two requests per storm-day for the box, then crop a 21x21 patch (+-2.5 deg at 0.25 deg)
centred on each storm in that day. Ocean fields are daily and slowly varying, so a same-day field
is causal enough -- unlike the atmosphere there is no 6-hourly analysis to be stale against.

~46% of the box is valid ocean; the rest is land, and those cells come back NaN. They are stored
as exact zeros behind a per-window mask, the same unavailable-means-zero convention the rest of
this project uses.
"""
import os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    import netCDF4
except ImportError:
    sys.exit("pip install netCDF4")

E = "https://upwell.pfeg.noaa.gov/erddap/griddap/noaa_aoml_6b09_4e6f_46dd"
VARS = ["Ocean_Heat_Content", "D26", "D20"]
HALF = 10                      # 21 x 21 at 0.25 deg = +-2.5 deg
OUT = "track_build/ohc"
W = int(os.environ.get("WORKERS", "6"))
TMP = os.environ.get("TMPDIR", "/tmp")

z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
N = len(bt)
DAY = int(24 * 3600 * 1e9)
day = bt // DAY
# The dataset ADVERTISES 2012-08-27, but 2012 comes back entirely NaN over our box while 2013
# onward is 70% valid. Probed 2012/2013/.../2026 directly rather than trusting the metadata --
# the listed range and the real data do not agree, which is exactly the failure the research
# flagged for a sibling CoastWatch dataset that silently died but stayed published.
T0 = np.datetime64("2013-01-01", "D").astype("datetime64[ns]").astype("int64") // DAY
T1 = np.datetime64("2026-07-19", "D").astype("datetime64[ns]").astype("int64") // DAY
bas = z["basin"].astype(str)
blo360 = blo % 360
# only the basins we actually score, and only where a full 21x21 patch fits inside the box
inbox = (bla >= 2.5) & (bla <= 57.5) & (blo360 >= 102.5) & (blo360 <= 257.5)
elig = (day >= T0) & (day <= T1) & np.isin(bas, ["WP", "EP"]) & inbox
days = np.unique(day[elig])
print(f"{N:,} windows | {elig.sum():,} WP+EP in box, 2013+ | {len(days):,} storm-days")
print(f"{len(days):,} days x 2 slabs at ~4.2 s / {W} workers -> ~{len(days)*8.4/W/60:.0f} min\n", flush=True)
os.makedirs(OUT, exist_ok=True)


def _one(ds, w0, w1, tag):
    """One lon slab for one day. Returns (array[3,nlat,nlon], lat, lon 0-360)."""
    import urllib.request
    q = f"%5B({ds}):1:({ds})%5D%5B(0):1:(60)%5D%5B({w0}):1:({w1})%5D"
    url = f"{E}.nc?" + ",".join(v + q for v in VARS)
    f = f"{TMP}/ohc_{ds}_{tag}.nc"
    urllib.request.urlretrieve(url, f)
    nc = netCDF4.Dataset(f)
    a = np.stack([np.asarray(nc.variables[v][0], "float32") for v in VARS])
    la = np.asarray(nc.variables["latitude"][:], "float64")
    lo = np.asarray(nc.variables["longitude"][:], "float64") % 360.0
    nc.close(); os.remove(f)
    return a, la, lo


def fetch(d):
    ds = str(np.datetime64(int(d) * DAY, "ns").astype("datetime64[D]"))
    for attempt in range(4):
        try:
            # the axis is -180..180, so 100-260E is two slabs stitched into one 0-360 frame
            aw, la, low = _one(ds, 100, 180, "w")
            # start at -179.75, NOT -180: -180 % 360 == 180, which duplicates the west slab's
            # last point and makes the stitched axis non-monotonic. The assertion below caught
            # this on every single day -- 152/152 failures -- rather than letting a duplicated
            # column through into the patches.
            ae, _, loe = _one(ds, -179.75, -100.25, "e")
            a = np.concatenate([aw, ae], axis=2)
            lo = np.concatenate([low, loe])
            o = np.argsort(lo)
            assert (np.diff(lo[o]) > 0).all(), "stitched longitude is not strictly increasing"
            return d, a[:, :, o], la, lo[o], None
        except Exception as ex:
            for t in ("w", "e"):
                try:
                    os.remove(f"{TMP}/ohc_{ds}_{t}.nc")
                except OSError:
                    pass
            if attempt == 3:
                return d, None, None, None, f"{ds}: {str(ex)[:80]}"
            time.sleep(3 * (attempt + 1))


by_year = {}
for i in np.where(elig)[0]:
    y = int(str(np.datetime64(int(bt[i]), "ns").astype("datetime64[Y]")))
    by_year.setdefault(y, []).append(i)

t_all = time.time()
for year in sorted(by_year):
    f = f"{OUT}/ohc_{year}.npz"
    if os.path.exists(f):
        print(f"  {year}: present, skipping", flush=True); continue
    idx = np.array(by_year[year]); t0 = time.time()
    P = np.zeros((len(idx), 3, 2 * HALF + 1, 2 * HALF + 1), "float32")
    got = np.zeros(len(idx), "float32")
    dmap = {}
    for s_, i in enumerate(idx):
        dmap.setdefault(int(day[i]), []).append(s_)
    fails = 0
    with ThreadPoolExecutor(max_workers=W) as ex:
        for d, a, la, lo, err in ex.map(fetch, sorted(dmap)):
            if err:
                fails += 1; continue
            for s_ in dmap[d]:
                i = idx[s_]
                r = int(np.abs(la - bla[i]).argmin())
                c = int(np.abs(lo - (blo[i] % 360)).argmin())
                if r - HALF < 0 or c - HALF < 0 or r + HALF + 1 > len(la) or c + HALF + 1 > len(lo):
                    continue
                p = a[:, r - HALF:r + HALF + 1, c - HALF:c + HALF + 1]
                # land/ice comes back NaN; store zeros and let the mask say so
                if np.isfinite(p).mean() < 0.10:
                    continue                      # storm essentially over land, no ocean signal
                P[s_] = np.nan_to_num(p, nan=0.0)
                got[s_] = 1.0
    sc = np.array([max(np.abs(P[:, v]).max(), 1e-6) / 127.0 for v in range(3)], "float32")
    q = np.clip(np.round(P / sc[None, :, None, None]), -127, 127).astype("int8")
    np.savez_compressed(f, q=q, scale=sc, got=got, widx=idx, vars=np.array(VARS))
    print(f"  {year}: {len(idx):6,} win  {100*got.mean():5.1f}% got  "
          f"{os.path.getsize(f)/1e6:6.1f} MB  {fails:3d} fail  {(time.time()-t0)/60:5.1f} min",
          flush=True)
    if got.mean() < 0.20:
        os.remove(f)
        raise SystemExit(f"ABORT: {year} got only {100*got.mean():.1f}% -- fix the cause first.")
print(f"\ndone in {(time.time()-t_all)/60:.1f} min -> {OUT}/")
