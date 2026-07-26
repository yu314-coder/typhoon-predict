"""Re-pull ONLY the z (geopotential) variable for years already extracted by extract_era5.py, to
fix a quantization bug found in merge_era5_steer.py's verification: the per-year z scale was sized
by the RAW absolute value (~122,300 m^2/s^2 at 200 hPa), giving an int8 step (~974) LARGER than the
actual within-patch gradient signal (std ~340-440) -- for most windows the whole 65x65 patch
collapsed to <=3 distinct int8 values. u and v are unaffected (verified 0% collapsed) and are left
untouched here -- z is on its OWN separate DAP request per day, so this refetches 1/3 the data of
the original pull.

THE FIX. Subtract each window's own per-level SPATIAL MEAN from z before quantizing (an anomaly,
exactly like merge_era5_steer.py already does for the OUTPUT tensor -- but doing it here, before
quantization, means the fix operates on full float32 precision instead of already-destroyed int8).

RESUMABLE. Same atomic-write + per-year skip pattern as extract_era5.py, but the skip marker is a
z_fixed flag INSIDE each file (the file already exists from the original pull), not file presence.
A window whose z re-fetch fails (server hiccup) is dropped from `got` entirely -- never a mix of
valid u/v with stale, still-broken z in the same row.
"""
import os, sys, time, socket
import numpy as np

socket.setdefaulttimeout(90)
import resource
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, _hard), _hard))

try:
    import netCDF4
except ImportError:
    sys.exit("pip install netCDF4")

RDA = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/e5.oper.an.pl"
Z_VAR = ("128_129_z", "Z", "ll025sc")
LAT0, LAT1 = 120, 361
LON0, LON1 = 400, 721
LEV_HPA = [200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850]
HALF = 32
OUT = "track_build/era5"
HOUR = int(3600 * 1e9)

zt = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
bt = zt["base_time"].astype("int64")
bla = zt["base_lat"].astype("float64"); blo = zt["base_lon"].astype("float64")
snap = (bt // HOUR) * HOUR
lat_g = 90.0 - 0.25 * np.arange(721)
lon_g = 0.25 * np.arange(1440)

W = int(os.environ.get("WORKERS", "4"))
Y0 = int(os.environ.get("Y0", "2001"))
YMAX = int(os.environ.get("YMAX", "9999"))
REVERSE = int(os.environ.get("REVERSE", "1"))
STRIDE = int(os.environ.get("STRIDE", "8"))
TSTRIDE = int(os.environ.get("TSTRIDE", "3"))
LEV0, LEV1 = 14, 31
LEV_SL = slice(LEV0, LEV1, STRIDE)
print(f"TMPDIR={os.environ.get('TMPDIR','(default)')}", flush=True)


def fetch_day_z(day_idx):
    """One day, z ONLY, strided levels. Returns (day_idx, array, err)."""
    d0 = int(day_idx) * 24 * HOUR
    ds = np.datetime64(d0, "ns").astype("datetime64[D]")
    y, m, dd = str(ds)[:4], str(ds)[5:7], str(ds)[8:10]
    code, dapname, grid = Z_VAR
    url = f"{RDA}/{y}{m}/e5.oper.an.pl.{code}.{grid}.{y}{m}{dd}00_{y}{m}{dd}23.nc"
    last = None
    for attempt in range(8):
        nc = None
        try:
            nc = netCDF4.Dataset(url)
            a = np.asarray(nc.variables[dapname][::TSTRIDE, LEV_SL, LAT0:LAT1, LON0:LON1], "float32")
            return day_idx, a, None
        except Exception as ex:
            time.sleep(min(4 * (attempt + 1), 20))
            last = str(ex)[:90]
        finally:
            if nc is not None:
                try:
                    nc.close()
                except Exception:
                    pass
    return day_idx, None, f"{ds} {dapname}: {last}"


from concurrent.futures import ThreadPoolExecutor

years = sorted(int(os.path.basename(f)[5:9]) for f in
                __import__("glob").glob(f"{OUT}/era5_*.npz"))
years = [y for y in years if Y0 <= y <= YMAX]
print(f"{len(years)} year files in range [{Y0},{YMAX}]: {years}", flush=True)

t_all = time.time()
for n_done, year in enumerate(sorted(years, reverse=bool(REVERSE))):
    f = f"{OUT}/era5_{year}.npz"
    with np.load(f) as d0:
        if bool(d0.get("z_fixed", np.array(False))):
            print(f"  {year}: z already fixed, skipping", flush=True); continue
        q_old, sc_old, got_old, widx = d0["q"].copy(), d0["scale"].copy(), d0["got"].copy(), d0["widx"].copy()
        levels, varsarr = d0["levels"], d0["vars"]

    live = np.where(got_old > 0.5)[0]           # only windows whose u/v are already valid
    if len(live) == 0:
        with open(f + ".tmp", "wb") as fh:
            np.savez_compressed(fh, q=q_old, scale=sc_old, got=got_old, widx=widx,
                                levels=levels, vars=varsarr, z_fixed=np.array(True))
        os.replace(f + ".tmp", f)
        print(f"  {year}: no live windows, marked fixed", flush=True); continue

    idx = widx[live]; t0 = time.time()
    nlev = q_old.shape[2]
    Pz = np.zeros((len(live), nlev, 2 * HALF + 1, 2 * HALF + 1), "float32")
    got_z = np.zeros(len(live), "float32")
    dmap = {}
    for s_, i in zip(range(len(live)), idx):
        dmap.setdefault(int(snap[i] // (24 * HOUR)), []).append(s_)
    fails = 0
    with ThreadPoolExecutor(max_workers=W) as ex:
        for di, a, err in ex.map(fetch_day_z, sorted(dmap)):
            if err:
                fails += 1; continue
            for s_ in dmap[di]:
                i = idx[s_]
                ti = int((snap[i] - int(di) * 24 * HOUR) // HOUR) // TSTRIDE
                if ti >= a.shape[0]:
                    continue
                r = int(np.abs(lat_g[LAT0:LAT1] - bla[i]).argmin())
                c = int(np.abs(lon_g[LON0:LON1] - (blo[i] % 360)).argmin())
                r0, r1, c0, c1 = r - HALF, r + HALF + 1, c - HALF, c + HALF + 1
                if r0 < 0 or c0 < 0 or r1 > (LAT1 - LAT0) or c1 > (LON1 - LON0):
                    continue
                Pz[s_] = a[ti][:, r0:r1, c0:c1]
                got_z[s_] = 1.0

    # THE FIX: per-window, per-level spatial-mean removal BEFORE quantizing -- an anomaly, at
    # full float32 precision, instead of destroying the gradient with a raw-magnitude scale.
    Pz -= Pz.mean(axis=(2, 3), keepdims=True)
    sc_z = max(np.abs(Pz[got_z > 0.5]).max(), 1e-6) / 127.0 if (got_z > 0.5).any() else 1e-6
    qz = np.clip(np.round(Pz / sc_z), -127, 127).astype("int8")

    q_new = q_old.copy(); sc_new = sc_old.copy(); got_new = got_old.copy()
    q_new[live, 0] = qz
    q_new[live[got_z < 0.5], 0] = 0              # a window whose z refetch failed: zero, not stale
    got_new[live[got_z < 0.5]] = 0.0             # ...and dropped from the usable set entirely
    sc_new[0] = sc_z

    lost = int((got_z < 0.5).sum())
    _tmp = f + ".tmp"
    with open(_tmp, "wb") as fh:
        np.savez_compressed(fh, q=q_new, scale=sc_new, got=got_new, widx=widx,
                            levels=levels, vars=varsarr, z_fixed=np.array(True))
    os.replace(_tmp, f)
    el = (time.time() - t0) / 60
    n_done += 1
    eta = (time.time() - t_all) / n_done * (len(years) - n_done) / 3600
    print(f"  {year}: {len(live):6,} win  {len(dmap):4d} days  {fails:3d} fail  "
          f"{lost:4d} lost (z-only refetch miss)  scale_z {sc_z:8.2f}  {el:5.1f} min   "
          f"ETA {eta:4.1f} h", flush=True)

print(f"\ndone in {(time.time()-t_all)/3600:.1f} h", flush=True)
