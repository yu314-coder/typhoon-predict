"""Per-lead steering targets for v21: the flow each storm ACTUALLY experienced, at every lead.

    !pip install -q netCDF4
    !wget -q -O v25e.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v25_extract.py
    exec(open('v25e.py').read())

v21 predicts the steering flow at each of 20 leads and DERIVES the track by integrating it, instead
of regressing displacement directly. That needs a supervision signal for the intermediate step, and
this builds it: for every window and every lead, the deep-layer-mean environmental flow at the
position the storm actually reached, at the time it reached it.

WHAT "STEERING" MEANS HERE. Not the wind at the storm centre -- at 2.5 deg the reanalysis partly
resolves the vortex, and its own circulation would swamp the reading. The environmental steering is
the deep-layer mean averaged over a 3-8 degree ANNULUS around the centre, which is the standard
operational definition and excludes the core.

VALIDATION BEFORE ANY TRAINING. The script ends by regressing observed storm motion on the
extracted flow. Theory says motion ~ 0.8 x steering + beta drift, so a correlation near 0.8-0.9 and
a slope near 0.8 means the extraction is right. A weak correlation means it is wrong, and v21 should
not be built on it. That check is the whole point of doing extraction first.

Cross-year leads are handled by grouping on the FUTURE time rather than the window's own year, so a
December storm forecasting into January is extracted from the correct file.
"""
import os, subprocess, time, urllib.request
import numpy as np
try:
    import netCDF4
except ImportError:
    subprocess.run(["pip", "install", "-q", "netCDF4"], check=True)
    import netCDF4

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
LEVELS = [850, 500, 200]
WGT = np.array([0.269, 0.500, 0.231], dtype="float32")
R_IN, R_OUT = 3.0, 8.0                 # steering annulus, degrees
MAX_DT = int(18 * 3600 * 1e9)
SIX_H = int(6 * 3600 * 1e9)
BASE = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure"

if not os.path.exists("/content/track_windows_v13.npz"):
    print("fetching v13 windows ...", flush=True)
    urllib.request.urlretrieve(f"{RAW}/track_build/track_windows_v13.npz",
                               "/content/track_windows_v13.npz")
z = np.load("/content/track_windows_v13.npz", allow_pickle=True)
bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
tgt = z["target"].astype("float32"); msk = z["target_mask"].astype("float32")
N = len(bt)

# ---- where and when the storm actually was, at each lead ----
R = 111.2
cE = np.cumsum(tgt[..., 0], 1); cN = np.cumsum(tgt[..., 1], 1)      # [N,20] km from base
FLAT = bla[:, None] + cN / R                                        # future lat
FLON = blo[:, None] + cE / (R * np.cos(np.radians((bla[:, None] + FLAT) / 2)))
FT = bt[:, None] + (np.arange(1, 21)[None, :] * SIX_H)              # future time
VALID = msk[..., 0] > 0.5                                           # lead is observed
print(f"{N:,} windows x 20 leads = {N*20:,} points, {VALID.sum():,} with observed truth")

FLOW = np.zeros((N, 20, 2), dtype="float32")
GOT = np.zeros((N, 20), dtype=bool)
fy = (FT / 1e9 / 86400 / 365.2425 + 1970).astype(int)               # approximate year, refined below
years = np.array([int(str(np.datetime64(int(t), "ns"))[:4]) for t in FT.ravel()]).reshape(N, 20)


def load_year(y):
    out = []
    for var in ("uwnd", "vwnd"):
        f = f"/content/tmp_{var}_{y}.nc"
        if not os.path.exists(f):
            for a in range(4):
                r = subprocess.run(["curl", "-sSL", "--max-time", "900", "-o", f,
                                    f"{BASE}/{var}.{y}.nc"])
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
        out.append((WGT[:, None, None, None] * arr).sum(0))         # deep-layer mean [T,lat,lon]
    return tns, lat, lon, out[0], out[1]


_ann_cache = {}
def annulus(lat_c, nlat, nlon, dlat, dlon):
    """Grid offsets forming a 3-8 deg ring at this latitude. Cached per rounded latitude."""
    key = int(round(lat_c / 5.0))
    if key in _ann_cache:
        return _ann_cache[key]
    la = key * 5.0
    rad = int(np.ceil(R_OUT / min(dlat, dlon * max(np.cos(np.radians(la)), 0.2)))) + 1
    offs = []
    for di in range(-rad, rad + 1):
        for dj in range(-rad, rad + 1):
            ddeg_lat = di * dlat
            ddeg_lon = dj * dlon * np.cos(np.radians(la))
            dist = np.hypot(ddeg_lat, ddeg_lon)
            if R_IN <= dist <= R_OUT:
                offs.append((di, dj))
    a = (np.array([o[0] for o in offs]), np.array([o[1] for o in offs]))
    _ann_cache[key] = a
    return a


t0 = time.time()
uy = sorted(set(years[VALID].ravel().tolist()))
for y in uy:
    sel = np.argwhere(VALID & (years == y))
    if len(sel) == 0:
        continue
    tns, lat, lon, U, V = load_year(y)
    dlat = abs(lat[1] - lat[0]); dlon = abs(lon[1] - lon[0])
    n_ok = 0
    for w, L in sel:
        ft = int(FT[w, L])
        ti = int(np.abs(tns - ft).argmin())
        if abs(int(tns[ti]) - ft) > MAX_DT:
            continue
        la_c = float(FLAT[w, L]); lo_c = float(FLON[w, L]) % 360.0
        li = int(np.abs(lat - la_c).argmin()); lj = int(np.abs(lon - lo_c).argmin())
        di, dj = annulus(la_c, len(lat), len(lon), dlat, dlon)
        rows = np.clip(li + di, 0, len(lat) - 1); cols = np.mod(lj + dj, len(lon))
        FLOW[w, L, 0] = U[ti][rows, cols].mean()
        FLOW[w, L, 1] = V[ti][rows, cols].mean()
        GOT[w, L] = True; n_ok += 1
    print(f"{y}: {n_ok:,}/{len(sel):,} lead-points | "
          f"u {FLOW[GOT][:,0].mean():+.1f} v {FLOW[GOT][:,1].mean():+.1f} m/s | "
          f"{time.time()-t0:.0f}s", flush=True)

np.savez_compressed("/content/lead_flow.npz", flow=FLOW.astype("float16"), got=GOT)
print(f"\nwrote /content/lead_flow.npz  {FLOW.shape}  ({GOT.sum():,} points, "
      f"{os.path.getsize('/content/lead_flow.npz')/1e6:.0f} MB)")

# ---- THE VALIDATION: does the extracted flow explain the motion the storm actually made? ----
mo_e = tgt[..., 0] * 1000.0 / (6 * 3600)      # observed motion, km/6h -> m/s
mo_n = tgt[..., 1] * 1000.0 / (6 * 3600)
g = GOT & VALID
print(f"\n--- validation on {g.sum():,} lead-points ---")
for nm, mo, fl in (("east", mo_e[g], FLOW[..., 0][g]), ("north", mo_n[g], FLOW[..., 1][g])):
    r = float(np.corrcoef(fl, mo)[0, 1])
    slope, icpt = np.polyfit(fl, mo, 1)
    print(f"  {nm:5s}: corr {r:+.3f}   motion = {slope:.2f} x flow {icpt:+.2f} m/s")
print("\n  theory: motion ~= 0.8 x deep-layer steering, plus a beta drift of about")
print("  1-2 m/s toward west-northwest (so a NEGATIVE east intercept, POSITIVE north).")
print("  corr >= 0.7 and slope near 0.8 means the extraction is sound and v21 can be built on it.")
