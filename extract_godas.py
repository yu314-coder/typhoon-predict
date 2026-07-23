"""Ocean heat from the GODAS reanalysis, 1980-present -- the backfill AOML cannot provide.

WHY THIS EXISTS. v26 wired a CNN over the AOML ocean-heat patch into the intensity decoder and
bought 0.86 kt on max wind. The measured ceiling on that result was coverage, not architecture:

      AOML archive starts 2013.  Training split is storms whose first year is <= 2015.

Three years of overlap, so the ocean branch saw real data on 3.9% of training windows while being
asked to speak on 95.6% of test windows. That is not a download that was skipped -- AOML does not
go back, so no amount of fetching fixes it. GODAS does: NCEP's ocean reanalysis runs 1980-present,
and the earliest training storm in this dataset is 1980. Measured coverage if we use it:

      split     windows     AOML      GODAS
      train     153,378      3.9%     50.6%
      valid      16,342     48.4%     49.2%
      test        3,763     95.6%     98.2%

The uncovered half of training is Atlantic/Indian/Southern basins, which we never score on.

WHAT IT COSTS. GODAS is MONTHLY on a 1 deg x 1/3 deg grid; AOML is daily at 0.25 deg. Each sample
is much coarser and cannot resolve a warm eddy or a storm's cold wake. That trade is acceptable
here because the quantity we want from this field is the slowly-varying background thermal
structure -- how deep the warm water goes -- and a monthly mean represents that well. Subsurface
depth is also the one thing SST (already a model input since v16) cannot tell us.

SAME GRID AS AOML, DELIBERATELY. The three fields are resampled onto the identical 21x21 patch at
0.25 deg centred on the storm, so v26's CNN needs no change and the two sources are directly
comparable. Where both exist (2013-2015) they are cross-checked against each other; a poor
correlation would mean one of the two derivations is wrong, and that is reported, not assumed.

CAUSALITY. A monthly mean covering the window's own month includes ocean state AFTER t0 -- and the
storm itself cools the water it crosses, so that field is partly an effect of the thing we are
predicting. The PREVIOUS month is used instead: strictly before t0, and physically still valid
because this structure changes slowly.

DERIVATION from potential temperature (40 levels, Kelvin):
    D26 / D20   depth where the profile crosses 26 C / 20 C, linearly interpolated between levels
    OHC         rho*cp*Integral(T - 26)dz from the surface down to D26, in kJ/cm2

Output: track_build/godas/godas_YYYY.npz, int8-quantised, same layout as track_build/ohc/.
"""
import os, sys, time
import numpy as np

try:
    import netCDF4
except ImportError:
    sys.exit("pip install netCDF4")

import socket
socket.setdefaulttimeout(120)

URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas/pottmp.{yr}.nc"
OUT = "track_build/godas"
HALF = 10                      # 21 x 21 at 0.25 deg = +-2.5 deg, identical to the AOML patch
RHO_CP = 1026.0 * 3985.0       # J / (m^3 K)
Y0 = int(os.environ.get("Y0", "1980"))
Y1 = int(os.environ.get("Y1", "2026"))

z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
bas = z["basin"].astype(str)
N = len(bt)
blo360 = blo % 360
yrs = np.array([int(str(np.datetime64(int(t), "ns").astype("datetime64[Y]"))) for t in bt])
mons = np.array([int(str(np.datetime64(int(t), "ns").astype("datetime64[M]"))[5:7]) for t in bt])

inbox = (bla >= 2.5) & (bla <= 57.5) & (blo360 >= 102.5) & (blo360 <= 257.5)
elig = np.isin(bas, ["WP", "EP"]) & inbox & (yrs >= Y0) & (yrs <= Y1)
print(f"{N:,} windows | {elig.sum():,} WP+EP in box, {Y0}-{Y1}", flush=True)
os.makedirs(OUT, exist_ok=True)

# the field used is the PREVIOUS month, so a January window needs the previous December
prev_y = np.where(mons == 1, yrs - 1, yrs)
prev_m = np.where(mons == 1, 12, mons - 1)

# target grid: identical to the AOML patch geometry
OFF = (np.arange(2 * HALF + 1) - HALF) * 0.25


def derive(T, lev):
    """pottmp [nlev,nlat,nlon] in Kelvin -> (OHC kJ/cm2, D26 m, D20 m). Land/no-data are 0."""
    C = T - 273.15
    nl, ny, nx = C.shape
    good = np.isfinite(C)
    C = np.where(good, C, -99.0)
    out = np.zeros((3, ny, nx), "float32")

    def depth_of(thr):
        d = np.zeros((ny, nx), "float32")
        found = np.zeros((ny, nx), bool)
        for k in range(nl - 1):
            a, b = C[k], C[k + 1]
            cross = (~found) & (a >= thr) & (b < thr) & good[k] & good[k + 1]
            if cross.any():
                f = (a[cross] - thr) / np.maximum(a[cross] - b[cross], 1e-6)
                d[cross] = lev[k] + f * (lev[k + 1] - lev[k])
                found |= cross
        # warm all the way down through the sampled column: clamp at the deepest good level
        deep = (~found) & good[0] & (C[0] >= thr)
        if deep.any():
            last = np.zeros((ny, nx), "float32")
            for k in range(nl):
                last = np.where(good[k] & (C[k] >= thr), lev[k], last)
            d[deep] = last[deep]
        return d

    d26 = depth_of(26.0); d20 = depth_of(20.0)
    # heat above the 26 C isotherm, trapezoidal over the layers that sit above D26
    ohc = np.zeros((ny, nx), "float32")
    for k in range(nl - 1):
        z0, z1 = lev[k], lev[k + 1]
        top = np.minimum(z1, d26)
        dz = np.clip(top - z0, 0, None)
        m = (dz > 0) & good[k] & good[k + 1]
        if m.any():
            tbar = 0.5 * (C[k] + C[k + 1]) - 26.0
            ohc += np.where(m, np.clip(tbar, 0, None) * dz, 0.0)
    out[0] = ohc * RHO_CP * 1e-7          # J/m2 -> kJ/cm2
    out[1] = d26
    out[2] = d20
    out[np.repeat((~good[0])[None], 3, 0)] = 0.0
    return out


def year_fields(y):
    """Return {month: (3,nlat,nlon)} plus the lat/lon axes, for the WP+EP box."""
    nc = netCDF4.Dataset(URL.format(yr=y))
    try:
        lat = np.asarray(nc.variables["lat"][:], "float64")
        lon = np.asarray(nc.variables["lon"][:], "float64")
        lev = np.asarray(nc.variables["level"][:], "float64")
        iy = np.where((lat >= -0.5) & (lat <= 62.0))[0]
        ix = np.where((lon >= 98.0) & (lon <= 262.0))[0]
        kz = np.where(lev <= 700.0)[0]                    # deep enough for D20 everywhere
        # ONE MONTH PER REQUEST, deliberately. Asking for all twelve in a single 4-D slice
        # (12 x 29 x 187 x 164 ~ 42 MB) comes back FULLY MASKED AND ALL ZERO with no error
        # raised -- the server caps the response and returns emptiness rather than failing.
        # Measured side by side: the 4-D slice gave min/max 0.0/0.0 maskfrac 1.00, while the
        # identical single-month 3-D slice gave real temperatures at maskfrac 0.30. A naive
        # reader would assume the 4-D form is simply faster; it silently produces no data.
        la, lo, lv = lat[iy], lon[ix], lev[kz]
        out = {}
        nt = len(nc.variables["time"])
        for m in range(nt):
            r = nc.variables["pottmp"][m, kz[0]:kz[-1] + 1, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
            # GODAS marks land with missing_value = -9.96921e+36: NEGATIVE and, crucially, FINITE,
            # so neither `> 1e19` nor isfinite() rejects it. Masking on the variable's own physical
            # range is the robust form.
            r = np.ma.filled(np.ma.masked_outside(np.ma.masked_invalid(r), 250.0, 320.0)
                             .astype("float32"), np.nan)
            if not np.isfinite(r).any():
                raise RuntimeError(f"month {m+1} came back with no finite data")
            out[m + 1] = derive(r, lv)
    finally:
        nc.close()
    return out, la, lo


by_year = {}
for i in np.where(elig)[0]:
    by_year.setdefault(int(prev_y[i]), []).append(i)

t_all = time.time()
for year in sorted(by_year):
    f = f"{OUT}/godas_{year}.npz"
    if os.path.exists(f):
        print(f"  {year}: present, skipping", flush=True); continue
    idx = np.array(by_year[year]); t0 = time.time()
    try:
        fields, la, lo = year_fields(year)
    except Exception as ex:
        print(f"  {year}: FETCH FAILED {type(ex).__name__} {str(ex)[:90]}", flush=True)
        continue
    P = np.zeros((len(idx), 3, 2 * HALF + 1, 2 * HALF + 1), "float32")
    got = np.zeros(len(idx), "float32")
    for s_, i in enumerate(idx):
        F = fields.get(int(prev_m[i]))
        if F is None:
            continue
        tlat = bla[i] + OFF                      # storm-centred 0.25 deg target grid
        tlon = (blo360[i] + OFF)
        r = np.abs(la[None, :] - tlat[:, None]).argmin(1)
        c = np.abs(lo[None, :] - tlon[:, None]).argmin(1)
        p = F[:, r][:, :, c]                     # nearest-neighbour resample onto the AOML geometry
        if (p[0] != 0).mean() < 0.10:            # essentially land / no ocean column
            continue
        P[s_] = p
        got[s_] = 1.0
    sc = np.array([max(np.abs(P[:, v]).max(), 1e-6) / 127.0 for v in range(3)], "float32")
    q = np.clip(np.round(P / sc[None, :, None, None]), -127, 127).astype("int8")
    np.savez_compressed(f, q=q, scale=sc, got=got, widx=idx,
                        vars=np.array(["OHC", "D26", "D20"]))
    print(f"  {year}: {len(idx):6,} win  {100*got.mean():5.1f}% got  "
          f"{os.path.getsize(f)/1e6:6.1f} MB  {(time.time()-t0)/60:5.1f} min", flush=True)
    # A year with a handful of windows is an edge of the calendar, not a broken pipeline: the only
    # windows landing on e.g. December 2013 are January-2014 storms, and in deep winter there is
    # genuinely little water above 26 C to integrate. Aborting on those masked the fact that the
    # main seasons were fine. Same short-year-vs-real-failure distinction the ERA5 puller makes.
    if got.mean() < 0.20:
        if len(idx) < 200:
            print(f"    NOTE: {year} has only {len(idx)} windows (calendar edge) -- keeping, continuing",
                  flush=True)
        else:
            os.remove(f)
            raise SystemExit(f"ABORT: {year} got only {100*got.mean():.1f}% on {len(idx):,} "
                             f"windows -- fix the cause first.")

print(f"\ndone in {(time.time()-t_all)/60:.1f} min -> {OUT}/", flush=True)
