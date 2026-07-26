"""Fetch REAL deep-layer-mean steering (850/500/200 hPa u/v, weighted 0.269/0.500/0.231 -- the
exact recipe extract_tip_dlm.py uses for Typhoon Tip) for Noul's 12 issue times, from NOAA's GFS
analysis via NOMADS' GRIB-filter service (OpenDAP was retired; this replaced it). Runs isolated
with eccodes on disk D (.pylibs) since it needs a numpy version that would conflict with the main
session's torch/numpy stack -- saves plain float32 arrays so the main pipeline reads them with its
own numpy, no cross-contamination.
"""
import sys, os, subprocess
sys.path.insert(0, "/Volumes/D/typhoon_predict/.pylibs")
import eccodes as ec
import numpy as np

W = np.array([0.269, 0.500, 0.231], dtype="float32")   # 850, 500, 200 hPa thickness weights
LEVELS = [850, 500, 200]
CACHE = "track_build/noul_gfs"
os.makedirs(CACHE, exist_ok=True)

ISSUES = [
    ("20260723", "06"), ("20260723", "12"), ("20260723", "18"),
    ("20260724", "00"), ("20260724", "06"), ("20260724", "12"), ("20260724", "18"),
    ("20260725", "00"), ("20260725", "06"), ("20260725", "12"), ("20260725", "18"),
    ("20260726", "00"),
]
LAT_A = [17.4, 18.0, 18.3, 18.7, 18.8, 19.7, 20.0, 20.8, 21.3, 21.8, 22.4, 22.9]
LON_A = [128.4, 126.9, 125.1, 123.5, 121.8, 120.4, 119.3, 118.3, 116.6, 115.9, 115.1, 114.5]


def fetch(date, hour, box):
    f = f"{CACHE}/gfs_{date}_{hour}z.grb2"
    if not os.path.exists(f) or os.path.getsize(f) < 10_000:
        toplat, leftlon, rightlon, bottomlat = box
        r = subprocess.run(["curl", "-sG", f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
                            "--data-urlencode", f"file=gfs.t{hour}z.pgrb2.0p25.f000",
                            "--data-urlencode", f"dir=/gfs.{date}/{hour}/atmos",
                            "--data-urlencode", "var_UGRD=on", "--data-urlencode", "var_VGRD=on",
                            "--data-urlencode", "lev_850_mb=on", "--data-urlencode", "lev_500_mb=on",
                            "--data-urlencode", "lev_200_mb=on",
                            "--data-urlencode", "subregion=",
                            "--data-urlencode", f"toplat={toplat}", "--data-urlencode", f"leftlon={leftlon}",
                            "--data-urlencode", f"rightlon={rightlon}", "--data-urlencode", f"bottomlat={bottomlat}",
                            "-o", f], capture_output=True)
        assert os.path.getsize(f) > 10_000, f"fetch failed for {date} {hour}z: {r.stderr[:200]}"
    U = {}; V = {}; lat = lon = None
    fh = open(f, "rb")
    while True:
        gid = ec.codes_grib_new_from_file(fh)
        if gid is None:
            break
        short = ec.codes_get(gid, "shortName"); lev = ec.codes_get(gid, "level")
        if short in ("u", "v"):
            Ni = ec.codes_get(gid, "Ni"); Nj = ec.codes_get(gid, "Nj")
            vals = np.array(ec.codes_get_values(gid), dtype="float32").reshape(Nj, Ni)
            la0 = ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
            lo0 = ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
            dla = ec.codes_get(gid, "jDirectionIncrementInDegrees")
            dlo = ec.codes_get(gid, "iDirectionIncrementInDegrees")
            if lat is None:
                lat = la0 - dla * np.arange(Nj)   # GFS grids north -> south
                lon = lo0 + dlo * np.arange(Ni)
            (U if short == "u" else V)[lev] = vals
        ec.codes_release(gid)
    fh.close()
    Ud = sum(W[i] * U[LEVELS[i]] for i in range(3))
    Vd = sum(W[i] * V[LEVELS[i]] for i in range(3))
    return lat, lon, Ud, Vd


HALF = 8; RES = 2.5   # match dlm4_int8.npz's 2.5-deg, 17x17 storm-centered patch exactly


def patch(lat, lon, Ud, Vd, blat, blon):
    """2.5-deg spaced 17x17 patch centered on (blat, blon), matching the training grid."""
    out_u = np.zeros((2 * HALF + 1, 2 * HALF + 1), "float32")
    out_v = np.zeros_like(out_u)
    for ri, dlat in enumerate(range(-HALF, HALF + 1)):
        for ci, dlon in enumerate(range(-HALF, HALF + 1)):
            qlat = blat + dlat * RES; qlon = (blon + dlon * RES) % 360
            li = int(np.abs(lat - qlat).argmin()); lj = int(np.abs(lon - qlon).argmin())
            out_u[ri, ci] = Ud[li, lj]; out_v[ri, ci] = Vd[li, lj]
    return out_u, out_v


results = {}
for i, ((date, hour), blat, blon) in enumerate(zip(ISSUES, LAT_A, LON_A)):
    box = (min(35, blat + 22), max(100, blon - 22), min(160, blon + 22), max(0, blat - 22))
    lat, lon, Ud, Vd = fetch(date, hour, box)
    pu, pv = patch(lat, lon, Ud, Vd, blat, blon)
    results[f"{date}_{hour}"] = np.stack([pu, pv])
    print(f"  {date} {hour}z  DLM at storm center: u {pu[HALF,HALF]:+.1f}  v {pv[HALF,HALF]:+.1f} m/s", flush=True)

np.savez_compressed("track_build/noul_dlm4_real.npz",
                    **{k: v for k, v in results.items()},
                    issues=np.array(list(results.keys())))
print("\nwrote track_build/noul_dlm4_real.npz")
