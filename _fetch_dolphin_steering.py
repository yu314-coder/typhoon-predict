"""Fetch REAL deep-layer-mean steering (850/500/200 hPa u/v) for Dolphin's 10 real issue times,
from NOAA's GFS analysis via NOMADS' GRIB-filter service -- same recipe _fetch_noul_steering.py
used (OpenDAP was retired, this replaced it; NOMADS keeps a rolling ~10-day window, and Dolphin is
happening right now, so its dates are well within range).

Runs isolated with eccodes on disk D (.pylibs) since it needs a numpy version that would conflict
with the main session's torch/numpy stack.
"""
import sys, os, subprocess
sys.path.insert(0, "/Volumes/D/typhoon_predict/.pylibs")
import eccodes as ec
import numpy as np

W = np.array([0.269, 0.500, 0.231], dtype="float32")
LEVELS = [850, 500, 200]
CACHE = "track_build/dolphin_gfs"
os.makedirs(CACHE, exist_ok=True)

ISSUES = [
    ("20260727", "00"), ("20260727", "06"), ("20260727", "12"), ("20260727", "18"),
    ("20260728", "00"), ("20260728", "06"), ("20260728", "12"), ("20260728", "18"),
    ("20260729", "00"), ("20260729", "06"),
]
LAT_A = [12.8, 13.4, 13.6, 13.2, 13.0, 13.3, 13.4, 13.7, 14.1, 14.5]
LON_A = [178.3, 176.7, 175.2, 173.7, 172.8, 171.7, 170.7, 169.9, 169.1, 168.4]


def fetch(date, hour, box):
    f = f"{CACHE}/gfs_{date}_{hour}z.grb2"
    if not os.path.exists(f) or os.path.getsize(f) < 10_000:
        toplat, leftlon, rightlon, bottomlat = box
        r = subprocess.run(["curl", "-sG", "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
                            "--data-urlencode", f"file=gfs.t{hour}z.pgrb2.0p25.f000",
                            "--data-urlencode", f"dir=/gfs.{date}/{hour}/atmos",
                            "--data-urlencode", "var_UGRD=on", "--data-urlencode", "var_VGRD=on",
                            "--data-urlencode", "lev_850_mb=on", "--data-urlencode", "lev_500_mb=on",
                            "--data-urlencode", "lev_200_mb=on",
                            "--data-urlencode", "subregion=",
                            "--data-urlencode", f"toplat={toplat}", "--data-urlencode", f"leftlon={leftlon}",
                            "--data-urlencode", f"rightlon={rightlon}", "--data-urlencode", f"bottomlat={bottomlat}",
                            "-o", f], capture_output=True)
        if os.path.getsize(f) < 10_000:
            print(f"  WARN: fetch failed for {date} {hour}z (likely aged out of NOMADS retention): "
                  f"{r.stderr[:200]}")
            return None
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
                lat = la0 - dla * np.arange(Nj)
                lon = lo0 + dlo * np.arange(Ni)
            (U if short == "u" else V)[lev] = vals
        ec.codes_release(gid)
    fh.close()
    if any(l not in U for l in LEVELS) or any(l not in V for l in LEVELS):
        return None
    Ud = sum(W[i] * U[LEVELS[i]] for i in range(3))
    Vd = sum(W[i] * V[LEVELS[i]] for i in range(3))
    return lat, lon, Ud, Vd


HALF = 8; RES = 2.5


def patch(lat, lon, Ud, Vd, blat, blon):
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
    box = (min(35, blat + 22), max(100, blon - 22), min(160, blon + 22) % 360 or 360, max(0, blat - 22))
    r = fetch(date, hour, box)
    if r is None:
        print(f"  {date} {hour}z  MISSING (not fabricated, will use have=0 fallback)")
        continue
    lat, lon, Ud, Vd = r
    pu, pv = patch(lat, lon, Ud, Vd, blat, blon)
    results[f"{date}_{hour}"] = np.stack([pu, pv])
    print(f"  {date} {hour}z  DLM at storm center: u {pu[HALF,HALF]:+.1f}  v {pv[HALF,HALF]:+.1f} m/s", flush=True)

np.savez_compressed("track_build/dolphin_dlm4_real.npz",
                    **{k: v for k, v in results.items()},
                    issues=np.array(list(results.keys())))
print(f"\nwrote track_build/dolphin_dlm4_real.npz ({len(results)}/{len(ISSUES)} issues fetched)")
