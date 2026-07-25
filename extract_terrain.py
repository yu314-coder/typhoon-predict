"""Regional TERRAIN + LAND-SEA MASK for the WP basin, from ERA5's time-invariant fields.

WHY THIS DATA. TrackFormer currently has NO land feature of any kind -- the 54 track columns are
kinematics/intensity, and every land reference in the repo is SST/OHC masking or coastline
plotting. Land interaction is a real track-speed modulator (a storm crossing flat Luzon coast and
one hitting Taiwan's 3.9 km Central Mountain Range are identical to a coastline polygon and
completely different in reality), and it is the leading candidate to explain the along-track speed
bias that _speedfix.py showed a GLOBAL correction cannot fix.

WHY ERA5 INVARIANTS rather than ETOPO/SRTM. These land on EXACTLY the 0.25-degree grid the steering
patches use, so terrain fuses with the flow fields without a regrid step -- and it is two single
timesteps, a few hundred KB after subsetting, from the server we are already pulling from.

    Z   surface geopotential (m^2/s^2)  -> orography in metres via / 9.80665
    LSM land-sea mask, 0..1 land fraction

REGION: the box covering Taiwan, the Philippines, coastal China, Vietnam and Japan.
OUTPUT: track_build/terrain_wp.npz on disk D (never the main disk).
"""
import os, socket, time, numpy as np, netCDF4

socket.setdefaulttimeout(60)
OUT = "track_build/terrain_wp.npz"
G0 = 9.80665

# region box: Taiwan, Philippines, China coast, Vietnam, Japan
LAT0, LAT1 = 3.0, 47.0
LON0, LON1 = 100.0, 148.0

INV = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/e5.oper.invariant/197901"
FILES = {
    "Z":   f"{INV}/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc",
    "LSM": f"{INV}/e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc",
}


def fetch(varname, url, tries=8):
    """RDA throws intermittent 503s -- the same flakiness the main pull retries through."""
    for t in range(tries):
        try:
            nc = netCDF4.Dataset(url)
            lat = nc.variables["latitude"][:]
            lon = nc.variables["longitude"][:]
            # ERA5 latitude runs 90 -> -90, longitude 0 -> 359.75
            la = np.where((lat >= LAT0) & (lat <= LAT1))[0]
            lo = np.where((lon >= LON0) & (lon <= LON1))[0]
            key = [v for v in nc.variables if v.upper() == varname][0]
            arr = nc.variables[key][0, la[0]:la[-1] + 1, lo[0]:lo[-1] + 1]
            out = (np.array(arr, dtype="float32"),
                   np.array(lat[la[0]:la[-1] + 1], dtype="float32"),
                   np.array(lon[lo[0]:lo[-1] + 1], dtype="float32"))
            nc.close()
            return out
        except Exception as e:
            if t == tries - 1:
                raise
            print(f"  {varname} attempt {t+1} failed ({str(e)[:60]}), retrying ...", flush=True)
            time.sleep(5)


print(f"region lat {LAT0}-{LAT1}N  lon {LON0}-{LON1}E  (Taiwan/Philippines/China/Vietnam/Japan)")
z, lat, lon = fetch("Z", FILES["Z"])
print(f"  Z   {z.shape} fetched")
lsm, _, _ = fetch("LSM", FILES["LSM"])
print(f"  LSM {lsm.shape} fetched")

elev = z / G0                       # geopotential -> metres
elev_land = np.where(lsm > 0.5, elev, 0.0)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
_tmp = OUT + ".tmp"
with open(_tmp, "wb") as fh:        # open handle: savez must not append .npz (see extract_era5.py)
    np.savez_compressed(fh, elev=elev.astype("float32"), lsm=lsm.astype("float32"),
                        lat=lat, lon=lon, res=np.float32(0.25))
os.replace(_tmp, OUT)

print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1e3:.0f} KB)")
print(f"  grid {elev.shape}  land cells {int((lsm>0.5).sum()):,} / {lsm.size:,}")


def peak(name, a, b, c, d):
    """max terrain height inside a lat/lon box -- validates the grid actually resolves the ridge."""
    m = (lat >= a) & (lat <= b); n = (lon >= c) & (lon <= d)
    sub = elev_land[np.ix_(m, n)]
    print(f"  {name:22s} peak {sub.max():7.0f} m   mean-land {sub[sub>0].mean() if (sub>0).any() else 0:6.0f} m")


print("\nsanity -- does 0.25 deg resolve the mountains that matter?")
peak("Taiwan (CMR)", 21.9, 25.3, 120.0, 122.0)
peak("Luzon (Cordillera)", 13.0, 18.6, 120.0, 122.5)
peak("Japan (Honshu Alps)", 34.5, 37.5, 136.0, 140.0)
peak("Vietnam (Annamite)", 11.0, 22.0, 103.0, 109.0)
peak("SE China coast", 22.0, 28.0, 112.0, 120.0)
