"""Parse SHIPS-WP into per-window environmental features. CPU only, disk D.

WHAT THIS IS. lsdiagw_1990_2021_5day.txt is the SHIPS developmental dataset for the western
Pacific: ~30 environmental predictors per storm per 6 h, already computed from reanalysis. A
verified ablation in the literature (Yuan et al. 2023, held-out 2018-2020) put SHIPS-style
environmental predictors at -55.7% intensity error against -3.0% for physics-as-inductive-bias.
TrackFormer currently has NONE of them -- no ocean, no shear, no humidity.

*** THE LEAKAGE TRAP ***
Every block is a table whose columns run -12, -6, 0, +6 ... +120 hours. The FORWARD columns are
not forecasts: they are best-track truth combined with a reanalysis of what actually happened.
DELV is literally the intensity change we are trying to predict. Using any column past 0 would
leak the answer and produce a spectacular, meaningless result.

    ONLY columns -12, -6 and 0 are read here, and it is asserted below.

PROVENANCE BREAK. WP predictors come from CFSR reanalysis 1990-2004 and operational GFS analyses
from 2005 on. That is a distribution shift inside the training data, so an era flag is emitted
alongside; a model can use it or ignore it, but it should not be invisible.

MATCHING. SHIPS keys on ATCF id (WP012020); our windows key on IBTrACS storm ids. Rather than
maintain a crosswalk, match on TIME and POSITION -- same synoptic hour, storm centre within 1.5 deg.
That is unambiguous in practice: two WP storms are essentially never that close at the same instant,
and the check is asserted by reporting the fraction matched.

Output: track_build/ships_features.npz aligned 1:1 with track_windows_v13.npz.
"""
import os, sys, time
import numpy as np

SRC = "track_build/ships/lsdiagw_1990_2021_5day.txt"
OUT = "track_build/ships_features.npz"
MISS = 9999.0

# Environmental predictors worth keeping. Deliberately EXCLUDES VMAX/MSLP/DELV/INCV/TYPE/HIST,
# which are storm state or the target, and LAT/LON/TLAT/TLON which are position.
KEEP = ["CSST", "CD20", "CD26", "COHC", "RSST", "PHCN",     # ocean
        "U200", "U20C", "V20C", "SHDC", "SHRD", "SHRS", "SHTS",  # shear / upper flow
        "RHLO", "RHMD", "RHHI",                             # humidity
        "T000", "R000", "Z000", "Z850", "D200", "REFC", "PEFC",  # thermo / divergence
        "EPOS", "ENEG", "EPSS", "ENSS", "E000",             # theta-e excess
        "G150", "G200", "G250", "TWAC", "TWXC", "PSLV",     # tangential wind, GOES, steering layer
        "DTL", "OAGE", "NAGE"]                              # distance to land, storm age

print(f"reading {SRC} ...", flush=True)
t0 = time.time()
recs = []          # (yymmddhh, lat, lon, atcf, {label: (v-12, v-6, v0)})
cur = None
nlab = 0
with open(SRC, "r", errors="replace") as fh:
    for line in fh:
        lab = line[115:127].strip()
        if not lab:
            continue
        if lab == "HEAD":
            f = line.split()
            # WP01 900112 00   25    7.4  152.8 9999 WP011990
            try:
                yymmdd, hh = f[1], f[2]
                lat, lon = float(f[4]), float(f[5])
                atcf = f[7] if len(f) > 7 else ""
            except (IndexError, ValueError):
                cur = None; continue
            yy = int(yymmdd[:2])
            year = 1900 + yy if yy >= 50 else 2000 + yy
            cur = {"t": f"{year:04d}-{yymmdd[2:4]}-{yymmdd[4:6]}T{hh}",
                   "lat": lat, "lon": lon, "atcf": atcf, "v": {}}
            recs.append(cur)
        elif cur is not None and lab in KEEP:
            # 23 columns of width 5 starting at char 1. Take ONLY the first three: -12, -6, 0.
            # Columns may be BLANK rather than 9999 -- U200/RHLO/SHDC all leave -12 and -6 empty
            # while carrying a valid value at 0. Parsing the three together and discarding the row
            # on any failure silently threw away ~25 of the 37 predictors, including every shear
            # and humidity term. Parse each column independently.
            def _num(k):
                t = line[1 + 5 * k: 1 + 5 * (k + 1)].strip()
                if not t:
                    return np.nan
                try:
                    v = float(t)
                except ValueError:
                    return np.nan
                return np.nan if v == MISS else v
            cur["v"][lab] = (_num(0), _num(1), _num(2))
            nlab += 1

print(f"  {len(recs):,} storm-times, {nlab:,} predictor rows, {time.time()-t0:.0f}s", flush=True)

# ---- the leakage guard -------------------------------------------------------------------
# Re-read one block and prove the columns we took are the ones we think they are.
with open(SRC) as fh:
    head = [next(fh) for _ in range(3)]
tline = [l for l in head if l[115:127].strip() == "TIME"][0]
tcols = [float(tline[1 + 5 * k: 1 + 5 * (k + 1)]) for k in range(23)]
assert tcols[:3] == [-12.0, -6.0, 0.0], f"column layout is not what was assumed: {tcols[:5]}"
assert max(tcols[:3]) <= 0.0, "a column at or past the forecast time was read -- LEAK"
print(f"  leakage guard: columns read are {tcols[:3]} h, all <= 0  OK", flush=True)

# ---- join to the windows -----------------------------------------------------------------
z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
N = len(bt)

st = np.array([np.datetime64(r["t"], "h").astype("datetime64[ns]").astype("int64") for r in recs])
sla = np.array([r["lat"] for r in recs])
slo = np.array([r["lon"] for r in recs])
o = np.argsort(st); st, sla, slo = st[o], sla[o], slo[o]
recs = [recs[i] for i in o]

F = np.full((N, len(KEEP)), np.nan, "float32")
AGE = np.full(N, np.nan, "float32")
ERA = np.zeros(N, "float32")
HOUR = int(3600 * 1e9)
matched = 0
for i in range(N):
    lo_ = np.searchsorted(st, bt[i] - 3 * HOUR)
    hi_ = np.searchsorted(st, bt[i] + 3 * HOUR)
    best, bd = -1, 1e9
    for k in range(lo_, hi_):
        d = abs(sla[k] - bla[i]) + abs(((slo[k] - blo[i] + 180) % 360) - 180)
        if d < bd:
            bd, best = d, k
    if best < 0 or bd > 1.5:
        continue
    r = recs[best]
    for c, lab in enumerate(KEEP):
        v = r["v"].get(lab)
        if v is not None and not np.isnan(v[2]):
            F[i, c] = v[2]                       # the t=0 column
    AGE[i] = abs(st[best] - bt[i]) / 3.6e12
    ERA[i] = 1.0 if int(r["t"][:4]) >= 2005 else 0.0
    matched += 1

got = (~np.isnan(F)).astype("float32")
F = np.nan_to_num(F, nan=0.0)                    # unavailable == exact zeros, with a mask
print(f"\nmatched {matched:,}/{N:,} windows ({100*matched/N:.1f}%)")
print(f"  (SHIPS covers WP 1990-2021; our windows span 1980-2026 and include EP)")
print(f"  time offset of matched records: max {np.nanmax(AGE):.2f} h")
print(f"\nper-predictor coverage over matched windows:")
m = matched > 0
for c, lab in enumerate(KEEP):
    cov = 100 * got[:, c].sum() / max(matched, 1)
    if c < 12 or cov < 50:
        print(f"    {lab:6s} {cov:5.1f}%")

np.savez_compressed(OUT, feat=F, got=got, era_gfs=ERA, names=np.array(KEEP),
                    matched=np.array([matched]))
print(f"\nwrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
