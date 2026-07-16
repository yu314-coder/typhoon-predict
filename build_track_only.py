#!/usr/bin/env python3
"""Build a TRACK-ONLY window dataset from all-basin IBTrACS. Self-contained: downloads
IBTrACS if missing, no ERA5. Fast (searchsorted nearest, not pandas idxmin).

Paths via env (Colab): TRACK_CSV (IBTrACS csv), TRACK_OUT (npz output).
Reuses the 9x40 history-feature / 20x17 target logic from the ERA5 pipeline.
"""
import os
import math
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

CSV = Path(os.environ.get("TRACK_CSV", "typhoon_build/data/ibtracs_ALL.csv"))
OUT = Path(os.environ.get("TRACK_OUT", "track_build/track_windows.npz"))
OUT.parent.mkdir(parents=True, exist_ok=True)
CSV.parent.mkdir(parents=True, exist_ok=True)
MIN_YEAR = int(os.environ.get("TRACK_MIN_YEAR", "1980"))
URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"

HISTORY_STEPS = 9
LEAD_HOURS = list(range(6, 121, 6))
RADIUS_NAMES = [f"r{t}_{q}" for t in (34, 50, 64) for q in ("ne", "se", "sw", "nw")]

if not CSV.exists():
    print(f"downloading IBTrACS (all basins) -> {CSV} ...")
    urllib.request.urlretrieve(URL, CSV)
    print("download done")


def numeric(s):
    v = pd.to_numeric(s, errors="coerce").astype("float32")
    return v.mask(v < -900).mask(v > 1e6)


def first_existing(frame, names):
    for n in names:
        if n in frame.columns:
            return n
    return None


print(f"loading {CSV} ...")
raw = pd.read_csv(CSV, skiprows=[1], low_memory=False)
cols = {
    "sid": raw["SID"].astype(str),
    "basin": raw["BASIN"].astype(str),
    "time": pd.to_datetime(raw["ISO_TIME"], errors="coerce", utc=True),
    "lat": numeric(raw["LAT"]),
    "lon": numeric(raw["LON"]),
    "vmax": numeric(raw[first_existing(raw, ["USA_WIND", "WMO_WIND"])]),
    "pressure": numeric(raw[first_existing(raw, ["USA_PRES", "WMO_PRES"])]),
}
gc = first_existing(raw, ["USA_GUST", "WMO_GUST"])
rc = first_existing(raw, ["USA_RMW", "WMO_RMW"])
cols["gust"] = numeric(raw[gc]) if gc else np.float32(np.nan)
cols["rmw"] = numeric(raw[rc]) if rc else np.float32(np.nan)
track = pd.DataFrame(cols)
for t in (34, 50, 64):
    for q in ("ne", "se", "sw", "nw"):
        c = first_existing(raw, [f"USA_R{t}_{q.upper()}", f"WMO_R{t}_{q.upper()}"])
        track[f"r{t}_{q}"] = numeric(raw[c]) if c else np.float32(np.nan)

track = track[track["time"].notna() & track["lat"].notna() & track["lon"].notna()].copy()
track["year"] = track["time"].dt.year.astype(int)
track = track[track["year"] >= MIN_YEAR].copy()
track["lon"] = ((track["lon"] + 180.0) % 360.0) - 180.0
track = track.sort_values(["sid", "time"]).reset_index(drop=True)
print(f"{track['sid'].nunique()} storms, {len(track)} rows, years {track['year'].min()}-{track['year'].max()}, basins {sorted(track['basin'].dropna().unique())}")

FEATCOLS = ["lat", "lon", "vmax", "pressure", "gust", "rmw"] + RADIUS_NAMES
TOL_NS = int(1.5 * 3600 * 1e9)


def valid_number(v):
    return v is not None and np.isfinite(v)


def local_motion_km(lat0, lon0, lat1, lon1):
    dlat = lat1 - lat0
    dlon = ((lon1 - lon0 + 180.0) % 360.0) - 180.0
    return dlon * 111.2 * math.cos(math.radians((lat0 + lat1) / 2.0)), dlat * 111.2


tracks, targets, masks, sids, years, basins = [], [], [], [], [], []
groups = list(track.groupby("sid", sort=False))
hist_off_ns = [int(-6 * i * 3600 * 1e9) for i in range(HISTORY_STEPS - 1, -1, -1)]
lead_off_ns = [int(h * 3600 * 1e9) for h in LEAD_HOURS]
atm_none = 0

for gi, (sid, g) in enumerate(groups):
    tns = g["time"].dt.tz_convert(None).to_numpy().astype("datetime64[ns]").astype("int64")  # ns, sorted
    feat = {c: g[c].values.astype("float64") for c in FEATCOLS}
    basin = str(g["basin"].values[0])
    doy = g["time"].dt.dayofyear.values

    def nidx(target_ns):
        p = np.searchsorted(tns, target_ns)
        best, bd = -1, TOL_NS + 1
        for cand in (p - 1, p):
            if 0 <= cand < len(tns):
                d = abs(int(tns[cand]) - target_ns)
                if d < bd:
                    bd, best = d, cand
        return best if bd <= TOL_NS else -1

    for k in range(len(tns)):
        t0 = int(tns[k]); base = k
        hidx = [nidx(t0 + o) for o in hist_off_ns]
        fidx = [nidx(t0 + o) for o in lead_off_ns]
        if -1 in hidx or -1 in fidx:
            continue
        # history features (9x40)
        seq = np.zeros((HISTORY_STEPS, 40), dtype="float32")
        prev = -1
        phase = 2.0 * math.pi * float(doy[base]) / 365.25
        for i, idx in enumerate(hidx):
            e, n = local_motion_km(feat["lat"][base], feat["lon"][base], feat["lat"][idx], feat["lon"][idx])
            if prev < 0:
                se, sn = 0.0, 0.0
            else:
                se, sn = local_motion_km(feat["lat"][prev], feat["lon"][prev], feat["lat"][idx], feat["lon"][idx])
            f = seq[i]
            f[0:4] = [e, n, se, sn]
            vals = [feat["vmax"][idx], feat["pressure"][idx], feat["gust"][idx], feat["rmw"][idx]]
            for j in range(4):
                f[4 + j] = vals[j] if valid_number(vals[j]) else 0.0
            for j, nm in enumerate(RADIUS_NAMES):
                rv = feat[nm][idx]
                f[8 + j] = rv if valid_number(rv) else 0.0
            f[20] = 0.0
            f[21:23] = [math.sin(phase), math.cos(phase)]
            f[23] = (t0 - int(tns[idx])) / 3.6e12
            fields = vals + [feat[nm][idx] for nm in RADIUS_NAMES]
            f[24:28] = [float(valid_number(x)) for x in fields[:4]]
            f[28:40] = [float(valid_number(x)) for x in fields[4:]]
            prev = idx
        # targets (20x17)
        tgt = np.zeros((len(fidx), 17), dtype="float32")
        msk = np.zeros((len(fidx), 17), dtype=bool)
        prev = base
        for i, idx in enumerate(fidx):
            e, n = local_motion_km(feat["lat"][prev], feat["lon"][prev], feat["lat"][idx], feat["lon"][idx])
            tgt[i, 0:2] = [e, n]; msk[i, 0:2] = True
            for j, nm in enumerate(["vmax", "pressure", "rmw"] + RADIUS_NAMES, start=2):
                rv = feat[nm][idx]
                if valid_number(rv):
                    tgt[i, j] = rv; msk[i, j] = True
            prev = idx
        tracks.append(seq); targets.append(tgt); masks.append(msk)
        sids.append(str(sid)); years.append(int(g["year"].values[base])); basins.append(basin)
    if gi % 1000 == 0:
        print(f"  {gi}/{len(groups)} storms, {len(tracks)} windows", flush=True)

track_arr = np.asarray(tracks, dtype="float32")
print(f"\nbuilt {len(track_arr)} track-only windows")

years_a = np.asarray(years, dtype="int16"); sids_a = np.asarray(sids)
fy = {s: int(years_a[sids_a == s].min()) for s in np.unique(sids_a)}
train_idx = np.where(np.isin(sids_a, [s for s, y in fy.items() if y <= 2015]))[0]
means = np.zeros(40, dtype="float32"); stds = np.ones(40, dtype="float32")
for c in range(40):
    col = track_arr[train_idx, :, c].astype("float64")
    m, s = np.nanmean(col), np.nanstd(col)
    means[c] = 0.0 if not np.isfinite(m) else m
    stds[c] = 1.0 if (not np.isfinite(s) or s < 1e-6) else s
    track_arr[:, :, c] = (track_arr[:, :, c] - means[c]) / stds[c]
np.nan_to_num(track_arr, copy=False)
print(f"normalized on {len(train_idx)} train windows; overall mean {track_arr.mean():+.4f} std {track_arr.std():.4f}")

np.savez_compressed(OUT, track=track_arr, target=np.asarray(targets, dtype="float32"),
                    target_mask=np.asarray(masks, dtype="bool"), storm_id=sids_a,
                    year=years_a, basin=np.asarray(basins), track_mean=means, track_std=stds)
tw = int((years_a <= 2015).sum()); vw = int(((years_a >= 2016) & (years_a <= 2019)).sum()); ew = int((years_a >= 2020).sum())
print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB) | split windows train={tw} valid={vw} test={ew} | storms={len(fy)}")
