#!/usr/bin/env python3
"""Standalone ERA5 window-cache builder for StormFusion-MT.

Extracted from typhoon_stormfusion_mt_colab.ipynb so the slow, CPU/network-bound
ERA5 download step can run locally (free) instead of burning paid A100 hours.

Output: <OUT_ROOT>/data/windows/stormfusion_windows.npz
Upload that file to your Drive folder
    MyDrive/typhoon_predict_stormfusion_mt/data/windows/stormfusion_windows.npz
and the Colab notebook will skip CDS entirely and go straight to training.

Requirements:
    pip install cdsapi xarray netCDF4 pandas numpy requests tqdm
    CDS auth: create ~/.cdsapirc with:
        url: https://cds.climate.copernicus.eu/api
        key: <YOUR_CDS_PERSONAL_TOKEN>
    (or set CDSAPI_URL / CDSAPI_KEY env vars).

Env overrides:
    ERA5_START_YEAR (default 1979)  -- drop storms before this year (ERA5 has no pre-1940 data)
    MAX_STORMS      (default none)  -- cap number of storms (quick test runs)
    MAX_WINDOWS     (default 10000) -- cap number of built windows
    OUT_ROOT        (default ./typhoon_build)
"""
import os
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
import cdsapi
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Connect to CDS directly. A transient local sandbox proxy may be injected via env
# vars; it dies on session restart and would stall every request behind 120s retries.
for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_proxy_var, None)

# --------------------------------------------------------------------------
# Config (mirrors the notebook)
# --------------------------------------------------------------------------
CONFIG = {
    "track_url": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv",
    "history_steps": 9,
    "atmosphere_times": [-18, -12, -6, 0],
    "lead_hours": list(range(6, 121, 6)),
    "patch_size": 65,
    "inner_spacing_km": 25.0,
    "outer_spacing_km": 50.0,
    "era5_start_year": int(os.environ.get("ERA5_START_YEAR", "1979")),
    "max_storms": int(os.environ["MAX_STORMS"]) if os.environ.get("MAX_STORMS") else None,
    "max_windows": int(os.environ.get("MAX_WINDOWS", "10000")),
    "max_windows_per_storm": int(os.environ["MAX_WINDOWS_PER_STORM"]) if os.environ.get("MAX_WINDOWS_PER_STORM") else None,
    "download_workers": int(os.environ.get("DOWNLOAD_WORKERS", "5")),
    # cached_only: build only windows whose 4 atmosphere frames are already on disk;
    # never download. Lets us expand the dataset for free from the existing frame cache.
    "cached_only": os.environ.get("CACHED_ONLY", "0") == "1",
}

OUT_ROOT = Path(os.environ.get("OUT_ROOT", "./typhoon_build")).resolve()
DATA_ROOT = OUT_ROOT / "data"
FRAME_ROOT = DATA_ROOT / "era5_frames"
WINDOW_ROOT = DATA_ROOT / "windows"
for folder in (DATA_ROOT, FRAME_ROOT, WINDOW_ROOT):
    folder.mkdir(parents=True, exist_ok=True)

# Allow cdsapi auth via env vars in addition to ~/.cdsapirc
if os.environ.get("CDSAPI_KEY") and not (Path.home() / ".cdsapirc").exists():
    url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    (Path.home() / ".cdsapirc").write_text(f"url: {url}\nkey: {os.environ['CDSAPI_KEY']}\n")

# --------------------------------------------------------------------------
# Track download + normalize + year filter
# --------------------------------------------------------------------------
TRACK_FILE = DATA_ROOT / "ibtracs_wp_v04r01.csv"
if not TRACK_FILE.exists():
    print("Downloading IBTrACS WP ...")
    response = requests.get(CONFIG["track_url"], timeout=180)
    response.raise_for_status()
    TRACK_FILE.write_bytes(response.content)

raw_track = pd.read_csv(TRACK_FILE, skiprows=[1], low_memory=False)
print("Raw shape:", raw_track.shape)


def first_existing(frame, names, required=True):
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise KeyError(f"None of these columns exist: {names}")
    return None


def numeric(series):
    value = pd.to_numeric(series, errors="coerce").astype("float32")
    value = value.mask(value < -900)
    value = value.mask(value > 1e6)
    return value


sid_col = first_existing(raw_track, ["SID", "SERIAL_NUM"])
time_col = first_existing(raw_track, ["ISO_TIME"])
lat_col = first_existing(raw_track, ["LAT"])
lon_col = first_existing(raw_track, ["LON"])
wind_col = first_existing(raw_track, ["USA_WIND", "WMO_WIND"], required=False)
pres_col = first_existing(raw_track, ["USA_PRES", "WMO_PRES"], required=False)
gust_col = first_existing(raw_track, ["USA_GUST", "WMO_GUST"], required=False)
rmw_col = first_existing(raw_track, ["USA_RMW", "WMO_RMW"], required=False)

track = pd.DataFrame({
    "sid": raw_track[sid_col].astype(str),
    "basin": raw_track[first_existing(raw_track, ["BASIN"], required=False)].astype(str) if "BASIN" in raw_track else "WP",
    "time": pd.to_datetime(raw_track[time_col], errors="coerce", utc=True),
    "lat": numeric(raw_track[lat_col]),
    "lon": numeric(raw_track[lon_col]),
    "vmax": numeric(raw_track[wind_col]) if wind_col else np.nan,
    "pressure": numeric(raw_track[pres_col]) if pres_col else np.nan,
    "gust": numeric(raw_track[gust_col]) if gust_col else np.nan,
    "rmw": numeric(raw_track[rmw_col]) if rmw_col else np.nan,
})

for threshold in (34, 50, 64):
    for quadrant in ("ne", "se", "sw", "nw"):
        candidates = [f"USA_R{threshold}_{quadrant.upper()}", f"WMO_R{threshold}_{quadrant.upper()}"]
        source = first_existing(raw_track, candidates, required=False)
        track[f"r{threshold}_{quadrant}"] = numeric(raw_track[source]) if source else np.nan

track = track[(track["basin"] == "WP") & track["time"].notna() & track["lat"].notna() & track["lon"].notna()].copy()
track["year"] = track["time"].dt.year.astype(int)
track["lon"] = ((track["lon"] + 180.0) % 360.0) - 180.0
track = track.sort_values(["sid", "time"]).reset_index(drop=True)

start_year = CONFIG["era5_start_year"]
before = track["sid"].nunique()
track = track[track["year"] >= start_year].copy()
print(f"ERA5 filter: kept {track['sid'].nunique()} of {before} storms; "
      f"years {int(track['year'].min())}-{int(track['year'].max())}")

if CONFIG["max_storms"] is not None:
    keep = track["sid"].drop_duplicates().iloc[:CONFIG["max_storms"]]
    track = track[track["sid"].isin(keep)].copy()

print("Normalized rows:", len(track), "Storms:", track["sid"].nunique())

# --------------------------------------------------------------------------
# ERA5 extraction (cell 8)
# --------------------------------------------------------------------------
PRESSURE_VARIABLES = ["u_component_of_wind", "v_component_of_wind", "specific_humidity", "temperature", "geopotential"]
SINGLE_VARIABLES = ["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure",
                    "sea_surface_temperature", "surface_pressure", "land_sea_mask", "geopotential"]
PRESSURE_LEVELS = ["925", "850", "700", "500", "200"]

# The modern CDS backend returns ERA5 NetCDF variables under short names.
# Map each requested CDS variable to the netCDF names it may appear under.
VAR_ALIASES = {
    "u_component_of_wind": ["u_component_of_wind", "u"],
    "v_component_of_wind": ["v_component_of_wind", "v"],
    "specific_humidity": ["specific_humidity", "q"],
    "temperature": ["temperature", "t"],
    "geopotential": ["geopotential", "z"],
    "10m_u_component_of_wind": ["10m_u_component_of_wind", "u10"],
    "10m_v_component_of_wind": ["10m_v_component_of_wind", "v10"],
    "mean_sea_level_pressure": ["mean_sea_level_pressure", "msl"],
    "sea_surface_temperature": ["sea_surface_temperature", "sst"],
    "surface_pressure": ["surface_pressure", "sp"],
    "land_sea_mask": ["land_sea_mask", "lsm"],
}


def frame_key(ts, lat, lon):
    ts = pd.Timestamp(ts).tz_convert("UTC")
    return f"{ts:%Y%m%d%H}_{float(lat):+.2f}_{float(lon):+.2f}".replace("-", "m").replace("+", "p").replace(".", "d")


def request_era5_frame(ts, lat, lon):
    ts = pd.Timestamp(ts).tz_convert("UTC")
    key = frame_key(ts, lat, lon)
    pressure_path = FRAME_ROOT / f"{key}_pressure.nc"
    single_path = FRAME_ROOT / f"{key}_single.nc"
    if pressure_path.exists() and single_path.exists():
        return pressure_path, single_path

    # sleep_max defaults to 120s, so every transient 502 costs a 2-minute idle sleep.
    # CDS 502s come in storms; a shorter backoff recovers ~8x faster.
    client = cdsapi.Client(quiet=True, sleep_max=15, retry_max=500)
    lon0 = float(lon) % 360.0
    west = max(0.0, lon0 - 22.0)
    east = min(360.0, lon0 + 22.0)
    north = min(90.0, float(lat) + 22.0)
    south = max(-90.0, float(lat) - 22.0)
    common = {
        "product_type": ["reanalysis"],
        "year": [f"{ts.year:04d}"], "month": [f"{ts.month:02d}"],
        "day": [f"{ts.day:02d}"], "time": [f"{ts.hour:02d}:00"],
        "area": [north, west, south, east],
        "data_format": "netcdf", "download_format": "unarchived",
    }
    if not pressure_path.exists():
        client.retrieve("reanalysis-era5-pressure-levels",
                        {**common, "variable": PRESSURE_VARIABLES, "pressure_level": PRESSURE_LEVELS},
                        str(pressure_path))
    if not single_path.exists():
        client.retrieve("reanalysis-era5-single-levels",
                        {**common, "variable": SINGLE_VARIABLES}, str(single_path))
    if not pressure_path.exists() or not single_path.exists():
        raise RuntimeError(f"ERA5 request did not create both files for {ts}")
    return pressure_path, single_path


def _coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"Could not find coordinate among {candidates}; found {list(ds.coords)}")


def _select_dataarray(ds, variable, level=None):
    candidates = VAR_ALIASES.get(variable, [variable])
    found = next((name for name in candidates if name in ds), None)
    if found is None:
        raise KeyError(f"ERA5 variable {variable} missing (tried {candidates}); found {list(ds.data_vars)}")
    da = ds[found]
    for tdim in ("time", "valid_time"):
        if tdim in da.dims:
            da = da.isel({tdim: 0})
    if level is not None:
        level_dim = _coord_name(ds, ["pressure_level", "isobaricInhPa", "level"])
        da = da.sel({level_dim: int(level)}, method="nearest")
    return da


def _interp_patch(da, lat_values, lon_values):
    lat_name = _coord_name(da.to_dataset(name="x"), ["latitude", "lat"])
    lon_name = _coord_name(da.to_dataset(name="x"), ["longitude", "lon"])
    da = da.sortby(lat_name)
    lon_coord = da[lon_name]
    if float(lon_coord.max()) > 180.0:
        left = da.assign_coords({lon_name: lon_coord - 360.0})
        right = da.assign_coords({lon_name: lon_coord + 360.0})
        da = xr.concat([left, da, right], dim=lon_name).sortby(lon_name)
        lon_values = ((np.asarray(lon_values) + 180.0) % 360.0) - 180.0
    lat_da = xr.DataArray(np.asarray(lat_values), dims="y")
    lon_da = xr.DataArray(np.asarray(lon_values), dims="x")
    out = da.interp({lat_name: lat_da, lon_name: lon_da}, method="linear")
    return np.asarray(out.values, dtype="float32")


def _local_grid(lat, lon, spacing_km, size):
    offsets = (np.arange(size, dtype="float32") - (size - 1) / 2.0) * spacing_km
    lat_values = float(lat) + offsets / 111.2
    cos_lat = max(0.2, math.cos(math.radians(float(lat))))
    lon_values = float(lon) + offsets / (111.2 * cos_lat)
    return lat_values, lon_values


def extract_frame_patches(ts, lat, lon):
    pressure_path, single_path = request_era5_frame(ts, lat, lon)
    with xr.open_dataset(pressure_path) as pressure_ds, xr.open_dataset(single_path) as single_ds:
        inner_lat, inner_lon = _local_grid(lat, lon, CONFIG["inner_spacing_km"], CONFIG["patch_size"])
        outer_lat, outer_lon = _local_grid(lat, lon, CONFIG["outer_spacing_km"], CONFIG["patch_size"])

        def p(var, level, lat_values, lon_values):
            return _interp_patch(_select_dataarray(pressure_ds, var, level), lat_values, lon_values)

        def s(var, lat_values, lon_values):
            return _interp_patch(_select_dataarray(single_ds, var), lat_values, lon_values)

        inner_parts = []
        for level in ("925", "850", "700", "500", "200"):
            inner_parts += [p("u_component_of_wind", level, inner_lat, inner_lon)]
            inner_parts += [p("v_component_of_wind", level, inner_lat, inner_lon)]
        for level in ("850", "700", "500"):
            inner_parts.append(p("specific_humidity", level, inner_lat, inner_lon))
        for level in ("850", "500", "200"):
            inner_parts.append(p("temperature", level, inner_lat, inner_lon))
        for level in ("850", "500", "200"):
            inner_parts.append(p("geopotential", level, inner_lat, inner_lon))
        inner_parts += [
            s("mean_sea_level_pressure", inner_lat, inner_lon),
            s("surface_pressure", inner_lat, inner_lon),
            s("sea_surface_temperature", inner_lat, inner_lon),
            s("10m_u_component_of_wind", inner_lat, inner_lon),
            s("10m_v_component_of_wind", inner_lat, inner_lon),
            s("land_sea_mask", inner_lat, inner_lon),
            s("geopotential", inner_lat, inner_lon),
        ]

        outer_parts = []
        for level in ("850", "700", "500", "200"):
            outer_parts += [p("u_component_of_wind", level, outer_lat, outer_lon)]
            outer_parts += [p("v_component_of_wind", level, outer_lat, outer_lon)]
        for level in ("850", "500", "200"):
            outer_parts.append(p("geopotential", level, outer_lat, outer_lon))
        outer_parts += [
            p("specific_humidity", "700", outer_lat, outer_lon),
            s("mean_sea_level_pressure", outer_lat, outer_lon),
            s("sea_surface_temperature", outer_lat, outer_lon),
        ]

    inner = np.stack(inner_parts).astype("float32")
    outer = np.stack(outer_parts).astype("float32")
    if inner.shape != (26, CONFIG["patch_size"], CONFIG["patch_size"]):
        raise ValueError(f"Inner patch shape mismatch: {inner.shape}")
    if outer.shape != (14, CONFIG["patch_size"], CONFIG["patch_size"]):
        raise ValueError(f"Outer patch shape mismatch: {outer.shape}")
    return inner, outer


def environmental_summary(inner):
    mean = lambda x: float(np.nanmean(x))
    return np.asarray([
        mean(inner[8] - inner[2]), mean(inner[9] - inner[3]), mean(inner[2]), mean(inner[3]),
        mean(inner[11]), mean(inner[13]), mean(inner[19]), mean(inner[21]),
        mean(inner[24]), mean(inner[25]),
    ], dtype="float32")


# --------------------------------------------------------------------------
# Window building (cell 10)
# --------------------------------------------------------------------------
RADIUS_NAMES = [f"r{threshold}_{quadrant}" for threshold in (34, 50, 64) for quadrant in ("ne", "se", "sw", "nw")]


def valid_number(value):
    return value is not None and np.isfinite(value)


def local_motion_km(lat0, lon0, lat1, lon1):
    dlat = float(lat1) - float(lat0)
    dlon = ((float(lon1) - float(lon0) + 180.0) % 360.0) - 180.0
    north = dlat * 111.2
    east = dlon * 111.2 * math.cos(math.radians((float(lat0) + float(lat1)) / 2.0))
    return east, north


def nearest_row(group, target_time, tolerance_hours=1.5):
    delta = (group["time"] - target_time).abs()
    index = delta.idxmin()
    if pd.isna(index) or delta.loc[index] > pd.Timedelta(hours=tolerance_hours):
        return None
    return group.loc[index]


def history_features(history_rows, base_row, t0):
    sequence = np.zeros((len(history_rows), 40), dtype="float32")
    previous = None
    for i, row in enumerate(history_rows):
        east, north = local_motion_km(base_row.lat, base_row.lon, row.lat, row.lon)
        if previous is None:
            step_east, step_north = 0.0, 0.0
        else:
            step_east, step_north = local_motion_km(previous.lat, previous.lon, row.lat, row.lon)
        features = sequence[i]
        features[0:4] = [east, north, step_east, step_north]
        values = [row.vmax, row.pressure, row.gust, row.rmw]
        features[4] = float(values[0]) if valid_number(values[0]) else 0.0
        features[5] = float(values[1]) if valid_number(values[1]) else 0.0
        features[6] = float(values[2]) if valid_number(values[2]) else 0.0
        features[7] = float(values[3]) if valid_number(values[3]) else 0.0
        for j, name in enumerate(RADIUS_NAMES):
            value = row[name]
            features[8 + j] = float(value) if valid_number(value) else 0.0
        features[20] = 0.0
        phase = 2.0 * math.pi * t0.dayofyear / 365.25
        features[21:23] = [math.sin(phase), math.cos(phase)]
        features[23] = float((t0 - row.time).total_seconds() / 3600.0)
        fields = [row.vmax, row.pressure, row.gust, row.rmw] + [row[name] for name in RADIUS_NAMES]
        features[24] = float(valid_number(fields[0]))
        features[25] = float(valid_number(fields[1]))
        features[26] = float(valid_number(fields[2]))
        features[27] = float(valid_number(fields[3]))
        features[28:40] = [float(valid_number(value)) for value in fields[4:]]
        previous = row
    return sequence


def future_targets(future_rows, base_row):
    target = np.zeros((len(future_rows), 17), dtype="float32")
    mask = np.zeros((len(future_rows), 17), dtype=bool)
    previous = base_row
    for i, row in enumerate(future_rows):
        east, north = local_motion_km(previous.lat, previous.lon, row.lat, row.lon)
        target[i, 0:2] = [east, north]
        mask[i, 0:2] = True
        for j, name in enumerate(["vmax", "pressure", "rmw"] + RADIUS_NAMES, start=2):
            value = row[name]
            if valid_number(value):
                target[i, j] = float(value)
                mask[i, j] = True
        previous = row
    return target, mask


patch_cache = {}


def cached_patch(row):
    key = frame_key(row.time, row.lat, row.lon)
    if key not in patch_cache:
        patch_cache[key] = extract_frame_patches(row.time, row.lat, row.lon)
    return patch_cache[key]


def build_windows(track_frame):
    inner_windows, outer_windows, track_windows, env_windows = [], [], [], []
    target_windows, mask_windows = [], []
    ids, years, init_times, init_lats, init_lons = [], [], [], [], []
    history_offsets = [pd.Timedelta(hours=-6 * i) for i in range(CONFIG["history_steps"] - 1, -1, -1)]
    lead_offsets = [pd.Timedelta(hours=h) for h in CONFIG["lead_hours"]]

    groups = list(track_frame.groupby("sid", sort=False))
    # Shuffle storms deterministically so a max_windows cap samples evenly across
    # all years (otherwise the chronological order fills the cap with only the
    # earliest storms and the 2016-2019 / >=2020 validation and test splits are empty).
    order = np.random.RandomState(42).permutation(len(groups))
    groups = [groups[i] for i in order]
    if CONFIG["max_storms"] is not None:
        groups = groups[:CONFIG["max_storms"]]

    # Phase 1: plan windows (pure pandas, no network) until max_windows reached.
    per_storm_cap = CONFIG["max_windows_per_storm"]
    plans = []  # (sid, t0, base, history_rows, future_rows, atmosphere_rows)
    for sid, group in tqdm(groups, desc="Planning windows"):
        group = group.sort_values("time")
        storm_plans = []
        for t0 in group["time"].drop_duplicates().tolist():
            base = nearest_row(group, t0)
            if base is None:
                continue
            history_rows = [nearest_row(group, t0 + offset) for offset in history_offsets]
            future_rows = [nearest_row(group, t0 + offset) for offset in lead_offsets]
            if any(row is None for row in history_rows + future_rows):
                continue
            atmosphere_rows = [nearest_row(group, t0 + pd.Timedelta(hours=h)) for h in CONFIG["atmosphere_times"]]
            if any(row is None for row in atmosphere_rows):
                continue
            if CONFIG["cached_only"]:
                keys = [frame_key(r.time, r.lat, r.lon) for r in atmosphere_rows]
                if not all((FRAME_ROOT / f"{k}_pressure.nc").exists()
                           and (FRAME_ROOT / f"{k}_single.nc").exists() for k in keys):
                    continue
            storm_plans.append((sid, t0, base, history_rows, future_rows, atmosphere_rows))
        # Subsample evenly across the storm's life so we sample genesis -> peak -> decay,
        # not just the earliest windows.
        if per_storm_cap and len(storm_plans) > per_storm_cap:
            picks = np.linspace(0, len(storm_plans) - 1, per_storm_cap).round().astype(int)
            picks = sorted(set(int(i) for i in picks))
            storm_plans = [storm_plans[i] for i in picks]
        plans.extend(storm_plans)
        if len(plans) >= CONFIG["max_windows"]:
            break
    plans = plans[:CONFIG["max_windows"]]

    if not plans:
        raise RuntimeError("No windows planned. Check track cadence, dates, and history/lead settings.")

    # Phase 2: download the unique ERA5 frames those windows need, in parallel.
    unique_frames = {}
    for _, _, _, _, _, atmosphere_rows in plans:
        for row in atmosphere_rows:
            unique_frames.setdefault(frame_key(row.time, row.lat, row.lon), row)
    if CONFIG["cached_only"]:
        print(f"Planned {len(plans)} windows from {len(unique_frames)} cached frames "
              f"(cached-only mode; no downloads).")
    else:
        print(f"Planned {len(plans)} windows needing {len(unique_frames)} unique ERA5 frames; "
              f"downloading with {CONFIG['download_workers']} workers.")

    # CDS limits queued requests per user per dataset; a burst gets rejected with
    # "The job has been rejected. Number queued requests ... temporarily limited".
    # Retry those (and transient 5xx/timeouts) with backoff so the frame re-queues
    # instead of permanently dropping its window.
    _transient = ("rejected", "temporarily limited", "429", "Too Many Requests",
                  "500", "502", "503", "504", "Bad Gateway", "timed out", "timeout")

    def _download(row):
        key = frame_key(row.time, row.lat, row.lon)
        for attempt in range(8):
            try:
                request_era5_frame(row.time, row.lat, row.lon)
                return None
            except Exception as error:  # noqa: BLE001
                message = str(error)
                if attempt < 7 and any(token in message for token in _transient):
                    time.sleep(min(60, 10 * (attempt + 1)))
                    continue
                return (key, message)
        return (key, "exhausted retries")

    if not CONFIG["cached_only"]:
        failures = 0
        with ThreadPoolExecutor(max_workers=CONFIG["download_workers"]) as executor:
            futures = [executor.submit(_download, row) for row in unique_frames.values()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading ERA5 frames"):
                result = future.result()
                if result is not None:
                    failures += 1
                    if failures <= 20:
                        print(f"Frame download failed ({result[0]}): {result[1][:160]}")
        if failures:
            print(f"{failures} frames failed to download; windows needing them will be skipped.")

    # Phase 3: assemble windows serially from the on-disk frame cache (fast, CPU only).
    for sid, t0, base, history_rows, future_rows, atmosphere_rows in tqdm(plans, desc="Assembling windows"):
        try:
            patches = [cached_patch(row) for row in atmosphere_rows]
        except Exception as error:
            print(f"Skipping {sid} {t0} because ERA5 extraction failed: {error}")
            continue

        inner = np.stack([pair[0] for pair in patches])
        outer = np.stack([pair[1] for pair in patches])
        env = np.stack([environmental_summary(pair[0]) for pair in patches])
        hist = history_features(history_rows, base, t0)
        target, target_mask = future_targets(future_rows, base)

        inner_windows.append(inner)
        outer_windows.append(outer)
        track_windows.append(hist)
        env_windows.append(env)
        target_windows.append(target)
        mask_windows.append(target_mask)
        ids.append(str(sid))
        years.append(int(t0.year))
        init_times.append(str(t0))
        init_lats.append(float(base.lat))
        init_lons.append(float(base.lon))

    if not inner_windows:
        raise RuntimeError("No windows were built. Check ERA5 credentials and download failures above.")
    return {
        "inner": np.asarray(inner_windows, dtype="float32"),
        "outer": np.asarray(outer_windows, dtype="float32"),
        "track": np.asarray(track_windows, dtype="float32"),
        "env": np.asarray(env_windows, dtype="float32"),
        "target": np.asarray(target_windows, dtype="float32"),
        "target_mask": np.asarray(mask_windows, dtype="bool"),
        "storm_id": np.asarray(ids),
        "year": np.asarray(years, dtype="int16"),
        "init_time": np.asarray(init_times),
        "init_lat": np.asarray(init_lats, dtype="float32"),
        "init_lon": np.asarray(init_lons, dtype="float32"),
    }


if __name__ == "__main__":
    WINDOW_FILE = WINDOW_ROOT / "stormfusion_windows.npz"
    if WINDOW_FILE.exists():
        print(f"{WINDOW_FILE} already exists; delete it to rebuild.")
    else:
        window_arrays = build_windows(track)
        np.savez_compressed(WINDOW_FILE, **window_arrays)
        for key, value in window_arrays.items():
            print(key, value.shape, value.dtype)
        print(f"\nSaved {WINDOW_FILE}")
        print("Upload it to Drive at: MyDrive/typhoon_predict_stormfusion_mt/data/windows/stormfusion_windows.npz")
