#!/usr/bin/env python3
"""Cache a small no-key GEFS ensemble for the local Dolphin route experiment.

The files are subset through NOAA NOMADS to the Western Pacific and contain
only 850/500/200 hPa u/v winds. The control and 30 perturbed members are kept
separately so route members can be integrated first and averaged afterward,
matching the ensemble-mean idea used by modern global weather models.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "track_build" / "dolphin_gefs_ensemble_current.npz"
RAW_ROOT = ROOT / "v37" / "current_gefs_ensemble"
RAW_ROOT.mkdir(parents=True, exist_ok=True)

# This is the same cycle used by the existing deterministic GFS Dolphin cache.
CYCLE_DATE = os.environ.get("GEFS_CYCLE_DATE", "20260731")
CYCLE_HOUR = os.environ.get("GEFS_CYCLE_HOUR", "12")
LEADS = tuple(range(6, 121, 6))
MEMBER_COUNT = int(os.environ.get("GEFS_MEMBER_COUNT", "31"))
MEMBERS = ("gec00",) + tuple(f"gep{index:02d}" for index in range(1, MEMBER_COUNT))
WEIGHTS = np.asarray([0.269, 0.500, 0.231], dtype="float32")
LEFT_LON, RIGHT_LON = 130, 180
BOTTOM_LAT, TOP_LAT = 5, 35


def gefs_url(member: str, lead: int) -> str:
    query = {
        "file": f"{member}.t{CYCLE_HOUR}z.pgrb2a.0p50.f{lead:03d}",
        "lev_850_mb": "on",
        "lev_500_mb": "on",
        "lev_200_mb": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "subregion": "",
        "leftlon": str(LEFT_LON),
        "rightlon": str(RIGHT_LON),
        "toplat": str(TOP_LAT),
        "bottomlat": str(BOTTOM_LAT),
        "dir": f"/gefs.{CYCLE_DATE}/{CYCLE_HOUR}/atmos/pgrb2ap5",
    }
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl?" + urllib.parse.urlencode(query)


def raw_path(member: str, lead: int) -> Path:
    return RAW_ROOT / f"{member}_{CYCLE_DATE}_{CYCLE_HOUR}_f{lead:03d}.grib2"


def s3_url(member: str, lead: int, suffix: str = "") -> str:
    filename = f"{member}.t{CYCLE_HOUR}z.pgrb2a.0p50.f{lead:03d}{suffix}"
    return f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{CYCLE_DATE}/{CYCLE_HOUR}/atmos/pgrb2ap5/{filename}"


def s3_subset(member: str, lead: int) -> bytes:
    """Read just the six U/V pressure-level messages from the public S3 file."""

    index_request = urllib.request.Request(
        s3_url(member, lead, ".idx"),
        headers={"User-Agent": "typhoon-predict-local/1.0"},
    )
    with urllib.request.urlopen(index_request, timeout=60) as response:
        index_text = response.read().decode("utf-8")
    all_offsets = []
    entries = []
    for line in index_text.splitlines():
        pieces = line.split(":", 5)
        if len(pieces) != 6:
            continue
        offset = int(pieces[1])
        all_offsets.append(offset)
        description = pieces[5]
        if any(f":{variable}:{level} mb:" in line for variable in ("UGRD", "VGRD") for level in (850, 500, 200)):
            entries.append((offset, description))
    if len(entries) != 6:
        raise RuntimeError(f"S3 index for {member} +{lead}h has {len(entries)} needed messages")
    offsets = [offset for offset, _ in entries]
    chunks = []
    offset_positions = {offset: index for index, offset in enumerate(all_offsets)}
    for start in offsets:
        position = offset_positions[start]
        end = all_offsets[position + 1] - 1 if position + 1 < len(all_offsets) else None
        headers = {"User-Agent": "typhoon-predict-local/1.0", "Range": f"bytes={start}-" if end is None else f"bytes={start}-{end}"}
        request = urllib.request.Request(s3_url(member, lead), headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            chunks.append(response.read())
    data = b"".join(chunks)
    if data[:4] != b"GRIB":
        raise RuntimeError(f"S3 subset for {member} +{lead}h is not GRIB2")
    return data


def download_one(member: str, lead: int) -> tuple[str, int, str]:
    path = raw_path(member, lead)
    url = gefs_url(member, lead)
    if path.exists() and path.stat().st_size > 10_000:
        return member, lead, str(path)
    data = None
    last_error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "typhoon-predict-local/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            break
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 302:
                break
            if error.code not in {302, 429, 500, 502, 503, 504}:
                raise
        except urllib.error.URLError as error:
            last_error = error
        time.sleep(min(30.0, 2.0 ** min(attempt, 4)))
    if data is None:
        print(f"filter unavailable for {member} +{lead}h ({last_error}); using public S3 ranges", flush=True)
        data = s3_subset(member, lead)
    if data[:1] == b"<" or len(data) < 10_000:
        raise RuntimeError(f"NOMADS returned an error for {url}: {data[:300]!r}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return member, lead, str(path)


def read_deep_layer(path: Path) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    dataset = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
            "indexpath": "",
        },
    )
    try:
        dataset = dataset.sortby("latitude")
        pressure_dim = "isobaricInhPa"
        levels = np.asarray(dataset[pressure_dim].values, dtype="float32").reshape(-1)
        wanted = (850.0, 500.0, 200.0)
        indices = [int(np.abs(levels - level).argmin()) for level in wanted]
        if any(abs(float(levels[index]) - level) > 0.1 for index, level in zip(indices, wanted)):
            raise RuntimeError(f"{path.name} is missing 850/500/200 hPa: {levels.tolist()}")
        u_levels = np.stack([dataset["u"].isel({pressure_dim: index}).values for index in indices])
        v_levels = np.stack([dataset["v"].isel({pressure_dim: index}).values for index in indices])
        u = np.tensordot(WEIGHTS, u_levels, axes=(0, 0)).astype("float32")
        v = np.tensordot(WEIGHTS, v_levels, axes=(0, 0)).astype("float32")
        return (
            dataset["latitude"].values.astype("float32"),
            dataset["longitude"].values.astype("float32"),
            u,
            v,
            u_levels.astype("float32"),
            v_levels.astype("float32"),
        )
    finally:
        dataset.close()


def main() -> None:
    jobs = [(member, lead) for member in MEMBERS for lead in LEADS]
    print(json.dumps({
        "cycle": f"{CYCLE_DATE}{CYCLE_HOUR}Z",
        "members": len(MEMBERS),
        "leads": len(LEADS),
        "jobs": len(jobs),
        "region": [LEFT_LON, RIGHT_LON, BOTTOM_LAT, TOP_LAT],
    }), flush=True)
    paths: dict[tuple[str, int], Path] = {}
    workers = min(int(os.environ.get("GEFS_WORKERS", "4")), len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, member, lead): (member, lead) for member, lead in jobs}
        for number, future in enumerate(as_completed(futures), start=1):
            member, lead, path = future.result()
            paths[(member, lead)] = Path(path)
            if number == 1 or number % 50 == 0 or number == len(jobs):
                print(f"downloaded {number}/{len(jobs)}", flush=True)

    lat = lon = None
    u_members = np.empty((len(MEMBERS), len(LEADS), 61, 101), dtype="float32")
    v_members = np.empty_like(u_members)
    u_level_members = np.empty((len(MEMBERS), len(LEADS), 3, 61, 101), dtype="float32")
    v_level_members = np.empty_like(u_level_members)
    for member_index, member in enumerate(MEMBERS):
        for lead_index, lead in enumerate(LEADS):
            frame_lat, frame_lon, u, v, u_levels, v_levels = read_deep_layer(paths[(member, lead)])
            if lat is None:
                lat, lon = frame_lat, frame_lon
                if u.shape != (lat.size, lon.size):
                    raise RuntimeError(f"unexpected GEFS frame shape {u.shape}")
            elif not (np.array_equal(lat, frame_lat) and np.array_equal(lon, frame_lon)):
                raise RuntimeError(f"GEFS grid changed at {member} +{lead}h")
            u_members[member_index, lead_index] = u
            v_members[member_index, lead_index] = v
            u_level_members[member_index, lead_index] = u_levels
            v_level_members[member_index, lead_index] = v_levels
        print(f"parsed {member} ({member_index + 1}/{len(MEMBERS)})", flush=True)

    metadata = {
        "source": "NOAA NOMADS GEFS 0.5 degree pgrb2ap5",
        "cycle": f"{CYCLE_DATE}{CYCLE_HOUR}Z",
        "members": list(MEMBERS),
        "leads_hours": list(LEADS),
        "levels_hpa": [850, 500, 200],
        "weights": WEIGHTS.tolist(),
        "region": [LEFT_LON, RIGHT_LON, BOTTOM_LAT, TOP_LAT],
        "urls": {f"{member}_{lead:03d}": gefs_url(member, lead) for member, lead in jobs},
    }
    np.savez_compressed(
        OUT,
        leads=np.asarray(LEADS, dtype="int16"),
        latitude=lat,
        longitude=lon,
        u=u_members,
        v=v_members,
        u_levels=u_level_members,
        v_levels=v_level_members,
        metadata=np.asarray(json.dumps(metadata), dtype="U"),
    )
    print(json.dumps({
        "output": str(OUT),
        "cycle": f"{CYCLE_DATE}{CYCLE_HOUR}Z",
        "members": len(MEMBERS),
        "leads": len(LEADS),
        "grid": [int(lat.size), int(lon.size)],
        "source": "NOAA NOMADS GEFS 0.5 degree, weighted 850/500/200 hPa u/v",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
