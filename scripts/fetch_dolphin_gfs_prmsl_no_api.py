#!/usr/bin/env python3
"""Cache public GFS mean-sea-level pressure for a local Dolphin route expert.

This is a no-key NOAA NOMADS path.  It is deliberately separate from the
existing wind cache so the route experiment can be reproduced without
re-downloading the pressure-level files.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "track_build" / "dolphin_gfs_prmsl_current.npz"
RAW_ROOT = ROOT / "v37" / "current_gfs_prmsl"
RAW_ROOT.mkdir(parents=True, exist_ok=True)

CYCLE_DATE = "20260731"
CYCLE_HOUR = "12"
LEADS = tuple(range(6, 121, 6))
LEFT_LON, RIGHT_LON = 130, 190
BOTTOM_LAT, TOP_LAT = 0, 45


def gfs_url(lead: int) -> str:
    query = {
        "dir": f"/gfs.{CYCLE_DATE}/{CYCLE_HOUR}/atmos",
        "file": f"gfs.t{CYCLE_HOUR}z.pgrb2.0p25.f{lead:03d}",
        "lev_mean_sea_level": "on",
        "var_PRMSL": "on",
        "leftlon": str(LEFT_LON),
        "rightlon": str(RIGHT_LON),
        "toplat": str(TOP_LAT),
        "bottomlat": str(BOTTOM_LAT),
    }
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urllib.parse.urlencode(query)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 20_000:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "typhoon-predict-local/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if data[:1] == b"<" or len(data) < 20_000:
        raise RuntimeError(f"NOMADS returned an error for {url}: {data[:300]!r}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def read_prmsl(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "meanSea"},
            "indexpath": "",
        },
    )
    try:
        dataset = dataset.sortby("latitude")
        if "prmsl" not in dataset:
            raise RuntimeError(f"{path.name} has no prmsl variable: {list(dataset.data_vars)}")
        return (
            dataset["latitude"].values.astype("float32"),
            dataset["longitude"].values.astype("float32"),
            (dataset["prmsl"].values.astype("float32") / 100.0),
        )
    finally:
        dataset.close()


def main() -> None:
    latitude = longitude = None
    pressure = []
    sources = {}
    for lead in LEADS:
        path = RAW_ROOT / f"gfs_{CYCLE_DATE}_{CYCLE_HOUR}_f{lead:03d}.grib2"
        url = gfs_url(lead)
        download(url, path)
        frame_lat, frame_lon, frame_pressure = read_prmsl(path)
        if latitude is None:
            latitude, longitude = frame_lat, frame_lon
        elif not (np.array_equal(latitude, frame_lat) and np.array_equal(longitude, frame_lon)):
            raise RuntimeError(f"GFS pressure grid changed at +{lead}h")
        pressure.append(frame_pressure)
        sources[str(lead)] = {"url": url, "lead_hours": lead}
        print(f"parsed +{lead:03d}h: prmsl={frame_pressure.shape}", flush=True)

    np.savez_compressed(
        OUT,
        leads=np.asarray(LEADS, dtype="int16"),
        latitude=latitude,
        longitude=longitude,
        prmsl_hpa=np.stack(pressure).astype("float32"),
        sources=np.asarray(json.dumps({"cycle": f"{CYCLE_DATE}{CYCLE_HOUR}Z", "items": sources}), dtype="U"),
    )
    print(json.dumps({
        "output": str(OUT),
        "cycle": f"{CYCLE_DATE}{CYCLE_HOUR}Z",
        "leads": len(LEADS),
        "grid": [int(latitude.size), int(longitude.size)],
        "source": "NOAA NOMADS GFS 0.25 degree public filter endpoint, mean-sea-level pressure",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
