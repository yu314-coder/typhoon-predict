#!/usr/bin/env python3
"""Cache a no-key GFS forecast cycle for a physical Dolphin route guide.

The learned v23 route only sees analysis-time steering.  This companion cache
keeps the current public GFS cycle's future 850/500/200 hPa winds so a local
route integrator can use the same pressure-thickness-weighted deep-layer mean
that v23 was trained on.  JMA/JTWC forecast positions are never used here.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "track_build" / "dolphin_gfs_forecast_current.npz"
RAW_ROOT = ROOT / "v37" / "current_gfs_forecast"
RAW_ROOT.mkdir(parents=True, exist_ok=True)

CYCLE_DATE = "20260731"
CYCLE_HOUR = "12"
ISSUE_LAT = 19.1
ISSUE_LON = 160.2
LEADS = tuple(range(6, 121, 6))
WEIGHTS = np.asarray([0.269, 0.500, 0.231], dtype="float32")


def gfs_url(lead: int) -> str:
    query = {
        "dir": f"/gfs.{CYCLE_DATE}/{CYCLE_HOUR}/atmos",
        "file": f"gfs.t{CYCLE_HOUR}z.pgrb2.0p25.f{lead:03d}",
        "lev_850_mb": "on",
        "lev_500_mb": "on",
        "lev_200_mb": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "leftlon": "130",
        "rightlon": "190",
        "toplat": "45",
        "bottomlat": "0",
    }
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urllib.parse.urlencode(query)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 100_000:
        return
    print("download:", url, flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "typhoon-predict-local/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if data[:1] == b"<" or len(data) < 100_000:
        raise RuntimeError(f"NOMADS returned an error for {url}: {data[:300]!r}")
    path.write_bytes(data)


def read_deep_layer(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        levels = np.asarray(dataset["isobaricInhPa"].values, dtype="float32").reshape(-1)
        indices = [int(np.abs(levels - level).argmin()) for level in (850.0, 500.0, 200.0)]
        if any(abs(float(levels[index]) - wanted) > 0.1 for index, wanted in zip(indices, (850.0, 500.0, 200.0))):
            raise RuntimeError(f"{path.name} is missing 850/500/200 hPa; found {levels.tolist()}")
        u = np.stack([dataset["u"].isel(isobaricInhPa=index).values for index in indices]).astype("float32")
        v = np.stack([dataset["v"].isel(isobaricInhPa=index).values for index in indices]).astype("float32")
        u = np.tensordot(WEIGHTS, u, axes=(0, 0)).astype("float32")
        v = np.tensordot(WEIGHTS, v, axes=(0, 0)).astype("float32")
        return (
            dataset["latitude"].values.astype("float32"),
            dataset["longitude"].values.astype("float32"),
            u,
            v,
        )
    finally:
        dataset.close()


def main() -> None:
    lat = lon = None
    u_frames = []
    v_frames = []
    sources = {}
    for lead in LEADS:
        path = RAW_ROOT / f"gfs_{CYCLE_DATE}_{CYCLE_HOUR}_f{lead:03d}.grib2"
        source = gfs_url(lead)
        download(source, path)
        frame_lat, frame_lon, u, v = read_deep_layer(path)
        if lat is None:
            lat, lon = frame_lat, frame_lon
        elif not (np.array_equal(lat, frame_lat) and np.array_equal(lon, frame_lon)):
            raise RuntimeError(f"GFS grid changed at +{lead}h")
        u_frames.append(u)
        v_frames.append(v)
        sources[str(lead)] = {"url": source, "lead_hours": lead}
        print(f"parsed +{lead:03d}h: u={u.shape} v={v.shape}", flush=True)

    np.savez_compressed(
        OUT,
        leads=np.asarray(LEADS, dtype="int16"),
        latitude=lat,
        longitude=lon,
        u=np.stack(u_frames).astype("float32"),
        v=np.stack(v_frames).astype("float32"),
        issue_lat=np.asarray(ISSUE_LAT, dtype="float32"),
        issue_lon=np.asarray(ISSUE_LON, dtype="float32"),
        sources=np.asarray(json.dumps({"cycle": f"{CYCLE_DATE}{CYCLE_HOUR}Z", "items": sources}), dtype="U"),
    )
    print(json.dumps({
        "output": str(OUT),
        "cycle": f"{CYCLE_DATE}{CYCLE_HOUR}Z",
        "leads": len(LEADS),
        "grid": [int(lat.size), int(lon.size)],
        "source": "NOAA NOMADS GFS 0.25 degree, weighted 850/500/200 hPa u/v",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
