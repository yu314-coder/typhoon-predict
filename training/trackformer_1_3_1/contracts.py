"""Fresh 2026-11-02 track/pressure research contracts; no trainer/model imports.

The archived 2026-10-02 experiment with the same display version is unrelated.
Only cached observations are reused. This module never downloads or fits data.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

EXPERIMENT = "trackformer-1.3.1-track-pressure-research-20261102"
CHANNELS = ["mslp", "surface_pressure", "u10", "v10", "t2m", "sst"] + [
    f"{name}{level}" for level in (850, 700, 500, 300, 200)
    for name in ("u", "v", "temperature", "humidity", "geopotential")]
UNITS = ["hPa", "hPa", "m/s", "m/s", "K", "K"] + [
    unit for _ in range(5) for unit in ("m/s", "m/s", "K", "kg/kg", "m2/s2")]
HISTORY_HOURS = tuple(range(-48, 1, 6))
LEAD_HOURS = tuple(range(6, 121, 6))
LATITUDE = np.linspace(60, 0, 241, dtype=np.float32)
LONGITUDE = np.linspace(100, 180, 321, dtype=np.float32)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def history_times(issue_seconds):
    if int(issue_seconds) != issue_seconds or int(issue_seconds) % 21600:
        raise ValueError("Issue must be an exact six-hour UTC analysis cycle")
    return tuple(int(issue_seconds) + h * 3600 for h in HISTORY_HOURS)


def check_times(issue_seconds, actual_seconds):
    if tuple(actual_seconds) != history_times(issue_seconds):
        raise ValueError("Require exactly nine ordered causal frames ending at issue time")


def effective_mask(values, valid):
    """Return new masks only: raw values are immutable, not clipped or imputed.

    The archived finite-value masks also mark extrapolated underground levels
    valid. Exclude those levels using each frame's physical surface pressure.
    Reanalysis negative specific humidity is unavailable, not negative water.
    """
    if values.shape != (31, 241, 321) or valid.shape != values.shape:
        raise ValueError("Require actual native 0.25-degree 31-channel node grid")
    if valid.dtype != np.bool_:
        raise ValueError("Explicit boolean source masks required")
    mask = valid & np.isfinite(values)
    ps = values[1]
    ps_valid = mask[1] & (ps > 0)
    for j, level in enumerate((850, 700, 500, 300, 200)):
        first = 6 + 5 * j
        mask[first:first + 5] &= (ps_valid & (ps >= level))[None]
        mask[first + 3] &= values[first + 3] >= 0
    return mask


@dataclass(frozen=True)
class Frame:
    seconds: int
    values: np.ndarray
    valid: np.ndarray
    source_sha256: str
    source_path: str


def load_frame(root, ledger_row):
    root = Path(root).resolve()
    path = (root / ledger_row["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Weather frame must exist within the declared cache")
    if path.stat().st_size != ledger_row["size"] or sha256(path) != ledger_row["sha256"]:
        raise ValueError("Weather frame content differs from its source ledger")
    with np.load(path, allow_pickle=False) as z:
        if int(z["seconds"]) != ledger_row["seconds"]:
            raise ValueError("Weather timestamp differs from its receipt")
        np.testing.assert_array_equal(z["latitude"], LATITUDE)
        np.testing.assert_array_equal(z["longitude"], LONGITUDE)
        a, v = z["values"], z["valid"]
    if a.dtype != np.float32:
        raise ValueError("Lossless source FP32 required")
    v = effective_mask(a, v)
    a.setflags(write=False)
    v.setflags(write=False)
    return Frame(int(ledger_row["seconds"]), a, v, ledger_row["sha256"], str(path))


class CausalWeatherReader:
    """Input access is separate from future-label access by construction."""
    def __init__(self, root):
        self.root = Path(root)
        prep = json.loads((self.root / "preparation.json").read_text())
        if prep["channels"] != CHANNELS or prep["native_grid_degrees"] != .25:
            raise ValueError("Weather input contract changed")
        audit = json.loads((self.root / "weather_audit.json").read_text())
        self.index = {r["seconds"]: r for r in audit["rows"]}
        if len(self.index) != len(audit["rows"]):
            raise ValueError("Duplicate source times")

    def history(self, issue_seconds):
        times = history_times(issue_seconds)
        frames = tuple(load_frame(self.root, self.index[t]) for t in times)
        check_times(issue_seconds, [f.seconds for f in frames])
        return frames


def label_frame(reader, issue_seconds, lead_hours):
    """Explicit supervision API. Never call from a model forward/inference path."""
    if lead_hours not in LEAD_HOURS:
        raise ValueError("Require one of the twenty +6 through +120-hour labels")
    seconds = int(issue_seconds) + 3600 * lead_hours
    return load_frame(reader.root, reader.index[seconds])


def sample_pressure(field_hpa, valid, latitude, longitude, center):
    """Read a pressure scalar from the exact exported physical map.

    This is an export-consistency primitive, not a cyclone tracker or forecast.
    It cannot create a vortex or move an existing field onto a preferred track.
    """
    p, m = np.asarray(field_hpa), np.asarray(valid)
    y, x = np.asarray(latitude), np.asarray(longitude)
    if p.shape != m.shape or p.shape != (len(y), len(x)) or m.dtype != np.bool_:
        raise ValueError("Pressure grid and masks disagree")
    if len(y) < 2 or len(x) < 2 or not np.all(np.diff(y) < 0) or not np.all(np.diff(x) > 0):
        raise ValueError("Require north-to-south latitude and unwrapped increasing longitude")
    lat, lon = map(float, center)
    if not np.isfinite([lat, lon]).all() or not y[-1] <= lat <= y[0] or not x[0] <= lon <= x[-1]:
        return None
    iy = int(np.clip(np.searchsorted(-y, -lat, side="right") - 1, 0, len(y) - 2))
    ix = int(np.clip(np.searchsorted(x, lon, side="right") - 1, 0, len(x) - 2))
    fy = (y[iy] - lat) / (y[iy] - y[iy + 1])
    fx = (lon - x[ix]) / (x[ix + 1] - x[ix])
    weights = np.array([[(1-fy)*(1-fx), (1-fy)*fx], [fy*(1-fx), fy*fx]])
    patch = p[iy:iy+2, ix:ix+2]
    supported = m[iy:iy+2, ix:ix+2] & np.isfinite(patch)
    if not np.all(supported | (weights == 0)):
        return None
    return float((np.where(supported, patch, 0) * weights).sum())
