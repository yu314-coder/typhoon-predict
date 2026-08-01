"""Run a local Dolphin forecast using IBTrACS plus JMA source checks only.

The model is the existing TrackFormer v10 track-only checkpoint.  No ERA5,
CDS, GFS, satellite, or JTWC fields are loaded.  JMA best-track data is parsed
and matched to IBTrACS by year, storm number, and UTC time.  If JMA has no
matching current storm, the output says so instead of silently substituting a
different agency's values.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
IBTRACS_CSV = ROOT / "data" / "ibtracs" / "ibtracs.WP.list.v04r01.csv"
JMA_TEXT = ROOT / "data" / "jma" / "bst_all.txt"
JMA_ZIP = ROOT / "data" / "jma" / "bst_all.zip"
JMA_TABLE = ROOT / "data" / "jma" / "table2026.csv"
CHECKPOINT = ROOT / "track_build" / "track_v10_best.pt"
OUTPUT = ROOT / "track_build" / "ibtracs_jma_only"
PAPER = ROOT / "paper"

KIN_COLS = [0, 1, 2, 3, 21, 22, 23, 40, 41, 42, 43]
THERMO_COLS = [4, 5, 6, 7] + list(range(8, 20)) + list(range(24, 40)) + [44, 45, 46, 47]
ENV_COLS = [48, 49, 50, 51, 52, 53]
TARGET_SCALE = torch.tensor([100.0, 100.0, 35.0, 20.0, 50.0] + [50.0] * 12)


def clean_float(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null", "-999", "-9999"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def finite_or_zero(value: Any) -> float:
    number = clean_float(value)
    return number if np.isfinite(number) else 0.0


def normalize_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def jma_year(two_digit_year: int) -> int:
    return 1900 + two_digit_year if two_digit_year >= 51 else 2000 + two_digit_year


def parse_jma_best_track(path: Path) -> list[dict[str, Any]]:
    """Parse JMA's fixed-width text using the published whitespace layout."""

    header: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    header_re = re.compile(r"^66666\s+(\d{4})\b")
    with path.open("r", encoding="ascii", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            header_match = header_re.match(line)
            if header_match:
                identifier = header_match.group(1)
                serial = int(identifier[2:])
                year = jma_year(int(identifier[:2]))
                name = line[35:55].strip() or "UNKNOWN"
                header = {
                    "year": year,
                    "number": serial,
                    "storm_key": f"{year:04d}-{serial:02d}",
                    "name": name,
                }
                continue
            if header is None or not re.match(r"^\d{8}\s", line):
                continue

            fields = line.split()
            if len(fields) < 6:
                continue
            timestamp = fields[0]
            yy = int(timestamp[:2])
            year = jma_year(yy)
            try:
                # datetime.strptime maps 51--68 to 2051--2068. JMA's two-digit
                # year convention is 1951--1999 for values >= 51.
                when = datetime(
                    year,
                    int(timestamp[2:4]),
                    int(timestamp[4:6]),
                    int(timestamp[6:8]),
                    tzinfo=timezone.utc,
                )
            except ValueError:
                continue

            def radius(token: str, position: int) -> float:
                if len(token) < 5 or token[0] == "0":
                    return float("nan")
                try:
                    return float(int(token[1:]))
                except ValueError:
                    return float("nan")

            r50_token = fields[7] if len(fields) > 7 else "00000"
            r50_short = fields[8] if len(fields) > 8 else "0000"
            r30_token = fields[9] if len(fields) > 9 else "00000"
            r30_short = fields[10] if len(fields) > 10 else "0000"
            records.append(
                {
                    "source": "JMA_RSMC_Tokyo",
                    "year": year,
                    "number": header["number"],
                    "storm_key": header["storm_key"],
                    "name": header["name"],
                    "time_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lat": clean_float(fields[3]) / 10.0,
                    "lon": clean_float(fields[4]) / 10.0,
                    "grade": int(fields[2]) if fields[2].isdigit() else 0,
                    "pressure_hpa": clean_float(fields[5]),
                    "vmax_kt": clean_float(fields[6]) if len(fields) > 6 else float("nan"),
                    "r50_long_nm": radius(r50_token, 0),
                    "r50_short_nm": clean_float(r50_short),
                    "r30_long_nm": radius(r30_token, 0),
                    "r30_short_nm": clean_float(r30_short),
                }
            )
    return records


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_ibtracs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as stream:
        reader = csv.DictReader(stream)
        next(reader, None)  # units row
        for row in reader:
            if str(row.get("NAME", "")).strip().upper() != "DOLPHIN":
                continue
            if int(float(row.get("SEASON", "0") or 0)) != 2026:
                continue
            time_text = str(row.get("ISO_TIME", "")).strip()
            try:
                when = datetime.strptime(time_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            lat = clean_float(row.get("LAT"))
            lon = clean_float(row.get("LON"))
            if not np.isfinite(lat) or not np.isfinite(lon):
                continue
            vmax = clean_float(row.get("USA_WIND"))
            pres = clean_float(row.get("USA_PRES"))
            source = "IBTrACS_USA_provisional"
            if not np.isfinite(vmax):
                vmax = clean_float(row.get("WMO_WIND"))
                source = "IBTrACS_WMO"
            if not np.isfinite(pres):
                pres = clean_float(row.get("WMO_PRES"))
            rows.append(
                {
                    "source": source,
                    "sid": str(row.get("SID", "")).strip(),
                    "year": 2026,
                    "number": int(float(row.get("NUMBER", "0") or 0)),
                    "name": "DOLPHIN",
                    "time_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lat": lat,
                    "lon": lon,
                    "vmax_kt": vmax,
                    "pressure_hpa": pres,
                    "dist2land_km": clean_float(row.get("DIST2LAND")),
                    "rmw_nm": clean_float(row.get("USA_RMW")),
                    "roci_nm": clean_float(row.get("USA_ROCI")),
                    "r34_ne_nm": clean_float(row.get("USA_R34_NE")),
                    "r34_se_nm": clean_float(row.get("USA_R34_SE")),
                    "r34_sw_nm": clean_float(row.get("USA_R34_SW")),
                    "r34_nw_nm": clean_float(row.get("USA_R34_NW")),
                    "r50_ne_nm": clean_float(row.get("USA_R50_NE")),
                    "r50_se_nm": clean_float(row.get("USA_R50_SE")),
                    "r50_sw_nm": clean_float(row.get("USA_R50_SW")),
                    "r50_nw_nm": clean_float(row.get("USA_R50_NW")),
                    "r64_ne_nm": clean_float(row.get("USA_R64_NE")),
                    "r64_se_nm": clean_float(row.get("USA_R64_SE")),
                    "r64_sw_nm": clean_float(row.get("USA_R64_SW")),
                    "r64_nw_nm": clean_float(row.get("USA_R64_NW")),
                }
            )
    return sorted(rows, key=lambda row: row["time_utc"])


def load_land_points() -> tuple[np.ndarray, np.ndarray]:
    """Use the project's checked-in land grid only for the IBTrACS distance fallback."""

    terrain = ROOT / "track_build" / "terrain_wp.npz"
    if not terrain.exists():
        return np.array([], dtype="float32"), np.array([], dtype="float32")
    data = np.load(terrain)
    rows, cols = np.where(data["lsm"] > 0.5)
    return data["lat"][rows].astype("float32"), data["lon"][cols].astype("float32")


def distance_to_land(lat: float, lon: float, land_lat: np.ndarray, land_lon: np.ndarray) -> float:
    if not len(land_lat):
        return 9999.0
    dlat = land_lat - lat
    dlon = (land_lon - lon) * math.cos(math.radians(lat))
    return float(np.hypot(dlon * 111.2, dlat * 111.2).min())


def displacement(lat0: float, lon0: float, lat1: float, lon1: float) -> tuple[float, float]:
    dlon = ((lon1 - lon0 + 180.0) % 360.0) - 180.0
    mean_lat = math.radians((lat0 + lat1) / 2.0)
    return dlon * 111.2 * math.cos(mean_lat), (lat1 - lat0) * 111.2


def load_checkpoint(path: Path) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"unsupported checkpoint format: {path}")
    return checkpoint["model"], np.asarray(checkpoint["track_mean"], dtype="float32"), np.asarray(
        checkpoint["track_std"], dtype="float32"
    )


def sinusoidal(length: int, dimension: int) -> torch.Tensor:
    position = torch.arange(length).unsqueeze(1).float()
    divisor = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension))
    encoding = torch.zeros(length, dimension)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding


def encoder(dimension: int, heads: int, ffn: int, depth: int, dropout: float) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        dimension, heads, ffn, dropout, batch_first=True, norm_first=True, activation="gelu"
    )
    return nn.TransformerEncoder(layer, depth)


def decoder(dimension: int, heads: int, ffn: int, depth: int, dropout: float) -> nn.Module:
    layer = nn.TransformerDecoderLayer(
        dimension, heads, ffn, dropout, batch_first=True, norm_first=True, activation="gelu"
    )
    return nn.TransformerDecoder(layer, depth)


class TrackFormerV9(nn.Module):
    """Inference-only copy of the architecture used to produce track_v10_best.pt."""

    def __init__(self, d: int = 256, heads: int = 8, ffn: int = 1024, leads: int = 20, dropout: float = 0.2):
        super().__init__()
        self.leads = leads
        self.kin_proj = nn.Linear(len(KIN_COLS), d)
        self.thermo_proj = nn.Linear(len(THERMO_COLS), d)
        self.env_proj = nn.Linear(len(ENV_COLS), d)
        self.kin_time = nn.Parameter(torch.zeros(1, 9, d))
        self.thermo_time = nn.Parameter(torch.zeros(1, 9, d))
        self.env_time = nn.Parameter(torch.zeros(1, 9, d))
        self.kin_enc = encoder(d, heads, ffn, 4, dropout)
        self.thermo_enc = encoder(d, heads, ffn, 3, dropout)
        self.env_enc = encoder(d, heads, ffn, 2, dropout)
        self.track_dec = decoder(d, heads, ffn, 4, dropout)
        self.int_dec = decoder(d, heads, ffn, 5, dropout)
        self.track_q = nn.Parameter(torch.randn(1, leads, d) * 0.02)
        self.int_q = nn.Parameter(torch.randn(1, leads, d) * 0.02)
        self.register_buffer("qpos", sinusoidal(leads, d))
        self.adapter = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(leads))
        self.rho = nn.Parameter(torch.ones(leads))
        self.gturn = nn.Parameter(torch.zeros(leads))
        self.track_res = nn.Linear(d, 2)
        nn.init.zeros_(self.track_res.weight)
        nn.init.zeros_(self.track_res.bias)
        self.int_state = nn.Linear(d, 15)
        self.int_logscale = nn.Linear(d, 15)

    def forward(self, track: torch.Tensor, vpair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = track.shape[0]
        kin = self.kin_enc(self.kin_proj(track[:, :, KIN_COLS]) + self.kin_time)
        thermo = self.thermo_enc(self.thermo_proj(track[:, :, THERMO_COLS]) + self.thermo_time)
        env = self.env_enc(self.env_proj(track[:, :, ENV_COLS]) + self.env_time)
        track_queries = (self.track_q + self.qpos.unsqueeze(0)).expand(batch, -1, -1)
        h_track = self.track_dec(track_queries, torch.cat([kin, env], dim=1))
        h_track = h_track + self.alpha.view(1, self.leads, 1) * self.adapter(thermo.mean(1).detach()).unsqueeze(1)
        v0, previous = vpair[:, :2], vpair[:, 2:]
        speed_now = v0.norm(dim=1, keepdim=True).clamp(min=1e-3)
        heading_now = torch.atan2(v0[:, 1], v0[:, 0])
        delta_heading = heading_now - torch.atan2(previous[:, 1], previous[:, 0])
        turn = torch.atan2(torch.sin(delta_heading), torch.cos(delta_heading))
        headings = heading_now.unsqueeze(1) + self.gturn.view(1, self.leads) * turn.unsqueeze(1)
        speed = self.rho.view(1, self.leads) * speed_now
        baseline = torch.stack([speed * torch.cos(headings), speed * torch.sin(headings)], dim=-1) / 100.0
        motion = baseline + self.track_res(h_track)
        int_queries = (self.int_q + self.qpos.unsqueeze(0)).expand(batch, -1, -1)
        h_int = self.int_dec(int_queries, torch.cat([thermo, env, kin.detach()], dim=1))
        state = self.int_state(h_int)
        logs = self.int_logscale(h_int).clamp(-5.0, 3.0)
        return torch.cat([motion, state], -1), torch.cat([torch.zeros_like(motion), logs], -1)


def build_track_window(
    records: list[dict[str, Any]],
    track_mean: np.ndarray,
    track_std: np.ndarray,
    land_lat: np.ndarray,
    land_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise RuntimeError("no IBTrACS Dolphin records available")
    base = len(records) - 1
    indices = [max(0, base - 8 + offset) for offset in range(9)]
    sequence = np.zeros((9, 54), dtype="float32")
    previous = -1
    previous_direction: tuple[float, float] | None = None
    issue_time = datetime.fromisoformat(records[base]["time_utc"].replace("Z", "+00:00"))
    issue_day = issue_time.timetuple().tm_yday
    for row_index, index in enumerate(indices):
        current = records[index]
        current_time = datetime.fromisoformat(current["time_utc"].replace("Z", "+00:00"))
        east, north = displacement(records[base]["lat"], records[base]["lon"], current["lat"], current["lon"])
        if previous < 0:
            step_east, step_north = 0.0, 0.0
        else:
            step_east, step_north = displacement(
                records[previous]["lat"], records[previous]["lon"], current["lat"], current["lon"]
            )
        feature = sequence[row_index]
        feature[0:4] = [east, north, step_east, step_north]
        values = [
            current["vmax_kt"],
            current["pressure_hpa"],
            current["rmw_nm"],
            current["roci_nm"],
            current["r34_ne_nm"], current["r34_se_nm"], current["r34_sw_nm"], current["r34_nw_nm"],
            current["r50_ne_nm"], current["r50_se_nm"], current["r50_sw_nm"], current["r50_nw_nm"],
            current["r64_ne_nm"], current["r64_se_nm"], current["r64_sw_nm"], current["r64_nw_nm"],
        ]
        for column, value in enumerate(values):
            feature[4 + column] = finite_or_zero(value)
        feature[24:40] = [float(np.isfinite(clean_float(value))) for value in values]
        phase = 2.0 * math.pi * current_time.timetuple().tm_yday / 365.25
        feature[21:23] = [math.sin(phase), math.cos(phase)]
        feature[23] = (issue_time - current_time).total_seconds() / 3600.0
        speed = math.hypot(step_east, step_north)
        heading_sin, heading_cos = (
            (step_east / speed, step_north / speed) if speed > 1e-3 and previous >= 0 else (0.0, 0.0)
        )
        feature[40:43] = [heading_sin, heading_cos, speed]
        if previous_direction is not None and (heading_sin or heading_cos):
            feature[43] = previous_direction[0] * heading_cos - previous_direction[1] * heading_sin
        if previous >= 0:
            previous_record = records[previous]
            previous_vmax = clean_float(previous_record["vmax_kt"])
            previous_pressure = clean_float(previous_record["pressure_hpa"])
            current_vmax = clean_float(current["vmax_kt"])
            current_pressure = clean_float(current["pressure_hpa"])
            if np.isfinite(previous_vmax) and np.isfinite(current_vmax):
                feature[44] = current_vmax - previous_vmax
                feature[46] = 1.0
            if np.isfinite(previous_pressure) and np.isfinite(current_pressure):
                feature[45] = current_pressure - previous_pressure
                feature[47] = 1.0
        feature[48:54] = [
            current["lat"],
            abs(current["lat"]),
            math.sin(math.radians(current["lon"])),
            math.cos(math.radians(current["lon"])),
            current["dist2land_km"]
            if np.isfinite(clean_float(current["dist2land_km"]))
            else distance_to_land(current["lat"], current["lon"] % 360.0, land_lat, land_lon),
            max(0.0, min(31.0, 30.0 - 0.30 * abs(current["lat"] - 0.5 * 23.44 * math.sin(2.0 * math.pi * (current_time.month - 3) / 12.0)) ** 1.4)),
        ]
        if heading_sin or heading_cos:
            previous_direction = (heading_sin, heading_cos)
        previous = index

    normalized = ((sequence - track_mean) / np.maximum(track_std, 1e-6)).astype("float32")
    velocity_pair = np.concatenate([sequence[-1, 2:4], sequence[-2, 2:4]]).astype("float32")
    return normalized, velocity_pair, sequence


def instantiate_model(checkpoint: Path) -> tuple[TrackFormerV9, np.ndarray, np.ndarray]:
    state, track_mean, track_std = load_checkpoint(checkpoint)
    model = TrackFormerV9()
    model.load_state_dict(state)
    return model, track_mean, track_std


def run_forecast(
    model: TrackFormerV9,
    normalized: np.ndarray,
    velocity_pair: np.ndarray,
    base_lat: float,
    base_lon: float,
    members: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track_tensor = torch.from_numpy(normalized[None])
    velocity_tensor = torch.from_numpy(velocity_pair[None])
    member_states: list[np.ndarray] = []
    for member in range(members):
        torch.manual_seed(4100 + member)
        model.train(member > 0)  # first path is deterministic; remaining paths use MC dropout.
        with torch.no_grad():
            state, _ = model(track_tensor, velocity_tensor)
        member_states.append((state[0] * TARGET_SCALE).numpy())
    states = np.stack(member_states, axis=0)
    member_lats = np.empty((members, 20), dtype="float64")
    member_lons = np.empty((members, 20), dtype="float64")
    for member, path in enumerate(states):
        latitude, longitude = base_lat, base_lon
        for lead, state in enumerate(path):
            east, north = state[:2]
            latitude += float(north) / 111.2
            longitude += float(east) / (111.2 * max(math.cos(math.radians(latitude)), 0.15))
            longitude = ((longitude + 180.0) % 360.0) - 180.0
            member_lats[member, lead] = latitude
            member_lons[member, lead] = longitude
    return states, member_lats, member_lons


def draw_png(path: Path, observed: list[dict[str, Any]], mean_lats: np.ndarray, mean_lons: np.ndarray, member_lats: np.ndarray, member_lons: np.ndarray, states: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    geojson_path = ROOT / "paper" / "ne_110m_admin_0_countries.geojson"
    if geojson_path.exists():
        try:
            features = json.loads(geojson_path.read_text(encoding="utf-8"))["features"]
            for feature in features:
                geometry = feature.get("geometry") or {}
                polygons = geometry.get("coordinates", [])
                if geometry.get("type") == "Polygon":
                    polygons = [polygons]
                for polygon in polygons:
                    for ring in polygon:
                        xs, ys = zip(*ring)
                        axis.fill(xs, ys, color="#eef1f4", zorder=0)
                        axis.plot(xs, ys, color="#98a2b3", linewidth=0.25, zorder=1)
        except (KeyError, json.JSONDecodeError):
            pass
    obs_lon = [point["lon"] for point in observed]
    obs_lat = [point["lat"] for point in observed]
    axis.plot(obs_lon, obs_lat, color="#111827", linewidth=2.2, marker="o", markersize=3.5, label="IBTrACS observed")
    for member in range(1, len(member_lats)):
        axis.plot(np.r_[obs_lon[-1], member_lons[member]], np.r_[obs_lat[-1], member_lats[member]], color="#f59e0b", alpha=0.28, linewidth=0.8)
    axis.plot(np.r_[obs_lon[-1], mean_lons], np.r_[obs_lat[-1], mean_lats], color="#dc2626", linewidth=2.8, marker="o", markersize=3.5, label="TrackFormer v10 MC mean")
    for lead in [3, 7, 11, 15, 19]:
        axis.text(mean_lons[lead] + 0.4, mean_lats[lead] + 0.2, f"+{(lead + 1) * 6}h\n{max(0.0, states[:, lead, 2].mean()):.0f} kt", fontsize=7, color="#991b1b")
    axis.set_xlim(140, 185)
    axis.set_ylim(5, 35)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    axis.set_title("Dolphin: IBTrACS + JMA-source-check forecast\nNo JMA Dolphin record was available at the downloaded issue time")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def draw_html(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Dolphin IBTrACS + JMA-only forecast</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{{margin:0;font-family:system-ui,sans-serif;color:#172033}}#map{{height:78vh;min-height:520px}}header{{padding:14px 18px;border-bottom:1px solid #d7dde5}}h1{{font-size:20px;margin:0 0 5px}}p{{margin:4px 0;font-size:13px}}.notice{{color:#8a2c0a}}</style></head>
<body><header><h1>Dolphin: IBTrACS + JMA-only local forecast</h1>
<p>Issue: {payload['issue_time_utc']} | source rows: IBTrACS {payload['ibtracs_rows']} | JMA Dolphin matches: {payload['jma_dolphin_rows']}</p>
<p class="notice">JMA has no matching Dolphin record in the downloaded official archive/current feed; the forecast input is therefore IBTrACS-only at this issue time. No ERA5, GFS, satellite, or JTWC fields were loaded.</p></header>
<div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const DATA={data}; const map=L.map('map').setView([DATA.base_lat,DATA.base_lon],4);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:8,attribution:'&copy; OpenStreetMap'}}).addTo(map);
function point(p){{return [p.lat,p.lon]}} function popup(p){{return `<b>${{p.label}}</b><br>${{p.time}}<br>${{p.lat.toFixed(2)}}N, ${{p.lon.toFixed(2)}}E<br>vmax ${{p.vmax_kt.toFixed(1)}} kt<br>pressure ${{p.pressure_hpa.toFixed(1)}} hPa`;}}
const observed=DATA.observed.map(p=>point(p)); L.polyline(observed,{{color:'#111827',weight:4}}).addTo(map);
DATA.members.forEach(path=>L.polyline(path.map(point),{{color:'#f59e0b',weight:1,opacity:.35}}).addTo(map));
const mean=L.polyline(DATA.forecast.map(point),{{color:'#dc2626',weight:4}}).addTo(map); mean.addTo(map);
DATA.observed.forEach(p=>L.circleMarker(point(p),{{radius:4,color:'#111827',fillColor:'#fff',fillOpacity:1}}).bindPopup(popup(p)).addTo(map));
DATA.forecast.forEach((p,i)=>L.circleMarker(point(p),{{radius:5,color:'#991b1b',fillColor:'#ef4444',fillOpacity:.9}}).bindTooltip(`+${{p.lead_hours}}h · ${{p.vmax_kt.toFixed(0)}} kt`,{{permanent:i%4===3,direction:'top'}}).bindPopup(popup(p)).addTo(map));
map.fitBounds(L.latLngBounds(observed.concat(DATA.forecast.map(point))),{{padding:[24,24]}});
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main() -> None:
    if not IBTRACS_CSV.exists():
        raise FileNotFoundError(IBTRACS_CSV)
    if not JMA_TEXT.exists():
        if JMA_ZIP.exists():
            import zipfile

            with zipfile.ZipFile(JMA_ZIP) as archive:
                archive.extract("bst_all.txt", JMA_TEXT.parent)
        else:
            raise FileNotFoundError(JMA_TEXT)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    jma_records = parse_jma_best_track(JMA_TEXT)
    jma_fields = [
        "source", "year", "number", "storm_key", "name", "time_utc", "lat", "lon", "grade",
        "pressure_hpa", "vmax_kt", "r50_long_nm", "r50_short_nm", "r30_long_nm", "r30_short_nm",
    ]
    jma_csv = ROOT / "data" / "ibtracs_jma" / "jma_best_track.csv"
    write_csv(jma_csv, jma_records, jma_fields)
    # IBTrACS NUMBER is an internal cross-agency identifier and does not equal
    # JMA's annual storm serial, so the source join uses year + normalized name.
    jma_keys = {(r["year"], normalize_name(r["name"]), r["time_utc"]) for r in jma_records}

    dolphin = parse_ibtracs(IBTRACS_CSV)
    if not dolphin:
        raise RuntimeError("no 2026 DOLPHIN rows found in official IBTrACS WP CSV")
    dolphin_fields = list(dolphin[0].keys())
    dolphin_csv = OUTPUT / "dolphin_ibtracs_records.csv"
    write_csv(dolphin_csv, dolphin, dolphin_fields)

    # Count overlap using the full WP file without retaining its large rows.
    ibtracs_rows = 0
    jma_overlap = 0
    with IBTRACS_CSV.open("r", newline="", encoding="utf-8-sig", errors="replace") as stream:
        reader = csv.DictReader(stream)
        next(reader, None)
        for row in reader:
            try:
                year = int(float(row.get("SEASON", "0") or 0))
                number = int(float(row.get("NUMBER", "0") or 0))
            except ValueError:
                continue
            time_text = str(row.get("ISO_TIME", "")).strip()
            if not time_text:
                continue
            when = datetime.strptime(time_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            ibtracs_rows += 1
            if (year, normalize_name(row.get("NAME")), when.strftime("%Y-%m-%dT%H:%M:%SZ")) in jma_keys:
                jma_overlap += 1

    latest = dolphin[-1]
    jma_dolphin = [r for r in jma_records if "DOLPHIN" in r["name"].upper() and r["year"] == 2026]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ibtracs_file": str(IBTRACS_CSV.relative_to(ROOT)),
        "jma_best_track_file": str(JMA_TEXT.relative_to(ROOT)),
        "ibtracs_wp_rows": ibtracs_rows,
        "jma_best_track_rows": len(jma_records),
        "ibtracs_rows_with_exact_jma_time_match": jma_overlap,
        "dolphin_ibtracs_rows": len(dolphin),
        "dolphin_latest_issue_time_utc": latest["time_utc"],
        "dolphin_latest_position": {"lat": latest["lat"], "lon": latest["lon"]},
        "jma_dolphin_rows": len(jma_dolphin),
        "jma_current_feed_checked": str((ROOT / "data" / "jma" / "typhoon.json").relative_to(ROOT)),
        "model_input_policy": "IBTrACS current Dolphin rows; JMA is parsed/matched for source QC and is never substituted from another storm.",
        "excluded_sources": ["ERA5", "Copernicus CDS", "GFS", "satellite", "JTWC"],
    }
    report_path = OUTPUT / "source_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    six_hour = [r for r in dolphin if datetime.fromisoformat(r["time_utc"].replace("Z", "+00:00")).hour % 6 == 0]
    land_lat, land_lon = load_land_points()
    model, track_mean, track_std = instantiate_model(CHECKPOINT)
    normalized, velocity_pair, raw_sequence = build_track_window(six_hour, track_mean, track_std, land_lat, land_lon)
    states, member_lats, member_lons = run_forecast(model, normalized, velocity_pair, six_hour[-1]["lat"], six_hour[-1]["lon"])
    mean_state = states.mean(axis=0)
    mean_lats = member_lats.mean(axis=0)
    mean_lons = member_lons.mean(axis=0)
    issue_time = datetime.fromisoformat(six_hour[-1]["time_utc"].replace("Z", "+00:00"))
    forecast: list[dict[str, Any]] = []
    for lead in range(20):
        valid = issue_time + timedelta(hours=6 * (lead + 1))
        spread_km = np.hypot(
            member_lats[:, lead] - mean_lats[lead],
            (member_lons[:, lead] - mean_lons[lead]) * math.cos(math.radians(mean_lats[lead])),
        ) * 111.2
        state = mean_state[lead]
        forecast.append(
            {
                "lead_hours": 6 * (lead + 1),
                "valid_time_utc": valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latitude": round(float(mean_lats[lead]), 4),
                "longitude": round(float(mean_lons[lead]), 4),
                "track_spread_km": round(float(spread_km.mean()), 2),
                "vmax_kt": round(float(np.clip(state[2], 0.0, 190.0)), 2),
                "vmax_spread_kt": round(float(states[:, lead, 2].std()), 2),
                "central_pressure_hpa": round(float(np.clip(state[3], 850.0, 1020.0)), 2),
                "pressure_spread_hpa": round(float(states[:, lead, 3].std()), 2),
                "rmw_km": round(float(np.clip(state[4] * 1.852, 8.0, 250.0)), 2),
                "wind_radii_km": np.clip(state[5:17] * 1.852, 0.0, 1000.0).round(2).tolist(),
            }
        )
    payload = {
        "model": "TrackFormer v10 track-only checkpoint with 8 deterministic/MC-dropout paths",
        "issue_time_utc": issue_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_policy": "IBTrACS Dolphin observations only at the current issue time; JMA files used for exact-match source QC.",
        "data_limitations": report["model_input_policy"],
        "ibtracs_rows": len(six_hour),
        "jma_dolphin_rows": len(jma_dolphin),
        "members": int(len(member_lats)),
        "base_lat": six_hour[-1]["lat"],
        "base_lon": six_hour[-1]["lon"],
        "observed": six_hour,
        "forecast": forecast,
        "sources": {
            "ibtracs": "https://www.ncei.noaa.gov/products/international-best-track-archive",
            "jma_best_track": "https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/besttrack.html",
            "jma_position_table": "https://www.data.jma.go.jp/typhoon/position_table/table2026.html",
        },
    }
    json_path = OUTPUT / "dolphin_ibtracs_jma_only_forecast.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    forecast_csv = OUTPUT / "dolphin_ibtracs_jma_only_forecast.csv"
    write_csv(forecast_csv, forecast, list(forecast[0].keys()))

    members_payload = []
    for member in range(len(member_lats)):
        members_payload.append(
            [
                {
                    "lead_hours": 6 * (lead + 1),
                    "lat": float(member_lats[member, lead]),
                    "lon": float(member_lons[member, lead]),
                    "vmax_kt": float(np.clip(states[member, lead, 2], 0.0, 190.0)),
                    "pressure_hpa": float(np.clip(states[member, lead, 3], 850.0, 1020.0)),
                }
                for lead in range(20)
            ]
        )
    map_payload = {
        "issue_time_utc": payload["issue_time_utc"],
        "base_lat": payload["base_lat"],
        "base_lon": payload["base_lon"],
        "ibtracs_rows": payload["ibtracs_rows"],
        "jma_dolphin_rows": payload["jma_dolphin_rows"],
        "observed": [{"label": "IBTrACS observed", "time": r["time_utc"], "lat": r["lat"], "lon": r["lon"], "vmax_kt": finite_or_zero(r["vmax_kt"]), "pressure_hpa": finite_or_zero(r["pressure_hpa"])} for r in six_hour],
        "forecast": [{"label": "TrackFormer v10 mean", "time": r["valid_time_utc"], "lat": r["latitude"], "lon": r["longitude"], "vmax_kt": r["vmax_kt"], "pressure_hpa": r["central_pressure_hpa"], "lead_hours": r["lead_hours"]} for r in forecast],
        "members": members_payload,
    }
    html_path = PAPER / "dolphin_ibtracs_jma_only_world_map.html"
    png_path = PAPER / "dolphin_ibtracs_jma_only_world_map.png"
    draw_html(html_path, map_payload)
    draw_png(png_path, six_hour, mean_lats, mean_lons, member_lats, member_lons, states)

    manifest = {
        "purpose": "Minimum Google Drive bundle for reproducing the local IBTrACS + JMA-only forecast",
        "generated_from": [
            str(json_path.relative_to(ROOT)),
            str(dolphin_csv.relative_to(ROOT)),
            str(jma_csv.relative_to(ROOT)),
            str(report_path.relative_to(ROOT)),
        ],
        "upload_now_inference": [
            "scripts/download_ibtracs_jma.py",
            "scripts/predict_ibtracs_jma_only.py",
            "track_build/track_v10_best.pt",
            "data/ibtracs/ibtracs.WP.list.v04r01.csv",
            "data/jma/bst_all.zip",
            "data/jma/table2026.csv",
            "data/jma/typhoon.json",
            "track_build/terrain_wp.npz",
        ],
        "upload_if_retraining": [
            "track_build/track_windows_v13.npz",
            "models/v10/train_track_v10.py",
            "data/ibtracs/IBTrACS.WP.v04r01.nc",
            "data/ibtracs_jma/jma_best_track.csv",
            "data/ibtracs_jma/source_report.json",
        ],
        "do_not_upload_for_this_path": [
            "track_build/era5/",
            "track_build/dolphin_gfs/",
            "v37_ncep/",
            "steer5_patches.npy",
            "CDS API keys or Colab secrets",
        ],
        "artifacts": [
            str(json_path.relative_to(ROOT)),
            str(forecast_csv.relative_to(ROOT)),
            str(html_path.relative_to(ROOT)),
            str(png_path.relative_to(ROOT)),
        ],
    }
    manifest_path = ROOT / "google_drive_upload_manifest_ibtracs_jma.md"
    lines = [
        "# Google Drive upload list: IBTrACS + JMA only",
        "",
        "The current official IBTrACS release contains Dolphin through `2026-07-28 00:00 UTC`. The downloaded JMA feeds do not contain a matching Dolphin record, so the current forecast is IBTrACS-driven and JMA is a source/QC layer only.",
        "",
        "## Minimum inference bundle",
        *[f"- `{item}`" for item in manifest["upload_now_inference"]],
        "",
        "## Additional files for retraining/fine-tuning",
        *[f"- `{item}`" for item in manifest["upload_if_retraining"]],
        "",
        "## Exclude",
        *[f"- `{item}`" for item in manifest["do_not_upload_for_this_path"]],
        "",
        "## Generated outputs",
        *[f"- `{item}`" for item in manifest["artifacts"]],
        "",
        "Do not upload any API key, token, cookie, or Colab secret.",
    ]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "data" / "ibtracs_jma" / "upload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"forecast": str(json_path), "map": str(html_path), "report": str(report_path), "upload_manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
