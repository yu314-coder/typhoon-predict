#!/usr/bin/env python3
"""v1.2.64 Western Pacific coarse-context/native-pressure forecaster.

Run: python v164_global_regional_train.py 10 3 1
Uses verified old coarse archives plus new native 0.25-degree pressure patches.
Predicts eight coarse fields and one fine pressure field, not 31 fine variables.
The legacy 31-channel definitions below are not the active CLI architecture.
No route-centered vortex is painted. This is not a numerical conservation solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader

VERSION = "1.2.64"
ARCHITECTURE = "global-regional-field-first-v1"
ROOT = Path(__file__).resolve().parent
LEVELS = [850, 700, 500, 300, 200]
CHANNELS = ["mslp", "surface_pressure", "u10", "v10", "t2m", "sst"] + [
    f"{v}{p}" for p in LEVELS for v in ("u", "v", "temperature", "humidity", "geopotential")]
UNITS = ["hPa", "hPa", "m/s", "m/s", "K", "K"] + [
    unit for _ in LEVELS for unit in ("m/s", "m/s", "K", "kg/kg", "m2/s2")]
HISTORY = 9
LEADS = 20
HOUR = 3_600_000_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_checkpoint(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python. Install a CUDA PyTorch wheel before training.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


class Block(nn.Module):
    def __init__(self, width, periodic=False):
        super().__init__()
        self.periodic = periodic
        self.depthwise = nn.Conv2d(width, width, 5, groups=width)
        self.net = nn.Sequential(nn.GroupNorm(8, width), nn.Conv2d(width, width * 4, 1),
                                 nn.GELU(), nn.Conv2d(width * 4, width, 1))
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), 0.01))

    def forward(self, x):
        padded = F.pad(x, (2, 2, 0, 0), mode="circular" if self.periodic else "replicate")
        padded = F.pad(padded, (0, 0, 2, 2), mode="replicate")
        return x + self.scale * self.net(self.depthwise(padded))


class FieldOperator(nn.Module):
    def __init__(self, channels, base, periodic, attention_layers=0, tiny=False):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 6]
        self.stem = nn.Conv2d(channels * 3 + 4, base, 3, padding=1)
        depths = [1, 1, 1, 1] if tiny else [2, 3, 5, 6]
        self.enc = nn.ModuleList([nn.Sequential(*[Block(w, periodic) for _ in range(n)]) for w, n in zip(widths, depths)])
        self.down = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(widths, widths[1:])])
        self.attention = nn.ModuleList([nn.TransformerEncoderLayer(widths[-1], 8,
                                      dim_feedforward=widths[-1] * 4, batch_first=True,
                                      dropout=0.0, norm_first=True) for _ in range(attention_layers)])
        self.up = nn.ModuleList([nn.Conv2d(widths[i + 1], widths[i], 1) for i in (2, 1, 0)])
        self.dec = nn.ModuleList([nn.Sequential(Block(widths[i], periodic), Block(widths[i], periodic)) for i in (2, 1, 0)])
        self.out = nn.Conv2d(base, channels, 1)
        self.object_features = nn.Conv2d(base, 32, 1)
        self.checkpointing = True
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def run(self, module, value):
        return checkpoint(module, value, use_reentrant=False) if self.training and self.checkpointing else module(value)

    def forward(self, state, context, mask, static):
        x = self.stem(torch.cat([state, context, mask, static], 1))
        skips = []
        for i in range(4):
            x = self.run(self.enc[i], x)
            skips.append(x)
            if i < 3:
                x = self.down[i](x)
        if self.attention:
            shape = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            for layer in self.attention:
                tokens = self.run(layer, tokens)
            x = tokens.transpose(1, 2).reshape(shape)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            x = self.run(dec, F.interpolate(up(x), size=skip.shape[-2:], mode="bilinear", align_corners=False) + skip)
        # A learned field tendency, with no fixed per-cell hPa cap or route warp.
        return state + self.out(x), self.object_features(x)


class GlobalRegionalForecaster(nn.Module):
    def __init__(self, global_lat, global_lon, regional_lat, regional_lon, mean, std, tiny=False):
        super().__init__()
        c = len(CHANNELS)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, c, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, c, 1, 1))
        self.register_buffer("regional_lat", torch.tensor(regional_lat, dtype=torch.float32))
        self.register_buffer("regional_lon", torch.tensor(regional_lon, dtype=torch.float32))
        self.global_time = nn.Conv2d(c * HISTORY, c, 1)
        self.regional_time = nn.Conv2d(c * HISTORY, c, 1)
        self.global_operator = FieldOperator(c, 8 if tiny else 80, True, 1 if tiny else 3, tiny)
        self.regional_operator = FieldOperator(c, 8 if tiny else 88, False, 0, tiny)
        self.parent_fusion = nn.Conv2d(c * 2, c, 1)
        self.object_score = nn.Conv2d(32 + 3, 1, 1)
        self.intensity = nn.Sequential(nn.Linear(32 + 3, 64), nn.GELU(), nn.Linear(64, 2))
        y, x = np.meshgrid(regional_lat, regional_lon, indexing="ij")
        grid = np.stack([2 * (x - global_lon[0]) / (global_lon[-1] - global_lon[0]) - 1,
                         2 * (y - global_lat[0]) / (global_lat[-1] - global_lat[0]) - 1], -1)
        self.register_buffer("parent_grid", torch.tensor(grid, dtype=torch.float32)[None])
        self.register_buffer("yy", torch.tensor(y, dtype=torch.float32)[None])
        self.register_buffer("xx", torch.tensor(x, dtype=torch.float32)[None])

    def initial(self, inputs):
        expected = {"global_history", "regional_history", "global_mask", "regional_mask",
                    "global_static", "regional_static", "center", "motion"}
        if set(inputs) != expected:
            raise ValueError("Forward inputs must contain only the declared causal input keys")
        gh, rh = inputs["global_history"], inputs["regional_history"]
        return {"g": gh[:, -1], "r": rh[:, -1],
                "gc": self.global_time(gh.flatten(1, 2)), "rc": self.regional_time(rh.flatten(1, 2)),
                "gm": inputs["global_mask"], "rm": inputs["regional_mask"],
                "gs": inputs["global_static"], "rs": inputs["regional_static"],
                "center": inputs["center"], "motion": inputs["motion"]}

    def step(self, state):
        g, _ = self.global_operator(state["g"], state["gc"], state["gm"], state["gs"])
        parent = F.grid_sample(g, self.parent_grid.expand(len(g), -1, -1, -1), align_corners=True)
        context = self.parent_fusion(torch.cat([state["rc"], parent], 1))
        r, features = self.regional_operator(state["r"], context, state["rm"], state["rs"])
        physical = r.float() * self.std + self.mean
        pressure = physical[:, 0]
        anomaly = pressure - F.avg_pool2d(F.pad(pressure[:, None], (4, 4, 4, 4), mode="replicate"), 9, stride=1)[:, 0]
        idx_u, idx_v = CHANNELS.index("u850"), CHANNELS.index("v850")
        u, v = physical[:, idx_u], physical[:, idx_v]
        evidence = torch.cat([features.float(), anomaly[:, None] / 10, u[:, None] / 30, v[:, None] / 30], 1)
        score = self.object_score(evidence)[:, 0] - anomaly / 10
        lat0, lon0 = state["center"][:, 0, None, None], state["center"][:, 1, None, None]
        prior_lat = lat0 + state["motion"][:, 1, None, None] / 111.2
        prior_lon = lon0 + state["motion"][:, 0, None, None] / (111.2 * lat0.deg2rad().cos().clamp_min(.2))
        distance2 = ((self.yy - prior_lat) * 111.2).square() + ((self.xx - prior_lon) * 111.2 * lat0.deg2rad().cos()).square()
        score = score - distance2 / (2 * 600**2)
        # Select one spatial mode, then refine locally; never average distinct lows.
        peak = score.flatten(1).argmax(1)
        h, w = pressure.shape[-2:]
        iy = torch.arange(h, device=r.device)[None, :, None]
        ix = torch.arange(w, device=r.device)[None, None, :]
        local = ((iy - (peak // w)[:, None, None]).abs() <= 2) & ((ix - (peak % w)[:, None, None]).abs() <= 2)
        pdf = torch.softmax(score.masked_fill(~local, -1e9).flatten(1), 1).reshape_as(score)
        center = torch.stack([(pdf * self.yy).sum((1, 2)), (pdf * self.xx).sum((1, 2))], 1)
        pooled = (evidence * pdf[:, None]).sum((2, 3))
        intensity = self.intensity(pooled)
        grid_pressure = (pressure * pdf).sum((1, 2))
        central_pressure = grid_pressure - F.softplus(intensity[:, 0]) * 5
        surface_speed = torch.sqrt(physical[:, 2].square() + physical[:, 3].square() + 1e-6)
        vmax = (surface_speed * pdf).sum((1, 2)) * 1.943844 + F.softplus(intensity[:, 1]) * 5
        mean_lat = (center[:, 0] + state["center"][:, 0]) / 2
        motion = torch.stack([(center[:, 1] - state["center"][:, 1]) * 111.2 * mean_lat.deg2rad().cos(),
                              (center[:, 0] - state["center"][:, 0]) * 111.2], 1)
        next_state = dict(state, g=g, r=r, center=center, motion=motion)
        return next_state, {"global": g, "regional": r, "center": center, "pressure": central_pressure,
                            "vmax": vmax, "grid_pressure": grid_pressure, "association": pdf}

    def forward(self, inputs, steps=20):
        if not 1 <= steps <= LEADS:
            raise ValueError("steps must be 1..20")
        state = self.initial(inputs)
        outputs = []
        for _ in range(steps):
            state, out = self.step(state)
            outputs.append(out)
        return {k: torch.stack([o[k] for o in outputs], 1) for k in outputs[0]}


def require_full_dataset(m, times=None, rows=None):
    if m.get('dataset_role') != 'full_training' or m.get('synthetic', False):
        raise ValueError('FULL DATA REQUIRED: pilot, synthetic or unclassified datasets cannot be used by the training CLI. Prepare the full 2000-2025 dataset on the Mac first.')
    if times is not None:
        expected = np.arange(np.datetime64('2000-01-01', 'ns').astype('int64'),
                             np.datetime64('2026-01-01', 'ns').astype('int64'), 6*HOUR, dtype='int64')
        if not np.array_equal(times, expected):
            raise ValueError('Full dataset must contain every six-hour field from 2000 through 2025; partial coverage is not full training')
    if rows is not None:
        for split, years in [('train', range(2000, 2022)), ('validation', range(2022, 2024)), ('test', range(2024, 2026))]:
            covered = {int(np.datetime64(int(times[r['issue_index']]), 'ns').astype('datetime64[Y]').astype(int))+1970
                       for r in rows if r['split'] == split}
            if covered != set(years):
                raise ValueError(f'Full dataset lacks sample coverage in {split}: {sorted(set(years)-covered)}')


def load_manifest(path, verify=True, smoke=False, require_full=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"V164 DATA NOT READY: {path}. Requires native global <=1-degree and regional <=0.25-degree multilevel fields. The v163 2.5-degree archive cannot substitute. See v164_dataset_contract.md.")
    m = json.loads(path.read_text())
    if require_full:
        require_full_dataset(m)
    if m.get("schema") != "v164-atmosphere-v1" or m.get("channels") != CHANNELS or m.get("units") != UNITS:
        raise ValueError("Dataset schema/channel order/physical units do not match v164")
    if m.get("future_weather_as_inputs") is not False or m.get("official_forecasts_as_inputs") is not False:
        raise ValueError("Dataset must explicitly disable future weather and official forecast inputs")
    if not smoke and m.get("synthetic", False):
        raise ValueError("Synthetic data is only permitted by the smoke-test API")
    for key in ("global", "regional", "global_static", "regional_static", "time", "samples"):
        item = m["files"][key]
        p = (path.parent / item["path"]).resolve()
        if not p.is_relative_to(path.parent.resolve()) or not p.is_file():
            raise ValueError(f"Missing or outside-dataset file: {key}")
        if verify and digest(p) != item["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {key}")
        item["resolved"] = str(p)
    times = np.load(m["files"]["time"]["resolved"], mmap_mode="r")
    if times.dtype != np.dtype("int64") or np.any(np.diff(times) <= 0) or np.any(times % (6 * HOUR)):
        raise ValueError("Time must be increasing int64 UTC nanoseconds aligned to six hours")
    if m["normalization"]["fit_split"] != "train" or m["normalization"]["end_year"] > 2021:
        raise ValueError("Normalization must be fitted exclusively to training years <=2021")
    mean, std = np.array(m["normalization"]["mean"]), np.array(m["normalization"]["std"])
    if mean.shape != (len(CHANNELS),) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid training normalization")
    for region, max_step in (("global", 1), ("regional", .25)):
        lat, lon = np.array(m[region + "_lat"]), np.array(m[region + "_lon"])
        if min(len(lat), len(lon)) < 2 or np.any(np.diff(lat) >= 0) or np.any(np.diff(lon) <= 0):
            raise ValueError("Coordinates must be north-to-south, west-to-east")
        if not np.allclose(np.diff(lat), lat[1]-lat[0], atol=1e-5) or not np.allclose(np.diff(lon), lon[1]-lon[0], atol=1e-5):
            raise ValueError("This operator requires regular latitude/longitude grids")
        if not smoke and (np.max(-np.diff(lat)) > max_step + 1e-5 or np.max(np.diff(lon)) > max_step + 1e-5):
            raise ValueError(f"{region} resolution too coarse")
        if region == "global" and not smoke and (lat[0] < 89 or lat[-1] > -89 or lon[0] != 0 or lon[-1] < 359):
            raise ValueError("Global parent must cover the globe without duplicated longitude seam")
        array = np.load(m["files"][region]["resolved"], mmap_mode="r")
        static = np.load(m["files"][region + "_static"]["resolved"], mmap_mode="r")
        if array.shape != (len(times), len(CHANNELS), len(lat), len(lon)) or static.shape != (4, len(lat), len(lon)):
            raise ValueError(f"{region} field/static shape mismatch")
        if not np.isfinite(static).all():
            raise ValueError("Non-finite static geography")
    if not (min(m["global_lat"]) <= min(m["regional_lat"]) <= max(m["regional_lat"]) <= max(m["global_lat"]) and
            min(m["global_lon"]) <= min(m["regional_lon"]) <= max(m["regional_lon"]) <= max(m["global_lon"])):
        raise ValueError("Regional grid lies outside global parent coverage")
    rows = [json.loads(line) for line in Path(m["files"]["samples"]["resolved"]).read_text().splitlines() if line.strip()]
    membership = {}
    for row in rows:
        i = row["issue_index"]
        if i < HISTORY - 1 or i + LEADS >= len(times):
            raise ValueError("Sample lacks full causal history or target coverage")
        if np.any(np.diff(times[i-8:i+21]) != 6 * HOUR):
            raise ValueError("Sample crosses a gap in causal history or future target coverage")
        year = int(np.datetime64(int(times[i]), "ns").astype("datetime64[Y]").astype(int)) + 1970
        valid_years = {"train": (2000, 2021), "validation": (2022, 2023), "test": (2024, 2025)}
        low, high = valid_years[row["split"]]
        first_year = int(np.datetime64(int(times[i-8]), "ns").astype("datetime64[Y]").astype(int)) + 1970
        last_year = int(np.datetime64(int(times[i+20]), "ns").astype("datetime64[Y]").astype(int)) + 1970
        if not smoke and not (low <= first_year <= year <= last_year <= high):
            raise ValueError("Sample violates fixed chronological split")
        storm = row.get("storm_id")
        if storm:
            if "1979" in storm or storm.upper() == "TIP":
                raise ValueError("Tip is excluded from training, validation and test")
            if storm in membership and membership[storm] != row["split"]:
                raise ValueError("Storm crosses dataset splits")
            membership[storm] = row["split"]
            if "center" not in row or "target_centers" not in row:
                raise ValueError("Storm samples require observed center and masked target centers")
            centers = np.asarray(row["target_centers"])
            route_mask = np.asarray(row.get("route_mask", [1] * LEADS))
            if centers.shape != (LEADS, 2) or route_mask.shape != (LEADS,) or not np.isfinite(centers).all():
                raise ValueError("Invalid storm target coordinates/mask")
        for field in ("target_pressure_hpa", "target_vmax_kt", "pressure_mask", "vmax_mask"):
            values = np.asarray(row.get(field, [0] * LEADS))
            if values.shape != (LEADS,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid target field: {field}")
        for field in ("route_mask", "pressure_mask", "vmax_mask"):
            values = np.asarray(row.get(field, [0] * LEADS))
            if not np.isin(values, [0, 1]).all():
                raise ValueError("Target masks must be binary")
        center = np.asarray(row.get("center", [20, 140]))
        motion = np.asarray(row.get("motion_km_per_6h", [0, 0]))
        if center.shape != (2,) or motion.shape != (2,) or not np.isfinite([center, motion]).all():
            raise ValueError("Invalid causal center or motion")
        if not (min(m["regional_lat"]) <= center[0] <= max(m["regional_lat"]) and min(m["regional_lon"]) <= center[1] <= max(m["regional_lon"])):
            raise ValueError("Issue center outside regional domain")
    if not smoke and not any(not r.get("storm_id") for r in rows if r["split"] == "train"):
        raise ValueError("Field pretraining requires non-storm atmospheric samples")
    if require_full:
        require_full_dataset(m, times, rows)
    m["rows"] = rows
    m["identity"] = digest(path)
    return m


class AtmosphereDataset(Dataset):
    def __init__(self, manifest, split, training=False):
        self.m, self.training = manifest, training
        self.rows = [r for r in manifest["rows"] if r["split"] == split]
        self.arrays = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.arrays is None:
            self.arrays = {k: np.load(self.m["files"][k]["resolved"], mmap_mode="r") for k in ("global", "regional", "global_static", "regional_static")}
        row = self.rows[index]
        i = row["issue_index"]
        mean = np.array(self.m["normalization"]["mean"], dtype="float32")[:, None, None]
        std = np.array(self.m["normalization"]["std"], dtype="float32")[:, None, None]
        inputs, targets = {}, {}
        drop_sst = self.training and np.random.random() < .15
        for name in ("global", "regional"):
            history = np.array(self.arrays[name][i - 8:i + 1], dtype="float32")
            available = np.isfinite(history)
            # SST can be masked; all atmospheric predictor channels are required.
            if not available[:, [k for k in range(len(CHANNELS)) if k != 5]].all():
                raise ValueError(f"Missing required atmospheric inputs at sample {i}")
            if drop_sst:
                available[:, 5] = False
            history = np.where(available, (history - mean) / std, 0)
            inputs[name + "_history"] = torch.from_numpy(history)
            inputs[name + "_mask"] = torch.from_numpy(available.all(0).astype("float32"))
            inputs[name + "_static"] = torch.from_numpy(np.array(self.arrays[name + "_static"], dtype="float32"))
            future = np.array(self.arrays[name][i + 1:i + 21], dtype="float32")
            mask = np.isfinite(future)
            if not mask[:, [k for k in range(len(CHANNELS)) if k != 5]].all():
                raise ValueError(f"Missing required atmospheric verification fields at sample {i}")
            targets[name] = torch.from_numpy(np.where(mask, (future - mean) / std, 0))
            targets[name + "_mask"] = torch.from_numpy(mask.astype("float32"))
        inputs["center"] = torch.tensor(row.get("center", [20, 140]), dtype=torch.float32)
        inputs["motion"] = torch.tensor(row.get("motion_km_per_6h", [0, 0]), dtype=torch.float32)
        storm = bool(row.get("storm_id"))
        targets["center"] = torch.tensor(row.get("target_centers", [[0, 0]] * 20), dtype=torch.float32)
        targets["pressure"] = torch.tensor(row.get("target_pressure_hpa", [0] * 20), dtype=torch.float32)
        targets["vmax"] = torch.tensor(row.get("target_vmax_kt", [0] * 20), dtype=torch.float32)
        targets["storm_mask"] = torch.tensor(row.get("route_mask", [float(storm)] * 20), dtype=torch.float32)
        targets["pressure_mask"] = torch.tensor(row.get("pressure_mask", [0] * 20), dtype=torch.float32)
        targets["vmax_mask"] = torch.tensor(row.get("vmax_mask", [0] * 20), dtype=torch.float32)
        return inputs, targets


def masked_mean(value, mask):
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1)


def objective(output, target, step, manifest, joint):
    terms = {}
    for region in ("global", "regional"):
        p, t = output[region].float(), target[region][:, step]
        mask = target[region + "_mask"][:, step]
        lat = torch.tensor(manifest[region + "_lat"], device=p.device, dtype=torch.float32).deg2rad().cos()[None, None, :, None]
        if region == "regional" and "regional_lat_weight" in target:
            lat = target["regional_lat_weight"][:, step]
        terms[region] = masked_mean(F.smooth_l1_loss(p, t, reduction="none"), mask * lat)
        dx = (p[..., 1:] - p[..., :-1]) - (t[..., 1:] - t[..., :-1])
        dy = (p[..., 1:, :] - p[..., :-1, :]) - (t[..., 1:, :] - t[..., :-1, :])
        terms[region + "_gradient"] = .5 * (masked_mean(dx.abs(), mask[..., 1:] * mask[..., :-1]) + masked_mean(dy.abs(), mask[..., 1:, :] * mask[..., :-1, :]))
    delta = output["center"] - target["center"][:, step]
    scale = torch.stack([torch.ones_like(delta[:, 0]), target["center"][:, step, 0].deg2rad().cos()], 1) * 111.2 / 100
    terms["route"] = masked_mean(F.smooth_l1_loss(delta * scale, torch.zeros_like(delta), reduction="none"), target["storm_mask"][:, step, None])
    terms["pressure"] = masked_mean((output["pressure"] - target["pressure"][:, step]).abs() / 20, target["pressure_mask"][:, step])
    terms["vmax"] = masked_mean((output["vmax"] - target["vmax"][:, step]).abs() / 30, target["vmax_mask"][:, step])
    value = terms["global"] + terms["regional"] + .2 * (terms["global_gradient"] + terms["regional_gradient"])
    if joint:
        value = value + .2 * terms["route"] + .1 * (terms["pressure"] + terms["vmax"])
    return value, terms


def make_model(m, tiny=False):
    return GlobalRegionalForecaster(*(m[k] for k in ("global_lat", "global_lon", "regional_lat", "regional_lon")),
                                    m["normalization"]["mean"], m["normalization"]["std"], tiny)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int, nargs="?", default=10, help="Effective examples per optimizer update; physical microbatch is 1")
    parser.add_argument("workers", type=int, nargs="?", default=3)
    parser.add_argument("amp", type=int, nargs="?", default=1, choices=[0, 1])
    persistent = Path("/content/drive/MyDrive/typhoon_predict") if Path("/content").is_dir() else ROOT
    default_manifest = persistent / "data/v164/manifest.json"
    parser.add_argument("--manifest", type=Path, default=Path(os.getenv("V164_MANIFEST", default_manifest)))
    parser.add_argument("--output", type=Path, default=os.getenv("V164_OUTPUT_DIR"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--model-count", action="store_true")
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or args.epochs < 1:
        parser.error("batch/epochs must be positive; workers must be nonnegative")
    if args.model_count:
        m = {"global_lat": np.linspace(90, -90, 181), "global_lon": np.arange(360),
             "regional_lat": np.linspace(60, 0, 241), "regional_lon": np.linspace(100, 180, 321),
             "normalization": {"mean": [0] * len(CHANNELS), "std": [1] * len(CHANNELS)}}
        model = make_model(m)
        print(json.dumps({"version": VERSION, "architecture": ARCHITECTURE, "parameters": sum(p.numel() for p in model.parameters()), "channels": len(CHANNELS)}))
        return
    m = load_manifest(args.manifest, require_full=True)
    if args.output is None:
        args.output = persistent / "training/v164"
    print(json.dumps({"version": VERSION, "dataset_role": m.get("dataset_role", "full"),
                      "pilot_only": False, "samples": len(m["rows"]), "output": str(args.output)}), flush=True)
    if args.preflight:
        for split in ("train", "validation", "test"):
            dataset = AtmosphereDataset(m, split)
            if not len(dataset):
                raise ValueError(f"Empty {split} split")
            dataset[0]
        print("PASS dataset integrity and first-sample checks. No training started.")
        return
    if Path("/content").is_dir() and not args.output.resolve().is_relative_to(Path("/content/drive/MyDrive/typhoon_predict")):
        raise ValueError("Colab outputs must persist in MyDrive/typhoon_predict")
    if Path("/content").is_dir() and not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Mount Drive before training; refusing temporary checkpoints")
    device = device_for(args.device)
    random.seed(164); np.random.seed(164); torch.manual_seed(164)
    model = make_model(m).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(args.amp))
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / "training.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Output locked: {lock}. Verify no other runtime is training before removing a stale lock.") from exc
    os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname()}).encode()); os.close(fd)
    try:
        train(model, optimizer, scaler, m, args, device)
    finally:
        lock.unlink(missing_ok=True)


def train(model, optimizer, scaler, m, args, device):
    last, best = args.output / "last.pt", args.output / "best.pt"
    start, history, best_value = 1, [], float("inf")
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["architecture"] != ARCHITECTURE or saved["dataset_sha256"] != m["identity"]:
            raise ValueError("Resume architecture/dataset mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        start, history, best_value = saved["epoch"] + 1, saved["history"], saved["best_value"]
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"]); torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    atomic_json(args.output / "manifest.json", {"version": VERSION, "architecture": ARCHITECTURE,
                "parameters": sum(p.numel() for p in model.parameters()), "dataset_sha256": m["identity"],
                "script_sha256": digest(__file__), "future_weather_as_inputs": False,
                "official_forecasts_as_inputs": False, "ensemble": False,
                "resume_policy": "last completed epoch with optimizer/scaler/RNG",
                "physical_microbatch": 1, "effective_batch": args.batch})
    workers = min(args.workers, max(0, (os.cpu_count() or 1) - 1), 2)
    for epoch in range(start, args.epochs + 1):
        horizon = [2, 4, 8, 12, 20][min(epoch - 1, 4)]
        joint = epoch > 2
        epoch_start = time.time()
        metrics = {}
        for split in ("train", "validation"):
            training = split == "train"
            model.train(training)
            dataset = AtmosphereDataset(m, split, training)
            if not len(dataset):
                raise ValueError(f"Empty {split} split")
            loader = DataLoader(dataset, batch_size=1, shuffle=training, num_workers=workers,
                                **({"prefetch_factor": 1} if workers else {}))
            optimizer.zero_grad(set_to_none=True)
            sums = []
            for index, (inputs, targets) in enumerate(loader, 1):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                steps = horizon if training else 20
                loss_block, total = 0, 0.0
                with torch.set_grad_enabled(training):
                    state = model.initial(inputs)
                    for step in range(steps):
                        with torch.autocast(device_type="cuda", dtype=torch.float16) if scaler.is_enabled() else nullcontext():
                            state, out = model.step(state)
                        # Transfer one target frame at a time to bound VRAM.
                        frame = {k: v[:, step:step+1].to(device) for k, v in targets.items()}
                        loss, _ = objective(out, frame, 0, m, joint)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite loss epoch={epoch} sample={index} step={step}; checkpoint not advanced")
                        total += float(loss.detach()) / steps
                        loss_block = loss_block + loss / (steps * args.batch)
                        if training and ((step + 1) % 2 == 0 or step + 1 == steps):
                            scaler.scale(loss_block).backward()
                            state = {k: v.detach() for k, v in state.items()}
                            loss_block = 0
                    if training and (index % args.batch == 0 or index == len(loader)):
                        scaler.unscale_(optimizer)
                        remainder = index % args.batch
                        if index == len(loader) and remainder:
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad.mul_(args.batch / remainder)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if not torch.isfinite(norm):
                            raise RuntimeError("Non-finite gradients; checkpoint not advanced. Retry last epoch with AMP=0.")
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                sums.append(total)
                if index == 1 or index % 10 == 0:
                    print(f"{split} epoch={epoch}/{args.epochs} sample={index}/{len(loader)} horizon={steps} loss={total:.5f} elapsed_s={time.time()-epoch_start:.0f}", flush=True)
            metrics[split] = float(np.mean(sums))
        # Selection only after reaching the full horizon, on a stable joint objective.
        promote = epoch >= 5 and metrics["validation"] < best_value
        if promote:
            best_value = metrics["validation"]
        history.append({"epoch": epoch, "horizon": horizon, "joint": joint, **metrics})
        payload = {"version": VERSION, "architecture": ARCHITECTURE, "dataset_sha256": m["identity"],
                   "data_contract": {k: m[k] for k in ("channels", "units", "global_lat", "global_lon", "regional_lat", "regional_lon", "normalization")},
                   "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "history": history, "best_value": best_value,
                   "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        atomic_checkpoint(last, payload)
        if promote:
            atomic_checkpoint(best, payload)
        atomic_json(args.output / "history.json", history)
        print(f"Saved completed epoch {epoch}: {last}; promoted_best={promote}", flush=True)
    print("Training completed. Checkpoints and history are persistent. Test set was not used for selection.", flush=True)


# BEGIN REUSE IMPLEMENTATION
"""Embedded in the standalone v164 trainer: coarse archive plus native MSLP patches."""
REUSE_ARCH = 'wp-coarse8-native-pressure1-v164'
REUSE_CHANNELS = ['mslp', 'hgt500', 'uwnd850', 'vwnd850', 'uwnd500', 'vwnd500', 'uwnd200', 'vwnd200']


def load_reuse(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Combined dataset missing. Run the updated Drive setup cell first: '+str(path))
    m = json.loads(path.read_text())
    if m.get('schema') != 'v164-reuse-pressure-v1' or m.get('pilot', True):
        raise ValueError('Combined non-pilot archive required; no pilot fallback')
    for name, entry in m['files'].items():
        p = path.parent/entry['path']
        print('Checking dataset file:', name, flush=True)
        if digest(p) != entry['sha256']:
            raise ValueError('Dataset checksum mismatch: '+name)
        entry['resolved'] = str(p)
    with np.load(m['files']['basin']['resolved'], allow_pickle=False) as z:
        if z['channels'].tolist() != REUSE_CHANNELS[1:]:
            raise ValueError('Coarse channel order mismatch')
        times, lat, lon = z['time'], z['lat'], z['lon']
        q = z['q']
        scale, offset = z['scale'], z['offset']
    if q.shape != (len(times),7,25,33) or not np.all(np.diff(times) == 6*HOUR):
        raise ValueError('Coarse grid/time mismatch')
    qpath = path.parent/'basin_q_verified.npy'
    # Refresh the compact worker mmap from the checksum-verified source every preflight.
    temporary = qpath.with_suffix('.pending.npy')
    np.save(temporary, q)
    os.replace(temporary, qpath)
    slp = np.load(m['files']['slp']['resolved'], mmap_mode='r')
    if slp.shape != (len(times),25,33) or not np.isfinite(slp).all():
        raise ValueError('Coarse pressure coverage invalid')
    with np.load(m['files']['track']['resolved'], allow_pickle=False) as z:
        track = {k:z[k] for k in ['base_time','storm_id','base_lat','base_lon','target','target_mask']}
        issue_intensity = z['track'][:,-1,4:6]*z['track_std'][4:6]+z['track_mean'][4:6]
    plan = json.loads(Path(m['files']['plan']['resolved']).read_text())
    if len(plan['rows']) != 1000 or plan['shape'] != [1000,29,121,121] or plan['lead_hours'] != list(range(-48,121,6)):
        raise ValueError('Expected the verified 1000-window native pressure package, not the six-sample pilot')
    pressure = np.load(m['files']['pressure']['resolved'], mmap_mode='r')
    if pressure.shape != (1000,29,121,121):
        raise ValueError('Native pressure shape mismatch')
    patches = {}
    for pi,row in enumerate(plan['rows']):
        ti = row['track_archive_row']
        if int(track['base_time'][ti]) != row['issue_ns'] or str(track['storm_id'][ti]) != row['storm_id']:
            raise ValueError('Native patch/track identity mismatch')
        patches[ti] = pi
    first_year = {}
    years = track['base_time'].astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970
    for sid,y in zip(track['storm_id'],years):
        first_year[str(sid)] = min(first_year.get(str(sid),9999),int(y))
    lookup = {(str(s),int(t)):i for i,(s,t) in enumerate(zip(track['storm_id'],track['base_time']))}
    rows = []
    for i,ns in enumerate(track['base_time']):
        sid = str(track['storm_id'][i])
        y = first_year[sid]
        split = 'train' if 2000 <= y <= 2021 else 'validation' if 2022 <= y <= 2023 else 'test' if 2024 <= y <= 2025 else None
        if split is None:
            continue
        low,high = {'train':(2000,2021),'validation':(2022,2023),'test':(2024,2025)}[split]
        bounds = np.array([ns-48*HOUR,ns+120*HOUR],dtype='int64').astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970
        ai = int(np.searchsorted(times,ns))
        center = [float(track['base_lat'][i]),float(track['base_lon'][i])%360]
        if bounds[0] < low or bounds[1] > high or ai < 8 or ai+20 >= len(times) or times[ai] != ns:
            continue
        if not np.isfinite(center).all() or not (0 < center[0] < 60 and 100 < center[1] < 180):
            continue
        previous = lookup.get((sid,int(ns-6*HOUR)))
        motion = [0.,0.]
        if previous is not None:
            old = [float(track['base_lat'][previous]),float(track['base_lon'][previous])%360]
            if np.isfinite(old).all():
                motion = [(center[1]-old[1])*111.2*np.cos(np.deg2rad(center[0])),(center[0]-old[0])*111.2]
        future = [lookup.get((sid,int(ns+j*6*HOUR))) for j in range(1,21)]
        target_centers = [[float(track['base_lat'][j]),float(track['base_lon'][j])%360] if j is not None else [0.,0.] for j in future]
        rm = [float(j is not None and np.isfinite(target_centers[k]).all()) for k,j in enumerate(future)]
        target_centers = np.nan_to_num(target_centers).tolist()
        wind,p = track['target'][i,:,2],track['target'][i,:,3]
        wm = track['target_mask'][i,:,2].astype(bool)&np.isfinite(wind)&(wind>=0)&(wind<250)
        pm = track['target_mask'][i,:,3].astype(bool)&np.isfinite(p)&(p>800)&(p<1100)
        pi = patches.get(i,-1)
        if pi >= 0 and plan['rows'][pi]['split'] != split:
            raise ValueError('Patch split mismatch')
        issue = issue_intensity[i]
        iv = [bool(np.isfinite(issue[0]) and 0<=issue[0]<250),bool(np.isfinite(issue[1]) and 800<issue[1]<1100)]
        rows.append({'split':split,'storm_id':sid,'atlas':ai,'patch':pi,'center':center,'motion':motion,
                     'issue_intensity':[float(issue[k]) if iv[k] else 0. for k in range(2)],'issue_mask':iv,
                     'target_centers':target_centers,'route_mask':rm,
                     'pressure':np.where(pm,p,0).tolist(),'pressure_mask':pm.tolist(),
                     'vmax':np.where(wm,wind,0).tolist(),'vmax_mask':wm.tolist()})
    counts = {s:sum(r['split']==s for r in rows) for s in ['train','validation','test']}
    patch_counts = {s:sum(r['split']==s and r['patch']>=0 for r in rows) for s in counts}
    if counts['train'] < 10000 or patch_counts != {'train':800,'validation':100,'test':100}:
        raise ValueError(f'Incomplete combined coverage: {counts}, native patches {patch_counts}')
    train_times = np.flatnonzero((times.astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970>=2000)&
                                (times.astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970<=2021))
    sums,squares = np.zeros(8),np.zeros(8)
    n = 0
    for start in range(0,len(train_times),256):
        ii = train_times[start:start+256]
        b = np.concatenate([np.asarray(slp[ii],dtype='float32')[:,None],q[ii].astype('float32')*scale[None,:,None,None]+offset[None,:,None,None]],1).astype('float64')
        if not np.isfinite(b).all():
            raise ValueError('Nonfinite coarse training fields')
        sums += b.sum((0,2,3)); squares += (b*b).sum((0,2,3)); n += len(ii)*25*33
    mean = sums/n
    std = np.maximum(np.sqrt(np.maximum(squares/n-mean*mean,0)),1e-4)
    m.update(rows=rows, plan=plan, qpath=str(qpath), scale=scale.tolist(), offset=offset.tolist(),
             channels=REUSE_CHANNELS,units=['hPa','m','m/s','m/s','m/s','m/s','m/s','m/s'],
             global_lat=lat.tolist(),global_lon=lon.tolist(),regional_lat=list(np.linspace(15,-15,121)),
             regional_lon=list(np.linspace(-15,15,121)),normalization={'mean':mean.tolist(),'std':std.tolist(),'fit_split':'train','end_year':2021},
             identity=digest(path),counts=counts,patch_counts=patch_counts)
    print(json.dumps({'all_archive_windows':counts,'native_pressure_windows':patch_counts,'architecture':REUSE_ARCH}),flush=True)
    return m


class ReuseDataset(Dataset):
    def __init__(self,m,split,training=False):
        self.m = m
        self.rows = [r for r in m['rows'] if r['split']==split]
        self.arrays = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        m,row = self.m,self.rows[i]
        if self.arrays is None:
            self.arrays = {k:np.load(m['files'][k]['resolved'],mmap_mode='r') for k in ['slp','pressure']}
            self.arrays['q'] = np.load(m['qpath'],mmap_mode='r')
            with np.load(m['files']['geography']['resolved'],allow_pickle=False) as geo:
                self.arrays['land'] = geo['land_fraction']
                self.arrays['height'] = geo['elevation_m']/8000
        ai = row['atlas']
        ii = slice(ai-8,ai+21)
        coarse = np.concatenate([np.array(self.arrays['slp'][ii],dtype='float32')[:,None],
              np.array(self.arrays['q'][ii],dtype='float32')*np.array(m['scale'],dtype='float32')[None,:,None,None]+np.array(m['offset'],dtype='float32')[None,:,None,None]],1)
        mean,std = np.array(m['normalization']['mean'],dtype='float32'),np.array(m['normalization']['std'],dtype='float32')
        coarse = (coarse-mean[None,:,None,None])/std[None,:,None,None]
        pi = row['patch']
        anchor = m['plan']['rows'][pi] if pi>=0 else {'center_lat':round(row['center'][0]*4)/4,'center_lon':round(row['center'][1]*4)/4}
        lat = anchor['center_lat']+np.linspace(15,-15,121,dtype='float32')
        lon = anchor['center_lon']+np.linspace(-15,15,121,dtype='float32')
        yy,xx = np.meshgrid(lat,lon,indexing='ij')
        y = np.rint((90-yy)*4).astype(int); x = np.rint(xx*4).astype(int)%1440
        rs = np.stack([yy/90,xx/180-1,self.arrays['land'][y,x],self.arrays['height'][y,x]]).astype('float32')
        cy,cx = np.meshgrid(m['global_lat'],m['global_lon'],indexing='ij')
        iy = np.rint((90-cy)*4).astype(int); ix = np.rint(cx*4).astype(int)%1440
        gs = np.stack([cy/90,cx/180-1,self.arrays['land'][iy,ix],self.arrays['height'][iy,ix]]).astype('float32')
        detail = np.zeros((29,1,121,121),dtype='float32')
        if pi>=0:
            detail[:,0] = (np.array(self.arrays['pressure'][pi],dtype='float32')-mean[0])/std[0]
            if not np.isfinite(detail).all():
                raise ValueError('Native pressure contains nonfinite fields')
        inputs = {'global_history':coarse[:9], 'regional_history':detail[:9],
                  'global_static':gs,'regional_static':rs,'detail_available':np.array([pi>=0],dtype='float32'),
                  'center':np.array(row['center'],dtype='float32'),'motion':np.array(row['motion'],dtype='float32'),
                  'issue_intensity':np.array(row['issue_intensity'],dtype='float32'),'issue_mask':np.array(row['issue_mask'],dtype='float32')}
        targets = {'global':coarse[9:],'global_mask':np.ones_like(coarse[9:]),
                   'regional_lat_weight':np.broadcast_to(np.cos(np.deg2rad(lat))[None,None,:,None],(20,1,121,1)).astype('float32'),
                   'regional':detail[9:],'regional_mask':np.full_like(detail[9:],float(pi>=0)),
                   'center':np.array(row['target_centers'],dtype='float32'),
                   'storm_mask':np.array(row['route_mask'],dtype='float32')}
        for k in ['pressure','pressure_mask','vmax','vmax_mask']:
            targets[k] = np.array(row[k],dtype='float32')
        return {k:torch.from_numpy(v.copy()) for k,v in inputs.items()},{k:torch.from_numpy(v.copy()) for k,v in targets.items()}


class ReuseForecaster(nn.Module):
    def __init__(self,m,tiny=False):
        super().__init__()
        self.global_time = nn.Conv2d(8*9,8,1)
        self.regional_time = nn.Conv2d(9,1,1)
        self.global_operator = FieldOperator(8,8 if tiny else 80,False,1 if tiny else 3,tiny)
        self.regional_operator = FieldOperator(1,8 if tiny else 88,False,0,tiny)
        self.parent_fusion = nn.Conv2d(1+8+1,1,1)
        self.object_score = nn.Conv2d(32+3,1,1)
        self.intensity = nn.Sequential(nn.Linear(32+3+4,64),nn.GELU(),nn.Linear(64,2))
        self.register_buffer('mean',torch.tensor(m['normalization']['mean'],dtype=torch.float32)[None,:,None,None])
        self.register_buffer('std',torch.tensor(m['normalization']['std'],dtype=torch.float32)[None,:,None,None])
        yy,xx = np.meshgrid(m['global_lat'],m['global_lon'],indexing='ij')
        self.register_buffer('yy',torch.tensor(yy,dtype=torch.float32)[None])
        self.register_buffer('xx',torch.tensor(xx,dtype=torch.float32)[None])

    def initial(self,inputs):
        required = {'global_history','regional_history','global_static','regional_static','detail_available','center','motion','issue_intensity','issue_mask'}
        if set(inputs)!=required:
            raise ValueError('Only declared causal inputs are allowed')
        gh,rh = inputs['global_history'],inputs['regional_history']
        rs = inputs['regional_static']
        yy,xx = rs[:,0]*90,(rs[:,1]+1)*180
        grid = torch.stack([(xx-100)/40-1,1-yy/30],-1)
        return {'g':gh[:,-1],'r':rh[:,-1],'gc':self.global_time(gh.flatten(1,2)),
                'rc':self.regional_time(rh.flatten(1,2)),'gs':inputs['global_static'],'rs':rs,
                'grid':grid,'available':inputs['detail_available'],'center':inputs['center'],
                'motion':inputs['motion'],'issue':inputs['issue_intensity'],'issue_mask':inputs['issue_mask']}

    def step(self,s):
        g,features = self.global_operator(s['g'],s['gc'],torch.ones_like(s['g']),s['gs'])
        valid = (s['grid'].abs()<=1).all(-1)[:,None].to(g.dtype)
        parent = F.grid_sample(g,s['grid'],align_corners=True,padding_mode='zeros')*valid
        r = s['r']
        if bool(s['available'].any()):
            context = self.parent_fusion(torch.cat([s['rc'],parent,valid],1))
            r,_ = self.regional_operator(r,context,torch.ones_like(r),s['rs'])
        physical = g.float()*self.std+self.mean
        p = physical[:,0]
        anomaly = p-F.avg_pool2d(F.pad(p[:,None],(2,2,2,2),mode='replicate'),5,1)[:,0]
        evidence = torch.cat([features.float(),anomaly[:,None]/10,physical[:,2:4]/30],1)
        lat0,lon0 = s['center'][:,0,None,None],s['center'][:,1,None,None]
        plat = lat0+s['motion'][:,1,None,None]/111.2
        plon = lon0+s['motion'][:,0,None,None]/(111.2*lat0.deg2rad().cos().clamp_min(.2))
        d2 = ((self.yy-plat)*111.2).square()+((self.xx-plon)*111.2*lat0.deg2rad().cos()).square()
        score = self.object_score(evidence)[:,0]-anomaly/10-d2/(2*600**2)
        peak = score.flatten(1).argmax(1)
        iy,ix = torch.arange(25,device=g.device)[None,:,None],torch.arange(33,device=g.device)[None,None,:]
        local = ((iy-(peak//33)[:,None,None]).abs()<=1)&((ix-(peak%33)[:,None,None]).abs()<=1)
        pdf = torch.softmax(score.masked_fill(~local,-1e9).flatten(1),1).reshape_as(score)
        center = torch.stack([(pdf*self.yy).sum((1,2)),(pdf*self.xx).sum((1,2))],1)
        pooled = (evidence*pdf[:,None]).sum((2,3))
        scalar = torch.cat([s['issue']/torch.tensor([100.,1000.],device=g.device),s['issue_mask']],1)
        intensity = self.intensity(torch.cat([pooled,scalar],1))
        grid_p = (p*pdf).sum((1,2))
        dy,dx = s['rs'][:,0]*90,(s['rs'][:,1]+1)*180
        sample = torch.stack([2*(center[:,1]-dx[:,0,0])/30-1,2*(dy[:,0,0]-center[:,0])/30-1],-1)[:,None,None]
        inside = (sample.abs()<=1).all((1,2,3)) & (s['available'][:,0]>0)
        detailed_p = F.grid_sample(r.float()*self.std[:,:1]+self.mean[:,:1],sample,align_corners=True)[:,0,0,0]
        grid_p = torch.where(inside,detailed_p,grid_p)
        pressure = grid_p-F.softplus(intensity[:,0])*5
        vmax = s['issue'][:,0]*s['issue_mask'][:,0]+intensity[:,1]*10
        meanlat = (center[:,0]+s['center'][:,0])/2
        motion = torch.stack([(center[:,1]-s['center'][:,1])*111.2*meanlat.deg2rad().cos(),(center[:,0]-s['center'][:,0])*111.2],1)
        return dict(s,g=g,r=r,center=center,motion=motion),{'global':g,'regional':r,'center':center,'pressure':pressure,'vmax':vmax,'grid_pressure':grid_p}

    def forward(self,inputs,steps=20):
        s = self.initial(inputs)
        out = []
        for _ in range(steps):
            s,o = self.step(s); out.append(o)
        return {k:torch.stack([o[k] for o in out],1) for k in out[0]}


def reuse_main():
    global ARCHITECTURE, AtmosphereDataset
    ARCHITECTURE,AtmosphereDataset = REUSE_ARCH,ReuseDataset
    p = argparse.ArgumentParser(description='v1.2.64 coarse weather + native pressure training; no pilot fallback')
    p.add_argument('batch',type=int,nargs='?',default=10)
    p.add_argument('workers',type=int,nargs='?',default=3)
    p.add_argument('amp',type=int,nargs='?',default=1,choices=[0,1])
    p.add_argument('--manifest',type=Path,default=ROOT/'data/v164_reuse/manifest.json')
    persistent = Path('/content/drive/MyDrive/typhoon_predict') if Path('/content').is_dir() else ROOT
    p.add_argument('--output',type=Path,default=persistent/'training/v164_reuse_pressure')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--preflight',action='store_true')
    args = p.parse_args()
    if args.batch<1 or args.workers<0 or args.epochs<1:
        p.error('Invalid batch/workers/epochs')
    m = load_reuse(args.manifest)
    for split in ['train','validation','test']:
        d = ReuseDataset(m,split)
        d[0]
        d[next(i for i,r in enumerate(d.rows) if r['patch']>=0)]
    if args.preflight:
        print('PASS combined non-pilot archives and native pressure data. No training started.',flush=True)
        return
    if Path('/content').is_dir() and not args.output.resolve().is_relative_to(persistent.resolve()):
        raise ValueError('Colab checkpoints must persist in typhoon_predict on Drive')
    device = device_for(args.device)
    random.seed(164); np.random.seed(164); torch.manual_seed(164)
    model = ReuseForecaster(m).to(device)
    print(json.dumps({'version':VERSION,'architecture':REUSE_ARCH,'parameters':sum(x.numel() for x in model.parameters()),'device':str(device),'windows':m['counts'],'native_windows':m['patch_counts'],'output':str(args.output)}),flush=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and bool(args.amp))
    args.output.mkdir(parents=True,exist_ok=True)
    lock = args.output/'training.lock'
    fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({'pid':os.getpid(),'host':socket.gethostname()}).encode()); os.close(fd)
    try:
        train(model,optimizer,scaler,m,args,device)
    finally:
        lock.unlink(missing_ok=True)

if __name__ == "__main__":
    reuse_main()
