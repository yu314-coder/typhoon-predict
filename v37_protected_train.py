#!/usr/bin/env python3
"""Train the v37C protected structure branch locally on Apple Silicon.

The route is intentionally not retrained here: v23 is the project's validated
track backbone. v37C keeps its route frozen and trains a separate transformer
for maximum wind, central pressure, RMW, and the R34/R50/R64 wind radii. This
prevents an unstable intensity head from changing the route direction.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
TRACK_PATH = ROOT / "track_build" / "track_windows_v13.npz"
OUT_ROOT = ROOT / "v37" / "protected"
CKPT_ROOT = OUT_ROOT / "checkpoints"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

LEADS = 20
VERSION = "v37C"
TARGET_SCALE = np.asarray([100.0, 100.0, 35.0, 20.0, 50.0] + [50.0] * 12, dtype="float32")
CONFIG = {
    "epochs": int(os.environ.get("V37C_EPOCHS", "14")),
    "patience": int(os.environ.get("V37C_PATIENCE", "4")),
    "batch": int(os.environ.get("V37C_BATCH", "512")),
    "seeds": int(os.environ.get("V37C_SEEDS", "3")),
    "width": int(os.environ.get("V37C_WIDTH", "256")),
    "layers": int(os.environ.get("V37C_LAYERS", "2")),
    "heads": int(os.environ.get("V37C_HEADS", "8")),
    "lr": float(os.environ.get("V37C_LR", "3e-4")),
    "weight_decay": float(os.environ.get("V37C_WEIGHT_DECAY", "0.02")),
}

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(min(8, os.cpu_count() or 4))


def sinusoidal(length: int, width: int) -> torch.Tensor:
    position = torch.arange(length).unsqueeze(1).float()
    divisor = torch.exp(torch.arange(0, width, 2).float() * (-np.log(10000.0) / width))
    result = torch.zeros(length, width)
    result[:, 0::2] = torch.sin(position * divisor)
    result[:, 1::2] = torch.cos(position * divisor)
    return result


def first_storm_year(storm_id: np.ndarray, years: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}
    for index in indices:
        key = str(storm_id[index])
        result[key] = min(result.get(key, int(years[index])), int(years[index]))
    return result


def load_arrays() -> dict:
    archive = np.load(TRACK_PATH, allow_pickle=True)
    track = archive["track"].astype("float32")
    target = archive["target"].astype("float32")
    target_mask = archive["target_mask"].astype("float32")
    storm_id = archive["storm_id"].astype(str)
    years = archive["year"].astype("int32")
    n_leads = archive["n_leads"].astype("int32")
    mean = archive["track_mean"].astype("float32")
    std = archive["track_std"].astype("float32")
    raw_track = track * std[None, None, :] + mean[None, None, :]

    # Structure training needs complete labels, but it does not require a
    # future atmospheric field. Keep the route split storm-held-out and stable.
    full = (n_leads >= LEADS)
    full &= np.isfinite(target[:, :, 2:]).all(axis=(1, 2))
    full_indices = np.where(full)[0]
    storm_year = first_storm_year(storm_id, years, full_indices)
    split_lists = {"train": [], "val": [], "test": []}
    for index in full_indices:
        year = storm_year[str(storm_id[index])]
        name = "train" if year <= 2015 else "val" if year <= 2019 else "test"
        split_lists[name].append(int(index))
    splits = {name: np.asarray(values, dtype="int64") for name, values in split_lists.items()}
    return {
        "track": track,
        "target": target,
        "mask": target_mask,
        "raw_track": raw_track,
        "splits": splits,
    }


class StructureDataset(Dataset):
    def __init__(self, arrays: dict, indices: np.ndarray):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype="int64")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        return (
            torch.from_numpy(self.arrays["track"][index]),
            torch.from_numpy(self.arrays["target"][index]),
            torch.from_numpy(self.arrays["mask"][index]),
        )


class StructureV37(nn.Module):
    """Track-conditioned multi-horizon intensity and wind-structure decoder."""

    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        thermo_cols = [4, 5, 6, 7] + list(range(8, 20)) + list(range(24, 40)) + [44, 45, 46, 47]
        env_cols = [48, 49, 50, 51, 52, 53]
        self.columns = thermo_cols + env_cols
        self.proj = nn.Linear(len(self.columns), width)
        self.register_buffer("time", sinusoidal(9, width).unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, 0.12, batch_first=True, norm_first=True, activation="gelu"
        )
        decoder_layer = nn.TransformerDecoderLayer(
            width, heads, width * 4, 0.12, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.query = nn.Parameter(torch.randn(1, LEADS, width) * 0.02)
        self.register_buffer("lead_time", sinusoidal(LEADS, width).unsqueeze(0))
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.state = nn.Linear(width, 15)
        self.log_scale = nn.Linear(width, 15)

    def forward(self, track: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encoder(self.proj(track[:, :, self.columns]) + self.time)
        query = (self.query + self.lead_time).expand(track.shape[0], -1, -1)
        hidden = self.decoder(query, context)
        return self.state(hidden), self.log_scale(hidden)


def structure_loss(state: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    scale = torch.as_tensor(TARGET_SCALE[2:], device=state.device, dtype=state.dtype)
    truth = target[..., 2:] / scale
    valid = mask[..., 2:]
    log_scale = log_scale.clamp(-4.0, 4.0)
    huber = (F.smooth_l1_loss(state, truth, reduction="none") * valid).sum() / valid.sum().clamp_min(1.0)
    nll = ((state - truth).abs() * torch.exp(-log_scale) + log_scale) * valid
    nll = nll.sum() / valid.sum().clamp_min(1.0)

    # Keep the predicted wind radii physically ordered at every lead.
    r34, r50, r64 = state[..., 3:7], state[..., 7:11], state[..., 11:15]
    m34, m50, m64 = valid[..., 3:7], valid[..., 7:11], valid[..., 11:15]
    mono50 = (F.relu(r50 - r34) * m34 * m50).sum() / (m34 * m50).sum().clamp_min(1.0)
    mono64 = (F.relu(r64 - r50) * m50 * m64).sum() / (m50 * m64).sum().clamp_min(1.0)

    adjacent = valid[:, 1:] * valid[:, :-1]
    smooth = F.smooth_l1_loss(state[:, 1:] - state[:, :-1], truth[:, 1:] - truth[:, :-1], reduction="none")
    smooth = (smooth * adjacent).sum() / adjacent.sum().clamp_min(1.0)
    return 0.7 * huber + 0.3 * nll + 0.08 * (mono50 + mono64) + 0.02 * smooth


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, arrays: dict) -> dict:
    model.eval()
    absolute = np.zeros((LEADS, 15), dtype="float64")
    valid_sum = np.zeros((LEADS, 15), dtype="float64")
    loss_sum = 0.0
    count = 0
    for track, target, mask in loader:
        track = track.to(DEVICE)
        target = target.to(DEVICE)
        mask = mask.to(DEVICE)
        state, logs = model(track)
        loss_sum += float(structure_loss(state, logs, target, mask)) * len(track)
        predicted = (state * torch.as_tensor(TARGET_SCALE[2:], device=DEVICE)).cpu().numpy()
        actual = target[:, :, 2:].cpu().numpy()
        valid = mask[:, :, 2:].cpu().numpy()
        absolute += (np.abs(predicted - actual) * valid).sum(axis=0)
        valid_sum += valid.sum(axis=0)
        count += len(track)
    mae = absolute / np.maximum(valid_sum, 1.0)
    return {
        "windows": count,
        "loss": round(loss_sum / max(count, 1), 6),
        "vmax_mae_120h_kt": round(float(mae[-1, 0]), 3),
        "pressure_mae_120h_hpa": round(float(mae[-1, 1]), 3),
        "vmax_mae_by_lead_kt": [round(float(x), 3) for x in mae[:, 0]],
        "pressure_mae_by_lead_hpa": [round(float(x), 3) for x in mae[:, 1]],
        "structure_mae_120h": [round(float(x), 3) for x in mae[-1]],
    }


class StructureEnsemble(nn.Module):
    """Average forecasts, not transformer weights."""

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, track: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states = []
        logs = []
        for model in self.models:
            state, log_scale = model(track)
            states.append(state)
            logs.append(log_scale)
        return torch.stack(states).mean(0), torch.stack(logs).mean(0)


def move_batch(batch):
    return tuple(value.to(DEVICE, non_blocking=DEVICE.type == "cuda") for value in batch)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for raw_batch in loader:
        track, target, mask = move_batch(raw_batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            state, logs = model(track)
            loss = structure_loss(state, logs, target, mask)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach()) * len(track)
        count += len(track)
    return total / max(count, 1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_seed(seed: int, arrays: dict, loaders: dict):
    seed_everything(seed)
    model = StructureV37(CONFIG["width"], CONFIG["layers"], CONFIG["heads"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    best = float("inf")
    stale = 0
    history = []
    path = CKPT_ROOT / f"structure_v37_seed{seed}.pt"
    for epoch in range(1, CONFIG["epochs"] + 1):
        started = time.time()
        train_loss = run_epoch(model, loaders["train"], optimizer)
        val_loss = run_epoch(model, loaders["val"])
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 2),
        }
        history.append(record)
        print(f"seed {seed} epoch {epoch:03d}: train={train_loss:.4f} val={val_loss:.4f} {record['seconds']:.1f}s", flush=True)
        if val_loss < best:
            best = val_loss
            stale = 0
            torch.save({
                "version": VERSION,
                "seed": seed,
                "config": CONFIG,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "history": history,
                "target_scale": TARGET_SCALE,
                "route_backbone": "TrackFormer v23 frozen route ensemble",
            }, path)
        else:
            stale += 1
            if stale >= CONFIG["patience"]:
                print(f"seed {seed}: early stopping", flush=True)
                break
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    model.to(DEVICE).eval()
    return model, path, history


def main():
    print("v37 version:", VERSION, "device:", DEVICE, "config:", CONFIG, flush=True)
    arrays = load_arrays()
    loaders = {}
    for name, indices in arrays["splits"].items():
        if len(indices):
            loaders[name] = DataLoader(
                StructureDataset(arrays, indices),
                batch_size=CONFIG["batch"],
                shuffle=name == "train",
                num_workers=0,
                drop_last=False,
            )
            print(f"{name}: {len(indices):,} windows", flush=True)
    models = []
    checkpoints = []
    histories = {}
    for seed in range(CONFIG["seeds"]):
        model, checkpoint, history = train_seed(seed, arrays, loaders)
        models.append(model)
        checkpoints.append(str(checkpoint))
        histories[str(seed)] = history

    ensemble = StructureEnsemble(models).to(DEVICE).eval()
    metrics = {name: evaluate(ensemble, loaders[name], arrays) for name in ("val", "test") if name in loaders}
    for name, value in metrics.items():
        print(name, json.dumps(value), flush=True)
    manifest = {
        "version": VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": str(DEVICE),
        "config": CONFIG,
        "data": str(TRACK_PATH),
        "split_sizes": {name: int(len(indices)) for name, indices in arrays["splits"].items()},
        "checkpoints": checkpoints,
        "metrics": metrics,
        "route_policy": "Use frozen v23 ensemble with correctly normalized public GFS steering fields",
    }
    (OUT_ROOT / "v37_protected_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_ROOT / "v37_protected_histories.json").write_text(json.dumps(histories, indent=2))
    print("manifest:", OUT_ROOT / "v37_protected_manifest.json", flush=True)


if __name__ == "__main__":
    main()
