#!/usr/bin/env python3
"""Train a spatial v37 intensity/structure expert on the Mac.

The route remains the public GFS/GEFS vortex ensemble.  This expert keeps
spatial field tokens and predicts a residual from the current wind and
pressure, which is easier to optimize for a strong storm than an unconstrained
absolute intensity head.
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

from v37_protected_train import DEVICE, LEADS, TARGET_SCALE, load_arrays


ROOT = Path(__file__).resolve().parent
TRACK_PATH = ROOT / "track_build" / "track_windows_v13.npz"
FIELD_PATH = ROOT / "track_build" / "dlm4_int8.npz"
OUT_ROOT = ROOT / "v37" / "structure_spatial"
CKPT_ROOT = OUT_ROOT / "checkpoints"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

VERSION = os.environ.get("V37G_VERSION", "v37G")
CONFIG = {
    "epochs": int(os.environ.get("V37G_EPOCHS", "14")),
    "patience": int(os.environ.get("V37G_PATIENCE", "4")),
    "batch": int(os.environ.get("V37G_BATCH", "512")),
    "seeds": int(os.environ.get("V37G_SEEDS", "3")),
    "width": int(os.environ.get("V37G_WIDTH", "192")),
    "layers": int(os.environ.get("V37G_LAYERS", "2")),
    "heads": int(os.environ.get("V37G_HEADS", "8")),
    "lr": float(os.environ.get("V37G_LR", "3e-4")),
    "weight_decay": float(os.environ.get("V37G_WEIGHT_DECAY", "0.02")),
    "field_decode": "q/31.75",
    "current_residual": True,
}

THERMO_ENV_COLS = [4, 5, 6, 7] + list(range(8, 20)) + list(range(24, 40)) + [44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
CURRENT_SCALE = torch.as_tensor(TARGET_SCALE[2:4], dtype=torch.float32)


def sinusoidal(length: int, width: int) -> torch.Tensor:
    position = torch.arange(length).unsqueeze(1).float()
    divisor = torch.exp(torch.arange(0, width, 2).float() * (-np.log(10000.0) / width))
    result = torch.zeros(length, width)
    result[:, 0::2] = torch.sin(position * divisor)
    result[:, 1::2] = torch.cos(position * divisor)
    return result


class SpatialDataset(Dataset):
    def __init__(self, arrays: dict, fields: np.ndarray, indices: np.ndarray):
        self.arrays = arrays
        self.fields = fields
        self.indices = np.asarray(indices, dtype="int64")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        field = np.asarray(self.fields[index], dtype="float32") / 31.75
        field = np.clip(field, -4.0, 4.0)
        current = self.arrays["raw_track"][index, -1, 4:6].astype("float32")
        current = current / np.asarray(TARGET_SCALE[2:4], dtype="float32")
        available = self.arrays["raw_track"][index, -1, 24:26].astype("float32")
        return (
            torch.from_numpy(self.arrays["track"][index]),
            torch.from_numpy(field),
            torch.from_numpy(current),
            torch.from_numpy(available),
            torch.from_numpy(self.arrays["target"][index]),
            torch.from_numpy(self.arrays["mask"][index]),
        )


class StructureSpatialV37G(nn.Module):
    """Track transformer plus spatial field tokens and current-state residual."""

    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.track_proj = nn.Linear(len(THERMO_ENV_COLS), width)
        self.register_buffer("track_time", sinusoidal(9, width).unsqueeze(0))
        track_layer = nn.TransformerEncoderLayer(
            width,
            heads,
            width * 4,
            0.12,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.track_encoder = nn.TransformerEncoder(track_layer, layers)
        self.field_encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        # The stride-2 convolution produces a 9x9 grid.  Fixed pooling gives
        # a 4x4 token grid and avoids MPS's non-divisible adaptive-pool limit.
        self.field_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.field_norm = nn.LayerNorm(width)
        self.field_pos = nn.Parameter(torch.randn(1, 16, width) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            width,
            heads,
            width * 4,
            0.12,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.query = nn.Parameter(torch.randn(1, LEADS, width) * 0.02)
        self.register_buffer("lead_time", sinusoidal(LEADS, width).unsqueeze(0))
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.state = nn.Linear(width, 15)
        self.log_scale = nn.Linear(width, 15)

    def forward(
        self,
        track: torch.Tensor,
        field: torch.Tensor,
        current: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        track_tokens = self.track_encoder(self.track_proj(track[:, :, THERMO_ENV_COLS]) + self.track_time)
        field_tokens = self.field_pool(self.field_encoder(field)).flatten(2).transpose(1, 2)
        field_tokens = self.field_norm(field_tokens + self.field_pos)
        memory = torch.cat([track_tokens, field_tokens], dim=1)
        query = (self.query + self.lead_time).expand(track.shape[0], -1, -1)
        hidden = self.decoder(query, memory)
        state = self.state(hidden)
        current = current * available
        state = state.clone()
        state[:, :, :2] = state[:, :, :2] + current[:, None, :]
        return state, self.log_scale(hidden)


def structure_loss(state, log_scale, target, mask):
    scale = torch.as_tensor(TARGET_SCALE[2:], device=state.device, dtype=state.dtype)
    truth = target[..., 2:] / scale
    valid = mask[..., 2:]
    log_scale = log_scale.clamp(-4.0, 4.0)
    huber = (F.smooth_l1_loss(state, truth, reduction="none") * valid).sum() / valid.sum().clamp_min(1.0)
    nll = (((state - truth).abs() * torch.exp(-log_scale)) + log_scale) * valid
    nll = nll.sum() / valid.sum().clamp_min(1.0)
    r34, r50, r64 = state[..., 3:7], state[..., 7:11], state[..., 11:15]
    m34, m50, m64 = valid[..., 3:7], valid[..., 7:11], valid[..., 11:15]
    mono50 = (F.relu(r50 - r34) * m34 * m50).sum() / (m34 * m50).sum().clamp_min(1.0)
    mono64 = (F.relu(r64 - r50) * m50 * m64).sum() / (m50 * m64).sum().clamp_min(1.0)
    adjacent = valid[:, 1:] * valid[:, :-1]
    smooth = F.smooth_l1_loss(state[:, 1:] - state[:, :-1], truth[:, 1:] - truth[:, :-1], reduction="none")
    smooth = (smooth * adjacent).sum() / adjacent.sum().clamp_min(1.0)
    return 0.7 * huber + 0.3 * nll + 0.08 * (mono50 + mono64) + 0.02 * smooth


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for track, field, current, available, target, mask in loader:
        track = track.to(DEVICE)
        field = field.to(DEVICE)
        current = current.to(DEVICE)
        available = available.to(DEVICE)
        target = target.to(DEVICE)
        mask = mask.to(DEVICE)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            state, logs = model(track, field, current, available)
            loss = structure_loss(state, logs, target, mask)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach()) * len(track)
        count += len(track)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(models, loader):
    for model in models:
        model.eval()
    absolute = np.zeros((LEADS, 15), dtype="float64")
    valid_sum = np.zeros((LEADS, 15), dtype="float64")
    loss_sum = 0.0
    count = 0
    for track, field, current, available, target, mask in loader:
        track = track.to(DEVICE)
        field = field.to(DEVICE)
        current = current.to(DEVICE)
        available = available.to(DEVICE)
        target = target.to(DEVICE)
        mask = mask.to(DEVICE)
        states = [model(track, field, current, available)[0] for model in models]
        state = torch.stack(states).mean(0)
        logs = torch.zeros_like(state)
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_seed(seed: int, loaders: dict, config: dict):
    seed_everything(seed)
    model = StructureSpatialV37G(config["width"], config["layers"], config["heads"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    path = CKPT_ROOT / f"structure_spatial_{VERSION.lower()}_seed{seed}.pt"
    best = float("inf")
    stale = 0
    history = []
    for epoch in range(1, config["epochs"] + 1):
        started = time.time()
        train_loss = run_epoch(model, loaders["train"], optimizer)
        val_loss = run_epoch(model, loaders.get("val", loaders["train"]))
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 2),
        }
        history.append(record)
        print(f"spatial seed {seed} epoch {epoch:03d}: train={train_loss:.4f} val={val_loss:.4f} {record['seconds']:.1f}s", flush=True)
        if val_loss < best:
            best = val_loss
            stale = 0
            torch.save(
                {
                    "version": f"{VERSION}-structure-spatial",
                    "seed": seed,
                    "config": config,
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "target_scale": TARGET_SCALE,
                    "field_normalization": "q/31.75; current wind/pressure residual anchor",
                    "history": history,
                },
                path,
            )
        else:
            stale += 1
            if stale >= config["patience"]:
                print(f"spatial seed {seed}: early stopping", flush=True)
                break
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    model.to(DEVICE).eval()
    return model, path, history


def main() -> None:
    print(f"{VERSION} spatial structure device:", DEVICE, "config:", CONFIG, flush=True)
    if not FIELD_PATH.exists():
        raise FileNotFoundError(FIELD_PATH)
    arrays = load_arrays()
    fields = np.load(FIELD_PATH, allow_pickle=True)["q"]
    loaders = {}
    for name, indices in arrays["splits"].items():
        if not len(indices):
            continue
        loaders[name] = DataLoader(
            SpatialDataset(arrays, fields, indices),
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
        model, path, history = train_seed(seed, loaders, CONFIG)
        models.append(model)
        checkpoints.append(str(path))
        histories[str(seed)] = history
    metrics = {
        name: evaluate(models, loaders[name])
        for name in ("val", "test")
        if name in loaders
    }
    for name, metric in metrics.items():
        print(name, json.dumps(metric), flush=True)
    manifest = {
        "version": f"{VERSION}-structure-spatial",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": str(DEVICE),
        "config": CONFIG,
        "data": {
            "track": str(TRACK_PATH),
            "fields": str(FIELD_PATH),
            "field_normalization": "q/31.75; current wind/pressure residual anchor",
        },
        "split_sizes": {name: int(len(indices)) for name, indices in arrays["splits"].items()},
        "checkpoints": checkpoints,
        "metrics": metrics,
        "ensemble_policy": "prediction-level averaging across spatial experts",
    }
    (OUT_ROOT / f"{VERSION.lower()}_structure_spatial_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_ROOT / f"{VERSION.lower()}_structure_spatial_histories.json").write_text(json.dumps(histories, indent=2))
    print("manifest:", OUT_ROOT / f"{VERSION.lower()}_structure_spatial_manifest.json", flush=True)


if __name__ == "__main__":
    main()
