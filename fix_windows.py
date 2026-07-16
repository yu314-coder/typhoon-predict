#!/usr/bin/env python3
"""Repair the StormFusion-MT window cache: fill NaN and standardize inputs.

Two defects made training collapse to NaN loss:

1. ERA5 sea_surface_temperature is undefined (NaN) over land, so inner ch21 was
   7.5% NaN and outer ch13 13.2% NaN. Those NaNs propagate through the conv
   encoders and every loss becomes NaN.
2. The notebook never normalizes its inputs (only the target). Channels span ~7
   orders of magnitude (q850 ~0.01 vs z200 ~122300). Under BF16 the ULP at
   122300 is ~478 while z200's std is ~492 -- the signal is quantized to noise.

Fix: fill NaN with the channel mean, then standardize per channel. Statistics are
computed over TRAINING windows only (storm first-year <= 2015, matching the
notebook's storm_year_split) so validation/test never leak into the stats.

The per-channel mean/std are stored back into the npz (inner_mean, inner_std,
outer_mean, outer_std, env_mean, env_std) so inference can reproduce the transform.
The notebook's cache check is a subset check, so the extra keys are harmless.

track/target are left untouched: they contain no NaN, are on moderate scales, and
the target is already scaled by TARGET_SCALE in the notebook.
"""
import numpy as np
from pathlib import Path

SRC = Path("typhoon_build/data/windows/stormfusion_windows.npz")
RAW_BACKUP = Path("typhoon_build/data/windows/stormfusion_windows_raw.npz")
OUT = Path("typhoon_build/data/windows/stormfusion_windows.npz")

print(f"loading {SRC} ...")
z = dict(np.load(SRC, allow_pickle=True))

# --- training split (storm-level, mirrors the notebook's storm_year_split) ---
years = z["year"].astype(int)
storm_ids = z["storm_id"].astype(str)
first_year = {s: int(years[storm_ids == s].min()) for s in np.unique(storm_ids)}
train_storms = {s for s, y in first_year.items() if y <= 2015}
train_idx = np.where(np.isin(storm_ids, list(train_storms)))[0]
print(f"train windows used for stats: {len(train_idx)} / {len(years)}")


def fix(name, channel_axis):
    """Fill NaN with the train mean, then standardize, per channel."""
    arr = z[name]
    n_ch = arr.shape[channel_axis]
    means = np.zeros(n_ch, dtype="float32")
    stds = np.ones(n_ch, dtype="float32")
    nan_before = np.isnan(arr).mean() * 100
    for c in range(n_ch):
        train_slice = arr[train_idx, :, c] if channel_axis == 2 else arr[train_idx, :, c]
        m = np.nanmean(train_slice.astype("float64"))
        s = np.nanstd(train_slice.astype("float64"))
        if not np.isfinite(m):
            m = 0.0
        # Guard against constant/degenerate channels (avoid divide-by-zero blowup).
        if not np.isfinite(s) or s < 1e-6:
            s = 1.0
        means[c], stds[c] = np.float32(m), np.float32(s)
        ch = arr[:, :, c]
        np.nan_to_num(ch, copy=False, nan=float(m), posinf=float(m), neginf=float(m))
        arr[:, :, c] = (ch - means[c]) / stds[c]
    z[f"{name}_mean"] = means
    z[f"{name}_std"] = stds
    print(f"{name:6s}: NaN {nan_before:.3f}% -> {np.isnan(arr).mean()*100:.3f}% | "
          f"after norm mean={arr.mean():+.4f} std={arr.std():.4f}")
    return arr


print("\n=== fixing inner / outer / env ===")
z["inner"] = fix("inner", 2)
z["outer"] = fix("outer", 2)
z["env"] = fix("env", 2)

# --- sanity checks before writing ---
for k in ("inner", "outer", "env", "track", "target"):
    a = z[k]
    assert not np.isnan(a).any(), f"{k} still has NaN"
    assert not np.isinf(a).any(), f"{k} has inf"
print("\nall arrays finite ✅")

# verify train-split stats really are ~0/1 (val/test may differ slightly - expected)
for k in ("inner", "outer", "env"):
    tr = z[k][train_idx]
    print(f"{k:6s} train mean={tr.mean():+.4f} std={tr.std():.4f}")

if not RAW_BACKUP.exists():
    print(f"\nbacking up raw -> {RAW_BACKUP}")
    import shutil
    shutil.copy2(SRC, RAW_BACKUP)

print(f"saving {OUT} ...")
np.savez_compressed(OUT, **z)
print("done:", OUT, f"{OUT.stat().st_size/1e9:.2f} GB")
