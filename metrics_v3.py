# Standalone metrics for the trained v3 model, run inside the Colab kernel via exec().
# Uses num_workers=0 DataLoaders to avoid the persistent-worker hang seen in the
# notebook's collect_predictions cell. Relies on globals already defined by the run:
# CHECKPOINT_ROOT, model, WindowDataset, window_arrays, valid_idx, test_idx,
# move_batch, amp_dtype, use_amp, TARGET_SCALE, DEVICE.
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

_bp = torch.load(CHECKPOINT_ROOT / "best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(_bp["model"])
model.eval()
print("Loaded best.pt (epoch", _bp.get("epoch"), ", best_val", round(float(_bp.get("best_val", float("nan"))), 5), ")")


def _loader(idx):
    return DataLoader(WindowDataset(window_arrays, idx), batch_size=32,
                      shuffle=False, num_workers=0)


def _collect(loader):
    preds, targs, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch)
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                pred, _ = model(batch["inner"], batch["outer"], batch["track"], batch["env"])
            preds.append((pred * TARGET_SCALE).float().cpu().numpy())
            targs.append(batch["target"].float().cpu().numpy())
            masks.append(batch["target_mask"].float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(targs), np.concatenate(masks)


def _metrics(pred, target, mask):
    m = {}
    pt = np.cumsum(pred[..., :2], axis=1)
    tt = np.cumsum(target[..., :2], axis=1)
    m["track_error_km"] = round(float(np.sqrt(((pt - tt) ** 2).sum(-1)).mean()), 2)
    for i, name in [(2, "vmax_mae_kt"), (3, "pressure_mae_hpa"), (4, "rmw_mae_km")]:
        v = mask[..., i] > 0.5
        m[name] = round(float(np.abs(pred[..., i][v] - target[..., i][v]).mean()), 2) if v.any() else None
    rm = mask[..., 5:17] > 0.5
    re = np.abs(pred[..., 5:17] - target[..., 5:17])
    m["radius_mae_km"] = round(float(re[rm].mean()), 2) if rm.any() else None
    return m


for _split, _idx in [("Validation", valid_idx), ("Test", test_idx)]:
    if len(_idx) == 0:
        continue
    _p, _t, _mk = _collect(_loader(_idx))
    print(_split, "metrics:", json.dumps(_metrics(_p, _t, _mk), indent=2))
