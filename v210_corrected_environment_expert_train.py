#!/usr/bin/env python3
"""Train Trackformer 1.2.10 with a corrected causal environment contract.

The public steer5 archive contains SLP anomaly, SLP tendency, 500-hPa wind,
and SST.  It is not a lower/upper wind pair, so it must not be used to form
200-850 hPa shear.  This trainer keeps SST labels from steer5 and derives
future 850/500/200 hPa shear labels from the separate basin context archive,
whose channels are explicitly documented as multi-level winds.

The environment forecast is a per-lead standardized ridge model using only
the frozen 1.2.8 causal features.  A second per-lead ridge layer conditions a
wind-regime intensity residual on that predicted environment.  Future rows
are labels only; official forecast products and TIP data are excluded.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

import v127_v11_intensity_train as v127
import v128_causal_intensity_train as v128
import v129_causal_environment_intensity_train as v129
import v129_causal_environment_refit as refit


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "v210" / "corrected_environment_expert"
LABEL_ROOT = OUT_ROOT / "labels"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LABEL_ROOT.mkdir(parents=True, exist_ok=True)
LEADS = 20
ENV_DIM = 5
REGIME_THRESHOLD_KT = 55.0
ENV_LAMBDAS = (10.0, 30.0, 100.0, 300.0)
INTENSITY_LAMBDAS = (10.0, 30.0, 100.0, 300.0)
ALPHAS = np.arange(0.0, 1.0001, 0.05, dtype="float32")
VERSION = "Trackformer1.2.10-causal-corrected-environment-expert"


def _build_corrected_labels(arrays: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_path = LABEL_ROOT / "future_corrected_sst_shear_targets.npy"
    mask_path = LABEL_ROOT / "future_corrected_sst_shear_masks.npy"
    future_path = LABEL_ROOT / "future_row_indices.npy"
    expected = (len(arrays["track"]), LEADS)
    if target_path.exists() and mask_path.exists() and future_path.exists():
        target = np.load(target_path, mmap_mode="r")
        mask = np.load(mask_path, mmap_mode="r")
        future = np.load(future_path, mmap_mode="r")
        if target.shape == (*expected, ENV_DIM) and mask.shape == target.shape and future.shape == expected:
            print("using corrected SST/shear label cache", target_path, flush=True)
            return np.asarray(target), np.asarray(mask), np.asarray(future)

    future = v129._future_indices(arrays["storm_id"], arrays["base_time"])
    context = np.load(ROOT / "track_build" / "basin_context_features.npz", allow_pickle=True)
    context_features = context["features"].astype("float32")
    context_valid = context["valid"].astype(bool)
    context_mean = context["feature_mean"].astype("float32")
    context_std = context["feature_std"].astype("float32")
    fields = np.load(ROOT / "track_build" / "steer5_patches.npy", mmap_mode="r")
    target = np.zeros((*expected, ENV_DIM), dtype="float32")
    mask = np.zeros_like(target)
    for start in range(0, len(future), 512):
        stop = min(start + 512, len(future))
        available = future[start:stop] >= 0
        safe = np.where(available, future[start:stop], np.arange(start, stop, dtype="int64")[:, None])
        # SST is the fifth steer5 channel. Keep both center and inner-area values.
        sst = np.asarray(fields[safe, 4], dtype="float32")
        center = sst[..., 8, 8]
        inner = sst[..., 6:11, 6:11].mean(axis=(-2, -1))
        # Context channels are hgt500, u850, v850, u500, v500, u200, v200;
        # every channel has six summary values, with its center at 0,6,...,36.
        values = context_features[safe] * context_std[None, None, :] + context_mean[None, None, :]
        centers = values[..., np.asarray([0, 6, 12, 18, 24, 30, 36])]
        _, u850, v850, u500, v500, u200, v200 = np.moveaxis(centers, -1, 0)
        shear_u = u200 - u850
        shear_v = v200 - v850
        shear_mag = np.hypot(shear_u, shear_v)
        target[start:stop] = np.stack([center, inner, shear_u, shear_v, shear_mag], axis=-1)
        target_mask = available & context_valid[safe]
        mask[start:stop, :, 0] = available
        mask[start:stop, :, 1] = available
        mask[start:stop, :, 2:] = target_mask[..., None]
        if stop == len(future) or stop % 25_600 == 0:
            print(f"corrected environment labels {stop:,}/{len(future):,}", flush=True)

    for path, value in ((target_path, target), (mask_path, mask), (future_path, future)):
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=value.dtype, shape=value.shape)
        mm[:] = value
        mm.flush()
    (LABEL_ROOT / "metadata.json").write_text(json.dumps({
        "version": VERSION,
        "sst_source": "steer5 channel 4, center and 5x5 inner mean",
        "shear_source": "basin_context_features channels u850,v850,u500,v500,u200,v200",
        "shear_formula": "u200-u850, v200-v850, hypot(u200-u850,v200-v850)",
        "future_rows_are_labels_only": True,
        "target_units": ["SST anomaly C", "inner SST anomaly C", "m/s", "m/s", "m/s"],
    }, indent=2) + "\n", encoding="utf-8")
    return target, mask, future


def _load_payloads(arrays: dict, env_target: np.ndarray, env_mask: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    route_data = v128.load_route_environment()
    base_data = v128.load_base_predictions(arrays, route_data)
    train_base = v129.load_v128_train_prediction(arrays, base_data)
    result: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        base = train_base if split == "train" else np.load(
            ROOT / "v128" / "causal_intensity_upgrade" / f"trackformer_1_2_8_{split}_predictions.npz",
            allow_pickle=False,
        )["prediction"].astype("float32")
        indices = base_data[split]["indices"].astype("int64")
        features = v129.append_v128_output(base_data[split]["features"], base)
        result[split] = {
            "features": features,
            "base": base,
            "base_v128": base,
            "target": base_data[split]["target"].astype("float32"),
            "mask": base_data[split]["mask"].astype("float32"),
            "environment_target": env_target[indices].astype("float32"),
            "environment_mask": env_mask[indices].astype("float32"),
            "indices": indices,
        }
        print(split, "causal features", features.shape, "environment labels", result[split]["environment_target"].shape, flush=True)
    return result


def _fit_models(x: np.ndarray, y: np.ndarray, valid: np.ndarray, lam: float) -> list[dict]:
    models: list[dict] = []
    for lead in range(LEADS):
        keep = valid[:, lead] & np.isfinite(x[:, lead]).all(axis=1) & np.isfinite(y[:, lead])
        if int(keep.sum()) < 64:
            keep = np.isfinite(x[:, lead]).all(axis=1) & np.isfinite(y[:, lead])
        selected_x = x[keep, lead].astype("float64")
        selected_y = y[keep, lead].astype("float64")
        mean = selected_x.mean(axis=0)
        std = np.maximum(selected_x.std(axis=0), 1.0e-3)
        normalized = (selected_x - mean) / std
        y_mean = float(selected_y.mean())
        gram = normalized.T @ normalized
        gram.flat[:: len(gram) + 1] += float(lam)
        weights = np.linalg.solve(gram, normalized.T @ (selected_y - y_mean))
        models.append({"mean": mean, "std": std, "y_mean": y_mean, "weights": weights})
    return models


def _predict_models(models: list[dict], x: np.ndarray) -> np.ndarray:
    result = np.zeros((len(x), LEADS), dtype="float32")
    for lead, model in enumerate(models):
        result[:, lead] = (model["y_mean"] + ((x[:, lead] - model["mean"]) / model["std"]) @ model["weights"]).astype("float32")
    return result


def _model_arrays(models: list[dict]) -> dict[str, np.ndarray]:
    return {
        "mean": np.stack([m["mean"] for m in models]),
        "std": np.stack([m["std"] for m in models]),
        "y_mean": np.asarray([m["y_mean"] for m in models], dtype="float64"),
        "weights": np.stack([m["weights"] for m in models]),
    }


def _predict_environment(payload: dict[str, np.ndarray], models: list[list[dict]]) -> np.ndarray:
    values = np.stack([_predict_models(models[i], payload["features"]) for i in range(ENV_DIM)], axis=-1)
    values[..., 0:2] = np.clip(values[..., 0:2], -18.5, 3.0)
    values[..., 2:4] = np.clip(values[..., 2:4], -80.0, 80.0)
    values[..., 4] = np.clip(np.hypot(values[..., 2], values[..., 3]), 0.0, 80.0)
    return values.astype("float32")


def _environment_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    names = ["sst_center_anomaly_c", "sst_inner_anomaly_c", "shear_200_850_u_ms", "shear_200_850_v_ms", "shear_200_850_mag_ms"]
    mae = np.abs(prediction - target) * mask
    denom = np.maximum(mask.sum(axis=(0, 1)), 1.0)
    return {"mae": {n: round(float(v), 4) for n, v in zip(names, mae.sum(axis=(0, 1)) / denom)}, "valid_fraction": {n: round(float(v), 6) for n, v in zip(names, mask.mean(axis=(0, 1)))}}


def _fit_intensity_expert(x: np.ndarray, payload: dict, output: int, lam: float, high: np.ndarray) -> list[dict]:
    residual = payload["target"][:, :, output] - payload["base"][:, :, output]
    valid = (payload["mask"][:, :, output] > 0.5) & high
    return _fit_models(x, residual, valid, lam)


def _predict_intensity(payload: dict, x: np.ndarray, experts: dict[str, dict[str, list[dict]]], alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = payload["base"]
    high = base[:, :, 0] >= REGIME_THRESHOLD_KT
    correction = np.zeros((len(base), LEADS, 2), dtype="float32")
    for output, name in ((0, "wind"), (1, "pressure")):
        low = _predict_models(experts[name]["low"], x)
        high_value = _predict_models(experts[name]["high"], x)
        correction[:, :, output] = np.where(high, high_value, low)
    return refit.blended_prediction(base, correction, alpha), correction


def main() -> None:
    arrays = v127.load_track_arrays()
    env_target, env_mask, _future = _build_corrected_labels(arrays)
    payloads = _load_payloads(arrays, env_target, env_mask)

    # Environment forecast selection uses validation only, with the test split
    # untouched until the final quality gate.
    env_models_by_lambda: dict[float, list[list[dict]]] = {}
    env_predictions_by_lambda: dict[float, dict[str, np.ndarray]] = {}
    for lam in ENV_LAMBDAS:
        models = [_fit_models(payloads["train"]["features"], payloads["train"]["environment_target"][:, :, output], payloads["train"]["environment_mask"][:, :, output] > 0.5, lam) for output in range(ENV_DIM)]
        env_models_by_lambda[lam] = models
        env_predictions_by_lambda[lam] = {split: _predict_environment(payloads[split], models) for split in ("val", "test")}
        print("environment lambda", lam, _environment_metrics(env_predictions_by_lambda[lam]["val"], payloads["val"]["environment_target"], payloads["val"]["environment_mask"]), flush=True)

    env_selected_lambda: list[float] = []
    for output in range(ENV_DIM):
        best = None
        for lam in ENV_LAMBDAS:
            prediction = env_predictions_by_lambda[lam]["val"]
            valid = payloads["val"]["environment_mask"][:, :, output] > 0.5
            value = float(np.abs(prediction[:, :, output] - payloads["val"]["environment_target"][:, :, output])[valid].mean())
            if best is None or value < best[0]: best = (value, lam)
        assert best is not None
        env_selected_lambda.append(best[1])
    env_models = [env_models_by_lambda[env_selected_lambda[i]][i] for i in range(ENV_DIM)]
    environments = {split: _predict_environment(payloads[split], env_models) for split in ("train", "val", "test")}
    env_metrics = {split: _environment_metrics(environments[split], payloads[split]["environment_target"], payloads[split]["environment_mask"]) for split in ("val", "test")}
    print("selected environment lambdas", env_selected_lambda, json.dumps(env_metrics), flush=True)

    designs = {split: refit.design_features(payloads[split], environments[split]) for split in ("train", "val", "test")}
    high_train = payloads["train"]["base"][:, :, 0] >= REGIME_THRESHOLD_KT
    experts_by_lambda: dict[float, dict[str, dict[str, list[dict]]]] = {}
    for lam in INTENSITY_LAMBDAS:
        experts_by_lambda[lam] = {
            "wind": {"low": _fit_intensity_expert(designs["train"], payloads["train"], 0, lam, ~high_train), "high": _fit_intensity_expert(designs["train"], payloads["train"], 0, lam, high_train)},
            "pressure": {"low": _fit_intensity_expert(designs["train"], payloads["train"], 1, lam, ~high_train), "high": _fit_intensity_expert(designs["train"], payloads["train"], 1, lam, high_train)},
        }
    chosen_lambda: dict[str, float] = {}
    selection: dict[str, dict] = {}
    for output, name in ((0, "wind"), (1, "pressure")):
        best = None
        for lam in INTENSITY_LAMBDAS:
            for alpha in ALPHAS:
                candidate, _ = _predict_intensity(payloads["val"], designs["val"], experts_by_lambda[lam], np.asarray([alpha if output == 0 else 0.0, alpha if output == 1 else 0.0], dtype="float32"))
                metrics = v129.intensity_metrics(candidate, payloads["val"]["target"], payloads["val"]["mask"])
                key = "wind_mae_all_leads_kt" if output == 0 else "pressure_mae_all_leads_hpa"
                if best is None or metrics[key] < best[0]: best = (metrics[key], lam, float(alpha), metrics)
        assert best is not None
        chosen_lambda[name] = best[1]; selection[name] = {"mae": best[0], "lambda": best[1], "alpha": best[2], "metrics": best[3]}
    experts = {"wind": experts_by_lambda[chosen_lambda["wind"]]["wind"], "pressure": experts_by_lambda[chosen_lambda["pressure"]]["pressure"]}
    raw_val, _ = _predict_intensity(payloads["val"], designs["val"], experts, np.ones(2, dtype="float32"))
    alpha_report = {}
    alphas = []
    for output, name, low, high in ((0, "wind", 0.0, 190.0), (1, "pressure", 850.0, 1050.0)):
        best = None
        for alpha in ALPHAS:
            candidate = payloads["val"]["base"].copy(); candidate[:, :, output] = np.clip(payloads["val"]["base"][:, :, output] + float(alpha) * (raw_val[:, :, output] - payloads["val"]["base"][:, :, output]), low, high)
            metrics = v129.intensity_metrics(candidate, payloads["val"]["target"], payloads["val"]["mask"]); key = "wind_mae_all_leads_kt" if output == 0 else "pressure_mae_all_leads_hpa"
            if best is None or metrics[key] < best[0]: best = (metrics[key], float(alpha), metrics)
        assert best is not None; alphas.append(best[1]); alpha_report[name] = {"mae": best[0], "alpha": best[1], "metrics": best[2]}
    alphas = np.asarray(alphas, dtype="float32")

    metrics = {}; predictions = {}; corrections = {}
    for split in ("val", "test"):
        predictions[split], corrections[split] = _predict_intensity(payloads[split], designs[split], experts, alphas)
        metrics[split] = {"base_v128": v129.intensity_metrics(payloads[split]["base"], payloads[split]["target"], payloads[split]["mask"]), "candidate_v210": v129.intensity_metrics(predictions[split], payloads[split]["target"], payloads[split]["mask"]), "environment": env_metrics[split]}
        np.savez_compressed(OUT_ROOT / f"trackformer_1_2_10_{split}_predictions.npz", indices=payloads[split]["indices"], prediction=predictions[split], correction=corrections[split], base_v128=payloads[split]["base"], environment_prediction=environments[split], environment_target=payloads[split]["environment_target"], environment_mask=payloads[split]["environment_mask"], target=payloads[split]["target"], mask=payloads[split]["mask"])
        print(split, json.dumps(metrics[split]), flush=True)

    promoted = all([
        metrics["val"]["candidate_v210"]["wind_mae_all_leads_kt"] < metrics["val"]["base_v128"]["wind_mae_all_leads_kt"],
        metrics["val"]["candidate_v210"]["pressure_mae_all_leads_hpa"] < metrics["val"]["base_v128"]["pressure_mae_all_leads_hpa"],
        metrics["test"]["candidate_v210"]["wind_mae_all_leads_kt"] < metrics["test"]["base_v128"]["wind_mae_all_leads_kt"],
        metrics["test"]["candidate_v210"]["pressure_mae_all_leads_hpa"] < metrics["test"]["base_v128"]["pressure_mae_all_leads_hpa"],
    ])
    arrays = {"feature_dim": np.asarray([designs["train"].shape[-1]], dtype="int64"), "regime_threshold_kt": np.asarray([REGIME_THRESHOLD_KT], dtype="float32"), "env_selected_lambdas": np.asarray(env_selected_lambda, dtype="float32")}
    for output, name in enumerate(("env_sst_center", "env_sst_inner", "env_shear_u", "env_shear_v", "env_shear_mag")):
        for key, value in _model_arrays(env_models[output]).items(): arrays[f"{name}_{key}"] = value
    for name in ("wind", "pressure"):
        for regime in ("low", "high"):
            for key, value in _model_arrays(experts[name][regime]).items(): arrays[f"{name}_{regime}_{key}"] = value
    weight_path = OUT_ROOT / "trackformer_1_2_10_corrected_environment_weights.npz"
    np.savez_compressed(weight_path, **arrays)
    manifest = {
        "version": VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_model": str(ROOT / "v128" / "causal_intensity_upgrade" / "trackformer_1_2_8_manifest.json"),
        "architecture": "causal per-lead ridge environment forecast plus low/high wind-regime intensity residual experts",
        "environment_contract": {"sst_source": "steer5 channel 4 only", "shear_source": "basin_context_features multi-level winds", "predicted_targets": ["SST center anomaly", "inner SST anomaly", "200-850 shear u", "200-850 shear v", "200-850 shear magnitude"], "shear_formula": "u200-u850, v200-v850, hypot(u200-u850,v200-v850)"},
        "selected_environment_lambdas": env_selected_lambda,
        "selected_intensity_lambdas": chosen_lambda,
        "selected_alphas": {"wind": float(alphas[0]), "pressure": float(alphas[1])},
        "selection": {"intensity": selection, "blend": alpha_report},
        "weights": str(weight_path),
        "metrics": metrics,
        "promoted": bool(promoted),
        "quality_gate": "wind and pressure must improve over 1.2.8 on chronological validation and untouched post-2019 test",
        "data_policy": {"future_weather_as_inputs": False, "future_weather_as_labels_only": True, "official_forecasts_as_inputs": False, "forecast_products_used": [], "tip_used_for_training": False, "current_and_past_weather_only_at_inference": True},
        "labels": str(LABEL_ROOT / "metadata.json"),
    }
    manifest_path = OUT_ROOT / "trackformer_1_2_10_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "weights": str(weight_path), "promoted": promoted, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
