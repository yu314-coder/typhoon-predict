#!/usr/bin/env python3
"""Train Trackformer 1.2.11 as an intensity-only upgrade over 1.2.10.

The corrected 1.2.10 SST and 200-850 hPa shear forecast is frozen.  This
trainer reconstructs that forecast from its serialized weights, then fits a
small causal residual head for wind and pressure.  The new head receives only
the existing issue-time/past feature contract, the frozen 1.2.10 output, and
the frozen model's own predicted environment.  Future weather rows are labels
only; official forecast products and TIP data are never inputs.

Selection is performed on the chronological validation split.  A second
train+validation refit is exported for later inference, while the promotion
gate is evaluated on a train-only validation/test pass so the post-2019 test
remains untouched by model selection.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

import v127_v11_intensity_train as v127
import v129_causal_environment_intensity_train as v129
import v129_causal_environment_refit as refit
import v210_corrected_environment_expert_train as v210


ROOT = Path(__file__).resolve().parent
V210_ROOT = ROOT / "v210" / "corrected_environment_expert"
OUT_ROOT = ROOT / "v211" / "intensity_only"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LEADS = 20
PHASES = 3
REGULARIZATIONS = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
ALPHAS = np.arange(0.0, 1.0001, 0.05, dtype="float32")
VERSION = "Trackformer1.2.11-causal-intensity-only"


def _ridge_models(x: np.ndarray, y: np.ndarray, valid: np.ndarray, lam: float) -> list[dict]:
    """Fit one standardized ridge residual model per lead."""

    models: list[dict] = []
    for lead in range(LEADS):
        keep = valid[:, lead] & np.isfinite(x[:, lead]).all(axis=1) & np.isfinite(y[:, lead])
        if int(keep.sum()) < 64:
            keep = np.isfinite(x[:, lead]).all(axis=1) & np.isfinite(y[:, lead])
        selected_x = x[keep, lead].astype("float64")
        selected_y = y[keep, lead].astype("float64")
        if len(selected_x) == 0:
            raise RuntimeError(f"no samples for lead {lead}")
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
        normalized = (x[:, lead] - model["mean"]) / model["std"]
        result[:, lead] = (model["y_mean"] + normalized @ model["weights"]).astype("float32")
    return result


def _fit_phase_models(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    phase: np.ndarray,
    lam: float,
) -> list[list[dict]]:
    """Fit strengthening, mature, and weakening experts with safe fallback."""

    fallback = _ridge_models(x, y, valid, lam)
    models = []
    for phase_id in range(PHASES):
        selected = valid & (phase == phase_id)
        if int(selected.sum()) < 512:
            models.append(fallback)
        else:
            models.append(_ridge_models(x, y, selected, lam))
    return models


def _predict_phase_models(models: list[list[dict]], x: np.ndarray, phase: np.ndarray) -> np.ndarray:
    values = np.stack([_predict_models(model, x) for model in models], axis=-1)
    selected = np.take_along_axis(values, phase[..., None], axis=-1)[..., 0]
    return selected.astype("float32")


def _model_arrays(models: list[dict]) -> dict[str, np.ndarray]:
    return {
        "mean": np.stack([item["mean"] for item in models]),
        "std": np.stack([item["std"] for item in models]),
        "y_mean": np.asarray([item["y_mean"] for item in models], dtype="float64"),
        "weights": np.stack([item["weights"] for item in models]),
    }


def _phase_model_arrays(models: list[list[dict]]) -> dict[str, np.ndarray]:
    values = {key: [] for key in ("mean", "std", "y_mean", "weights")}
    for model in models:
        arrays = _model_arrays(model)
        for key in values:
            values[key].append(arrays[key])
    return {key: np.stack(value) for key, value in values.items()}


def _archive_models(archive: np.lib.npyio.NpzFile, prefix: str) -> list[dict]:
    return [
        {
            "mean": archive[f"{prefix}_mean"][lead],
            "std": archive[f"{prefix}_std"][lead],
            "y_mean": archive[f"{prefix}_y_mean"][lead],
            "weights": archive[f"{prefix}_weights"][lead],
        }
        for lead in range(LEADS)
    ]


def _archive_phase_models(archive: np.lib.npyio.NpzFile, prefix: str) -> list[list[dict]]:
    return [
        [
            {
                "mean": archive[f"{prefix}_mean"][phase_id, lead],
                "std": archive[f"{prefix}_std"][phase_id, lead],
                "y_mean": archive[f"{prefix}_y_mean"][phase_id, lead],
                "weights": archive[f"{prefix}_weights"][phase_id, lead],
            }
            for lead in range(LEADS)
        ]
        for phase_id in range(PHASES)
    ]


def _predict_environment(archive: np.lib.npyio.NpzFile, features: np.ndarray) -> np.ndarray:
    values = np.stack([
        _predict_models(_archive_models(archive, prefix), features)
        for prefix in ("env_sst_center", "env_sst_inner", "env_shear_u", "env_shear_v", "env_shear_mag")
    ], axis=-1)
    values[..., 0:2] = np.clip(values[..., 0:2], -18.5, 3.0)
    values[..., 2:4] = np.clip(values[..., 2:4], -80.0, 80.0)
    values[..., 4] = np.clip(np.hypot(values[..., 2], values[..., 3]), 0.0, 80.0)
    return values.astype("float32")


def _predict_v210(
    archive: np.lib.npyio.NpzFile,
    manifest: dict,
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    environment = _predict_environment(archive, payload["features"])
    design = refit.design_features(payload, environment)
    threshold = float(archive["regime_threshold_kt"][0])
    high = payload["base"][:, :, 0] >= threshold
    correction = np.zeros((len(payload["base"]), LEADS, 2), dtype="float32")
    for output, name in ((0, "wind"), (1, "pressure")):
        low = _predict_models(_archive_models(archive, f"{name}_low"), design)
        high_value = _predict_models(_archive_models(archive, f"{name}_high"), design)
        correction[:, :, output] = np.where(high, high_value, low)
    prediction = refit.blended_prediction(
        payload["base"], correction,
        np.asarray([manifest["selected_alphas"]["wind"], manifest["selected_alphas"]["pressure"]], dtype="float32"),
    )
    return prediction.astype("float32"), environment


def _load_payloads_and_frozen_v210() -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays = v127.load_track_arrays()
    env_target, env_mask, _future = v210._build_corrected_labels(arrays)
    payloads = v210._load_payloads(arrays, env_target, env_mask)
    manifest = json.loads((V210_ROOT / "trackformer_1_2_10_manifest.json").read_text(encoding="utf-8"))
    with np.load(V210_ROOT / "trackformer_1_2_10_corrected_environment_weights.npz", allow_pickle=False) as archive:
        frozen: dict[str, np.ndarray] = {}
        environments: dict[str, np.ndarray] = {}
        for split in ("train", "val", "test"):
            frozen[split], environments[split] = _predict_v210(archive, manifest, payloads[split])
    # The saved val/test archives are a byte-level contract for the frozen
    # environment/intensity stage.  Stop if reconstruction drifts.
    for split in ("val", "test"):
        with np.load(V210_ROOT / f"trackformer_1_2_10_{split}_predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(payloads[split]["indices"], saved["indices"]):
                raise RuntimeError(f"frozen 1.2.10 {split} indices changed")
            difference = float(np.max(np.abs(frozen[split] - saved["prediction"])))
            print(f"frozen 1.2.10 {split} reconstruction max_abs_error={difference:.7g}", flush=True)
            if difference > 2.0e-4:
                raise RuntimeError(f"frozen 1.2.10 {split} reconstruction mismatch: {difference}")
    return arrays, payloads, {"prediction": frozen, "environment": environments, "manifest": manifest}


def _phase_labels(
    arrays: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    frozen: np.ndarray,
) -> np.ndarray:
    """Causal phase gate from the frozen forecast and issue-time intensity."""

    indices = payload["indices"].astype("int64")
    current = arrays["raw_track"][indices, -1, 4:6].astype("float32")
    # Use the actual issue-time track anchor for the first forecast slope, then
    # one-step forecast differences afterwards.
    wind_step = np.diff(frozen[:, :, 0], axis=1, prepend=current[:, 0, None])
    pressure_step = np.diff(frozen[:, :, 1], axis=1, prepend=current[:, 1, None])
    strengthening = (wind_step >= 1.0) | (pressure_step <= -0.8)
    weakening = (wind_step <= -1.0) | (pressure_step >= 0.8)
    phase = np.ones(wind_step.shape, dtype="int8")
    phase[strengthening & ~weakening] = 0
    phase[weakening & ~strengthening] = 2
    return phase


def _build_intensity_features(
    arrays: dict[str, np.ndarray],
    payload: dict[str, np.ndarray],
    frozen: np.ndarray,
    environment: np.ndarray,
) -> np.ndarray:
    """Add causal state, forecast tendency, and environment interaction terms."""

    original = payload["features"].astype("float32")
    indices = payload["indices"].astype("int64")
    raw = arrays["raw_track"][indices].astype("float32")
    current = raw[:, -1, 4:6]
    persistence = refit.persistence_environment(original)
    env_delta = environment - persistence
    env_step = np.diff(environment, axis=1, prepend=persistence[:, :1])
    frozen_wp = frozen[:, :, :2]
    base_wp = payload["base"][:, :, :2]
    frozen_step = np.diff(frozen_wp, axis=1, prepend=current[:, None, :])
    base_step = np.diff(base_wp, axis=1, prepend=current[:, None, :])
    frozen_accel = np.diff(frozen_step, axis=1, prepend=np.zeros_like(frozen_step[:, :1]))

    history = []
    for back in (1, 2, 4, 8):
        history.append(current - raw[:, -1 - back, 4:6])
    history = np.concatenate(history, axis=-1)
    availability = raw[:, -1, 24:26]
    geography = raw[:, -1, 48:54]
    current_repeat = np.broadcast_to(current[:, None, :], (len(current), LEADS, 2))
    history_repeat = np.broadcast_to(history[:, None, :], (len(current), LEADS, history.shape[-1]))
    availability_repeat = np.broadcast_to(availability[:, None, :], (len(current), LEADS, availability.shape[-1]))
    geography_repeat = np.broadcast_to(geography[:, None, :], (len(current), LEADS, geography.shape[-1]))
    final_sst = environment[:, :, 0:2]
    shear = environment[:, :, 2:5]
    interactions = np.concatenate([
        frozen_wp[:, :, 0:1] * final_sst,
        frozen_wp[:, :, 0:1] * shear,
        frozen_step[:, :, 0:1] * final_sst,
        frozen_step[:, :, 0:1] * shear,
        env_delta * frozen_step[:, :, 0:1],
    ], axis=-1)
    derived = np.concatenate([
        current_repeat,
        history_repeat,
        availability_repeat,
        geography_repeat,
        base_wp,
        frozen_wp,
        base_step,
        frozen_step,
        frozen_accel,
        environment,
        persistence,
        env_delta,
        env_step,
        interactions,
    ], axis=-1)
    result = np.concatenate([original, derived], axis=-1)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    if not np.isfinite(result).all():
        raise RuntimeError("1.2.11 causal intensity features contain non-finite values")
    return result


def _blend(base: np.ndarray, correction: np.ndarray, alpha: float) -> np.ndarray:
    result = base.copy()
    result[:, :, 0] = np.clip(base[:, :, 0] + float(alpha) * correction[:, :, 0], 0.0, 190.0)
    result[:, :, 1] = np.clip(base[:, :, 1] + float(alpha) * correction[:, :, 1], 850.0, 1050.0)
    return result


def _metrics(prediction: np.ndarray, payload: dict[str, np.ndarray]) -> dict:
    return v129.intensity_metrics(prediction, payload["target"], payload["mask"])


def _select_output(
    name: str,
    output: int,
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_payload: dict,
    val_payload: dict,
    train_base: np.ndarray,
    val_base: np.ndarray,
    train_phase: np.ndarray,
    val_phase: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    residual_train = train_payload["target"][:, :, output] - train_base[:, :, output]
    valid_train = train_payload["mask"][:, :, output] > 0.5
    selection: list[dict] = []
    best = None
    for variant in ("generic", "phase"):
        for lam in REGULARIZATIONS:
            if variant == "generic":
                models = _ridge_models(train_features, residual_train, valid_train, lam)
                train_correction = _predict_models(models, train_features)
                val_correction = _predict_models(models, val_features)
            else:
                models = _fit_phase_models(train_features, residual_train, valid_train, train_phase, lam)
                train_correction = _predict_phase_models(models, train_features, train_phase)
                val_correction = _predict_phase_models(models, val_features, val_phase)
            for alpha in ALPHAS:
                train_candidate = train_base.copy()
                val_candidate = val_base.copy()
                train_candidate[:, :, output] = np.clip(train_base[:, :, output] + float(alpha) * train_correction, 0.0 if output == 0 else 850.0, 190.0 if output == 0 else 1050.0)
                val_candidate[:, :, output] = np.clip(val_base[:, :, output] + float(alpha) * val_correction, 0.0 if output == 0 else 850.0, 190.0 if output == 0 else 1050.0)
                metrics = _metrics(val_candidate, val_payload)
                key = "wind_mae_all_leads_kt" if output == 0 else "pressure_mae_all_leads_hpa"
                endpoint_key = "wind_mae_120h_kt" if output == 0 else "pressure_mae_120h_hpa"
                score = float(metrics[key]) + 0.20 * float(metrics[endpoint_key])
                record = {"variant": variant, "lambda": lam, "alpha": float(alpha), "score": score, "metrics": metrics}
                selection.append(record)
                if best is None or score < best[0]:
                    best = (score, variant, lam, float(alpha), models, train_correction, val_correction, record)
    if best is None:
        raise RuntimeError(f"no 1.2.11 {name} candidate")
    _, variant, lam, alpha, models, train_correction, val_correction, record = best
    print(f"selected {name}: variant={variant} lambda={lam:g} alpha={alpha:.2f} score={record['score']:.4f}", flush=True)
    return {"variant": variant, "lambda": lam, "alpha": alpha, "models": models}, train_correction, val_correction, selection


def _predict_head(config: dict, features: np.ndarray, base: np.ndarray, phase: np.ndarray) -> np.ndarray:
    if config["variant"] == "generic":
        return _predict_models(config["models"], features)
    return _predict_phase_models(config["models"], features, phase)


def _fit_final(config: dict, x: np.ndarray, payload: dict, base: np.ndarray, phase: np.ndarray) -> dict:
    output = 0 if config["name"] == "wind" else 1
    residual = payload["target"][:, :, output] - base[:, :, output]
    valid = payload["mask"][:, :, output] > 0.5
    if config["variant"] == "generic":
        models = _ridge_models(x, residual, valid, config["lambda"])
    else:
        models = _fit_phase_models(x, residual, valid, phase, config["lambda"])
    result = dict(config)
    result["models"] = models
    return result


def _save_model_arrays(configs: dict[str, dict]) -> dict[str, np.ndarray]:
    arrays = {
        "feature_dim": np.asarray([configs["wind"]["feature_dim"]], dtype="int64"),
        "phase_count": np.asarray([PHASES], dtype="int64"),
    }
    for name in ("wind", "pressure"):
        config = configs[name]
        arrays[f"{name}_variant"] = np.asarray([0 if config["variant"] == "generic" else 1], dtype="int8")
        arrays[f"{name}_lambda"] = np.asarray([config["lambda"]], dtype="float32")
        arrays[f"{name}_alpha"] = np.asarray([config["alpha"]], dtype="float32")
        if config["variant"] == "generic":
            for key, value in _model_arrays(config["models"]).items():
                arrays[f"{name}_{key}"] = value
        else:
            for key, value in _phase_model_arrays(config["models"]).items():
                arrays[f"{name}_{key}"] = value
    return arrays


def main() -> None:
    arrays, payloads, frozen = _load_payloads_and_frozen_v210()
    features = {}
    phases = {}
    for split in ("train", "val", "test"):
        features[split] = _build_intensity_features(
            arrays, payloads[split], frozen["prediction"][split], frozen["environment"][split]
        )
        phases[split] = _phase_labels(arrays, payloads[split], frozen["prediction"][split])
        print(split, "1.2.11 feature shape", features[split].shape, "phase counts", np.bincount(phases[split].reshape(-1), minlength=PHASES).tolist(), flush=True)

    train_configs = {}
    train_corrections = {}
    val_corrections = {}
    selections = {}
    for output, name in ((0, "wind"), (1, "pressure")):
        config, train_correction, val_correction, selection = _select_output(
            name, output,
            features["train"], features["val"],
            payloads["train"], payloads["val"],
            frozen["prediction"]["train"], frozen["prediction"]["val"],
            phases["train"], phases["val"],
        )
        config["name"] = name
        config["feature_dim"] = int(features["train"].shape[-1])
        train_configs[name] = config
        train_corrections[name] = train_correction
        val_corrections[name] = val_correction
        selections[name] = selection

    selection_val_prediction = frozen["prediction"]["val"].copy()
    for output, name in ((0, "wind"), (1, "pressure")):
        alpha = train_configs[name]["alpha"]
        selection_val_prediction[:, :, output] = np.clip(
            frozen["prediction"]["val"][:, :, output] + alpha * val_corrections[name],
            0.0 if output == 0 else 850.0,
            190.0 if output == 0 else 1050.0,
        )

    # The selected configuration is now refit on train+validation.  The
    # frozen 1.2.10 environment prediction itself is never refit or changed.
    train_val_payload = {
        key: np.concatenate([payloads["train"][key], payloads["val"][key]])
        for key in ("target", "mask")
    }
    train_val_payload["indices"] = np.concatenate([payloads["train"]["indices"], payloads["val"]["indices"]])
    train_val_features = np.concatenate([features["train"], features["val"]])
    train_val_base = np.concatenate([frozen["prediction"]["train"], frozen["prediction"]["val"]])
    train_val_phase = np.concatenate([phases["train"], phases["val"]])
    final_configs = {}
    for name in ("wind", "pressure"):
        final_configs[name] = _fit_final(train_configs[name], train_val_features, train_val_payload, train_val_base, train_val_phase)

    test_train_only = frozen["prediction"]["test"].copy()
    test_final = frozen["prediction"]["test"].copy()
    corrections_test_train_only = {}
    corrections_test_final = {}
    # Refit a train-only copy for an apples-to-apples untouched-test gate.
    for output, name in ((0, "wind"), (1, "pressure")):
        train_only = _fit_final(train_configs[name], features["train"], payloads["train"], frozen["prediction"]["train"], phases["train"])
        train_correction = _predict_head(train_only, features["test"], frozen["prediction"]["test"], phases["test"])
        final_correction = _predict_head(final_configs[name], features["test"], frozen["prediction"]["test"], phases["test"])
        corrections_test_train_only[name] = train_correction
        corrections_test_final[name] = final_correction
        alpha = train_configs[name]["alpha"]
        test_train_only[:, :, output] = np.clip(
            frozen["prediction"]["test"][:, :, output] + alpha * train_correction,
            0.0 if output == 0 else 850.0,
            190.0 if output == 0 else 1050.0,
        )
        test_final[:, :, output] = np.clip(
            frozen["prediction"]["test"][:, :, output] + alpha * final_correction,
            0.0 if output == 0 else 850.0,
            190.0 if output == 0 else 1050.0,
        )

    baseline_metrics = {
        split: _metrics(frozen["prediction"][split], payloads[split])
        for split in ("val", "test")
    }
    candidate_metrics = {
        "validation_train_only": _metrics(selection_val_prediction, payloads["val"]),
        "test_train_only": _metrics(test_train_only, payloads["test"]),
        "test_train_plus_validation": _metrics(test_final, payloads["test"]),
    }
    promoted = all([
        candidate_metrics["validation_train_only"]["wind_mae_all_leads_kt"] < baseline_metrics["val"]["wind_mae_all_leads_kt"],
        candidate_metrics["validation_train_only"]["pressure_mae_all_leads_hpa"] < baseline_metrics["val"]["pressure_mae_all_leads_hpa"],
        candidate_metrics["test_train_only"]["wind_mae_all_leads_kt"] < baseline_metrics["test"]["wind_mae_all_leads_kt"],
        candidate_metrics["test_train_only"]["pressure_mae_all_leads_hpa"] < baseline_metrics["test"]["pressure_mae_all_leads_hpa"],
    ])

    for split, prediction, correction in (
        ("val", selection_val_prediction, {name: val_corrections[name] for name in ("wind", "pressure")}),
        ("test_train_only", test_train_only, corrections_test_train_only),
        ("test_train_plus_validation", test_final, corrections_test_final),
    ):
        correction_array = np.zeros((len(prediction), LEADS, 2), dtype="float32")
        for output, name in ((0, "wind"), (1, "pressure")):
            correction_array[:, :, output] = correction[name]
        np.savez_compressed(
            OUT_ROOT / f"trackformer_1_2_11_{split}_predictions.npz",
            prediction=prediction.astype("float32"),
            base_v210=frozen["prediction"]["val" if split == "val" else "test"].astype("float32"),
            correction=correction_array,
            environment_prediction=frozen["environment"]["val" if split == "val" else "test"].astype("float32"),
            target=payloads["val" if split == "val" else "test"]["target"].astype("float32"),
            mask=payloads["val" if split == "val" else "test"]["mask"].astype("float32"),
            indices=payloads["val" if split == "val" else "test"]["indices"].astype("int64"),
        )

    arrays_out = _save_model_arrays(final_configs)
    weight_path = OUT_ROOT / "trackformer_1_2_11_intensity_weights.npz"
    np.savez_compressed(weight_path, **arrays_out)
    selection_summary = {}
    metric_keys = (
        "wind_mae_all_leads_kt",
        "pressure_mae_all_leads_hpa",
        "wind_mae_120h_kt",
        "pressure_mae_120h_hpa",
    )
    for name, rows in selections.items():
        selection_summary[name] = [
            {
                "variant": row["variant"],
                "lambda": row["lambda"],
                "alpha": row["alpha"],
                "score": row["score"],
                "metrics": {key: row["metrics"][key] for key in metric_keys},
            }
            for row in sorted(rows, key=lambda item: item["score"])[:8]
        ]
    manifest = {
        "version": VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "promoted" if promoted else "candidate_not_promoted",
        "base_model": str(V210_ROOT / "trackformer_1_2_10_manifest.json"),
        "frozen_environment_model": str(V210_ROOT / "trackformer_1_2_10_corrected_environment_weights.npz"),
        "architecture": "frozen Trackformer 1.2.10 causal environment/intensity output plus per-lead causal residual head with generic or strengthening/mature/weakening ridge experts",
        "selected_heads": {
            name: {key: value for key, value in config.items() if key not in ("models",)}
            for name, config in train_configs.items()
        },
        "feature_dim": int(features["train"].shape[-1]),
        "weights": str(weight_path),
        "metrics": {"base_v210": baseline_metrics, "candidate_v211": candidate_metrics},
        "selection": selection_summary,
        "promoted": bool(promoted),
        "quality_gate": "both wind and pressure must improve over frozen 1.2.10 on train-only chronological validation and untouched post-2019 test",
        "data_policy": {
            "future_weather_as_inputs": False,
            "future_weather_as_labels_only": True,
            "official_forecasts_as_inputs": False,
            "forecast_products_used": [],
            "tip_used_for_training": False,
            "current_and_past_weather_only_at_inference": True,
            "environment_forecast_frozen_from": "Trackformer 1.2.10 corrected causal environment head",
        },
        "artifacts": {
            "validation_train_only": str(OUT_ROOT / "trackformer_1_2_11_val_predictions.npz"),
            "test_train_only": str(OUT_ROOT / "trackformer_1_2_11_test_train_only_predictions.npz"),
            "test_train_plus_validation": str(OUT_ROOT / "trackformer_1_2_11_test_train_plus_validation_predictions.npz"),
        },
    }
    path = OUT_ROOT / "trackformer_1_2_11_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "weights": str(weight_path), "promoted": promoted, "metrics": manifest["metrics"], "selected_heads": manifest["selected_heads"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
