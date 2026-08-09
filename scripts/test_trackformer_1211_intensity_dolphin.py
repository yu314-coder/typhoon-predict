#!/usr/bin/env python3
"""Evaluate Trackformer 1.2.11 causal intensity head on Dolphin."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import v128_causal_intensity_train as v128  # noqa: E402
import v129_causal_environment_intensity_train as v129  # noqa: E402
import v129_causal_environment_refit as refit  # noqa: E402
import v211_intensity_only_train as v211  # noqa: E402
import scripts.test_trackformer_126_dolphin_20260807 as dolphin  # noqa: E402
import scripts.test_trackformer_127_v11_intensity_dolphin as v127_case  # noqa: E402
import scripts.test_trackformer_128_causal_intensity_dolphin as v128_case  # noqa: E402
from render_trackformer_125_graph_world_map import draw_coastlines  # noqa: E402


OUT_ROOT = ROOT / "v211" / "intensity_only"
V210_ROOT = ROOT / "v210" / "corrected_environment_expert"
COMPARISON_ENV_JSON = ROOT / "v129" / "causal_environment_intensity" / "dolphin_20260807_environment_forecast.json"
MANIFEST = OUT_ROOT / "trackformer_1_2_11_manifest.json"
WEIGHTS = OUT_ROOT / "trackformer_1_2_11_intensity_weights.npz"
V210_MANIFEST = V210_ROOT / "trackformer_1_2_10_manifest.json"
V210_WEIGHTS = V210_ROOT / "trackformer_1_2_10_corrected_environment_weights.npz"
OUT_JSON = OUT_ROOT / "dolphin_20260807_evaluation.json"
OUT_GRAPH = ROOT / "paper" / "dolphin_trackformer_1_2_11_intensity_graph.png"
OUT_MAP = ROOT / "paper" / "dolphin_trackformer_1_2_11_world_map.png"
OUT_HTML = ROOT / "paper" / "dolphin_trackformer_1_2_11_intensity.html"
LEADS = np.arange(6, 126, 6, dtype="int32")


def _ridge_models(archive: dict, prefix: str) -> list[dict]:
    return [
        {"mean": archive[f"{prefix}_mean"][lead], "std": archive[f"{prefix}_std"][lead], "y_mean": archive[f"{prefix}_y_mean"][lead], "weights": archive[f"{prefix}_weights"][lead]}
        for lead in range(len(LEADS))
    ]


def _expert_models(archive: dict, name: str, regime: str) -> list[dict]:
    return _ridge_models(archive, f"{name}_{regime}")


def _predict_models(models: list[dict], features: np.ndarray) -> np.ndarray:
    result = np.zeros((len(features), len(LEADS)), dtype="float32")
    for lead, model in enumerate(models):
        result[:, lead] = (model["y_mean"] + ((features[:, lead] - model["mean"]) / model["std"]) @ model["weights"]).astype("float32")
    return result


def _environment_models(archive: dict) -> list[list[dict]]:
    return [_ridge_models(archive, name) for name in ("env_sst_center", "env_sst_inner", "env_shear_u", "env_shear_v", "env_shear_mag")]


def _predict_environment(models: list[list[dict]], features: np.ndarray) -> np.ndarray:
    values = np.stack([_predict_models(model, features) for model in models], axis=-1)
    values[..., 0:2] = np.clip(values[..., 0:2], -18.5, 3.0)
    values[..., 2:4] = np.clip(values[..., 2:4], -80.0, 80.0)
    values[..., 4] = np.clip(np.hypot(values[..., 2], values[..., 3]), 0.0, 80.0)
    return values.astype("float32")


def _load_case() -> dict:
    v210_manifest = json.loads(V210_MANIFEST.read_text(encoding="utf-8"))
    v211_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    history = dolphin._canonical_history()
    field_archive = np.load(dolphin.FIELD_PATH, allow_pickle=False)
    try:
        inputs = dolphin._model_inputs(history, field_archive)
    finally:
        field_archive.close()

    points, environment, route_diagnostics = v127_case._route_and_environment(history, inputs, np.asarray(inputs["sst"], dtype="float32"))
    model_inputs = v127_case._structure_inputs(inputs)
    v127_state, v127_metadata = v127_case._predict_new(model_inputs, environment)
    v128_state, v128_metadata = v128_case._predict_v128(inputs["raw_track"], environment, v127_state)
    raw_features = v128_case._case_features(inputs["raw_track"], environment, v127_state)
    features = v129.append_v128_output(raw_features, v128_state[None, ...])
    with np.load(V210_WEIGHTS, allow_pickle=False) as archive:
        environment_forecast = _predict_environment(_environment_models(archive), features)
        design = refit.design_features({"features": features, "base": v128_state[None, ...]}, environment_forecast)
        threshold = float(archive["regime_threshold_kt"][0])
        experts = {
            "wind": {regime: _expert_models(archive, "wind", regime) for regime in ("low", "high")},
            "pressure": {regime: _expert_models(archive, "pressure", regime) for regime in ("low", "high")},
        }
    high = v128_state[None, :, 0] >= threshold
    correction = np.zeros((1, len(LEADS), 2), dtype="float32")
    for output, name in ((0, "wind"), (1, "pressure")):
        low_value = refit.predict_ridge(experts[name]["low"], design)
        high_value = refit.predict_ridge(experts[name]["high"], design)
        correction[:, :, output] = np.where(high, high_value, low_value)
    v210_state = refit.blended_prediction(
        v128_state[None, ...], correction,
        np.asarray([v210_manifest["selected_alphas"]["wind"], v210_manifest["selected_alphas"]["pressure"]], dtype="float32"),
    )[0]

    # Apply the serialized 1.2.11 intensity-only head to the frozen 1.2.10
    # output.  The route and environment forecast are unchanged.
    causal_arrays = {"raw_track": inputs["raw_track"][None, ...]}
    payload = {
        "indices": np.asarray([0], dtype="int64"),
        "features": features,
        "base": v128_state[None, ...],
    }
    frozen_v210 = v210_state[None, ...]
    intensity_features = v211._build_intensity_features(causal_arrays, payload, frozen_v210, environment_forecast)
    phase = v211._phase_labels(causal_arrays, payload, frozen_v210)
    correction_1211 = np.zeros((1, len(LEADS), 2), dtype="float32")
    with np.load(WEIGHTS, allow_pickle=False) as archive:
        for output, name in ((0, "wind"), (1, "pressure")):
            if int(archive[f"{name}_variant"][0]) == 0:
                models = v211._archive_models(archive, name)
                value = v211._predict_models(models, intensity_features)
            else:
                models = v211._archive_phase_models(archive, name)
                value = v211._predict_phase_models(models, intensity_features, phase)
            alpha = float(v211_manifest["selected_heads"][name]["alpha"])
            correction_1211[:, :, output] = alpha * value
    v211_state = frozen_v210.copy()
    v211_state[:, :, :2] = np.clip(frozen_v210[:, :, :2] + correction_1211, [0.0, 850.0], [190.0, 1050.0])
    v211_state = v211_state[0]

    v128_forecast = v128_case._enrich(v127_case._rows(points, v128_state), "Trackformer 1.2.8")
    v210_forecast = v128_case._enrich(v127_case._rows(points, v210_state), "Trackformer 1.2.10")
    v211_forecast = v128_case._enrich(v127_case._rows(points, v211_state), "Trackformer 1.2.11")
    actual = v127_case._actual_rows()
    jma = dolphin._jma_official()
    jtwc = dolphin._jtwc_official()
    comparison_environment = json.loads(COMPARISON_ENV_JSON.read_text(encoding="utf-8")).get("official_environment", {"jma": [], "jtwc": []})
    return {
        "version": "Trackformer1.2.11-causal-intensity-only-Dolphin",
        "issue_time_utc": dolphin.ISSUE_TIME.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "v128_forecast": v128_forecast,
        "v210_forecast": v210_forecast,
        "v211_forecast": v211_forecast,
        "jma_official": jma,
        "jtwc_official": jtwc,
        "actual_rows": actual,
        "environment_forecast": [{"lead_hours": int(lead), "sst_anomaly_c": float(value[0]), "sst_c": float(value[0] + 28.0), "sst_inner_anomaly_c": float(value[1]), "shear_u_ms": float(value[2]), "shear_v_ms": float(value[3]), "shear_200_850_ms": float(value[4])} for lead, value in zip(LEADS, environment_forecast[0])],
        "environment_persistence": [{"lead_hours": int(lead), "sst_anomaly_c": float(value[0]), "sst_c": float(value[0] + 28.0), "sst_inner_anomaly_c": float(value[1]), "shear_u_ms": float(value[2]), "shear_v_ms": float(value[3]), "shear_200_850_ms": float(value[4])} for lead, value in zip(LEADS, refit.persistence_environment(features)[0])],
        "official_environment": comparison_environment,
        "routes": {
            "observed": [[row["lon"], row["lat"]] for row in history],
            "model": points.tolist(),
            "jma": [[history[-1]["lon"], history[-1]["lat"]]] + [[row["lon"], row["lat"]] for row in jma if row["lead_hours"] > 0],
            "jtwc": [[history[-1]["lon"], history[-1]["lat"]]] + [[row["lon"], row["lat"]] for row in jtwc],
        },
        "scores": {
            "trackformer_1_2_8": v128_case._score(v128_forecast, actual),
            "trackformer_1_2_10": v128_case._score(v210_forecast, actual),
            "trackformer_1_2_11": v128_case._score(v211_forecast, actual),
        },
        "environment_model": {
            "manifest": str(V210_MANIFEST.relative_to(ROOT)),
            "targets": ["SST center anomaly", "inner SST anomaly", "200-850 shear u", "200-850 shear v", "200-850 shear magnitude"],
            "threshold_kt": threshold,
        },
        "intensity_model": {
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "weights": str(WEIGHTS.relative_to(ROOT)),
            "base_intensity": "frozen Trackformer 1.2.10 corrected environment expert",
            "selected_heads": v211_manifest["selected_heads"],
        },
        "data_policy": v210_manifest["data_policy"],
        "route_diagnostics": route_diagnostics,
        "models": {"v128": v128_metadata, "v127": v127_metadata},
    }


def _draw_graph(payload: dict) -> None:
    x = np.asarray(LEADS)
    actual = payload["actual_rows"]
    candidate = payload["v211_forecast"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 13.5), sharex=True, dpi=170)
    fig.suptitle("Dolphin | Trackformer 1.2.11 causal intensity", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.955, "Future SST/shear are frozen model forecasts generated from issue-time/past inputs; JMA/JTWC are comparison-only", ha="center", fontsize=8, color="#334155")
    specs = [
        (payload["v128_forecast"], "#dc2626", "-", "Trackformer 1.2.8"),
        (payload["v210_forecast"], "#64748b", "--", "Trackformer 1.2.10 corrected"),
        (candidate, "#0891b2", "-", "Trackformer 1.2.11"),
    ]
    for rows, color, style, label in specs:
        axes[0].plot(x, [r["vmax_1min_kt"] for r in rows], color=color, linestyle=style, marker="o", label=f"{label} 1-min")
        axes[0].plot(x, [r["vmax_10min_kt"] for r in rows], color=color, linestyle=style, marker="x", alpha=0.48, label=f"{label} 10-min eq.")
    for rows, color, style, label in ((payload["jma_official"], "#2563eb", "-.", "JMA official 10-min"), (payload["jtwc_official"], "#7c3aed", (0, (2, 3)), "JTWC official 1-min")):
        rows = [r for r in rows if r.get("vmax_kt") is not None]
        axes[0].plot([r["lead_hours"] for r in rows], [r["vmax_kt"] for r in rows], color=color, linestyle=style, marker="s", label=label)
    truth = [(r["lead_hours"], actual.get(r["valid_time_utc"], {}).get("vmax_kt")) for r in candidate]
    truth = [(a, b) for a, b in truth if b is not None]
    if truth: axes[0].plot(*zip(*truth), color="#111827", marker="s", label="Later JMA observation (score only)")
    axes[0].set_ylabel("Wind (kt)"); axes[0].grid(alpha=.3); axes[0].legend(fontsize=7, ncol=3)

    for rows, color, style, label in (
        (payload["v128_forecast"], "#dc2626", "-", "Trackformer 1.2.8"),
        (payload["v210_forecast"], "#64748b", "--", "Trackformer 1.2.10 corrected"),
        (candidate, "#0891b2", "-", "Trackformer 1.2.11"),
    ):
        axes[1].plot(x, [r["pressure_hpa"] for r in rows], color=color, linestyle=style, marker="o", label=label)
    rows = [r for r in payload["jma_official"] if r.get("pressure_hpa") is not None]
    axes[1].plot([r["lead_hours"] for r in rows], [r["pressure_hpa"] for r in rows], color="#2563eb", linestyle="-.", marker="s", label="JMA official")
    truth = [(r["lead_hours"], actual.get(r["valid_time_utc"], {}).get("pressure_hpa")) for r in candidate]
    truth = [(a, b) for a, b in truth if b is not None]
    if truth: axes[1].plot(*zip(*truth), color="#111827", marker="s", label="Later JMA observation (score only)")
    axes[1].invert_yaxis(); axes[1].set_ylabel("Pressure (hPa)"); axes[1].grid(alpha=.3); axes[1].legend(fontsize=7.5, ncol=3)

    env = payload["environment_forecast"]; persistence = payload["environment_persistence"]
    axes[2].plot(x, [r["sst_c"] for r in env], color="#0891b2", marker="o", label="Frozen 1.2.10 predicted SST")
    axes[2].plot(x, [r["sst_c"] for r in persistence], color="#dc2626", linestyle="--", marker="x", label="Issue-time SST persistence")
    for key, color, style in (("jma", "#2563eb", "-."), ("jtwc", "#7c3aed", (0, (2, 3)))):
        rows = payload["official_environment"][key]
        axes[2].plot(x, [r["sst_c"] for r in rows], color=color, linestyle=style, label=f"analysis along {key.upper()} route")
    axes[2].axhline(26.0, color="#64748b", linestyle=":", label="26 C reference")
    axes[2].set_ylabel("SST (C)"); axes[2].grid(alpha=.3); axes[2].legend(fontsize=7, ncol=2)

    axes[3].plot(x, [r["shear_200_850_ms"] for r in env], color="#0891b2", marker="o", label="Frozen 1.2.10 predicted 200-850 shear")
    axes[3].plot(x, [r["shear_200_850_ms"] for r in persistence], color="#dc2626", linestyle="--", marker="x", label="Issue-time shear persistence")
    for key, color, style in (("jma", "#2563eb", "-."), ("jtwc", "#7c3aed", (0, (2, 3)))):
        rows = payload["official_environment"][key]
        axes[3].plot(x, [r["shear_200_850_ms"] for r in rows], color=color, linestyle=style, label=f"analysis along {key.upper()} route")
    axes[3].set_ylabel("Shear (m/s)"); axes[3].set_xlabel("Forecast lead (hours)"); axes[3].set_xlim(0, 120); axes[3].grid(alpha=.3); axes[3].legend(fontsize=7, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(OUT_GRAPH, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _draw_map(payload: dict) -> None:
    observed = np.asarray(payload["routes"]["observed"], dtype="float64")
    model = np.asarray(payload["routes"]["model"], dtype="float64")
    jma = np.asarray(payload["routes"]["jma"], dtype="float64")
    jtwc = np.asarray(payload["routes"]["jtwc"], dtype="float64")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=170, gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.suptitle("Dolphin | Trackformer 1.2.11 intensity route comparison", fontsize=14, fontweight="bold")
    for axis, xlim, ylim, title in zip(axes, [(-180, 180), (105, 180)], [(-10, 75), (5, 55)], ["World overview", "Western Pacific detail"]):
        draw_coastlines(axis); axis.set_xlim(*xlim); axis.set_ylim(*ylim); axis.grid(color="#cbd5e1", linewidth=.45, alpha=.65)
        axis.plot(observed[:, 0], observed[:, 1], color="#111827", linewidth=2.4)
        axis.plot(model[:, 0], model[:, 1], color="#0891b2", linewidth=2.8)
        axis.plot(jma[:, 0], jma[:, 1], color="#2563eb", linewidth=2, linestyle="-.")
        axis.plot(jtwc[:, 0], jtwc[:, 1], color="#7c3aed", linewidth=2, linestyle=(0, (2, 3)))
        wind = np.asarray([r["vmax_1min_kt"] for r in payload["v211_forecast"]])
        axis.scatter(model[1:, 0], model[1:, 1], c=wind, cmap="magma", vmin=30, vmax=140, s=31, edgecolors="white", linewidths=.35, zorder=5)
        axis.set_title(title, loc="left", fontsize=11); axis.set_xlabel("Longitude (deg)"); axis.set_ylabel("Latitude (deg)")
    axes[0].legend(handles=[Line2D([0], [0], color="#111827", linewidth=2.4, label="Observed history"), Line2D([0], [0], color="#0891b2", linewidth=2.8, label="Trackformer route (frozen geometry)"), Line2D([0], [0], color="#2563eb", linestyle="-.", label="JMA official"), Line2D([0], [0], color="#7c3aed", linestyle=(0, (2, 3)), label="JTWC official")], fontsize=7.5, loc="lower left")
    scalar = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(30, 140)); scalar.set_array([]); fig.colorbar(scalar, ax=axes, shrink=.82, pad=.02, label="1-min wind (kt)")
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(OUT_MAP, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _write_html(payload: dict) -> None:
    score = html.escape(json.dumps(payload["scores"], indent=2))
    OUT_HTML.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Dolphin Trackformer 1.2.11</title><style>body{font:15px system-ui;margin:24px auto;max-width:1400px;color:#0f172a;background:#f8fafc}img{max-width:100%;display:block;background:#fff;border:1px solid #cbd5e1;margin:12px 0 24px}.notice{padding:12px;border-left:4px solid #0891b2;background:#fff}pre{background:#fff;padding:12px;overflow:auto}</style></head><body>"
        + "<h1>Dolphin | Trackformer 1.2.11 causal intensity</h1>"
        + f"<p>Issue: <code>{html.escape(payload['issue_time_utc'])}</code>.</p><p class='notice'><b>Inference policy:</b> Trackformer 1.2.11 uses current/past track and weather analysis, then the frozen 1.2.10 causal SST and 200-850 hPa shear forecasts. Future weather rows are training labels only. JMA/JTWC forecast paths are loaded after inference as comparison overlays and are never model inputs. The route and radius remain the frozen route model outputs; this version upgrades only the wind/pressure intensity residual.</p>"
        + f"<h2>World map</h2><img src='{OUT_MAP.name}' alt='Dolphin Trackformer 1.2.11 route comparison'><h2>Intensity and predicted environment</h2><img src='{OUT_GRAPH.name}' alt='Dolphin Trackformer 1.2.11 intensity, SST, and shear graph'><h2>Dolphin scores</h2><pre>{score}</pre><h2>Environment contract</h2><pre>{html.escape(json.dumps(payload['environment_model'], indent=2))}</pre><h2>Intensity contract</h2><pre>{html.escape(json.dumps(payload['intensity_model'], indent=2))}</pre></body></html>",
        encoding="utf-8",
    )


def main() -> None:
    payload = _load_case()
    OUT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    _draw_graph(payload); _draw_map(payload); _write_html(payload)
    print(json.dumps({"json": str(OUT_JSON), "graph": str(OUT_GRAPH), "map": str(OUT_MAP), "html": str(OUT_HTML), "scores": payload["scores"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
