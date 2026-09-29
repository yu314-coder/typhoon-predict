"""Reproduce the public 270-case chart and recent isobar example.

Use --source-root to prepare plot data from the local research archives. Without
it, only the small published arrays under evaluation/release_data are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def prepare(root, data):
    old_path = root / "benchmark_strict_causal_post2021/routes_and_truth.npz"
    new_path = root / "benchmark_ensemble50/v173_e4/v173_e4_causal_ensemble50.npz"
    report = json.loads((root / "benchmark_strict_causal_post2021/trackformer_100_storms_3_days_gpu.json").read_text())
    with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
        assert np.array_equal(old["source_rows"], new["source_rows"])
        assert np.array_equal(old["truth_local"], new["truth_local"])
        assert len(old["source_rows"]) == 270
        # The archived builder returned absolute [latitude, longitude], but its
        # caller stored it as v11_local and scored it as [east_km, north_km].
        # Recover that original route, then apply the same issue-latitude local
        # projection used for the 1.2 saved predictions. Never edit the archive.
        absolute = old["v11_local"].astype("float64")
        base_lat = new["base_lat"].astype("float64")[:, None]
        base_lon = new["base_lon"].astype("float64")[:, None]
        assert np.all(np.abs(absolute[..., 0]) <= 90)
        assert np.all((absolute[..., 1] >= 0) & (absolute[..., 1] <= 360))
        delta_lon = (absolute[..., 1] - base_lon + 180) % 360 - 180
        scale = 111.2 * np.maximum(np.cos(np.deg2rad(base_lat)), .2)
        corrected = np.stack([delta_lon * scale, (absolute[..., 0] - base_lat) * 111.2], axis=-1)
        recovered = np.stack([base_lat + corrected[..., 1]/111.2,
                              (base_lon + corrected[..., 0]/scale) % 360], axis=-1)
        assert np.allclose(recovered, absolute, atol=1e-8, rtol=0)
        np.savez_compressed(data / "routes_270.npz", source_rows=old["source_rows"],
            issue_times=old["issue_times"], truth_local=old["truth_local"],
            v11_local=corrected, v12_local=new["ensemble50_local"],
            v11_original_lat_lon=absolute, base_lat=base_lat[:,0], base_lon=base_lon[:,0])
    meta = {"case_count": 270, "storm_count": len({c["storm_id"] for c in report["cohort"]}),
        "cases": [{k: c[k] for k in ("storm_id", "issue_time_utc", "source_row")} for c in report["cohort"]],
        "leads_hours": list(range(6, 121, 6)), "member_policy": {
            "1.1": "saved released causal route pipeline", "1.2": "50 causal historical-input perturbations"},
        "case_and_truth_alignment": "Exact source-row order and truth-array equality verified",
        "v11_coordinate_correction": "Archived v11_local held absolute latitude/longitude from build_v11_routes; converted to issue-centred east/north km using the same projection as 1.2. Round-trip back to original coordinates verified. Original archives preserved.",
        "coordinate_bug_evidence": "benchmark_100_storms_3_days_gpu.py build_v11_routes returns local_to_absolute; benchmark_strict_causal_post2021.py stores and scores that output as local",
        "source_sha256": {old_path.name: digest(old_path), new_path.name: digest(new_path)},
        "evaluation": "Previously inspected development cohort; not an untouched test",
        "pressure_comparison": "No verified same-270 1.1 central-pressure predictions in these route artifacts"}
    (data / "cohort_270.json").write_text(json.dumps(meta, indent=2) + "\n")
    run_path = root / "local_forecast_archive/runs/2026092712-TC2632-2026092706-local-v17x-corrected-static.json"
    run = json.loads(run_path.read_text())
    selected = run["trackformer_runs"]["v173"]
    assert selected["forecast"]["checkpoint_sha256"] == "f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0"
    field = selected["field"]
    indices = np.array([1, 4, 6])  # +6/+24/+36 stay inside the regional map.
    route = selected["forecast"]["route"]
    np.savez_compressed(data / "surigae_pressure.npz", latitude=field["latitude"],
        longitude=field["longitude"], pressure_hpa=np.asarray(field["pressure_hpa"])[indices],
        lead_hours=indices * 6, route=np.array([[p["lat"], p["lon"]] for p in route[:7]]),
        central_pressure_hpa=np.array([route[i]["pressure_hpa"] for i in indices]))
    provenance = {"storm_name": run["storms"][0]["name"], "issue_time_utc": route[0]["valid_time_utc"],
        "valid_times_utc": [field["valid_times_utc"][i] for i in indices],
        "members": selected["forecast"]["members"], "source_sha256": digest(run_path),
        "checkpoint_sha256": selected["forecast"]["checkpoint_sha256"],
        "source_note": field["source_note"], "pressure_display": "Unmodified model regional field with 4 hPa contours"}
    (data / "surigae_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rings = json.loads((root / "trackformer-weatherlab-site/public/data/history/coastlines.json").read_text())
    rings = [ring for ring in rings if any(114 <= x <= 144 and 11 <= y <= 41 for x, y in ring)]
    (data / "map_boundaries.json").write_text(json.dumps(rings, separators=(",", ":")) + "\n")


def frechet(a, b):
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    cache = np.empty_like(distances)
    for i in range(len(a)):
        for j in range(len(b)):
            prior = 0 if i == j == 0 else cache[i, j-1] if i == 0 else cache[i-1, j] if j == 0 else min(cache[i-1, j], cache[i-1, j-1], cache[i, j-1])
            cache[i, j] = max(prior, distances[i, j])
    return cache[-1, -1]


def similarity_metrics(routes, truth):
    steps = [np.diff(np.concatenate([np.zeros((270, 1, 2)), p], axis=1), axis=1) for p in routes]
    truth_step = np.diff(np.concatenate([np.zeros((270, 1, 2)), truth], axis=1), axis=1)
    # Use exactly the same non-stationary case-leads for both heading scores.
    valid = np.linalg.norm(truth_step, axis=-1) > 1
    for step in steps:
        valid &= np.linalg.norm(step, axis=-1) > 1
    result = []
    for route, step in zip(routes, steps):
        delta = np.arctan2(step[..., 1], step[..., 0]) - np.arctan2(truth_step[..., 1], truth_step[..., 0])
        angle = np.degrees(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))
        a = (route - route.mean(axis=1, keepdims=True)).reshape(270, -1)
        b = (truth - truth.mean(axis=1, keepdims=True)).reshape(270, -1)
        corr = np.sum(a*b, axis=1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-8)
        distance = np.array([frechet(p, t) for p, t in zip(route, truth)])
        length = np.linalg.norm(truth_step, axis=-1).sum(axis=1)
        result.append({"mean_direction_error_deg": float(angle[valid].mean()),
            "direction_valid_case_leads": int(valid.sum()),
            "direction_error_by_lead_deg": [float(angle[:,i][valid[:,i]].mean()) for i in range(20)],
            "shape_similarity": float(np.clip((corr+1)/2, 0, 1).mean()),
            "path_similarity": float(np.exp(-distance/np.maximum(length, 1)).mean()),
            "frechet_distance_km": float(distance.mean())})
    return result


def select_showcase(routes, truth, meta, base_lat, base_lon):
    steps = [np.diff(np.concatenate([np.zeros((len(p), 1, 2)), p], axis=1), axis=1) for p in [truth, *routes]]
    valid = np.logical_and.reduce([np.linalg.norm(s, axis=-1) > 1 for s in steps])
    all_cases = []
    for i, case in enumerate(meta["cases"]):
        row = {"case_index": i, **case, "base_lat": float(base_lat[i]), "base_lon": float(base_lon[i]),
               "truth_path_length_km": float(np.linalg.norm(steps[0][i], axis=-1).sum()), "models": {}}
        for key, route, step in zip(("1.1", "1.2"), routes, steps[1:]):
            a = (route[i] - route[i].mean(axis=0)).ravel()
            b = (truth[i] - truth[i].mean(axis=0)).ravel()
            cosine = np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-8)
            delta = np.arctan2(step[i,:,1], step[i,:,0]) - np.arctan2(steps[0][i,:,1], steps[0][i,:,0])
            angles = np.degrees(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))
            error = np.linalg.norm(route[i]-truth[i], axis=-1)
            row["models"][key] = {"mean_track_error_km": float(error.mean()),
                "track_error_120h_km": float(error[-1]),
                "shape_similarity": float(np.clip((1+cosine)/2, 0, 1)),
                "direction_error_deg": float(angles[valid[i]].mean()) if valid[i].any() else None,
                "direction_valid_steps": int(valid[i].sum())}
        all_cases.append(row)
    eligible = [c for c in all_cases if 0 <= c["base_lat"] <= 60 and 100 <= c["base_lon"] <= 180
                and c["truth_path_length_km"] >= 300 and c["models"]["1.2"]["shape_similarity"] >= .9
                and c["models"]["1.2"]["direction_valid_steps"] >= 18
                and c["models"]["1.2"]["direction_error_deg"] <= 30]
    chosen, storms = [], set()
    for case in sorted(eligible, key=lambda c: (c["models"]["1.2"]["mean_track_error_km"], c["case_index"])):
        if case["storm_id"] not in storms:
            chosen.append(case)
            storms.add(case["storm_id"])
        if len(chosen) == 6:
            break
    if len(chosen) != 6:
        raise ValueError("Fewer than six distinct storms meet the declared showcase thresholds")
    return {"selection": "Selected best-performing examples, not a representative sample",
        "rule": "Within 0-60N / 100-180E; observed path length >=300 km; 1.2 shape >=0.90; direction error <=30 degrees on >=18 common valid leads; ascending 1.2 mean track error; one case per storm; first six distinct storms",
        "eligible_case_count": len(eligible), "selected": chosen, "all_case_metrics": all_cases}


def plot_benchmark(data, output):
    meta = json.loads((data / "cohort_270.json").read_text())
    with np.load(data / "routes_270.npz", allow_pickle=False) as z:
        truth = z["truth_local"].astype("float64")
        routes = [z[k].astype("float64") for k in ("v11_local", "v12_local")]
        base_lat, base_lon = z["base_lat"], z["base_lon"]
        errors = [np.linalg.norm(p - truth, axis=-1) for p in routes]
    similarity = similarity_metrics(routes, truth)
    assert all(e.shape == (270, 20) and np.isfinite(e).all() for e in errors)
    colors = ["#65758b", "#00857d"]
    labels = ["Trackformer 1.1", "Trackformer 1.2 · mean of 50"]
    fig = plt.figure(figsize=(13, 9), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.25))
    panels = [
        ("Track-direction error", [s["mean_direction_error_deg"] for s in similarity], "degrees · lower is better", None),
        ("Route-shape similarity", [s["shape_similarity"] for s in similarity], "score 0–1 · higher is better", 1.15),
        ("Path similarity (Fréchet)", [s["path_similarity"] for s in similarity], "score 0–1 · higher is better", 1.15),
        ("Mean position error", [e.mean() for e in errors], "km · lower is better", None)]
    for panel, (title, vals, units, limit) in enumerate(panels):
        ax = fig.add_subplot(gs[panel//2, panel%2])
        ax.barh([0, 1], vals, color=colors, height=.5)
        ax.set_yticks([0, 1], labels)
        ax.invert_yaxis()
        ax.set_xlim(0, limit or max(vals) * 1.23)
        ax.set_xlabel(units)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
        for i, value in enumerate(vals):
            ax.text(value + max(vals)*.025, i, f"{value:.3f}" if limit else f"{value:,.1f}", va="center", fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
    ax = fig.add_subplot(gs[2, :])
    leads = np.arange(6, 121, 6)
    for e, c, label in zip(errors, colors, labels):
        ax.plot(leads, e.mean(axis=0), color=c, label=label, linewidth=2.6, marker="o", markersize=3)
    ax.set(xlabel="Forecast lead (hours)", ylabel="Mean track error (km)", xlim=(6, 120), ylim=(0, None))
    ax.set_xticks([6, 12, 24, 48, 72, 96, 120])
    ax.grid(alpha=.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.suptitle("Trackformer 1.2 vs 1.1 · 270 forecast cases / 90 storms", fontsize=17, fontweight="bold")
    fig.text(.5, -.04, "Same cases, truth and 20 leads • Direction: 5,382 common valid steps • 270 cases ≠ 50 ensemble members", ha="center", fontsize=10)
    fig.savefig(output / "trackformer_1_2_vs_1_1_270_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    meta["metrics"] = {label: {**sim, "mean_track_error_km": float(e.mean()),
        "track_error_120h_km": float(e[:, -1].mean()), "track_error_by_lead_km": e.mean(axis=0).tolist()}
        for label, e, sim in zip(("1.1", "1.2"), errors, similarity)}
    meta["similarity_definitions"] = {
        "direction": "Mean absolute wrapped 6-hour heading difference, common mask: both models and truth move >1 km; issue origin included",
        "shape": "Mean (1 + cosine similarity)/2 of centered flattened 20-lead routes; translation and scalar scale removed, orientation preserved",
        "path": "Mean exp(-discrete Frechet distance / truth route length); same 20 forecast points and issue-origin-inclusive truth length",
        "position": "Euclidean displacement error in the existing benchmark local-km coordinate system"}
    meta["mean_track_error_reduction_percent"] = float(100*(1-errors[1].mean()/errors[0].mean()))
    (output / "trackformer_1_2_vs_1_1_270_metrics.json").write_text(json.dumps(meta, indent=2) + "\n")
    showcase = select_showcase(routes, truth, meta, base_lat, base_lon)
    (output / "trackformer_1_2_showcase_selection.json").write_text(json.dumps(showcase, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    for ax, selected in zip(axes.ravel(), showcase["selected"]):
        case_index = selected["case_index"]
        for route, color, label in [(truth, "#182c42", "Observed"), (routes[0], colors[0], "1.1"), (routes[1], colors[1], "1.2 · mean of 50")]:
            points = np.vstack([np.zeros((1,2)), route[case_index]])
            ax.plot(points[:,0], points[:,1], color=color, label=label, linewidth=2,
                    linestyle="--" if label=="1.1" else "-", marker="o", markersize=2.5)
            ax.scatter(*points[-1], color=color, s=24, zorder=5)
        c = meta["cases"][case_index]
        scores = selected["models"]["1.2"]
        ax.set_title(f"{c['storm_id']} · {c['issue_time_utc'][:10]}\n1.2: {scores['mean_track_error_km']:.0f} km · shape {scores['shape_similarity']:.3f} · direction {scores['direction_error_deg']:.1f}°", fontsize=10)
        ax.set(xlabel="East displacement (km)", ylabel="North displacement (km)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.18)
    axes[0,0].legend(frameon=False, fontsize=9)
    fig.suptitle("Trackformer 1.2 · selected best-performing route examples\nSix distinct storms from the 270-case benchmark", fontsize=16, fontweight="bold")
    fig.text(.5, -.045, "Selected for low 1.2 track error, high shape similarity and low direction error. These are showcase examples, not typical performance.", ha="center", fontsize=10)
    fig.savefig(output / "trackformer_1_2_vs_1_1_route_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pressure(data, output):
    meta = json.loads((data / "surigae_provenance.json").read_text())
    rings = json.loads((data / "map_boundaries.json").read_text())
    with np.load(data / "surigae_pressure.npz", allow_pickle=False) as z:
        lat, lon, maps, leads, route = (z[k] for k in ("latitude", "longitude", "pressure_hpa", "lead_hours", "route"))
        fig, axes = plt.subplots(1, 3, figsize=(14, 6), layout="constrained")
        for ax, field, lead, valid in zip(axes, maps, leads, meta["valid_times_utc"]):
            im = ax.contourf(lon, lat, field, levels=np.arange(948, 1021, 2), cmap="RdYlBu_r", extend="both")
            contours = ax.contour(lon, lat, field, levels=np.arange(948, 1021, 4), colors="#26374b", linewidths=.65)
            ax.clabel(contours, levels=np.arange(948, 1021, 12), fmt="%d", fontsize=8, inline_spacing=5)
            for ring in rings:
                coords = np.asarray(ring)
                ax.plot(coords[:, 0], coords[:, 1], color="#28343c", linewidth=.8)
            end = int(lead // 6)
            ax.plot(route[:end+1, 1], route[:end+1, 0], color="white", linewidth=4)
            ax.plot(route[:end+1, 1], route[:end+1, 0], color="#9d1752", linewidth=1.9)
            ax.scatter(route[end, 1], route[end, 0], s=60, c="#9d1752", edgecolor="white", zorder=5)
            ax.set(xlim=(114, 144), ylim=(11, 41), xlabel="Longitude °E")
            ax.set_ylabel("Latitude °N")
            ax.set_aspect(1/np.cos(np.deg2rad(26)))
            ax.set_title(f"+{lead} h · {valid[5:16].replace('T', ' ')} UTC", fontsize=11, fontweight="bold")
            ax.set_xticks(np.arange(115, 145, 5))
            ax.set_yticks(np.arange(15, 41, 5))
            ax.grid(alpha=.15)
        fig.colorbar(im, ax=axes, label="Sea-level pressure (hPa) · labelled isobars every 4 hPa", shrink=.78, pad=.015)
        fig.suptitle("Trackformer 1.2 · Surigae pressure forecast\nIssue: 27 September 2026, 12:00 UTC · one forecast", fontsize=16, fontweight="bold")
        fig.text(.5, -.015, "Magenta: forecast track and centre • Actual model regional field • Experimental GFS-input transfer", ha="center", fontsize=10)
        fig.savefig(output / "trackformer_1_2_surigae_isobars.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "evaluation")
    args = parser.parse_args()
    data = args.output_dir / "release_data"
    data.mkdir(parents=True, exist_ok=True)
    if args.source_root:
        prepare(args.source_root, data)
    plot_benchmark(data, args.output_dir)
    plot_pressure(data, args.output_dir)


if __name__ == "__main__":
    main()
