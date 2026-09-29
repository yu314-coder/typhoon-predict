"""Render saved TIP comparison bars and a real Trackformer 1.2 pressure map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v11_tip", type=Path)
    parser.add_argument("v12_tip", type=Path)
    parser.add_argument("showcase_four", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    old = json.loads(args.v11_tip.read_text())
    new = json.loads(args.v12_tip.read_text())
    assert old["case_count"] == new["case_count"] == 10
    assert [c["issue"] for c in old["cases"]] == [c["issue"] for c in new["cases"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for ax, key, title, unit in (
        (axes[0], "mean_track_error_km", "Mean track error", "km"),
        (axes[1], "central_pressure_mae_hpa", "Central-pressure MAE", "hPa"),
    ):
        values = [old[key], new[key]]
        bars = ax.barh(["1.1", "1.2 · 50 input members"], values,
                       color=["#708396", "#0d827b"], height=0.52)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.25)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(f"{unit} · lower is better")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + max(values) * .025, bar.get_y() + bar.get_height()/2,
                    f"{value:.1f}", va="center", fontweight="bold")
    fig.suptitle("TIP: same ten overlapping issue times, +6 to +120 h", fontweight="bold")
    fig.text(.5, -.07, "Development diagnostic, not ten independent storms or untouched validation.",
             ha="center", fontsize=9)
    fig.savefig(args.output_dir / "trackformer_1_2_vs_1_1_tip_bars.png", dpi=175, bbox_inches="tight")
    plt.close(fig)

    with np.load(args.showcase_four, allow_pickle=False) as z:
        storm = str(z["storms"][0])
        fields = z["fields"][0]
        route = z["route"][0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), layout="constrained")
    for ax, lead_index in zip(axes, (3, 11, 19)):
        image = ax.imshow(fields[lead_index], origin="upper", extent=(100, 180, 0, 60),
                          cmap="viridis_r", vmin=960, vmax=1025, aspect="auto")
        ax.plot(route[:lead_index+1, 1], route[:lead_index+1, 0],
                color="#ffed9a", linewidth=1.8, marker="o", markersize=2)
        ax.set_title(f"{storm.title()} · +{(lead_index+1)*6} h")
        ax.set_xlabel("Longitude °E")
        ax.set_ylabel("Latitude °N")
        ax.set_xlim(100, 180)
        ax.set_ylim(0, 60)
    fig.colorbar(image, ax=axes, label="Forecast basin MSLP (hPa)", shrink=.76)
    fig.suptitle("Trackformer 1.2 · model-generated pressure-field example", fontweight="bold")
    fig.savefig(args.output_dir / "trackformer_1_2_pressure_map_example.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
