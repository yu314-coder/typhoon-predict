"""Plot the released model's saved geographic pressure fields without inference.

The basin remains at 2.5 degrees. The moving core is the model's learned 20-km
reconstruction, not extra native observations. Invalid coverage stays masked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def render_pressure_map(data, output, lead=120, interval=4):
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Isobar interval must be positive and finite")
    indices = np.flatnonzero(np.asarray(data["lead_hours"]) == lead)
    if len(indices) != 1:
        raise ValueError("Choose a saved +6 to +120-hour forecast lead")
    required = {"core_mslp_hpa", "core_latitude_deg", "core_longitude_deg", "core_valid",
                "basin_latitude_deg", "basin_longitude_deg", "track_valid", "valid_time_ns", "member_count"}
    if not required.issubset(data):
        raise ValueError("This renderer requires the moving-core pressure export v2")
    if int(np.asarray(data["member_count"])) != 1:
        raise ValueError("This renderer accepts the clean one-member export, not unregistered ensemble cores")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    i = int(indices[0])
    basin = np.ma.masked_invalid(data["basin_mslp_hpa"][i])
    core = np.ma.masked_where(~np.asarray(data["core_valid"][i], bool), data["core_mslp_hpa"][i])
    core = np.ma.masked_invalid(core)
    finite = np.concatenate((basin.compressed(), core.compressed()))
    if not finite.size:
        raise ValueError("No supported pressure field for this lead")
    low = np.floor(finite.min() / interval) * interval
    high = max(low + interval, np.ceil(finite.max() / interval) * interval)
    levels = np.arange(low, high + interval * .5, interval)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    plots = (
        (basin, data["basin_longitude_deg"], data["basin_latitude_deg"], "Western Pacific · 2.5° basin field"),
        (core, data["core_longitude_deg"][i], data["core_latitude_deg"][i], "Moving core · learned 20-km reconstruction"),
    )
    route = np.asarray(data["track_lat_lon"][:i + 1])
    valid_route = np.asarray(data["track_valid"][:i + 1], bool)
    for ax, (pressure, lon, lat, title) in zip(axes, plots):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude °E")
        ax.set_ylabel("Latitude °N")
        if pressure.count():
            image = ax.pcolormesh(lon, lat, pressure, cmap="RdBu_r", vmin=low, vmax=high,
                                  shading="auto", rasterized=True)
            if pressure.max() > pressure.min():
                lines = ax.contour(lon, lat, pressure, levels=levels, colors="#35424d", linewidths=.65)
                ax.clabel(lines, levels[::2], fmt="%d", fontsize=7)
        else:
            ax.text(.5, .5, "Moving core outside supported basin coverage", ha="center", va="center", transform=ax.transAxes)
        ax.plot(np.where(valid_route, route[:, 1], np.nan), np.where(valid_route, route[:, 0], np.nan),
                color="#ad287b", linewidth=1.6)
        if valid_route[i] and (ax is axes[0] or bool(np.asarray(data["core_valid"][i]).any())):
            ax.scatter(route[i, 1], route[i, 0], s=25, c="#ad287b", edgecolors="white", zorder=4)
        ax.grid(alpha=.2)
    axes[0].set_xlim(100, 180)
    axes[0].set_ylim(0, 60)
    axes[1].set_xlim(float(np.min(plots[1][1])), float(np.max(plots[1][1])))
    axes[1].set_ylim(float(np.min(plots[1][2])), float(np.max(plots[1][2])))
    valid_time = datetime.fromtimestamp(int(np.asarray(data["valid_time_ns"])[i]) / 1e9, timezone.utc)
    value = (f"{float(data['central_pressure_hpa'][i]):.1f} hPa" if valid_route[i] else "unavailable outside domain")
    fig.suptitle(f"Trackformer 1.2 · +{lead:03d} h · {valid_time:%Y-%m-%d %H:%M UTC}\n"
                 f"1 clean member · central pressure {value}", fontsize=12)
    fig.colorbar(image, ax=axes, label="Model MSLP (hPa) · blue low / red high", shrink=.85)
    fig.supxlabel(f"{interval:g} hPa isobars · original model geography · unsupported core cells masked; no pressure or route shifting", fontsize=9)
    fig.savefig(Path(output), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lead", type=int, default=120, choices=range(6, 121, 6))
    parser.add_argument("--interval", type=float, default=4)
    args = parser.parse_args()
    with np.load(args.forecast, allow_pickle=False) as saved:
        render_pressure_map({key: saved[key] for key in saved.files}, args.output, args.lead, args.interval)


if __name__ == "__main__":
    main()
