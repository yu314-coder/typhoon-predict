"""Render the v9 storm case studies (Bavi / Co-may / Wayne) as a vector PDF for the LaTeX paper."""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OCEAN = "#eaf2f8"; LAND = "#e9e2d0"; LAND_EDGE = "#c9bd9c"; GRID = "#c7d4dc"
FCAST = "#0f7d8c"; OBS = "#e08214"; ENS = "#7fb8c2"; HIST = "#8a97a2"; ISSUE = "#d1495b"

PANELS = [("bavi", "Bavi (2026)", "well-behaved recurver"),
          ("comay", "Co-may (2025)", "erratic recurvature"),
          ("wayne", "Wayne (1986)", "five loops")]

fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5))
for ax, (key, name, tag) in zip(axes, PANELS):
    d = json.load(open(f"track_build/{key}_v9_geo.json"))
    coast = json.load(open(f"track_build/geo/coast_{key}_v9.json"))
    LON0, LON1, LAT0, LAT1 = d["extent"]
    ax.set_facecolor(OCEAN)
    for poly in coast:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        ax.fill(xs, ys, facecolor=LAND, edgecolor=LAND_EDGE, linewidth=0.5, zorder=1)
    ax.set_xticks(np.arange(math.ceil(LON0 / 5) * 5, LON1 + 1, 5))
    ax.set_yticks(np.arange(math.ceil(LAT0 / 5) * 5, LAT1 + 1, 5))
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=7, length=2)
    # ensemble (faint)
    for e in d["ensemble"]:
        ax.plot([p[1] for p in e], [p[0] for p in e], color=ENS, linewidth=0.5, alpha=0.28, zorder=2)
    # history
    h = d["history"]; ax.plot([p[1] for p in h], [p[0] for p in h], color=HIST, linewidth=1.3, ls=(0, (3, 2)), zorder=3)
    # forecast mean
    f = d["forecast"]; ax.plot([p[1] for p in f], [p[0] for p in f], color=FCAST, linewidth=2.0, ls=(0, (5, 2)), zorder=5)
    for k, lab in [(4, "24h"), (12, "72h"), (20, "120h")]:
        ax.plot(f[k][1], f[k][0], "o", color=FCAST, ms=3, zorder=6)
        ax.annotate(lab, (f[k][1], f[k][0]), textcoords="offset points", xytext=(3, 3), fontsize=6, color=FCAST)
    # observed
    o = [p for p in d["observed"] if p]; ax.plot([p[1] for p in o], [p[0] for p in o], color=OBS, linewidth=2.4, zorder=4)
    ax.plot(o[-1][1], o[-1][0], "o", color=OBS, ms=4, zorder=6)
    # issue marker
    ax.plot(d["issue"][1], d["issue"][0], "*", color=ISSUE, ms=13, mec="white", mew=0.6, zorder=7)
    ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
    ax.set_aspect(1.0 / math.cos(math.radians((LAT0 + LAT1) / 2)))
    e120 = d["errors"].get("120") or d["errors"].get("96"); e8 = d["errors_v8"].get("120") or d["errors_v8"].get("96")
    delta = "improved" if e120 < e8 else "regressed"
    ax.set_title(f"{name} — {tag}\n120 h error: {e120} km (v8 {e8}, {delta})", fontsize=8.5, pad=6)

handles = [Line2D([0], [0], color=HIST, ls=(0, (3, 2)), lw=1.3, label="history (input)"),
           Line2D([0], [0], marker="*", color="w", mfc=ISSUE, ms=11, label="forecast issued"),
           Line2D([0], [0], color=ENS, lw=1.2, alpha=.6, label="ensemble (50)"),
           Line2D([0], [0], color=FCAST, ls=(0, (5, 2)), lw=2, label="v9 forecast"),
           Line2D([0], [0], color=OBS, lw=2.4, label="observed")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("paper/storms_v9.pdf", bbox_inches="tight", dpi=200)
print("saved paper/storms_v9.pdf")
