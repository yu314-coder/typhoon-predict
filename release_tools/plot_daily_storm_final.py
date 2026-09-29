"""Plot the completed equal-storm development benchmark; no new inference."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / 'evaluation/daily_storm_final.json').read_text())
assert report['status'] == 'complete' and report['completed_storms'] == 270
metrics = report['aggregate_equal_storm']
panels = [
    ('mean_track_error_km', 'Mean position error ↓', 'km', 1),
    ('track_error_120h_km', '+120 h position error ↓', 'km', 1),
    ('direction_error_deg', 'Six-hour heading error ↓', 'degrees', 2),
    ('shape_similarity', 'Centred route shape ↑', 'similarity (not accuracy)', 4),
    ('path_similarity', 'Geographic path similarity ↑', 'similarity (not accuracy)', 4),
    ('frechet_distance_km', 'Fréchet distance ↓', 'km', 1),
]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, (key, title, unit, digits) in zip(axes.flat, panels):
    values = [metrics[m][key] for m in ['1.1', '1.2']]
    bars = ax.bar(['1.1', '1.2 · mean of 50'], values, width=.55, color=['#65758b', '#007d78'])
    ax.bar_label(bars, labels=[f'{v:,.{digits}f}' for v in values], padding=6, fontweight='bold')
    ax.set_title(title, loc='left', pad=14, fontweight='bold')
    ax.set_ylabel(unit)
    ax.set_ylim(0, 1.06 if 'similarity' in key else max(values)*1.22)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.16)
fig.suptitle('Trackformer 1.2 vs 1.1 | 270 storms · 1,473 daily forecasts', x=.07, ha='left', fontsize=17, fontweight='bold')
fig.text(.07, .922, 'Complete +6 to +120 h rollouts · each storm has equal weight · broader development evidence', color='#526174')
fig.text(.07, .022, 'Different input pipelines; not an architecture ablation or certified untouched test. No matched 1.1 pressure comparison.', fontsize=9, color='#526174')
fig.subplots_adjust(top=.84, bottom=.11, left=.07, right=.98, hspace=.42, wspace=.35)
fig.savefig(ROOT / 'evaluation/trackformer_1_2_vs_1_1_270_storms_bars.png', dpi=180, facecolor='white')
