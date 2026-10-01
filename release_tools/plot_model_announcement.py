"""Three release metrics from frozen scores; pending models have no fake bars."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'evaluation/released_daily/released_daily_benchmark.json'
DEST = ROOT/'evaluation/released_daily/model_1_2_benchmark.png'


def main():
    source = json.loads(SOURCE.read_text())
    if source['status'] != 'complete_verified' or source['cohort']['daily_issues'] != 1473:
        raise ValueError('Chart requires verified daily-cohort results')
    metrics = {m['key']: m for m in source['metrics']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.7))
    fig.set_facecolor('#f6f8fc')
    specs = [('pressure_JMA_hpa', 'Pressure intensity', 'Central-pressure MAE · hPa', 18, 2),
             ('mean_track_error_km', 'Track position', 'Mean position error · km', 1050, 1),
             ('direction_error_deg', 'Track direction', 'Six-hour heading error · degrees', 65, 2)]
    for ax, (key, title, label, maximum, digits) in zip(axes, specs):
        row = metrics[key]
        values = [row['values'][k] for k in ('1.1', '1.2')]
        ax.set_facecolor('#f6f8fc')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#dfe5ee', linewidth=.8)
        ax.bar([0, 1], values, width=.53, color=['#9aa9bc', '#3c67d6'], zorder=3)
        if key == 'pressure_JMA_hpa':
            intervals = [row['intervals'][k] for k in ('1.1', '1.2')]
            ax.errorbar([0, 1], values, yerr=[[v-low for v, (low, high) in zip(values, intervals)],
                        [high-v for v, (low, high) in zip(values, intervals)]],
                        fmt='none', color='#253853', capsize=5, linewidth=1.2, zorder=4)
        for x, value in enumerate(values):
            top = row['intervals'][('1.1', '1.2')[x]][1] if key == 'pressure_JMA_hpa' else value
            ax.text(x, top+maximum*.038, f'{value:,.{digits}f}', ha='center',
                    fontsize=17, fontweight='bold', color='#21334c')
        ax.set_ylim(0, maximum)
        ax.set_xticks([0, 1], ['1.1', '1.2 · mean of 50'])
        ax.tick_params(axis='both', length=0, labelcolor='#53647c')
        ax.set_title(title, loc='left', fontsize=16, fontweight='bold', color='#21334c', pad=38)
        ax.text(0, 1.03, label, transform=ax.transAxes, color='#53647c', fontsize=10)
        coverage = row['coverage']
        ax.text(.5, -.16, f"{coverage['daily_issues']:,} daily starts / {coverage['storms']} storms",
                transform=ax.transAxes, ha='center', color='#53647c', fontsize=10)
    fig.text(.055, .945, 'Trackformer 1.2', fontsize=23, fontweight='bold', color='#182b45')
    fig.text(.055, .895, 'Matched five-day forecasts · equal weight per storm · lower is better', fontsize=12, color='#53647c')
    fig.text(.055, .095, 'Pressure: JMA reference; bars show 95% whole-storm intervals. Paired improvement is uncertain.', fontsize=10, color='#53647c')
    fig.text(.055, .057, 'DeepMind Mini: matched run in progress — no score yet. Different pipelines/member policies; development evidence.', fontsize=10, color='#53647c')
    fig.subplots_adjust(left=.055, right=.975, top=.735, bottom=.25, wspace=.36)
    fig.savefig(DEST, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    receipt = {'source': SOURCE.relative_to(ROOT).as_posix(), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'chart': DEST.relative_to(ROOT).as_posix(), 'chart_sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(),
               'metrics': [s[0] for s in specs], 'values': {k: metrics[k]['values'] for k, *_ in specs},
               'deepmind_score': None, 'zero_fill': False, 'model_forecasts_modified': False}
    DEST.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
