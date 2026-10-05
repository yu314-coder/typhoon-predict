"""Three release metrics from the independently audited three-model cohort."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'evaluation/released_daily/released_daily_benchmark.json'
DEST = ROOT/'evaluation/released_daily/model_1_2_benchmark.png'


def main():
    source = json.loads(SOURCE.read_text())
    if source['status'] != 'complete_verified' or source['cohort']['daily_issues'] != 1473:
        raise ValueError('Chart requires verified daily-cohort results')
    reference = next(m for m in source['models'] if m['key'] == 'deepmind')
    audit = json.loads((ROOT/'evaluation/deepmind_daily/publication_audit.json').read_text())
    if reference['status'] != 'complete_verified' or audit['snapshot_sha256'] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
        raise ValueError('DeepMind bars require the matching independent publication audit')
    metrics = {m['key']: m for m in source['metrics']}
    models = ('1.1', '1.2', 'deepmind')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.1))
    fig.set_facecolor('#f6f8fc')
    specs = [('pressure_JMA_hpa', 'Pressure intensity', 'Central-pressure MAE · hPa', 18, 2),
             ('mean_track_error_km', 'Track position', 'Mean position error · km', 1050, 1),
             ('direction_error_deg', 'Track direction', 'Six-hour heading error · degrees', 65, 2)]
    for ax, (key, title, label, maximum, digits) in zip(axes, specs):
        row = metrics[key]
        values = [row['values'][k] for k in models]
        if any(v is None or not np.isfinite(v) or v < 0 for v in values):
            raise ValueError('Missing or invalid metric cannot become a zero bar')
        ax.set_facecolor('#f6f8fc')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#dfe5ee', linewidth=.8)
        ax.bar([0, 1, 2], values, width=.55, color=['#9aa9bc', '#3c67d6', '#9073cf'], zorder=3)
        for x, value in enumerate(values):
            ax.text(x, value+maximum*.038, f'{value:,.{digits}f}', ha='center',
                    fontsize=17, fontweight='bold', color='#21334c')
        ax.set_ylim(0, maximum)
        ax.set_xticks([0, 1, 2], ['1.1', '1.2\nmean of 50', 'DeepMind Mini\none member'])
        ax.tick_params(axis='both', length=0, labelcolor='#53647c')
        ax.set_title(title, loc='left', fontsize=16, fontweight='bold', color='#21334c', pad=38)
        ax.text(0, 1.03, label, transform=ax.transAxes, color='#53647c', fontsize=10)
        coverage = row['coverage']
        ax.text(.5, -.225, f"{coverage['daily_issues']:,} daily starts / {coverage['storms']} storms",
                transform=ax.transAxes, ha='center', color='#53647c', fontsize=10)
    fig.text(.055, .945, 'Trackformer 1.2 · matched model comparison', fontsize=23, fontweight='bold', color='#182b45')
    fig.text(.055, .895, 'Matched five-day forecasts · equal weight per storm · lower is better', fontsize=12, color='#53647c')
    fig.text(.055, .095, 'DeepMind: WeatherNext Cyclones Mini <2024 · software v0.3.0 · 1° · RTX 3070 CUDA · all 1,473 forecasts verified.', fontsize=10, color='#53647c')
    fig.text(.055, .057, 'Mean errors, not significance claims. Mixed historical/recent cohort; Mini has better recent-only track scores. Different pipelines/member policies.', fontsize=10, color='#53647c')
    fig.subplots_adjust(left=.055, right=.975, top=.735, bottom=.29, wspace=.36)
    fig.savefig(DEST, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    receipt = {'source': SOURCE.relative_to(ROOT).as_posix(), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'chart': DEST.relative_to(ROOT).as_posix(), 'chart_sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(),
               'metrics': [s[0] for s in specs], 'values': {k: metrics[k]['values'] for k, *_ in specs},
               'deepmind_status': 'complete_verified', 'deepmind_model': reference,
               'models': list(models), 'bars_per_panel': 3, 'error_bars': False,
               'zero_fill': False, 'model_forecasts_modified': False,
               'publication_audit_sha256': hashlib.sha256((ROOT/'evaluation/deepmind_daily/publication_audit.json').read_bytes()).hexdigest()}
    DEST.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
