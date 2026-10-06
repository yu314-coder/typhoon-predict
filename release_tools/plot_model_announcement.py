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
PERIOD_SOURCE = ROOT/'evaluation/deepmind_daily/period_comparison.json'
POST_DEST = ROOT/'evaluation/released_daily/model_1_2_after_2024_benchmark.png'


def period_metrics(period):
    scores = period['scores']
    pressure = scores['pressure']['JMA']
    return {
        'pressure_JMA_hpa': {
            'values': {m: pressure['models'][m]['mae_hpa']['value'] for m in ('1.1', '1.2', 'deepmind')},
            'coverage': {'daily_issues': pressure['daily_cases'], 'storms': pressure['storms']},
        },
        **{key: {'values': {m: scores['route'][m][key]['value'] for m in ('1.1', '1.2', 'deepmind')},
                  'coverage': {'daily_issues': period['daily_issues'], 'storms': period['storms']}}
           for key in ('mean_track_error_km', 'direction_error_deg')},
    }


def main():
    source = json.loads(SOURCE.read_text())
    if source['status'] != 'complete_verified' or source['cohort']['daily_issues'] != 1473:
        raise ValueError('Chart requires verified daily-cohort results')
    reference = next(m for m in source['models'] if m['key'] == 'deepmind')
    audit = json.loads((ROOT/'evaluation/deepmind_daily/publication_audit.json').read_text())
    if reference['status'] != 'complete_verified' or audit['snapshot_sha256'] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
        raise ValueError('DeepMind bars require the matching independent publication audit')
    periods = json.loads(PERIOD_SOURCE.read_text())
    if periods['status'] != 'complete_verified' or periods['source_hashes']['snapshot_sha256'] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
        raise ValueError('Period comparison requires the exact original audited snapshot')
    if periods['model'] != reference or not periods['single_checkpoint_across_all_periods']:
        raise ValueError('Period groups are not different checkpoint generations')
    for key, dest, title in (('total', DEST, 'Total · mixed-year comparison'),
                             ('after_2024', POST_DEST, 'After 2024 · 2025–2026 comparison')):
        metrics = period_metrics(periods['periods'][key])
        draw_chart(metrics, reference, periods, key, dest, title)


def draw_chart(metrics, reference, periods, period_key, dest, title):
    models = ('1.1', '1.2', 'deepmind')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.1))
    fig.set_facecolor('#f6f8fc')
    specs = [('pressure_JMA_hpa', 'Pressure intensity', 'Central-pressure MAE · hPa', 20, 2),
             ('mean_track_error_km', 'Track position', 'Mean position error · km', 1200, 1),
             ('direction_error_deg', 'Track direction', 'Six-hour heading error · degrees', 75, 2)]
    for ax, (key, panel_title, label, maximum, digits) in zip(axes, specs):
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
        ax.set_xticks([0, 1, 2], ['1.1', '1.2\nmean of 50', 'DeepMind Mini\n<2024 · 1 member'])
        ax.tick_params(axis='both', length=0, labelcolor='#53647c')
        ax.set_title(panel_title, loc='left', fontsize=16, fontweight='bold', color='#21334c', pad=38)
        ax.text(0, 1.03, label, transform=ax.transAxes, color='#53647c', fontsize=10)
        coverage = row['coverage']
        ax.text(.5, -.33, f"{coverage['daily_issues']:,} daily starts / {coverage['storms']} storms",
                transform=ax.transAxes, ha='center', color='#53647c', fontsize=10)
    fig.text(.055, .945, title, fontsize=23, fontweight='bold', color='#182b45')
    fig.text(.055, .89, 'DeepMind = WeatherNext Cyclones Mini <2024 · software v0.3.0 · 1° · one CUDA member', fontsize=12, color='#53647c')
    fig.text(.055, .84, 'Same saved forecasts and valid times · +6 to +120 h · equal storm weight · lower is better', fontsize=11, color='#53647c')
    if period_key == 'total':
        note = 'Storm proportions: <2024 85.2% · 2024 8.1% · >2024 6.7%. Historical dates overlap Mini fitting years.'
    else:
        note = 'Strict UTC issue year >2024: 61 starts / 18 storms. Calendar 2024 is excluded, not counted as >2024.'
    fig.text(.055, .09, note, fontsize=10, color='#53647c')
    fig.text(.055, .052, 'Pressure coverage is smaller than track coverage. Missing is not zero. Different pipelines/member counts; no significance or unused-holdout claim.', fontsize=9.5, color='#53647c')
    fig.subplots_adjust(left=.055, right=.975, top=.705, bottom=.32, wspace=.36)
    fig.savefig(dest, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    receipt = {'source': PERIOD_SOURCE.relative_to(ROOT).as_posix(), 'source_sha256': hashlib.sha256(PERIOD_SOURCE.read_bytes()).hexdigest(),
               'chart': dest.relative_to(ROOT).as_posix(), 'chart_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
               'period': period_key, 'period_definition': periods['period_definition'][period_key],
               'metrics': [s[0] for s in specs], 'values': {k: metrics[k]['values'] for k, *_ in specs},
               'deepmind_status': 'complete_verified', 'deepmind_model': reference,
               'models': list(models), 'bars_per_panel': 3, 'error_bars': False,
               'zero_fill': False, 'model_forecasts_modified': False,
               'publication_audit_sha256': hashlib.sha256((ROOT/'evaluation/deepmind_daily/publication_audit.json').read_bytes()).hexdigest()}
    dest.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'chart': receipt['chart'], 'period': period_key, 'values': receipt['values']}))


if __name__ == '__main__':
    main()
