"""Verify and plot daily -> storm -> equal-storm intensity/curve diagnostics.

This is a read-only postprocessor for the separately frozen GPU runner. It
never changes its inputs, weights, member sampling, calibration or cases.
Native 1.2 has no radius output: absent bars are N/A, never zero error.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = ('1.1', '1.2')
LEADS = list(range(6, 121, 6))
QUADRANTS = ('NE', 'SE', 'SW', 'NW')
CURVE_PROTOCOL = {
    'version': 'unshifted-six-hour-intensity-curves-v1',
    'minimum_curve_points': 6,
    'shape_error': '(1 - centred cosine similarity) / 2; exact common UTC leads; '
                   'remove mean and scalar magnitude, preserve time order and sign; '
                   'no lag optimization, time warping, shift or fit; flat truth or '
                   'either flat model makes both shape scores unavailable.',
    'trend_error': 'Fraction of adjacent exact six-hour changes with a different '
                   'sign, including flat vs non-flat. Sign deadband = 0 '
                   '(strict native changes); no joining across missing leads.',
    'tendency_mae': 'MAE of predicted minus observed adjacent six-hour changes, '
                    'in kt/6h, hPa/6h or km/6h.',
    'aggregation': 'Paired common lead mask -> daily issue mean -> mean of days '
                   'per storm -> equal storm weights, same as track. Radius '
                   'quadrants are averaged within the daily issue; no pooling '
                   'long-lived storms or quadrant counts across storms.',
    'radius': 'Native 1.1 RMW and NE/SE/SW/NW R34/R50/R64 kilometre outputs '
              'versus exact USA reports in nautical miles * 1.852; radii, not '
              'diameters. No native 1.2 radius head: do not score experimental '
              'pressure radii as a calibrated official equivalent. RMW must '
              'be positive; reported zero wind-radius extents are valid.',
    'uncertainty': 'Whole-storm bootstrap, fixed 2000 replicates, seed 4712; '
                   'development uncertainty, not untouched-holdout certification.',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def numeric(values):
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def curve_scores(predictions, truth, *, low, high):
    """Return per-issue scores with exactly the same mask for both models."""
    truth = numeric(truth)
    predicted = {m: numeric(v) for m, v in predictions.items()}
    if not predicted or any(v.shape != truth.shape for v in predicted.values()):
        raise ValueError('Curve lengths differ')
    valid = np.isfinite(truth) & (truth >= low) & (truth <= high)
    for values in predicted.values():
        valid &= np.isfinite(values)
    adjacent = valid[:-1] & valid[1:]
    output = {m: {'mae': None, 'shape_error': None, 'trend_error': None,
                  'tendency_mae': None} for m in predicted}
    if not valid.any():
        return output, {'valid_leads': 0, 'valid_adjacent_steps': 0}
    centred_truth = truth[valid] - truth[valid].mean()
    centred = {m: v[valid] - v[valid].mean() for m, v in predicted.items()}
    shape_ok = valid.sum() >= CURVE_PROTOCOL['minimum_curve_points']
    shape_ok &= np.linalg.norm(centred_truth) > 1e-6
    shape_ok &= all(np.linalg.norm(v) > 1e-6 for v in centred.values())
    truth_change = np.diff(truth)[adjacent]
    for model, values in predicted.items():
        output[model]['mae'] = float(np.abs(values[valid] - truth[valid]).mean())
        if shape_ok:
            cosine = float(np.dot(centred[model], centred_truth) /
                           (np.linalg.norm(centred[model]) * np.linalg.norm(centred_truth)))
            output[model]['shape_error'] = float((1 - np.clip(cosine, -1, 1)) / 2)
        if adjacent.any():
            change = np.diff(values)[adjacent]
            output[model]['trend_error'] = float((np.sign(change) != np.sign(truth_change)).mean())
            output[model]['tendency_mae'] = float(np.abs(change - truth_change).mean())
    return output, {'valid_leads': int(valid.sum()), 'valid_adjacent_steps': int(adjacent.sum()),
                    'shape_available': bool(shape_ok)}


def aggregate_daily(records, models=MODELS):
    grouped = defaultdict(list)
    for record in records:
        grouped[record['storm_id']].append(record)
    result = {}
    for key in ('mae', 'shape_error', 'trend_error', 'tendency_mae'):
        storms = []
        count = 0
        for sid, days in grouped.items():
            paired = [r for r in days if all(r['models'][m].get(key) is not None for m in models)]
            if paired:
                count += len(paired)
                storms.append({'storm_id': sid, 'daily_issues': len(paired),
                    'models': {m: float(np.mean([r['models'][m][key] for r in paired])) for m in models}})
        scores = {}
        for model in models:
            values = np.array([s['models'][model] for s in storms])
            if len(values):
                rng = np.random.default_rng(4712)
                boot = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
                scores[model] = {'equal_storm_mean': float(values.mean()),
                    'bootstrap_95_percent_interval': np.quantile(boot, [.025, .975]).tolist()}
        result[key] = {'valid_storms': len(storms), 'valid_daily_issues': count,
                       'models': scores, 'storm_scores': storms}
        if len(models) == 2 and storms:
            delta = np.array([s['models']['1.2'] - s['models']['1.1'] for s in storms])
            rng = np.random.default_rng(4712)
            boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
            result[key]['paired_delta_1_2_minus_1_1'] = {'mean': float(delta.mean()),
                'bootstrap_95_percent_interval': np.quantile(boot, [.025, .975]).tolist()}
    return result


def observation_number(row, name, low, high):
    try:
        v = float(row.get(name, ''))
        return v if np.isfinite(v) and low <= v <= high else None
    except (ValueError, TypeError):
        return None


def summarize_radius(records, observations):
    by_type = defaultdict(list)
    for record in records:
        if record.get('v11') is None:
            continue
        issue = datetime.fromisoformat(record['issue_time_utc'].replace('Z', '+00:00'))
        rows = [observations.get((record['storm_id'],
                 (issue + timedelta(hours=h)).strftime('%Y-%m-%dT%H:%M:%SZ')), {}) for h in LEADS]
        for target in ('RMW', 'R34', 'R50', 'R64'):
            daily = []
            components = ['RMW'] if target == 'RMW' else list(QUADRANTS)
            for component in components:
                column = 'USA_RMW' if target == 'RMW' else f'USA_{target}_{component}'
                obs = [observation_number(r, column, 0.001 if target == 'RMW' else 0, 1000) for r in rows]
                obs = [None if v is None else v*1.852 for v in obs]
                component_index = None if target == 'RMW' else (('R34', 'R50', 'R64').index(target)*4 + QUADRANTS.index(component))
                values = [p['rmw_km'] if component_index is None else p['wind_radii_km'][component_index] for p in record['v11']]
                scored, coverage = curve_scores({'1.1': values}, obs, low=0, high=1852)
                daily.append(scored['1.1'])
            # Equal component weight inside one issue. Missing components remain
            # absent; a zero observed extent is included in its lead error.
            combined = {key: float(np.mean(v)) if (v := [d[key] for d in daily if d[key] is not None]) else None
                        for key in ('mae', 'shape_error', 'trend_error', 'tendency_mae')}
            by_type[target].append({'storm_id': record['storm_id'], 'case_index': record['case_index'],
                                    'models': {'1.1': combined}})
    return {target: aggregate_daily(days, ('1.1',)) for target, days in by_type.items()}


def verify_and_summarize(source, project):
    from benchmark_intensity_v12_v11 import observations
    plan = json.loads((source/'protocol.json').read_text())
    report = json.loads((source/'report.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('Do not publish partial wind replay scores as final')
    if report['protocol_sha256'] != sha(source/'protocol.json'):
        raise ValueError('Protocol hash mismatch')
    for name, item in plan['source_hashes'].items():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Source changed: '+name)
    records = [json.loads(p.read_text()) for p in sorted((source/'cases').glob('*.json'))]
    if len(records) != len(plan['cases']) or {r['case_index'] for r in records} != {c['case_index'] for c in plan['cases']}:
        raise ValueError('Frozen case coverage differs')
    verified_wind, maximum_delta = 0, 0.
    for record in records:
        original = project/'benchmark_daily_storms_1_2/cases'/f"{record['case_index']:05d}.npz"
        if sha(original) != record['original_forecast_sha256']:
            raise ValueError('Original forecast mutated')
        if not record['initial_intensity_available']:
            if record['v11'] is not None or record.get('v12_auxiliary_wind_kt') is not None:
                raise ValueError('Excluded issue unexpectedly has intensity outputs')
            continue
        members_path = source/'cases'/f"{record['case_index']:05d}_wind_members.npz"
        if sha(members_path) != record['wind_member_file_sha256'] or record['v12_distinct_input_members'] != 50:
            raise ValueError('Member proof mismatch')
        if len(set(record['v12_member_input_sha256'])) != 50:
            raise ValueError('Duplicate input members')
        wanted = record['issue_ns'] + np.arange(-8, 1, dtype=np.int64)*6*3600*10**9
        if not np.array_equal(wanted, record['v12_causal_analysis_times_ns']):
            raise ValueError('Noncausal analysis history')
        with np.load(members_path, allow_pickle=False) as z:
            if z['wind_kt'].shape != (50, 20) or z['pressure_hpa'].shape != (50, 20):
                raise ValueError('Member lead shape mismatch')
            if not np.isfinite(z['wind_kt']).all() or not np.isfinite(z['pressure_hpa']).all():
                raise ValueError('Nonfinite member output')
            if not np.allclose(z['wind_kt'].mean(0, dtype='float64'), record['v12_auxiliary_wind_kt'], rtol=0, atol=1e-8):
                raise ValueError('Wind mean mismatch')
            delta = float(np.max(np.abs(z['pressure_hpa'].mean(0, dtype='float64') - record['v12_pressure_hpa'])))
            if delta > 1e-3:
                raise ValueError('Original pressure replay differs')
            maximum_delta = max(maximum_delta, delta)
        verified_wind += 1
    metrics = {}
    specifications = [
        ('pressure_USA_hpa', 'USA central pressure', 'hPa', 800, 1100),
        ('pressure_JMA_hpa', 'JMA central pressure', 'hPa', 800, 1100),
        ('auxiliary_wind_vs_USA_1min_kt', 'Wind against USA 1-minute reference', 'kt', 0, 249),
    ]
    for key, label, unit, low, high in specifications:
        days = []
        for record in records:
            if record['v11'] is None:
                continue
            forecast = {'1.1': [p['vmax_kt' if unit == 'kt' else 'central_pressure_hpa'] for p in record['v11']],
                        '1.2': record['v12_auxiliary_wind_kt' if unit == 'kt' else 'v12_pressure_hpa']}
            scores, coverage = curve_scores(forecast, record['truth'][key], low=low, high=high)
            days.append({'storm_id': record['storm_id'], 'case_index': record['case_index'],
                         'models': scores, 'coverage': coverage})
        metrics[key] = {'label': label, 'unit': unit, **aggregate_daily(days)}
        # The independent postprocessor must reproduce the runner's MAE.
        for model in MODELS:
            actual = metrics[key]['mae']['models'][model]['equal_storm_mean']
            expected = report[key]['models'][model]['equal_storm_mae']
            if not np.isclose(actual, expected, rtol=0, atol=1e-9):
                raise ValueError('Equal-storm aggregation differs from frozen runner')
    obs = observations(project/'data/ibtracs/ibtracs.WP.list.v04r01.csv', {r['storm_id'] for r in records})
    return {'status': 'complete_verified', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'cohort_storms': plan['storm_count'], 'cohort_daily_issues': len(plan['cases']),
        'eligible_intensity_daily_issues': verified_wind,
        'excluded_missing_current_intensity': len(records)-verified_wind,
        'protocol_sha256': sha(source/'protocol.json'), 'runner_report_sha256': sha(source/'report.json'),
        'postprocessor_sha256': sha(__file__), 'curve_protocol': CURVE_PROTOCOL,
        'source_hashes': {k: v['sha256'] for k, v in plan['source_hashes'].items()},
        'verification': {'all_frozen_original_forecasts_sha256_valid': True,
            'actual_distinct_input_members_per_replay': 50, 'verified_member_replays': verified_wind,
            'maximum_original_pressure_difference_hpa': maximum_delta,
            'all_wind_means_from_members_verified': True},
        'metrics': metrics, 'native_1_1_radius_USA_km': summarize_radius(records, obs),
        'native_1_2_radius': {'status': 'unavailable', 'reason': 'No native radius head; not zero error.'},
        'limits': [plan['evaluation_label'], report['wind_note'], report['radius_note'],
                   'Temporal shape ignores level and amplitude. Read it with MAE and unshifted timelines.',
                   'Curve metric protocol is a separately versioned post-hoc descriptive addition, '
                   'not a preregistered untouched-test hypothesis. No selection or fitting on forecast errors.']}


def plot(report, runner, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#65758b', '#007d78']
    metrics = report['metrics']
    keys = list(metrics)
    titles = ['USA pressure', 'JMA pressure', 'Wind / USA 1-minute']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8))
    for j, (key, title) in enumerate(zip(keys, titles)):
        for i, score in enumerate(('mae', 'shape_error')):
            ax = axes[i, j]
            result = metrics[key][score]
            values = [result['models'][m]['equal_storm_mean'] for m in MODELS]
            intervals = [result['models'][m]['bootstrap_95_percent_interval'] for m in MODELS]
            error = np.array([[max(0, v-p[0]) for v, p in zip(values, intervals)],
                              [max(0, p[1]-v) for v, p in zip(values, intervals)]])
            bars = ax.bar(['1.1', '1.2 · mean of 50'], values, yerr=error, capsize=5, width=.55, color=colors)
            ax.bar_label(bars, labels=[f'{v:.2f}' if i == 0 else f'{v:.3f}' for v in values], padding=25, fontweight='bold')
            ax.set_title(title + ('\nMagnitude error ↓' if i == 0 else '\nTime-curve shape error ↓'), loc='left', fontweight='bold', fontsize=11)
            ax.set_ylabel(metrics[key]['unit'] if i == 0 else '(1 − centred cosine) / 2')
            ax.set_ylim(0, max(max(p[1] for p in intervals)*1.42, .05))
            ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
            ax.text(.02, .95, f"{result['valid_storms']} storms · {result['valid_daily_issues']} days", transform=ax.transAxes, va='top', color='#526174', fontsize=9)
    fig.suptitle('Intensity comparison | same daily → storm → equal-storm method as track', x=.06, ha='left', fontsize=16, fontweight='bold')
    fig.text(.06, .927, f"Frozen 270-storm / 1,473-day plan · {report['eligible_intensity_daily_issues']} starts usable by native 1.1 intensity · +6 to +120 h", color='#526174')
    fig.text(.06, .018, 'Development comparison, not untouched test; different pipelines. Error bars: whole-storm bootstrap 95%.\nWind is an auxiliary scalar diagnostic, not certified surface wind. Shape removes level/scale but never shifts time.', color='#526174', fontsize=9)
    fig.subplots_adjust(top=.84, bottom=.13, left=.06, right=.98, hspace=.43, wspace=.33)
    fig.savefig(output/'intensity_error_bars.png', dpi=175, facecolor='white'); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8))
    for j, (key, title) in enumerate(zip(keys, titles)):
        ax = axes[0, j]
        for model, color in zip(MODELS, colors):
            ax.plot(LEADS, runner[key]['models'][model]['equal_storm_mae_by_lead'], label=model, color=color, lw=2.4)
        ax.set_title(title + '\nPer-lead MAE ↓', loc='left', fontweight='bold', fontsize=11)
        ax.set_ylabel(metrics[key]['unit']); ax.set_xlabel('Forecast lead · h'); ax.legend(frameon=False)
        ax.set_xticks([6, 24, 48, 72, 96, 120]); ax.grid(alpha=.18)
        ax = axes[1, j]; result = metrics[key]['trend_error']
        vals = [100*result['models'][m]['equal_storm_mean'] for m in MODELS]
        bars = ax.bar(['1.1', '1.2 · mean of 50'], vals, width=.55, color=colors)
        ax.bar_label(bars, labels=[f'{v:.1f}%' for v in vals], padding=6, fontweight='bold')
        ax.set_ylim(0, 100); ax.set_ylabel('% adjacent changes with different sign')
        ax.set_title(title + '\nTrend-direction mismatch ↓', loc='left', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
        ax.text(.02, .95, f"{result['valid_storms']} storms · {result['valid_daily_issues']} days", transform=ax.transAxes, va='top', fontsize=9, color='#526174')
    fig.suptitle('Intensity lead errors & six-hour change direction', x=.06, ha='left', fontsize=17, fontweight='bold')
    fig.text(.06, .927, 'No time shifting, nearest labels or wind-period conversion · same common valid masks for both systems', color='#526174')
    fig.text(.06, .024, 'Changes are scored only across adjacent valid six-hour leads; flat vs changing counts as a mismatch.\nUSA and JMA pressure references remain separate; USA 1-minute and JMA 10-minute winds are not pooled.', color='#526174', fontsize=9)
    fig.subplots_adjust(top=.84, bottom=.12, left=.06, right=.98, hspace=.49, wspace=.35)
    fig.savefig(output/'intensity_leads_and_trends.png', dpi=175, facecolor='white'); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5))
    for j, (target, result) in enumerate(report['native_1_1_radius_USA_km'].items()):
        for i, score in enumerate(('mae', 'shape_error')):
            ax = axes[i, j]; r = result[score]
            value = r['models'].get('1.1', {}).get('equal_storm_mean')
            if value is not None:
                bars = ax.bar([0], [value], color=colors[0], width=.55)
                ax.bar_label(bars, labels=[f'{value:.1f}' if i == 0 else f'{value:.3f}'], padding=6, fontweight='bold')
            ax.set_xlim(-.6, 1.6); ax.set_xticks([0, 1], ['1.1', '1.2'])
            ax.set_ylim(0, max((value or 0)*1.45, .08))
            ax.text(1, .28, 'N/A\nNo radius head', transform=ax.get_xaxis_transform(), ha='center', color='#a24537', fontsize=10, fontweight='bold')
            ax.set_title(target + (' · radius MAE ↓' if i == 0 else ' · curve shape error ↓'), loc='left', fontsize=11, fontweight='bold')
            ax.set_ylabel('km' if i == 0 else '(1 − centred cosine) / 2')
            ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
            ax.text(.03, .97, f"{r['valid_storms']} storms\n{r['valid_daily_issues']} days", transform=ax.transAxes, va='top', fontsize=9, color='#526174')
    fig.suptitle('Radius error | native 1.1 vs matching USA definitions; 1.2 unavailable', x=.06, ha='left', fontsize=15, fontweight='bold')
    fig.text(.06, .92, 'RMW; R34 / R50 / R64 NE–SE–SW–NW maximum quadrant extents · nmi × 1.852 = km · radii, not diameters', color='#526174', fontsize=10)
    fig.text(.06, .018, 'Same daily → storm → equal-storm hierarchy as track, using valid reports. Native 1.2 N/A is NOT zero error.\nSeparate pressure-scaled experimental radius examples are not substituted into this official-definition score.', color='#526174', fontsize=9)
    fig.subplots_adjust(top=.83, bottom=.13, left=.06, right=.98, hspace=.42, wspace=.44)
    fig.savefig(output/'native_radius_error_bars.png', dpi=175, facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not str(args.output.resolve()).startswith('/Volumes/D/'):
        raise ValueError('Artifacts must stay on D')
    args.output.mkdir(parents=True, exist_ok=True)
    summary = verify_and_summarize(args.source, args.project)
    write(args.output/'intensity_final.json', summary)
    plot(summary, json.loads((args.source/'report.json').read_text()), args.output)
    receipt = {'status': 'complete_verified', 'summary_sha256': sha(args.output/'intensity_final.json'),
        'images': {p.name: sha(p) for p in sorted(args.output.glob('*.png'))},
        'aggregation': CURVE_PROTOCOL['aggregation'], 'native_1_2_radius_status': 'unavailable, not zero'}
    write(args.output/'verification.json', receipt)
    print(json.dumps({'status': summary['status'], 'verified_replays': summary['verification']['verified_member_replays'],
                      'output': str(args.output), 'receipt': receipt}, indent=2))
