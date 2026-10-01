"""Verify saved daily forecasts and publish a small, cohort-explicit Site snapshot.

Read-only evaluation: no inference, fitting, forecast replacement or model changes.
The pressure postprocessor checks the original inputs and 50-member replay files.
DeepMind is explicitly unscored on this daily cohort, not borrowed from an older one.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = 'f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0'
TRACK_METRICS = (
    ('mean_track_error_km', 'Mean track error', 'km', 1, 'lower'),
    ('track_error_120h_km', '+120 h track error', 'km', 1, 'lower'),
    ('direction_error_deg', 'Six-hour direction error', 'degrees', 2, 'lower'),
    ('shape_similarity', 'Centred route-shape similarity', 'similarity', 4, 'higher'),
    ('path_similarity', 'Geographic path similarity', 'similarity', 4, 'higher'),
    ('frechet_distance_km', 'Frechet path distance', 'km', 1, 'lower'),
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError('Unavailable value replaced by a score')
    elif not np.allclose(actual, expected, rtol=0, atol=1e-8):
        raise ValueError('Saved metric does not reproduce: ' + str((actual, expected)))


def check_daily(project, report, plan):
    """Recompute routes, coordinate conversion, days and equal-storm means."""
    sys.path.insert(0, str(project))
    from benchmark_daily_storms_1_2 import metrics, local
    if report['status'] != 'complete' or report['completed_daily_cases'] != len(plan['cases']):
        raise ValueError('Daily benchmark is not complete')
    if report['cohort_sha256'] != sha(project/'benchmark_daily_storms_1_2/cohort.json'):
        raise ValueError('Daily cohort identity differs')
    if plan['models']['1.2']['checkpoint_sha256'] != CHECKPOINT or plan['models']['1.2']['members'] != 50:
        raise ValueError('Wrong release checkpoint or member count')
    groups = defaultdict(list)
    files = {}
    daily_keys = [(c['storm_id'], c['day_utc']) for c in plan['cases']]
    if len(set(daily_keys)) != len(daily_keys):
        raise ValueError('More than one start per storm-day')
    for case in plan['cases']:
        stem = f"{case['case_index']:05d}"
        path = project/'benchmark_daily_storms_1_2/cases'/stem
        saved = read(path.with_suffix('.json'))
        if any(saved[k] != v for k, v in case.items()):
            raise ValueError('Daily case identity changed')
        digest = sha(path.with_suffix('.npz'))
        if digest != saved['forecast_sha256'] or saved['members'] != 50:
            raise ValueError('Saved forecast or member count changed')
        files[stem] = digest
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            arrays = [z[k] for k in ('v11_local', 'v12_local', 'truth_local')]
            if any(a.shape != (20, 2) or not np.isfinite(a).all() for a in arrays):
                raise ValueError('Missing route must not be zero-scored')
            for model, route in zip(('v11', 'v12'), arrays[:2]):
                projected = local(z[model+'_lat_lon'], case['base_lat'], case['base_lon'])
                close(route, projected)
            calculated = metrics(*arrays)
        for model in ('1.1', '1.2'):
            for key in [k[0] for k in TRACK_METRICS] + ['track_error_by_lead_km']:
                close(calculated[model][key], saved['metrics'][model][key])
        groups[case['storm_id']].append(calculated)
    if len(groups) != plan['storm_count'] or len(groups) != report['completed_storms']:
        raise ValueError('Incomplete storm coverage')
    expected_storms = {s['storm_id']: s for s in report['storm_scores']}
    for model in ('1.1', '1.2'):
        for key in [k[0] for k in TRACK_METRICS] + ['track_error_by_lead_km']:
            storm_values = []
            for sid, days in groups.items():
                values = [d[model][key] for d in days if d[model][key] is not None]
                value = np.mean(values, axis=0).tolist() if values else None
                close(value, expected_storms[sid]['models'][model][key])
                if value is not None:
                    storm_values.append(value)
            close(np.mean(storm_values, axis=0).tolist(), report['aggregate_equal_storm'][model][key])
    return files


def snapshot(daily, plan, intensity, runner, identities):
    if daily['cohort_sha256'] != intensity['source_hashes']['cohort']:
        raise ValueError('Track and pressure use different frozen daily plans')
    if daily['completed_storms'] != 270 or daily['completed_daily_cases'] != 1473:
        raise ValueError('Unexpected released daily coverage')
    if intensity['status'] != 'complete_verified' or not intensity['verification']['all_frozen_original_forecasts_sha256_valid']:
        raise ValueError('Pressure comparison is not verified')
    metrics = []
    curve_metrics = []
    for key, label, unit, digits, preferred in TRACK_METRICS:
        metrics.append(dict(key=key, label=label, unit=unit, decimals=digits,
            preferred=preferred, coverage={'storms': 270, 'daily_issues': 1473},
            values={m: daily['aggregate_equal_storm'][m][key] for m in ('1.1', '1.2')}))
    for agency in ('JMA', 'USA'):
        key = 'pressure_'+agency+'_hpa'
        source = intensity['metrics'][key]['mae']
        shape = intensity['metrics'][key]['shape_error']
        # The published error is (1 - centred cosine) / 2. Its complement is
        # a similarity; this linear transform preserves the exact paired mask.
        delta = shape['paired_delta_1_2_minus_1_1']
        curve_metrics.append(dict(key='pressure_'+agency+'_curve_similarity',
            label=agency+' pressure-curve similarity', unit='similarity',
            decimals=4, preferred='higher', reference=agency,
            coverage={'storms': shape['valid_storms'], 'daily_issues': shape['valid_daily_issues']},
            values={m: 1-shape['models'][m]['equal_storm_mean'] for m in ('1.1', '1.2')},
            intervals={m: [1-shape['models'][m]['bootstrap_95_percent_interval'][1],
                          1-shape['models'][m]['bootstrap_95_percent_interval'][0]] for m in ('1.1', '1.2')},
            paired_delta={'mean': -delta['mean'],
                'bootstrap_95_percent_interval': [-delta['bootstrap_95_percent_interval'][1],
                    -delta['bootstrap_95_percent_interval'][0]]},
            definition='(1 + centred cosine similarity) / 2 at exact common UTC leads; no shift, time warp or lag optimization. At least six common points and non-flat truth and both forecasts. Removes level and amplitude, so read with pressure MAE and actual hPa curves.'))
        metrics.append(dict(key=key, label=agency+' central-pressure MAE', unit='hPa',
            decimals=2, preferred='lower',
            coverage={'storms': source['valid_storms'], 'daily_issues': source['valid_daily_issues'],
                'valid_issue_leads': sum(runner[key]['paired_issue_leads_by_lead']),
                'issue_leads_by_lead': runner[key]['paired_issue_leads_by_lead']},
            values={m: source['models'][m]['equal_storm_mean'] for m in ('1.1', '1.2')},
            intervals={m: source['models'][m]['bootstrap_95_percent_interval'] for m in ('1.1', '1.2')},
            paired_delta=source['paired_delta_1_2_minus_1_1'],
            per_lead={m: runner[key]['models'][m]['equal_storm_mae_by_lead'] for m in ('1.1', '1.2')}))
    return dict(schema='released-daily-benchmark-v1', status='complete_verified',
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        cohort={'sha256': daily['cohort_sha256'], 'storms': 270, 'daily_issues': 1473,
            'leads_hours': plan['leads_hours'], 'aggregation': daily['aggregation']},
        models=[{'key': 'v11', 'label': 'Trackformer 1.1',
            'method': 'Frozen weighted route / native intensity ensemble'},
            {'key': 'v12', 'label': 'Trackformer 1.2', 'method': '50 distinct input-perturbed members',
             'member_count': 50, 'checkpoint_sha256': CHECKPOINT},
            {'key': 'deepmind', 'label': 'DeepMind Mini', 'method': 'Not run on this daily cohort',
             'status': 'unavailable', 'reason': 'Older starts and different weighting; no transferred score.'}],
        metrics=curve_metrics+metrics,
        intensity_coverage={'eligible_daily_issues': intensity['eligible_intensity_daily_issues'],
            'unavailable_daily_issues': intensity['excluded_missing_current_intensity'],
            'reason': 'Frozen 1.1 pipeline requires valid issue-time wind and pressure.'},
        per_lead_track={m: daily['aggregate_equal_storm'][m]['track_error_by_lead_km'] for m in ('1.1', '1.2')},
        sources=identities, notes=[
            'Development evidence, not a certified unused holdout or an architecture ablation.',
            'Metric-specific common truth masks; missing outputs and labels never become zero error.',
            'No clean one-member daily evaluation: these 1.2 scores are always the verified 50-member mean.',
            'Pressure metrics use 40 storms / 134 issues, not all 270 storms / 1473 issues.',
            'No wind or radius bars in the Site main chart; diagnostic reports remain separately downloadable.',
            'The pressure display repair does not change these immutable forecasts or central-pressure results.'])


def build(project, output):
    from plot_intensity_benchmark import verify_and_summarize
    plan = read(project/'benchmark_daily_storms_1_2/cohort.json')
    daily = read(ROOT/'evaluation/daily_storm_final.json')
    originals = check_daily(project, daily, plan)
    source = project/'output/intensity-v12-v11-20261001'
    intensity = read(ROOT/'evaluation/intensity/intensity_final.json')
    receipt = read(ROOT/'evaluation/intensity/verification.json')
    if sha(ROOT/'evaluation/intensity/intensity_final.json') != receipt['summary_sha256']:
        raise ValueError('Published pressure report hash differs')
    reproduced = verify_and_summarize(source, project)
    for key in ('metrics', 'native_1_1_radius_USA_km', 'source_hashes', 'verification'):
        if reproduced[key] != intensity[key]:
            raise ValueError('Verified pressure report no longer reproduces: '+key)
    runner = read(source/'report.json')
    identities = {'daily_report_sha256': sha(ROOT/'evaluation/daily_storm_final.json'),
        'intensity_report_sha256': sha(ROOT/'evaluation/intensity/intensity_final.json'),
        'intensity_receipt_sha256': sha(ROOT/'evaluation/intensity/verification.json'),
        'intensity_protocol_sha256': sha(source/'protocol.json'),
        'intensity_runner_report_sha256': sha(source/'report.json')}
    data = snapshot(daily, plan, intensity, runner, identities)
    import csv
    names = {}
    with (project/'data/ibtracs/ibtracs.WP.list.v04r01.csv').open() as stream:
        rows = csv.DictReader(stream)
        next(rows)  # units row, not a storm
        for row in rows:
            if row['NAME'].strip() not in ('', 'NOT_NAMED'):
                names[row['SID'].strip()] = row['NAME'].strip()
    records = [read(p) for p in sorted((source/'cases').glob('*.json'))]
    data['pressure_curves'] = [dict(case_index=r['case_index'], storm_id=r['storm_id'],
        name=names.get(r['storm_id'], r['storm_id']), issue_time_utc=r['issue_time_utc'],
        observed={agency:r['truth']['pressure_'+agency+'_hpa'] for agency in ('JMA','USA')},
        forecasts={'1.1':[p['central_pressure_hpa'] for p in r['v11']],
                   '1.2':r['v12_pressure_hpa']}) for r in records if r['v11'] is not None]
    output = output.resolve()
    if not output.is_relative_to(Path('/Volumes/D')):
        raise ValueError('Artifacts must stay on D')
    output.mkdir(parents=True, exist_ok=True)
    (output/'released_daily_benchmark.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    audit = {'status': 'complete_verified', **identities,
        'snapshot_sha256': sha(output/'released_daily_benchmark.json'),
        'verified_original_daily_forecasts': len(originals),
        'all_route_scores_and_absolute_local_conversions_reproduced': True,
        'original_forecast_sha256': originals, 'verified_pressure_member_replays': 134,
        'missing_results_scored_as_zero': 0, 'new_inference_performed': False}
    (output/'released_daily_verification.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k != 'original_forecast_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT.parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.project, args.output)
