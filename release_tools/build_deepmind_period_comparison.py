"""Audit and partition completed CUDA scores; never run or change a forecast.

The frozen worker export stays byte-for-byte intact. This derivative uses UTC
issue years, not model version strings or storm-name matching. Calendar 2024
is a separate boundary group: after_2024 always means 2025 onward.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

from deepmind_daily_benchmark import aggregate, STEP_NS
from import_deepmind_release_results import ROOT, same, validate_identity

FOLDER = ROOT / 'evaluation/deepmind_daily'
DEST = FOLDER / 'period_comparison.json'
MODELS = ('1.1', '1.2', 'deepmind')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def period_key(issue_time):
    issue = datetime.fromisoformat(issue_time.replace('Z', '+00:00'))
    require(issue.utcoffset().total_seconds() == 0, 'Issue date must be UTC')
    return 'before_2024' if issue.year < 2024 else 'year_2024' if issue.year == 2024 else 'after_2024'


def summarize(records, total_issues, total_storms):
    groups = defaultdict(list)
    for record in records:
        groups[record['storm_id']].append(record)
    return {
        'daily_issues': len(records),
        'storms': len(groups),
        'daily_issue_percentage': 100 * len(records) / total_issues,
        'storm_percentage': 100 * len(groups) / total_storms,
        'years': sorted({int(r['issue_time_utc'][:4]) for r in records}),
        'case_indices': sorted(r['case_index'] for r in records),
        'scores': aggregate(groups),
    }


def build(archive):
    protocol = json.loads((FOLDER / 'protocol.json').read_text())
    exported = json.loads((FOLDER / 'benchmark.json').read_text())
    receipt = json.loads((FOLDER / 'verification.json').read_text())
    audit = json.loads((FOLDER / 'publication_audit.json').read_text())
    snapshot_path = ROOT / 'evaluation/released_daily/released_daily_benchmark.json'
    snapshot = json.loads(snapshot_path.read_text())
    validate_identity(protocol, receipt, exported)
    require(audit['status'] == 'complete_verified', 'Missing original publication audit')
    require(sha(archive) == audit['archive_sha256'], 'Different result archive')
    require(sha(snapshot_path) == audit['snapshot_sha256'], 'Different original comparison')
    for name, expected in audit['imported_files_sha256'].items():
        require(sha(FOLDER / name) == expected, 'Changed frozen export: ' + name)

    records = []
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP entries')
        manifest_bytes = bundle.read('results/case-manifest.json')
        require(hashlib.sha256(manifest_bytes).hexdigest() == sha(FOLDER / 'case-manifest.json'),
                'Archive case manifest differs')
        items = {r['path']: r for r in json.loads(manifest_bytes)['files']}
        for case in protocol['cases']:
            path = f"cases/{case['case_index']:05d}.json"
            content = bundle.read('results/' + path)
            require(len(content) == items[path]['bytes'] and
                    hashlib.sha256(content).hexdigest() == items[path]['sha256'],
                    'Case score hash differs: ' + path)
            record = json.loads(content)
            require(all(record.get(k) == v for k, v in case.items()), 'Case identity differs: ' + path)
            require(record['status'] == 'complete' and record['backend'] == 'cuda' and
                    record['members'] == 1 and record['gpu_verified_steps'] == 20 and
                    record['verified_device_steps'] == 20, 'Incomplete CUDA case: ' + path)
            require(record['checkpoint_sha256'] == protocol['checkpoint_sha256'] and
                    record['protocol_sha256'] == receipt['protocol_sha256'], 'Model identity differs')
            require(record['input_time_ns'] == [case['issue_ns'] - STEP_NS, case['issue_ns']] and
                    not record['future_weather_used'] and not record['future_storm_labels_used'],
                    'Noncausal input metadata')
            for agency, pressure in record['pressure_metrics'].items():
                if not pressure['valid_leads']:
                    continue
                masks = [[value is not None for value in pressure['models'][m]['error_by_lead_hpa']]
                         for m in MODELS]
                require(masks[0] == masks[1] == masks[2] and sum(masks[0]) == pressure['valid_leads'],
                        'Different pressure support: ' + agency)
                shapes = [pressure['models'][m]['curve_similarity'] is not None for m in MODELS]
                require(shapes[0] == shapes[1] == shapes[2], 'Different pressure-curve support')
            records.append(record)

    total_issues, total_storms = 1473, 270
    require(len(records) == total_issues and len({r['storm_id'] for r in records}) == total_storms,
            'Incomplete case/storm coverage')
    bins = {key: [r for r in records if period_key(r['issue_time_utc']) == key]
            for key in ('before_2024', 'year_2024', 'after_2024')}
    periods = {'total': summarize(records, total_issues, total_storms),
               **{key: summarize(rows, total_issues, total_storms) for key, rows in bins.items()}}
    same(periods['total']['scores'], exported['aggregate_equal_complete_storm'], 'Original total scores')
    original_groups = defaultdict(list)
    for record in records:
        original_groups[record['storm_id']].append(record)
    for key, expected in exported['periods'].items():
        same(aggregate({sid: days for sid, days in original_groups.items() if days[0]['period'] == key}),
             expected, 'Original period scores: ' + key)
    require(set(periods['before_2024']['case_indices']).isdisjoint(periods['after_2024']['case_indices']),
            'Overlapping temporal subsets')
    require(sum(len(rows) for rows in bins.values()) == total_issues, 'Partition does not cover total')
    require(sum(len({r['storm_id'] for r in rows}) for rows in bins.values()) == total_storms,
            'Storm crosses year bins; do not imply additive equal-storm weights')
    reference = next(m for m in snapshot['models'] if m['key'] == 'deepmind')
    return {
        'schema': 'deepmind-completed-utc-period-comparison-v1',
        'status': 'complete_verified',
        'model': reference,
        'single_checkpoint_across_all_periods': True,
        'cohort': exported['cohort'],
        'period_definition': {'before_2024': 'UTC issue year < 2024',
                              'year_2024': 'UTC issue year == 2024',
                              'after_2024': 'UTC issue year > 2024 (2025 onward)',
                              'total': 'All 1,473 frozen daily starts'},
        'periods': periods,
        'source_hashes': {'archive_sha256': audit['archive_sha256'],
                          'publication_audit_sha256': sha(FOLDER / 'publication_audit.json'),
                          'snapshot_sha256': sha(snapshot_path),
                          **{name: sha(FOLDER / name) for name in
                             ('protocol.json', 'verification.json', 'case-manifest.json', 'benchmark.json')}},
        'verified_case_score_files': total_issues,
        'aggregation': 'Valid leads within day, days within storm, then equal storm weight; identical original masks.',
        'proportion_definition': 'Fractions of evaluation starts and storms, not proportions of model training data.',
        'pressure_definition': 'Agency-specific central-pressure MAE/centred time-curve similarity, not map-grid MAE.',
        'missing_results_scored_as_zero': 0,
        'new_inference_performed': False,
        'model_weights_changed': False,
        'forecast_arrays_modified': False,
        'raw_inputs_reaudited': False,
        'notes': ['Historical years overlap Mini fitting years; do not call the combined cohort an unused temporal test.',
                  '2024 is already after the checkpoint training cutoff, but is excluded from strict >2024.',
                  'All-time pressure has smaller valid coverage than all-time track; counts are metric-specific.',
                  'Different causal input pipelines and 50-member 1.2 versus one-member Mini; not an equal-compute comparison.',
                  'Post-2024 dates alone do not certify an untouched holdout across earlier development experiments.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.archive)
    DEST.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(DEST),
                      'periods': {k: {n: v[n] for n in
                                     ('daily_issues', 'storms', 'daily_issue_percentage', 'storm_percentage')}
                                  for k, v in result['periods'].items()}}, indent=2))


if __name__ == '__main__':
    main()
