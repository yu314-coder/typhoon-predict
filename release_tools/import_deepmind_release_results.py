"""Audit the returned CUDA ZIP and merge documentation data, without inference.

Read immutable local reference forecasts and labels; never execute a model,
modify a forecast, publish a Site, or extract paths supplied by the archive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np

from deepmind_daily_benchmark import (
    COHORT, WEIGHTS, CHECKPOINT, STEP_NS, aggregate, curve_similarity,
    local, pressure_mask, route_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
CODE_COMMIT = '89c4b2a77a1c57b328b909c575550fd2e5aadc9c'
MODEL_LABEL = 'Google DeepMind WeatherNext Cyclones Mini (<2024)'
LEADS = np.arange(6, 121, 6)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def same(actual, expected, context):
    """Compare nested scores without treating missing values as zero."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), context)
        for key in expected:
            same(actual[key], expected[key], context + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), context)
        for index, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, context + '/' + str(index))
    elif expected is None or isinstance(expected, (str, bool)):
        require(actual == expected, context)
    else:
        require(actual is not None and np.isfinite(actual)
                and np.isclose(actual, expected, rtol=0, atol=1e-8), context)


def validate_identity(protocol, receipt, exported):
    require(receipt['status'] == exported['status'] == 'complete_verified', 'Incomplete run')
    require(receipt['verified_daily_cases'] == 1473 and receipt['verified_storms'] == 270,
            'Incomplete coverage')
    require(receipt['case_indices'] == list(range(1473)), 'Missing or duplicate case IDs')
    require([c['case_index'] for c in protocol['cases']] == list(range(1473)), 'Wrong plan IDs')
    require(protocol['target_daily_cases'] == 1473 and protocol['target_storms'] == 270,
            'Wrong target scope')
    require(protocol['leads_hours'] == LEADS.tolist(), 'Wrong lead schedule')
    require(protocol['vendor_commit'] == CODE_COMMIT, 'Wrong official software commit')
    require(protocol['resolution_degrees'] == 1 and protocol['trained_through'] == 2023,
            'Wrong Mini checkpoint generation')
    require(protocol['reference_v12_checkpoint_sha256'] == CHECKPOINT,
            'Wrong Trackformer reference')
    require(protocol['members'] == receipt['members'] == 1, 'Wrong member count')
    require(protocol['checkpoint_sha256'] == receipt['checkpoint_sha256'] == WEIGHTS,
            'Wrong official checkpoint')
    require(protocol['cohort_sha256'] == receipt['cohort_sha256']
            == exported['cohort']['sha256'] == COHORT, 'Wrong daily cohort')
    require(not receipt['cpu_predictions_reused_as_cuda'], 'CPU results relabelled as CUDA')
    require(receipt['cpu_numerical_equivalence'], 'Same-runtime numerical gate did not pass')


def merge_snapshot(base, exported, receipt, pressures):
    """Preserve every old 1.1/1.2 score and unsupported metric explicitly."""
    result = copy.deepcopy(base)
    require(all(base['cohort'][k] == v for k, v in exported['cohort'].items()),
            'Published baseline cohort differs')
    rows = {row['key']: row for row in result['metrics']}
    for row in exported['metrics']:
        target = rows[row['key']]
        for model in ('1.1', '1.2'):
            same(target['values'][model], row['values'][model], 'Original score changed')
        target['values']['deepmind'] = row['values']['deepmind']
        if 'common_lead_points' in row:
            target['coverage']['three_way_valid_issue_leads'] = row['common_lead_points']
    rows['track_error_120h_km']['values']['deepmind'] = (
        exported['aggregate_equal_complete_storm']['route']['deepmind']['track_error_120h_km']['value'])
    model = next(m for m in result['models'] if m['key'] == 'deepmind')
    model.clear()
    model.update(key='deepmind', label=MODEL_LABEL, model_checkpoint='WeatherNextCyclones_Mini_<2024',
                 software_version='0.3.0', software_commit=CODE_COMMIT,
                 trained_through=2023, resolution_degrees=1, member_count=1,
                 member_policy='One official Mini member, seed 0', checkpoint_sha256=WEIGHTS,
                 method='Official predictor and initialized-storm tracker; causal ERA5; RTX 3070 CUDA',
                 status='complete_verified')
    result['per_lead_track']['deepmind'] = (
        exported['aggregate_equal_complete_storm']['route']['deepmind']['track_error_by_lead_km'])
    for row in result['pressure_curves']:
        row['forecasts']['deepmind'] = pressures[row['case_index']]
    result.update(deepmind_verification=receipt, deepmind_periods=exported['periods'],
                  deepmind_interpretation=exported['interpretation'])
    result['notes'].append('WeatherNext Cyclones Mini <2024, software v0.3.0: one member; '
                           'not full-sized WeatherNext. Historic fitting-year overlap; '
                           'recent 2024+ results are separate. DeepMind path/Frechet scores '
                           'and confidence intervals are not supplied by this export.')
    return result


def audit(archive, project, destination):
    destination = destination.resolve()
    require(destination.is_relative_to(Path('/Volumes/D')), 'Diagnostics must stay on D')
    snapshot_path = ROOT / 'evaluation/released_daily/released_daily_benchmark.json'
    base_bytes = snapshot_path.read_bytes()
    base = json.loads(base_bytes)
    groups = defaultdict(list)
    pressures = {}
    shapes = {agency: {m: set() for m in ('1.1', '1.2', 'deepmind')} for agency in ('JMA', 'USA')}
    with zipfile.ZipFile(archive) as bundle:
        require(len(bundle.namelist()) == len(set(bundle.namelist())), 'Duplicate ZIP entries')
        require(bundle.testzip() is None, 'ZIP CRC failure')

        def read(name):
            require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts,
                    'Unsafe ZIP path')
            return bundle.read('results/' + name)

        def load(name):
            return json.loads(read(name))

        receipt = load('verification.json')
        exported = load('site_export/benchmark.json')
        protocol = load('protocol.json')
        validate_identity(protocol, receipt, exported)
        require(digest(read('verification.json')) == exported['verification_sha256'], 'Receipt hash')
        require(digest(read('protocol.json')) == receipt['protocol_sha256'], 'Protocol hash')
        require(digest(read('case-manifest.json')) == receipt['case_manifest_sha256'], 'Case manifest hash')
        canary = load('canary.json')
        require(digest(read('canary.json')) == receipt['canary_sha256'], 'Canary hash')
        require(canary['field_gate_passed'] and canary['runtime_match']
                and canary['input_bytes_match'] and canary['protocol_match'], 'Canary did not pass')
        require(canary['device']['device_kind'] == 'NVIDIA GeForce RTX 3070', 'Wrong GPU')
        require(not canary['cpu_predictions_reused_as_cuda'], 'Wrong CPU/CUDA provenance')
        case_manifest = load('case-manifest.json')
        require({f['path'] for f in case_manifest['files']} == {
            f'cases/{i:05d}.{ext}' for i in range(1473) for ext in ('npz', 'json')},
            'Case manifest is not the exact full archive')
        for manifest in (case_manifest, load('site_export/manifest.json')):
            for item in manifest['files']:
                content = read(item['path'])
                require(len(content) == item['bytes'] and digest(content) == item['sha256'],
                        'Manifest mismatch: ' + item['path'])
        for case in protocol['cases']:
            index = case['case_index']
            stem = f'{index:05d}'
            record = load('cases/' + stem + '.json')
            require(all(record.get(k) == v for k, v in case.items()), 'Case identity: ' + stem)
            require(record['status'] == 'complete' and record['members'] == 1
                    and record['backend'] == 'cuda' and record['gpu_verified_steps'] == 20
                    and record['verified_device_steps'] == 20, 'CUDA/coverage: ' + stem)
            require(record['device']['gpu_matmul_verified']
                    and record['device']['device_kind'] == 'NVIDIA GeForce RTX 3070', 'GPU: ' + stem)
            require(record['checkpoint_sha256'] == WEIGHTS
                    and record['protocol_sha256'] == receipt['protocol_sha256'], 'Model: ' + stem)
            require(record['input_time_ns'] == [case['issue_ns'] - STEP_NS, case['issue_ns']]
                    and not record['future_weather_used'] and not record['future_storm_labels_used'],
                    'Input causality metadata: ' + stem)
            require(re.fullmatch('[0-9a-f]{64}', record['input_state_sha256']) is not None,
                    'Missing saved input hash: ' + stem)
            content = read('cases/' + stem + '.npz')
            require(digest(content) == record['forecast_sha256'], 'Forecast hash: ' + stem)
            reference_content = (project / 'benchmark_daily_storms_1_2/cases' / (stem + '.npz')).read_bytes()
            require(digest(reference_content) == case['original_forecast_sha256'], 'Baseline changed: ' + stem)
            with np.load(io.BytesIO(content), allow_pickle=False) as fields, np.load(
                    io.BytesIO(reference_content), allow_pickle=False) as reference:
                require(np.array_equal(fields['lead_hours'], LEADS) and np.array_equal(
                    fields['valid_time_ns'], case['issue_ns'] + LEADS * 3600 * 10**9), 'Lead times: ' + stem)
                route = fields['route_lat_lon']
                field = fields['wp_pressure_hpa']
                require(route.shape == (20, 2) and np.isfinite(route).all(), 'Route missing: ' + stem)
                require(field.shape == (20, 61, 81) and np.isfinite(field).all()
                        and field.min() >= 800 and field.max() <= 1100, 'Physical fields: ' + stem)
                require(np.array_equal(np.sort(fields['latitude']), np.arange(61))
                        and np.array_equal(np.sort(fields['longitude']), np.arange(100, 181)),
                        'Native one-degree grid: ' + stem)
                p = fields['central_pressure_hpa'].copy()
                valid = fields['pressure_valid'].astype(bool)
                require(p.shape == valid.shape == (20,), 'Pressure schedule: ' + stem)
                require(np.all(~valid | (np.isfinite(p) & (p >= 800) & (p <= 1100))),
                        'Invalid pressure marked available: ' + stem)
                require(np.all(valid | ~np.isfinite(p)), 'Unavailable pressure not masked: ' + stem)
                routes = route_metrics({'1.1': reference['v11_local'], '1.2': reference['v12_local'],
                                        'deepmind': local(route, case['base_lat'], case['base_lon'])},
                                       reference['truth_local'])
            same(routes, record['route_metrics'], 'Route scoring: ' + stem)
            source = json.loads((project / 'output/intensity-v12-v11-20261001/cases' / (stem + '.json')).read_bytes())
            pressure_scores = {}
            if source['v11'] is not None:
                curves = {'1.1': np.array([r['central_pressure_hpa'] for r in source['v11']], dtype=float),
                          '1.2': np.array(source['v12_pressure_hpa'], dtype=float), 'deepmind': p}
                for agency in ('JMA', 'USA'):
                    truth = np.array(source['truth']['pressure_' + agency + '_hpa'], dtype=float)
                    mask = pressure_mask(truth, curves)
                    pressure_scores[agency] = {'valid_leads': int(mask.sum()), 'models': {}}
                    for model, prediction in curves.items():
                        similarity = curve_similarity(prediction[mask], truth[mask])
                        if similarity is not None:
                            shapes[agency][model].add(index)
                        pressure_scores[agency]['models'][model] = {
                            'mae_hpa': float(np.abs(prediction[mask] - truth[mask]).mean()) if mask.any() else None,
                            'curve_similarity': similarity,
                            'error_by_lead_hpa': [float(abs(prediction[i] - truth[i])) if mask[i] else None for i in range(20)]}
            same(pressure_scores, record['pressure_metrics'], 'Pressure scoring: ' + stem)
            pressures[index] = [float(value) if ok else None for value, ok in zip(p, valid)]
            groups[case['storm_id']].append(dict(record, route_metrics=routes, pressure_metrics=pressure_scores))
        require(len(groups) == 270 and sum(map(len, groups.values())) == 1473, 'Incomplete storms')
        for agency, models in shapes.items():
            require(models['1.1'] == models['1.2'] == models['deepmind'],
                    'Pressure shape scores do not use the same cases: ' + agency)
        same(aggregate(groups), exported['aggregate_equal_complete_storm'], 'Equal-storm aggregate')
        for period, expected in exported['periods'].items():
            same(aggregate({sid: days for sid, days in groups.items() if days[0]['period'] == period}),
                 expected, 'Period scoring: ' + period)
        result = merge_snapshot(base, exported, receipt, pressures)
        destination.mkdir(parents=True, exist_ok=True)
        saved = {}
        for target, original in (
                ('benchmark.json', 'site_export/benchmark.json'), ('verification.json', 'verification.json'),
                ('protocol.json', 'protocol.json'), ('case-manifest.json', 'case-manifest.json'),
                ('canary.json', 'canary.json')):
            content = read(original)
            (destination / target).write_bytes(content)
            saved[target] = digest(content)
    snapshot_content = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    review = dict(status='complete_verified', reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                  archive_sha256=file_sha(archive), verified_daily_cases=1473, verified_storms=270,
                  hashed_case_files=2946, verified_physical_pressure_grids=1473 * 20,
                  model=MODEL_LABEL, software_version='0.3.0', software_commit=CODE_COMMIT,
                  checkpoint_sha256=WEIGHTS, cohort_sha256=COHORT, members=1,
                  backend='NVIDIA GeForce RTX 3070 / CUDA',
                  all_route_and_pressure_scores_recomputed=True,
                  exact_common_curve_case_ids=True, all_original_forecasts_sha256_verified=True,
                  causal_input_metadata_and_saved_hashes_verified=True,
                  raw_era5_states_in_return_zip=False,
                  raw_input_audit_scope='Original worker receipt and saved input hashes/times; '
                                       'raw ERA5 states were not returned in this result-only ZIP.',
                  new_inference_performed=False, model_weights_changed=False,
                  forecasts_modified=False, missing_results_scored_as_zero=0,
                  previous_snapshot_sha256=digest(base_bytes), snapshot_sha256=digest(snapshot_content),
                  imported_files_sha256=saved,
                  period_coverage={period: dict(daily_issues=sum(len(d) for d in groups.values() if d[0]['period'] == period),
                      storms=sum(d[0]['period'] == period for d in groups.values())) for period in exported['periods']})
    snapshot_path.write_bytes(snapshot_content)
    (destination / 'publication_audit.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=ROOT.parent)
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation/deepmind_daily')
    args = parser.parse_args()
    audit(args.archive, args.project, args.output)
