"""Resumable, frozen daily-cohort intensity comparison; no training or tuning.

Reuse verified 1.2 fifty-member pressure forecasts, replay the original 1.1
intensity experts, then replay the IDENTICAL 1.2 members to recover auxiliary
wind that the original route benchmark did not save. All outputs stay on D.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

CHECKPOINT = 'f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0'
LEADS = list(range(6, 121, 6))
HOUR = 3600 * 10**9


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def initial_available(raw):
    return bool(np.isfinite(raw).all() and 0 <= raw[0] < 250 and 800 < raw[1] < 1100)


def number(row, name, low, high):
    try:
        value = float(row.get(name, ''))
        return value if np.isfinite(value) and low <= value <= high else None
    except (TypeError, ValueError):
        return None


def common_errors(a, b, truth, *, low, high):
    a, b, truth = (np.asarray(v, dtype=float) for v in (a, b, truth))
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(truth)
    valid &= (truth >= low) & (truth <= high)
    return {key: [float(v) if ok else None for v, ok in zip(values, valid)]
            for key, values in [('1.1', np.abs(a-truth)), ('1.2', np.abs(b-truth))]}


def aggregate(records, key):
    """Leads -> daily issue -> storm -> equal storm. Never pool storm lengths."""
    grouped = defaultdict(list)
    for record in records:
        if record.get('errors', {}).get(key):
            grouped[record['storm_id']].append(record['errors'][key])
    storms = []
    counts = np.zeros(20, dtype=int)
    total_daily = 0
    for sid, issues in grouped.items():
        values = {}
        for model in ('1.1', '1.2'):
            daily = [np.array([np.nan if v is None else v for v in d[model]]) for d in issues]
            usable = [d for d in daily if np.isfinite(d).any()]
            if not usable:
                values = {}; break
            leads = []
            for i in range(20):
                valid = [d[i] for d in usable if np.isfinite(d[i])]
                leads.append(float(np.mean(valid)) if valid else None)
            values[model] = {'mae': float(np.mean([d[np.isfinite(d)].mean() for d in usable])),
                             'mae_by_lead': leads}
        if values:
            total_daily += sum(any(v is not None for v in d['1.1']) for d in issues)
            counts += np.array([sum(d['1.1'][i] is not None for d in issues) for i in range(20)])
            storms.append({'storm_id': sid, 'models': values})
    result = {'valid_storms': len(storms), 'valid_daily_issues': total_daily,
              'paired_issue_leads_by_lead': counts.tolist(), 'models': {}, 'storm_scores': storms}
    if not storms:
        return result
    for model in ('1.1', '1.2'):
        result['models'][model] = {'equal_storm_mae': float(np.mean([s['models'][model]['mae'] for s in storms])),
            'equal_storm_mae_by_lead': [float(np.mean(v)) if (v := [s['models'][model]['mae_by_lead'][i]
                for s in storms if s['models'][model]['mae_by_lead'][i] is not None]) else None for i in range(20)]}
    differences = np.array([s['models']['1.2']['mae']-s['models']['1.1']['mae'] for s in storms])
    random = np.random.default_rng(4712)
    boot = differences[random.integers(0, len(differences), (2000, len(differences)))].mean(1)
    result['paired_storm_delta_1_2_minus_1_1'] = {'mean': float(differences.mean()),
        'bootstrap_95_percent_interval': np.quantile(boot, [.025, .975]).tolist(),
        'resampling_unit': 'whole storm; fixed 2000 replicates, seed 4712; development uncertainty only'}
    return result


def summarize(output, plan, phase):
    records = [json.loads(p.read_text()) for p in sorted((output/'cases').glob('*.json'))]
    done = {r['case_index'] for r in records}
    eligible = {c['case_index'] for c in plan['cases'] if c['initial_intensity_available']}
    wind_done = {r['case_index'] for r in records if r.get('v12_auxiliary_wind_kt') is not None}
    report = {'status': 'complete' if len(done) == len(plan['cases']) and wind_done == eligible else 'running_partial',
        'phase': phase, 'updated_at_utc': utc(), 'protocol_sha256': sha(output/'protocol.json'),
        'cohort_storms': plan['storm_count'], 'cohort_daily_issues': len(plan['cases']),
        'processed_pressure_issues': len(done), 'eligible_native_1_1_intensity_issues': len(eligible),
        'wind_replays_completed': len(wind_done),
        'missing_current_1_1_inputs': len(plan['cases'])-len(eligible),
        'member_counts': {'1.2': 50, '1.1': 'frozen 3 primary + 3 structure + 3 temporal experts, not 50'},
        'aggregation': plan['aggregation'], 'evaluation_label': plan['evaluation_label'],
        'pressure_USA_hpa': aggregate(records, 'pressure_USA_hpa'),
        'pressure_JMA_hpa': aggregate(records, 'pressure_JMA_hpa'),
        'auxiliary_wind_vs_USA_1min_kt': aggregate(records, 'auxiliary_wind_vs_USA_1min_kt'),
        'wind_note': '1.1 nominal one-minute output versus 1.2 auxiliary scalar against USA_WIND. '
                     'Not a calibrated surface-wind or architecture-ablation claim. JMA 10-minute wind is not converted or pooled.',
        'radius_note': 'No matched native 1.2 radius head. No radius MAE or radius-skill ranking is claimed.'}
    write(output/'report.json', report)
    return report


def observations(path, storm_ids):
    import csv
    rows = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row.get('SID') not in storm_ids:
                continue
            key = (row['SID'], row['ISO_TIME'].replace(' ', 'T')+'Z')
            if key in rows:
                raise ValueError('Ambiguous exact observation: '+str(key))
            rows[key] = row
    return rows


def prepare(project, output):
    source = project/'benchmark_daily_storms_1_2'
    cohort = json.loads((source/'cohort.json').read_text())
    with np.load(project/'track_build/track_windows_v13.npz', allow_pickle=False) as z:
        raw = z['track'][:, -1, 4:6]*z['track_std'][4:6]+z['track_mean'][4:6]
        ids, times = z['storm_id'].astype(str), z['base_time'].astype('int64')
    cases = []
    for c in cohort['cases']:
        row = c['source_row']
        if ids[row] != c['storm_id'] or times[row] != c['issue_ns']:
            raise ValueError('Frozen case identity differs from archive')
        cases.append({**c, 'initial_intensity_available': initial_available(raw[row]),
                      'current_input_wind_kt': float(raw[row, 0]), 'current_input_pressure_hpa': float(raw[row, 1])})
    sources = {key: item for key, item in cohort['source_hashes'].items() if key != 'benchmark_code'}
    for key, path in {'cohort': source/'cohort.json', 'ibtracs': project/'data/ibtracs/ibtracs.WP.list.v04r01.csv',
        'intensity_code': project/'public-release/trackformer_1_1_intensity.py',
        'temporal_code': project/'public-release/trackformer_1_1_temporal.py',
        'local_weather_q': project/'cache/dlm4_q.npy', 'runner': Path(__file__).resolve()}.items():
        sources[key] = {'path': str(path), 'sha256': sha(path)}
    for path in sorted((project/'public-release/models/trackformer_1_1').glob('*')):
        if path.is_file():
            sources['v11_'+path.name] = {'path': str(path), 'sha256': sha(path)}
    for key, item in sources.items():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Source changed: '+key)
    if sha(project/'v173/checkpoints/epoch_004.pt') != CHECKPOINT:
        raise ValueError('Wrong release checkpoint')
    plan = {'schema': 'trackformer-intensity-daily-matched-v1', 'created_utc': utc(),
        'cohort_sha256': sha(source/'cohort.json'), 'storm_count': cohort['storm_count'], 'cases': cases,
        'source_hashes': sources, 'checkpoint_sha256': CHECKPOINT, 'leads_hours': LEADS,
        'aggregation': 'Mean paired valid leads per daily issue, daily scores per storm, then equal storm weights; same masks for both models.',
        'selection': 'Inherited complete frozen 270-storm / 1473-daily-issue cohort; no forecast-error selection. '
                     '1.1 native inference requires finite current wind and pressure; excluded inputs are reported, not repaired.',
        'evaluation_label': 'Development comparison, not an unused holdout: older storms overlap original 1.1 training; '
                            'the later 1.2 fitting period makes earlier hindcasts retrospective. Pipelines differ.',
        'truth_policy': 'Exact SID and valid UTC; native USA pressure, TOKYO pressure separately. '
                        'USA one-minute wind diagnostic, never mixed with JMA ten-minute wind; blanks remain null.',
        'v12_policy': 'Verify original saved pressure SHA, replay same seeds 2043+source_row*100+member, '
                      'same nine analyses, detail masks, issue inputs and frozen preprocessing. Require replayed mean pressure match.',
        'v11_policy': 'Original public frozen nine experts and frozen calibration, including original validation-fit coefficients; no new fit.'}
    write(output/'protocol.json', plan)
    return plan


def run(project, output, chunk, limit):
    import torch
    output.mkdir(parents=True, exist_ok=True)
    if not str(output.resolve()).startswith('/Volumes/D/'):
        raise ValueError('Artifacts must stay on /Volumes/D')
    lock = (output/'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This intensity benchmark is already running')
    if not torch.backends.mps.is_available():
        raise RuntimeError('Requested Mac GPU benchmark requires MPS')
    plan = json.loads((output/'protocol.json').read_text()) if (output/'protocol.json').exists() else prepare(project, output)
    for key, item in plan['source_hashes'].items():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen source changed: '+key)
    sys.path.insert(0, str(project/'public-release'))
    from trackformer_1_1_intensity import Trackformer11IntensityEnsemble
    sys.path.insert(0, str(project))
    import benchmark_ensemble50_field as ens
    with np.load(project/'track_build/track_windows_v13.npz', allow_pickle=False) as z:
        track, times, ids, lat, lon = (z[k] for k in ['track', 'base_time', 'storm_id', 'base_lat', 'base_lon'])
        track = track.astype('float32'); raw_track = track*z['track_std']+z['track_mean']
    local_fields = np.load(project/'cache/dlm4_q.npy', mmap_mode='r')
    lookup = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(ids, times))}
    source = project/'benchmark_daily_storms_1_2/cases'
    obs = observations(project/'data/ibtracs/ibtracs.WP.list.v04r01.csv', {c['storm_id'] for c in plan['cases']})
    truth_by_case = {}
    # Labels are not passed to either inference function.
    for c in plan['cases']:
        issue = datetime.fromisoformat(c['issue_time_utc'].replace('Z', '+00:00'))
        rows = [obs.get((c['storm_id'], (issue+timedelta(hours=h)).strftime('%Y-%m-%dT%H:%M:%SZ')), {}) for h in LEADS]
        truth_by_case[c['case_index']] = {'pressure_USA_hpa': [number(r, 'USA_PRES', 800, 1100) for r in rows],
            'pressure_JMA_hpa': [number(r, 'TOKYO_PRES', 800, 1100) for r in rows],
            'auxiliary_wind_vs_USA_1min_kt': [number(r, 'USA_WIND', 0, 249) for r in rows]}
    model_root = project/'public-release/models/trackformer_1_1'
    old = Trackformer11IntensityEnsemble(model_root, model_root/'trackformer_1_1_calibration.json', device='mps')
    print(json.dumps({'event': 'started', 'pid': os.getpid(), 'device': 'mps',
        'cohort': len(plan['cases']), 'eligible': sum(c['initial_intensity_available'] for c in plan['cases'])}), flush=True)
    (output/'cases').mkdir(exist_ok=True)
    for c in plan['cases']:
        index, row = c['case_index'], c['source_row']
        dest = output/'cases'/f'{index:05d}.json'
        if dest.exists():
            continue
        original = json.loads((source/f'{index:05d}.json').read_text())
        archive = source/f'{index:05d}.npz'
        if original['forecast_sha256'] != sha(archive) or original['cohort_sha256'] != plan['cohort_sha256'] or original['members'] != 50:
            raise ValueError('Original fifty-member forecast changed')
        with np.load(archive, allow_pickle=False) as z:
            if int(z['source_row']) != row:
                raise ValueError('Original forecast row mismatch')
            pressure = z['central_pressure_hpa'].tolist()
        result = {**c, 'v12_pressure_hpa': pressure, 'original_forecast_sha256': sha(archive),
            'truth': truth_by_case[index], 'errors': {}, 'v11': None,
            'excluded_reason': None if c['initial_intensity_available'] else 'Original 1.1 pipeline lacks valid issue wind/pressure; no future fill.'}
        if c['initial_intensity_available']:
            history = np.zeros((8, 17, 17), dtype='float32'); available = np.zeros(2, dtype='float32'); history_rows = []
            for j, h in enumerate((12, 24)):
                past = lookup.get((str(ids[row]), int(times[row])-h*HOUR))
                history_rows.append(past)
                if past is not None:
                    history[j*4:(j+1)*4] = np.clip(local_fields[past].astype('float32')/31.75, -4, 4)
                    available[j] = 1
            raw = raw_track[row]
            predicted, provenance = old.predict(track[row], np.clip(local_fields[row].astype('float32')/31.75, -4, 4),
                float(raw[-1, 4]), float(raw[-1, 5]), float(raw[-2, 4]), float(raw[-2, 5]),
                current_structure=np.r_[raw[-1, 6], raw[-1, 8:20]], history_field=history, history_available=available)
            result.update(v11=predicted, v11_provenance=provenance, v11_past_source_rows=history_rows,
                          v11_input_sha256=hashlib.sha256(track[row].tobytes()+local_fields[row].tobytes()+history.tobytes()).hexdigest())
            old_pressure = [p['central_pressure_hpa'] for p in predicted]
            for key in ('pressure_USA_hpa', 'pressure_JMA_hpa'):
                result['errors'][key] = common_errors(old_pressure, pressure, result['truth'][key], low=800, high=1100)
        write(dest, result)
        if index % 50 == 0:
            summarize(output, plan, 'original_1_1_native_pressure')
    summarize(output, plan, 'recovering_original_1_2_auxiliary_wind')
    del old
    torch.mps.empty_cache()
    sys.path.insert(0, str(project/'v173'))
    from model import CoreForecaster
    payload = torch.load(project/'v173/checkpoints/epoch_004.pt', map_location='cpu', weights_only=False)
    if (payload['version'], payload['architecture'], payload['epoch']) != ('1.2.73', 'v173-moving-core-multiscale-attention', 4):
        raise ValueError('Wrong release architecture')
    contract = payload['data_contract']
    model = CoreForecaster(contract).to('mps').eval(); model.load_state_dict(payload['model'], strict=True)
    del payload
    with np.load(project/'track_build/basin_all_int8.npz', allow_pickle=False) as z:
        atlas, scale, offset = z['time'], z['scale'], z['offset']
    q = np.load(project/'data/v164_reuse/basin_q_verified.npy', mmap_mode='r')
    slp = np.load(project/'track_build/basin_slp_atlas_float16.npy', mmap_mode='r')
    native = np.load(project/'data/v164_reuse/pressure_hpa.npy', mmap_mode='r')
    patches = {int(r['track_archive_row']): (i, r) for i, r in enumerate(json.loads((project/'data/v164_reuse/plan.json').read_text())['rows'])}
    geography = np.load(project/'data/v164_pressure_geography/geography.npz', allow_pickle=False)
    static = ens.static(geography, np.asarray(contract['global_lat']), np.asarray(contract['global_lon']))
    done = 0
    for c in plan['cases']:
        if not c['initial_intensity_available']:
            continue
        index, row = c['case_index'], c['source_row']; dest = output/'cases'/f'{index:05d}.json'
        result = json.loads(dest.read_text())
        if result.get('v12_auxiliary_wind_kt') is not None:
            continue
        started = time.monotonic(); ai = int(np.searchsorted(atlas, times[row]))
        wanted = int(times[row])+np.arange(-8, 1)*6*HOUR
        if ai < 8 or not np.array_equal(atlas[ai-8:ai+1].astype('int64'), wanted):
            raise ValueError('Nine exact causal analyses required')
        weather = np.concatenate([slp[ai-8:ai+1, None].astype('float32'),
            q[ai-8:ai+1].astype('float32')*scale[None, :, None, None]+offset[None, :, None, None]], axis=1)
        patch = patches.get(row)
        if patch:
            pi, pm = patch
            if int(pm['issue_ns']) != int(times[row]): raise ValueError('Native patch issue mismatch')
            rlat = float(pm['center_lat'])+np.linspace(15, -15, 121, dtype='float32')
            rlon = float(pm['center_lon'])+np.linspace(-15, 15, 121, dtype='float32')
            regional = native[pi, :9].astype('float32'); detail = 1.
        else:
            rlat = round(float(lat[row])*4)/4+np.linspace(15, -15, 121, dtype='float32')
            rlon = round(float(lon[row])*4)/4+np.linspace(-15, 15, 121, dtype='float32')
            regional = np.zeros((9, 121, 121), dtype='float32'); detail = 0.
        previous = lookup.get((str(ids[row]), int(times[row])-6*HOUR))
        motion = np.zeros(2, dtype='float32') if previous is None else np.array([
            ((lon[row]-lon[previous]+180)%360-180)*111.2*np.cos(np.deg2rad(lat[row])),
            (lat[row]-lat[previous])*111.2], dtype='float32')
        base = ens.normalized_inputs(contract, static, [{'global': weather, 'regional': regional,
            'regional_static': ens.static(geography, rlat, rlon), 'detail': detail,
            'center': np.array([lat[row], lon[row]], dtype='float32'), 'motion': motion,
            'intensity': raw_track[row, -1, 4:6].copy(), 'mask': np.ones(2, dtype='float32')}])
        ens.NOISE_SEED_BASE = 2043+row*100
        wind_members, pressure_members, input_hashes = [], [], []
        for first in range(0, 50, chunk):
            inputs = ens.member_inputs(base, list(range(first, min(50, first+chunk))))
            for j in range(len(inputs['center'])):
                input_hashes.append(hashlib.sha256(inputs['global_history'][j].tobytes()+inputs['regional_history'][j].tobytes()).hexdigest())
            tensors = {k: torch.from_numpy(v.astype('float32', copy=False)).to('mps') for k, v in inputs.items()}
            with torch.inference_mode():
                state = model.initial(tensors); outputs = []
                for _ in LEADS:
                    state, values = model.step(state); outputs.append(values)
            wind_members.append(torch.stack([v['vmax'] for v in outputs], 1).cpu().numpy())
            pressure_members.append(torch.stack([v['pressure'] for v in outputs], 1).cpu().numpy())
            del state, outputs, tensors
        wind = np.concatenate(wind_members); pressure = np.concatenate(pressure_members)
        if wind.shape != (50, 20) or pressure.shape != (50, 20) or len(set(input_hashes)) != 50 or not np.isfinite(wind).all():
            raise ValueError('Invalid real-member wind replay')
        mean_pressure = pressure.mean(0, dtype='float64')
        delta = float(np.max(np.abs(mean_pressure-np.asarray(result['v12_pressure_hpa']))))
        if delta > 1e-3:
            raise ValueError(f'Original fifty-member mean differs on replay: {delta} hPa; no scoring changed inputs')
        wind_mean = wind.mean(0, dtype='float64')
        result.update(v12_auxiliary_wind_kt=wind_mean.tolist(), pressure_replay_max_difference_hpa=delta,
            v12_causal_analysis_times_ns=wanted.tolist(), v12_distinct_input_members=50,
            v12_member_input_sha256=input_hashes, v12_member_seeds=list(range(ens.NOISE_SEED_BASE, ens.NOISE_SEED_BASE+50)))
        result['errors']['auxiliary_wind_vs_USA_1min_kt'] = common_errors(
            [p['vmax_kt'] for p in result['v11']], wind_mean, result['truth']['auxiliary_wind_vs_USA_1min_kt'], low=0, high=249)
        saved = output/'cases'/f'{index:05d}_wind_members.npz'
        np.savez_compressed(saved, wind_kt=wind, pressure_hpa=pressure)
        result['wind_member_file_sha256'] = sha(saved)
        write(dest, result); report = summarize(output, plan, 'recovering_original_1_2_auxiliary_wind'); done += 1
        print(json.dumps({'event': 'wind_case_complete', 'case': index,
            'wind_replays_completed': report['wind_replays_completed'], 'pressure_replay_difference_hpa': delta,
            'seconds': round(time.monotonic()-started, 2)}), flush=True)
        if limit and done >= limit:
            break
    report = summarize(output, plan, 'finished' if not limit else 'bounded_run')
    print(json.dumps({'event': 'run_finished', 'status': report['status']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=10, choices=range(1, 11))
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    run(args.project.resolve(), args.output.resolve(), args.chunk, args.limit)
