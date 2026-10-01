"""Frozen WeatherNext Cyclones Mini daily benchmark. No fitting or old scores.

The supervisor is resumable and CPU-only on the Mac. Each isolated worker runs
the official released predictor and tracker, using only two causal ERA5 states
and the frozen issue-time storm anchor. Truth is loaded afterwards for scoring.
All output, source caches and compilation caches remain on /Volumes/D.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

COHORT = '965184f1e5ab3fcb52e304f39b38583a14f753870aa2295f2d01942a441dcfd6'
WEIGHTS = 'a1bb151457077248d70a458b1a7b19deacd2926fbcb46eee4ff2691444e3596e'
CHECKPOINT = 'f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0'
STEP_NS = 6 * 3600 * 10**9
HERE = Path(__file__).resolve()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def assets(project):
    return {
        'weights': project/'v226/weathernext_global/weights/WeatherNextCyclones_Mini_<2024.npz',
        'shape_template': project/'v226/weathernext_global/data/hres_2024-10-07_00z_1deg_steps20.nc',
        'initial_state_reader': project/'v226_build_arco_weathernext_state.py',
        'inference_adapter': project/'v226_weathernext_cyclones_baseline.py',
        'scoring_definition': project/'benchmark_daily_storms_1_2.py',
        'runner': HERE,
    }


def prepare(project, out):
    if (out/'protocol.json').exists():
        raise ValueError('Protocol already frozen; run or status, not prepare again')
    original = project/'benchmark_daily_storms_1_2/cohort.json'
    plan = read(original)
    if sha(original) != COHORT or len(plan['cases']) != 1473 or plan['storm_count'] != 270:
        raise ValueError('Wrong frozen daily cohort')
    if plan['models']['1.2']['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Wrong Trackformer reference')
    sources = {key: {'path': str(path), 'sha256': sha(path)} for key, path in assets(project).items()}
    if sources['weights']['sha256'] != WEIGHTS:
        raise ValueError('Wrong official WeatherNext Mini checkpoint')
    rows = []
    for case in plan['cases']:
        path = project/'benchmark_daily_storms_1_2/cases'/f"{case['case_index']:05d}.npz"
        metadata = read(path.with_suffix('.json'))
        if any(metadata[key] != value for key, value in case.items()) or sha(path) != metadata['forecast_sha256']:
            raise ValueError('Original daily forecast differs')
        rows.append(dict(case, original_forecast_sha256=metadata['forecast_sha256']))
    out.mkdir(parents=True, exist_ok=True)
    vendor = subprocess.check_output(['git', '-C', str(project/'vendor/weathernext'), 'rev-parse', 'HEAD'], text=True).strip()
    write(out/'protocol.json', {
        'schema': 'weathernext-mini-daily-equal-storm-v1', 'frozen_at_utc': stamp(),
        'cohort_sha256': COHORT, 'target_daily_cases': 1473, 'target_storms': 270,
        'model': 'Google DeepMind WeatherNext Cyclones Mini <2024', 'members': 1,
        'seed': 0, 'trained_through': 2023, 'resolution_degrees': 1,
        'official_weights_url': 'https://storage.googleapis.com/dm_graphcast/weathernext2/params/WeatherNextCyclones_Mini_%3C2024.npz',
        'vendor_commit': vendor, 'sources': sources, 'cases': rows,
        'leads_hours': list(range(6, 121, 6)), 'backend': 'JAX CPU',
        'causal_policy': 'Exactly issue -6h and issue ERA5 analyses; frozen observed issue centre only. Calendar forcings only after issue. No future weather, truth or official forecast route in inference.',
        'tracker_policy': 'Official direct_tracker_6h_v1 with initialized-storm continuity and no cyclogenesis; dissipation and nearby pruning disabled, disclosed rather than official operational tracker policy.',
        'aggregation': 'Mean valid leads per day, days within storm, then equal storm weights. All three direction scores recomputed on common moving steps. Pressure uses the original matched native intensity cases, common leads and separate JMA/USA labels.',
        'failure_policy': 'Failures stay visible and unscored. No zero fills, source substitutions, calibration or case selection by performance. Full-cohort completion requires all 1473 forecasts.',
        'interpretation': 'Development comparison, different input pipelines and member policies. Historical 1980-1999 cases overlap WeatherNext fitting years; recent 2024+ cases are separately reported. Not a certified untouched test.',
    })
    print(json.dumps({'status': 'protocol_frozen', 'cases': len(rows), 'storms': 270, 'protocol_sha256': sha(out/'protocol.json')}), flush=True)


def validate_state(path, case):
    import xarray as xr
    with xr.open_dataset(path, engine='h5netcdf') as state:
        times = state.datetime.values.astype('datetime64[ns]').astype('int64')
        expected = int(case['issue_ns']) + np.array([-STEP_NS, 0])
        if times.shape != (2,) or not np.array_equal(times, expected):
            raise ValueError('Weather initial state is not the exact causal two-analysis history')
        for key in ('future_weather_fields_used', 'official_forecast_fields_used'):
            if str(state.attrs.get(key)).lower() != 'false':
                raise ValueError('Weather source does not declare causal analysis inputs')


def state_path(project, out, case):
    tag = case['issue_time_utc'][:16].replace('-', '').replace(':', '').replace('T', '_')
    filename = f'arco_era5_1deg_{tag}.nc'
    for directory in (project/'benchmark_weathernext_cyclones_mini_same270/arco_states',
                      project/'benchmark_tip_ten/deepmind_states', out/'states'):
        candidate = directory/filename
        if candidate.exists():
            validate_state(candidate, case)
            return candidate
    dest = out/'states'
    subprocess.run([sys.executable, str(project/'v226_build_arco_weathernext_state.py'),
        '--init', case['issue_time_utc'], '--output-dir', str(dest), '--skip-land-cache',
        '--download-workers', '2', '--download-retries', '2'], check=True, cwd=project)
    candidate = dest/filename
    validate_state(candidate, case)
    return candidate


def worker(project, out, index):
    """Inference has no access to any future labels or original forecast outputs."""
    import dataclasses
    import haiku as hk
    import jax
    import pandas as pd
    import xarray as xr
    import xarray_jax
    sys.path.insert(0, str(project))
    import v226_weathernext_cyclones_baseline as adapter
    from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
    from weathernext.weathernext2 import fgn
    from weathernext.cyclones import direct_tracker, direct_tracker_6h_v1_config
    protocol = read(out/'protocol.json')
    case = protocol['cases'][index]
    initial = state_path(project, out, case)
    config = fiddle_config_io.get_fiddle_config_by_name(adapter.CONFIG_NAME)
    if jax.default_backend() != 'cpu':
        raise ValueError('This frozen run requires its declared CPU backend')
    jax.config.update('jax_compilation_cache_dir', str(out/'jax-cache'))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
    attention = adapter.configure_attention(config, 'cpu')
    paths = assets(project)
    with paths['weights'].open('rb') as stream:
        ckpt = checkpoint.load(stream, fgn.CheckPoint)
    template = xr.load_dataset(paths['shape_template'], engine='h5netcdf').compute()
    template, issue = adapter.inject_initial_state(template, initial, config.task)
    if int(np.datetime64(issue, 'ns').astype('int64')) != case['issue_ns']:
        raise ValueError('Issue does not match frozen cohort')
    if set(config.task.forcing_variables) != adapter.ALLOWED_FORCINGS:
        raise ValueError('Unexpected future forcing')
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(template,
        target_lead_times=slice('6h', '120h'), **dataclasses.asdict(config.task))
    if not np.array_equal(inputs.time.values.astype('timedelta64[h]').astype(int), [-6, 0]):
        raise ValueError('Non-causal predictor input times')
    prediction_config = fgn.PredictorConfig(task=config.task,
        predictor_constructor=config.predictor_constructor,
        predictor_kwargs=config.predictor_kwargs, predictor_wrappers=config.predictor_wrappers[:-1])

    @hk.transform
    def forward(history, shape, forcing):
        return fgn.construct_predictor(prediction_config)(history, targets_template=shape, forcings=forcing)

    compiled = jax.jit(lambda rng, history, shape, forcing: forward.apply(ckpt.params, rng, history, shape, forcing))
    mapped = xarray_jax.pmap(compiled, dim='sample')
    rngs = np.stack([jax.random.fold_in(jax.random.PRNGKey(protocol['seed']), 0)])
    chunks = []
    started = time.monotonic()
    for i, chunk in enumerate(rollout.chunked_prediction_generator_multiple_runs(
            predictor_fn=mapped, rngs=rngs, inputs=inputs,
            targets_template=targets*np.nan, forcings=forcings,
            num_steps_per_chunk=1, num_samples=1, pmap_devices=jax.local_devices())):
        if i >= 20:
            break
        chunks.append(jax.device_get(chunk))
        print(json.dumps({'case': index, 'step': i+1, 'steps': 20, 'elapsed_seconds': round(time.monotonic()-started, 1)}), flush=True)
    predictions = xr.combine_by_coords(chunks).isel(time=slice(0, 20), batch=0, sample=0)
    if predictions.sizes.get('time') != 20:
        raise ValueError('Incomplete model rollout')
    config_tracker = direct_tracker_6h_v1_config.get_config()
    kwargs = dict(config_tracker.tracker_kwargs,
        dissipation_min_mean_probability_threshold=None, prune_nearby_cyclones=False)
    tracker = config_tracker.tracker_constructor(**kwargs)
    scalar = predictions['cyclone_exists_gaussian_unit_mode']
    for key in direct_tracker.IBTRACS_VARIABLE_NAME_TO_GRIDDED_VARIABLE_NAME.values():
        if key not in predictions:
            predictions[key] = xr.full_like(scalar, np.nan)
    grid = tracker.preprocess_gridded_ds(predictions).expand_dims(forecast_datetime=[np.datetime64(issue)])
    grid = grid.assign_coords(lead_time_secs=grid.time.astype('timedelta64[s]').astype(int),
        date_time=np.datetime64(issue)+grid.time).as_numpy()
    anchor = pd.DataFrame([dict(track_id=case['storm_id'], lat=case['base_lat'],
        lon=case['base_lon'], valid_time=issue, init_time=issue)])
    tracks = tracker(gridded_ds=grid, initial_storms_df=anchor, do_cyclogenesis=False)
    tracks = tracks[tracks.track_id == case['storm_id']].copy()
    wanted = pd.to_datetime(case['issue_ns'] + np.arange(1, 21)*STEP_NS)
    rows = tracks.set_index('valid_time').reindex(wanted)
    route = rows[['lat', 'lon']].to_numpy(dtype='float64')
    route[:, 1] %= 360
    if route.shape != (20, 2) or not np.isfinite(route).all():
        raise ValueError('Tracker did not produce twenty finite exact-time points')
    pressure = rows['minimum_sea_level_pressure_hpa'].to_numpy(dtype='float64')
    pressure_valid = np.isfinite(pressure) & (pressure >= 800) & (pressure <= 1100)
    native = predictions.mean_sea_level_pressure.sel(lat=slice(0, 60), lon=slice(100, 180)) / 100
    values = native.transpose('time', 'lat', 'lon').values.astype('float32')
    if values.shape != (20, 61, 81) or not np.isfinite(values).all() or values.min() < 800 or values.max() > 1100:
        raise ValueError('Nonphysical or incomplete native Western Pacific pressure fields')
    dest = out/'cases'/f'{index:05d}.npz'
    temporary = dest.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, route_lat_lon=route,
            central_pressure_hpa=np.where(pressure_valid, pressure, np.nan), pressure_valid=pressure_valid,
            wp_pressure_hpa=values, latitude=native.lat.values, longitude=native.lon.values,
            lead_hours=np.arange(6, 121, 6), valid_time_ns=case['issue_ns']+np.arange(1, 21)*STEP_NS)
    temporary.replace(dest)
    write(dest.with_suffix('.json'), dict(case, status='complete', members=1,
        model=protocol['model'], checkpoint_sha256=WEIGHTS, protocol_sha256=sha(out/'protocol.json'),
        forecast_sha256=sha(dest), input_state_sha256=sha(initial), input_state_path=str(initial),
        input_time_ns=[case['issue_ns']-STEP_NS, case['issue_ns']], backend='cpu',
        attention_override=attention, runtime_seconds=time.monotonic()-started, completed_at_utc=stamp(),
        future_weather_used=False, future_storm_labels_used=False))


def route_metrics(routes, truth):
    steps = [np.diff(np.vstack([np.zeros((1, 2)), p]), axis=0) for p in [*routes.values(), truth]]
    valid = np.logical_and.reduce([np.linalg.norm(p, axis=-1) > 1 for p in steps])
    result = {}
    for (key, path), step in zip(routes.items(), steps[:-1]):
        error = np.linalg.norm(path-truth, axis=-1)
        angle = np.arctan2(step[:, 1], step[:, 0])-np.arctan2(steps[-1][:, 1], steps[-1][:, 0])
        angle = np.degrees(np.abs(np.arctan2(np.sin(angle), np.cos(angle))))
        a, b = (path-path.mean(0)).ravel(), (truth-truth.mean(0)).ravel()
        similarity = (1+np.dot(a, b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-8))/2
        result[key] = dict(mean_track_error_km=float(error.mean()), track_error_120h_km=float(error[-1]),
            track_error_by_lead_km=error.tolist(), direction_error_deg=float(angle[valid].mean()) if valid.any() else None,
            direction_valid_steps=int(valid.sum()), shape_similarity=float(np.clip(similarity, 0, 1)))
    return result


def local(route, lat, lon):
    return np.stack([((route[:, 1]-lon+180)%360-180)*111.2*max(np.cos(np.deg2rad(lat)), .2),
                     (route[:, 0]-lat)*111.2], -1)


def curve_similarity(forecast, truth):
    if len(truth) < 6:
        return None
    a, b = forecast-forecast.mean(), truth-truth.mean()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip((1+np.dot(a, b)/norm)/2, 0, 1)) if norm > 1e-8 else None


def score(project, out, case):
    index = case['case_index']
    path = out/'cases'/f'{index:05d}.npz'
    record = read(path.with_suffix('.json'))
    if record['forecast_sha256'] != sha(path) or record['protocol_sha256'] != sha(out/'protocol.json'):
        raise ValueError('Saved candidate identity changed')
    original = project/'benchmark_daily_storms_1_2/cases'/path.name
    if sha(original) != case['original_forecast_sha256']:
        raise ValueError('Frozen baseline forecast changed')
    with np.load(path, allow_pickle=False) as z, np.load(original, allow_pickle=False) as reference:
        candidate = local(z['route_lat_lon'], case['base_lat'], case['base_lon'])
        metrics = route_metrics({'1.1': reference['v11_local'], '1.2': reference['v12_local'], 'deepmind': candidate}, reference['truth_local'])
        pressure = z['central_pressure_hpa'].copy()
    source = read(project/'output/intensity-v12-v11-20261001/cases'/f'{index:05d}.json')
    pressure_scores = {}
    if source['v11'] is not None:
        curves = {'1.1': np.array([p['central_pressure_hpa'] for p in source['v11']], dtype=float),
                  '1.2': np.array(source['v12_pressure_hpa'], dtype=float), 'deepmind': pressure}
        for agency in ('JMA', 'USA'):
            truth = np.array(source['truth']['pressure_'+agency+'_hpa'], dtype=float)
            mask = pressure_mask(truth, curves)
            pressure_scores[agency] = {'valid_leads': int(mask.sum()), 'models': {
                key: {'mae_hpa': float(np.abs(p[mask]-truth[mask]).mean()) if mask.any() else None,
                      'curve_similarity': curve_similarity(p[mask], truth[mask]),
                      'error_by_lead_hpa': [float(abs(p[i]-truth[i])) if mask[i] else None for i in range(20)]}
                for key, p in curves.items()}}
    record.update(route_metrics=metrics, pressure_metrics=pressure_scores)
    write(path.with_suffix('.json'), record)
    return record


def pressure_mask(truth, curves):
    arrays = [np.asarray(p, dtype=float) for p in [truth, *curves.values()]]
    if any(p.shape != (20,) for p in arrays):
        raise ValueError('Pressure comparison requires twenty matching exact leads')
    return np.logical_and.reduce([np.isfinite(p) & (p >= 800) & (p <= 1100) for p in arrays])


def aggregate(groups):
    scores = {}
    for model in ('1.1', '1.2', 'deepmind'):
        scores[model] = {}
        for metric in ('mean_track_error_km', 'track_error_120h_km', 'direction_error_deg', 'shape_similarity'):
            storms = []
            for days in groups.values():
                values = [d['route_metrics'][model][metric] for d in days if d['route_metrics'][model][metric] is not None]
                if values:
                    storms.append(float(np.mean(values)))
            scores[model][metric] = {'value': float(np.mean(storms)) if storms else None, 'storms': len(storms)}
        scores[model]['track_error_by_lead_km'] = np.mean([
            np.mean([d['route_metrics'][model]['track_error_by_lead_km'] for d in days], axis=0)
            for days in groups.values()], axis=0).tolist() if groups else [None]*20
    pressure = {}
    for agency in ('JMA', 'USA'):
        eligible = {sid: [d['pressure_metrics'][agency] for d in days
                         if agency in d['pressure_metrics'] and d['pressure_metrics'][agency]['valid_leads']]
                    for sid, days in groups.items()}
        eligible = {sid: days for sid, days in eligible.items() if days}
        pressure[agency] = {'daily_cases': sum(map(len, eligible.values())), 'storms': len(eligible),
                           'common_lead_points': sum(d['valid_leads'] for days in eligible.values() for d in days), 'models': {}}
        for model in ('1.1', '1.2', 'deepmind'):
            values = {}
            for metric in ('mae_hpa', 'curve_similarity'):
                storm_values = []
                case_count = 0
                for days in eligible.values():
                    valid = [d['models'][model][metric] for d in days if d['models'][model][metric] is not None]
                    if valid:
                        storm_values.append(float(np.mean(valid)))
                        case_count += len(valid)
                values[metric] = {'value': float(np.mean(storm_values)) if storm_values else None,
                                  'storms': len(storm_values), 'daily_cases': case_count}
            pressure[agency]['models'][model] = values
    return {'route': scores, 'pressure': pressure}


def status(project, out, active=None):
    protocol = read(out/'protocol.json')
    groups = defaultdict(list)
    expected = defaultdict(int)
    failures = []
    for case in protocol['cases']:
        expected[case['storm_id']] += 1
        path = out/'cases'/f"{case['case_index']:05d}.json"
        completed = False
        if path.exists():
            record = read(path)
            if 'route_metrics' in record:
                groups[case['storm_id']].append(record)
                completed = True
        failure = out/'failures'/f"{case['case_index']:05d}.json"
        if failure.exists() and not completed:
            failures.append(read(failure))
    full = {key: days for key, days in groups.items() if len(days) == expected[key]}
    done = sum(map(len, groups.values()))
    receipt = read(out/'verification.json') if (out/'verification.json').exists() else {}
    verified = (receipt.get('verified_daily_cases') == 1473 and receipt.get('protocol_sha256') == sha(out/'protocol.json'))
    report = {'state': 'complete_verified' if verified else 'forecasts_finished_audit_pending' if done == 1473 else 'running_partial' if active is not False else 'finished_with_unavailable',
        'updated_at_utc': stamp(), 'cohort_sha256': COHORT, 'protocol_sha256': sha(out/'protocol.json'),
        'completed_daily_cases': done, 'target_daily_cases': 1473, 'completed_storms': len(full),
        'target_storms': 270, 'failed_daily_cases': len(failures), 'failures': failures,
        'checkpoint_sha256': WEIGHTS, 'members': 1, 'backend': 'JAX CPU',
        'aggregate_equal_complete_storm': aggregate(full), 'final_results': verified,
        'periods': {period: aggregate({sid: days for sid, days in full.items() if days[0]['period'] == period})
                   for period in sorted({c['period'] for c in protocol['cases']})},
        'original_forecasts_modified': False, 'original_model_weights_modified': False,
        'missing_results_scored_as_zero': 0, 'pressure_note': 'Per-case three-way common-support pressure MAE and curve similarity in cases/*.json; separate agency masks. Final publication requires coverage audit, including pressure-specific coverage.'}
    write(out/'progress.json', report)
    return report


def verify(project, out):
    protocol = read(out/'protocol.json')
    verified = []
    for case in protocol['cases']:
        path = out/'cases'/f"{case['case_index']:05d}.npz"
        record = read(path.with_suffix('.json'))
        if any(record.get(k) != v for k, v in case.items()):
            raise ValueError('Forecast case does not match the frozen definition')
        if record.get('checkpoint_sha256') != WEIGHTS or record.get('members') != 1:
            raise ValueError('Forecast model/member identity differs')
        if record.get('input_time_ns') != [case['issue_ns']-STEP_NS, case['issue_ns']]:
            raise ValueError('Weather history is noncausal')
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['lead_hours'], np.arange(6, 121, 6)) or not np.array_equal(
                    z['valid_time_ns'], case['issue_ns']+np.arange(1, 21)*STEP_NS):
                raise ValueError('Forecast leads are not exact +6 to +120h')
            if z['route_lat_lon'].shape != (20, 2) or not np.isfinite(z['route_lat_lon']).all():
                raise ValueError('Incomplete route')
            p = z['wp_pressure_hpa']
            if p.shape != (20, 61, 81) or not np.isfinite(p).all() or p.min() < 800 or p.max() > 1100:
                raise ValueError('Incomplete physical native pressure fields')
        score(project, out, case)  # Recheck hashes and recompute common-support metrics.
        verified.append(case['case_index'])
    if len(set(verified)) != 1473:
        raise ValueError('Audit did not cover every planned case')
    write(out/'verification.json', {'status': 'complete_verified', 'verified_at_utc': stamp(),
        'protocol_sha256': sha(out/'protocol.json'), 'cohort_sha256': COHORT,
        'checkpoint_sha256': WEIGHTS, 'verified_daily_cases': 1473, 'verified_storms': 270,
        'case_indices': verified, 'checks': ['exact frozen issue identity', 'forecast and baseline SHA256',
        'one official Mini member', 'causal two-analysis history', 'twenty exact leads',
        'finite unshifted routes', 'twenty native physical WP pressure grids',
        'recomputed matched three-way route and pressure masks']})


def run(project, out, limit=0):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(out/'protocol.json')
    for row in protocol['sources'].values():
        if sha(row['path']) != row['sha256']:
            raise ValueError('Frozen source changed: '+row['path'])
    for name in ('cases', 'failures', 'logs', 'states', 'jax-cache'):
        (out/name).mkdir(exist_ok=True)
    write(out/'runtime.json', {'pid': os.getpid(), 'started_at_utc': stamp(), 'backend': 'JAX CPU',
        'python': sys.executable, 'protocol_sha256': sha(out/'protocol.json')})
    count = 0
    status(project, out, True)
    for case in protocol['cases']:
        index = case['case_index']
        saved = out/'cases'/f'{index:05d}.json'
        if saved.exists() and 'route_metrics' in read(saved):
            continue
        if limit and count >= limit:
            break
        count += 1
        try:
            if not saved.exists():
                with (out/'logs'/f'{index:05d}.log').open('a') as log:
                    subprocess.run([sys.executable, str(HERE), 'worker', '--project', str(project),
                        '--output', str(out), '--case-index', str(index)], cwd=project,
                        env=dict(os.environ, JAX_PLATFORMS='cpu', PYTHONUNBUFFERED='1',
                                 XLA_PYTHON_CLIENT_PREALLOCATE='false'), stdout=log, stderr=subprocess.STDOUT, check=True)
            score(project, out, case)
            print(json.dumps({'case_complete': index, 'storm_id': case['storm_id'], 'issue': case['issue_time_utc']}), flush=True)
        except Exception as error:
            write(out/'failures'/f'{index:05d}.json', dict(case_index=index, storm_id=case['storm_id'],
                issue_time_utc=case['issue_time_utc'], error=str(error), at_utc=stamp(), result=None))
            print(json.dumps({'case_failed': index, 'error': str(error)}), flush=True)
        status(project, out, True)
    report = status(project, out, False)
    if report['completed_daily_cases'] == 1473:
        verify(project, out)
        report = status(project, out, False)
    print(json.dumps({k: v for k, v in report.items() if k not in ('failures', 'aggregate_equal_complete_storm')}), flush=True)


def launch(project, out):
    # Lock is acquired by run; launch never stops or replaces an existing job.
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log = (out/'run.log').open('a')
    command = [sys.executable, str(HERE), 'run', '--project', str(project), '--output', str(out)]
    if sys.platform == 'darwin':
        command = ['/usr/bin/caffeinate', '-i', '-s', *command]
    child = subprocess.Popen(command, cwd=project, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        env=dict(os.environ, JAX_PLATFORMS='cpu', PYTHONUNBUFFERED='1', XLA_PYTHON_CLIENT_PREALLOCATE='false'))
    write(out/'launcher.json', {'pid': child.pid, 'launched_at_utc': stamp(), 'command': command})
    print(json.dumps({'launched_pid': child.pid, 'output': str(out), 'backend': 'JAX CPU'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'worker', 'status', 'launch'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-index', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    project, out = args.project.resolve(), args.output.resolve()
    if not out.is_relative_to(Path('/Volumes/D')) or not project.is_relative_to(Path('/Volumes/D')):
        raise ValueError('Project and all generated artifacts must remain on D')
    if args.mode == 'prepare': prepare(project, out)
    elif args.mode == 'worker': worker(project, out, args.case_index)
    elif args.mode == 'run': run(project, out, args.limit)
    elif args.mode == 'launch': launch(project, out)
    else: print(json.dumps(status(project, out), indent=2))


if __name__ == '__main__':
    main()
