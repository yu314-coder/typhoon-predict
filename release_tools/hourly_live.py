"""Hourly cloud inference independent of changes to the JMA analysis.

The UTC run slot is NOT a new observed initialization. Recompute all 50 members
from causal inputs, preserve original +0 and valid times, and retain every run.
Owns only the live-hourly-data branch, never the historical forecast-data branch.
"""
import argparse
import base64
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

CHECKPOINT = 'f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0'
WORKFLOW = 'hourly-live.yml'
BRANCH = 'live-hourly-data'


def utc(t):
    return t.astimezone(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def parse(t):
    return datetime.fromisoformat(t.replace('Z', '+00:00'))


def hourly_row(row, now):
    slot = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if parse(row['issue_time_utc']) > slot:
        raise ValueError('Observed initialization is later than this run slot')
    return {**row, 'id': row['id'] + '-r' + slot.strftime('%Y%m%dT%H%M'), 'run_slot_utc': utc(slot)}


def verify_record(f, field, row):
    """Read back the completed record before advertising it or publishing it."""
    ident = row['id'].replace('auto-live-', 'auto-live50-', 1)
    assert f['id'] == field['forecast_id'] == ident
    assert f['members'] == field['members'] == 50
    assert f['checkpoint_sha256'] == field['checkpoint_sha256'] == CHECKPOINT
    assert f['issue_time_utc'] == f['initialization_time_utc'] == row['issue_time_utc']
    assert f['run_slot_utc'] == row['run_slot_utc']
    assert parse(f['issue_time_utc']) <= parse(f['run_slot_utc']) <= parse(f['generated_at_utc'])
    assert f['route'][0]['lat'] == row['lat'] and f['route'][0]['lon'] == row['lon']
    assert f['route'][0]['pressure_hpa'] == row.get('pressure_hpa')
    origin = parse(f['issue_time_utc'])
    history = [parse(t) for t in f['input_history_times_utc']]
    assert len(history) == 9 and timedelta(0) <= origin-history[-1] <= timedelta(hours=12)
    assert all(b-a == timedelta(hours=6) for a, b in zip(history, history[1:]))
    policy = f['ensemble_policy']
    assert policy['member_count'] == 50 and policy['seeds'] == list(range(2043, 2093))
    assert policy['weights'] == [.02]*50
    for key in ('input_sha256', 'route_sha256', 'basin_field_sha256'):
        assert len(policy[key]) == len(set(policy[key])) == 50
    assert len(f['route']) == 21 and len(field['pressure_hpa']) == len(field['valid_times_utc']) == 20
    assert field['units'] == 'hPa'
    assert field['latitude'] == [60-i*2.5 for i in range(25)]
    assert field['longitude'] == [100+i*2.5 for i in range(33)]
    for i, p in enumerate(f['route']):
        assert p['lead_hours'] == i*6 and parse(p['valid_time_utc']) == origin+timedelta(hours=i*6)
        assert all(math.isfinite(p[k]) for k in ('lat', 'lon'))
        if i:
            assert math.isfinite(p['pressure_hpa']) and 800 < p['pressure_hpa'] < 1100
            assert field['valid_times_utc'][i-1] == p['valid_time_utc']
    for grid in [field['issue_pressure_hpa'], *field['pressure_hpa']]:
        assert len(grid) == 25 and all(len(r) == 33 for r in grid)
        assert all(math.isfinite(v) and 800 <= v <= 1100 for r in grid for v in r)
    core = field['core_reconstruction']
    assert core['forecast_id'] == ident and core['input_tensor_sha256'] == f['input_tensor_sha256']
    assert core['members'] == 50 and core['checkpoint_sha256'] == CHECKPOINT
    assert core['scalar_pressure_inserted'] is False and core['route_or_truth_alignment'] is False
    assert len(set(core['common_grid_policy']['member_state_sha256'])) == 50
    assert len(core['frames']) == 20
    for i, frame in enumerate([core['issue'], *core['frames']]):
        assert frame['lead_hours'] == 6*i and frame['valid_time_utc'] == f['route'][i]['valid_time_utc']
        if not frame['available']:
            continue  # Explicit geographic noncoverage remains missing.
        codec = frame['pressure_encoding']
        assert codec['format'] == 'delta-int16-le-base64'
        assert codec['shape'] == [len(frame['latitude']), len(frame['longitude'])]
        values = base64.b64decode(frame['pressure_delta_base64'], validate=True)
        assert len(values) == 2*math.prod(codec['shape'])
        total = 0
        for delta, in struct.iter_unpack('<h', values):
            total += delta
            assert 800 <= codec['offset_hpa']+total*codec['scale_hpa'] <= 1100


def run_once(output, cache):
    # Imported only in the cloud inference subprocess; gate/tests stay lightweight.
    import numpy as np
    import torch
    import automatic_forecasts as a
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    slot = utc(now.replace(minute=0, second=0, microsecond=0))
    old_path = output/'live50/status.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    if old.get('last_completed_slot_utc') == slot:
        print('This hourly slot is already complete; no duplicate inference', flush=True)
        return
    status = dict(model='Trackformer 1.2', members=50, checkpoint_sha256=CHECKPOINT,
                  run_slot_utc=slot, policy='Fresh inference hourly; unchanged JMA analysis does not suppress a run.',
                  last_completed_slot_utc=old.get('last_completed_slot_utc'),
                  last_successful_inference_utc=old.get('last_successful_inference_utc'),
                  runner='GitHub-hosted CPU, independent of the Mac and historical backfill',
                  run_url=os.environ.get('RUN_URL'), completed=[], errors={})
    try:
        rows = [hourly_row(r, now) for r in a.live_rows()]
        torch.set_num_threads(2)
        meta = json.loads((a.MODEL/'manifest.json').read_text())
        assert meta['source_checkpoint_sha256'] == CHECKPOINT
        for file, digest in meta['source_module_sha256'].items():
            assert a.digest((a.MODEL/file).read_bytes()) == digest
        cfg = json.loads((a.ROOT/'release_tools/history_inputs.json').read_text())
        plan = json.loads(a.asset(cache, 'input-manifest.json', cfg['base_url']+'/manifest.json', cfg['manifest_sha256']).read_text())
        geography = np.load(a.asset(cache, 'geography.npz', cfg['base_url']+'/geography.npz', plan['geography']['sha256']), allow_pickle=False)
        weights = a.asset(cache, 'weights.pt', a.HF+'/models/trackformer_1_2_field/weights.pt', meta['inference_weights_sha256'])
        contract = meta['data_contract']
        model = a.CoreForecaster(contract).eval()
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        for row in rows:
            try:
                issue = parse(row['issue_time_utc'])
                end = min(issue, now-timedelta(hours=3))
                end = end.replace(hour=end.hour//6*6, minute=0, second=0, microsecond=0)
                weather = None
                for lag in (0, 6):
                    last = end-timedelta(hours=lag)
                    if issue-last > timedelta(hours=12):
                        continue
                    dates = [last-timedelta(hours=6*(8-i)) for i in range(9)]
                    try:
                        weather = np.stack([a.gfs(d, cache, contract) for d in dates])
                        break
                    except Exception:
                        continue
                if weather is None:
                    raise ValueError('Nine causal GFS analyses unavailable; no substitute or relabelled forecast')
                times = [a.ns(utc(d)) for d in dates]
                provenance = dict(provider='NOAA GFS f000 analysis', experimental_transfer=True,
                    analyses=[dict(valid_time_utc=utc(d), subset_sha256=a.digest((cache/f"gfs-{d.strftime('%Y%m%d%H')}.grib2").read_bytes()),
                        source='NOAA GFS f000; exact GRIB timestamp, channels and units validated') for d in dates])
                a.infer_live50(model, contract, geography, weather, times, row, output,
                    'Fresh hourly computation using nine exact causal NOAA GFS analyses and the latest available JMA analysis. The run time is separate from the observed initialization.', provenance)
                ident = row['id'].replace('auto-live-', 'auto-live50-', 1)
                fp = output/'live50/forecasts'/f'{ident}.json'
                gp = output/'live50/fields'/f'{ident}.json.gz'
                f, field = json.loads(fp.read_text()), json.loads(gzip.decompress(gp.read_bytes()))
                verify_record(f, field, row)
                a.write(output/'live50/verification'/f'{ident}.json', dict(forecast_id=ident, verified=True,
                    checkpoint_sha256=CHECKPOINT, run_slot_utc=slot, initialization_time_utc=f['issue_time_utc'],
                    forecast_sha256=a.digest(fp.read_bytes()), field_gzip_sha256=a.digest(gp.read_bytes()),
                    checked_at_utc=utc(datetime.now(timezone.utc))))
                status['completed'].append(dict(id=ident, storm_id=row['storm_id'], initialization_time_utc=row['issue_time_utc']))
                status['last_successful_inference_utc'] = f['generated_at_utc']
            except Exception as exc:
                status['errors'][row['storm_id']] = str(exc)[:700]
        # Advertise verified outputs only; failed records are retained for diagnosis.
        storms = {}
        for p in sorted((output/'live50/verification').glob('*.json')):
            receipt = json.loads(p.read_text())
            if not receipt['verified']:
                continue
            f = json.loads((output/'live50/forecasts'/f"{receipt['forecast_id']}.json").read_text())
            s = storms.setdefault(f['storm_id'], dict(id=f['storm_id'], name=f['name'], season=f['season'], issues=[]))
            s['issues'].append({k:f[k] for k in ('id','issue_time_utc','members','kind','run_slot_utc','generated_at_utc')})
        for storm in storms.values():
            storm['issues'].sort(key=lambda i:(i['run_slot_utc'],i['issue_time_utc']), reverse=True)
        a.write(output/'live50/catalog.json', dict(model='Trackformer 1.2', members=50,
            checkpoint_sha256=CHECKPOINT, storms=list(storms.values()), updated_at_utc=utc(datetime.now(timezone.utc))))
        status['state'] = 'complete' if not status['errors'] else 'partial_failure'
        if not status['errors']:
            status['last_completed_slot_utc'] = slot
        if not rows:
            status['state'] = 'no_active_supported_storm'
    except Exception as exc:
        status['state'] = 'source_or_runner_failure'
        status['errors']['run'] = str(exc)[:700]
    status['live_checked_at_utc'] = utc(datetime.now(timezone.utc))
    a.write(old_path, status)
    print(json.dumps(status), flush=True)


def git_publish(output):
    checkout = output.parent
    def git(*args):
        return subprocess.run(['git', '-C', str(checkout), *args], check=True, capture_output=True, text=True)
    if git('branch', '--show-current').stdout.strip() != BRANCH:
        raise RuntimeError('Refusing to publish outside the hourly output branch')
    git('add', 'data/live50')
    if git('diff', '--cached', '--name-only').stdout.strip():
        git('commit', '-m', 'Publish verified hourly 1.2 inference '+utc(datetime.now(timezone.utc)))
        git('push', 'origin', 'HEAD:'+BRANCH)


def run_session(output, cache, minutes):
    """In-process hourly clock removes reliance on every GitHub cron firing."""
    deadline = time.monotonic()+minutes*60
    while time.monotonic() < deadline:
        began = datetime.now(timezone.utc)
        subprocess.run([sys.executable, '-B', __file__, '--output', str(output), '--cache', str(cache)],
                       check=True, timeout=min(2700, max(60, deadline-time.monotonic())))
        git_publish(output)
        wake = began.replace(minute=0, second=0, microsecond=0)+timedelta(hours=1)
        print(json.dumps({'next_run_utc':utc(wake),'JMA_change_required':False}), flush=True)
        while datetime.now(timezone.utc) < wake and time.monotonic() < deadline:
            time.sleep(min(30, max(0, (wake-datetime.now(timezone.utc)).total_seconds())))


def handoff_ready(jobs):
    """A finished inference session may hand off while its cache upload ends.

    The job-level concurrency lock still prevents simultaneous inference. Do
    not ignore a running parent until its successful compute step has ended
    and its explicit handoff step has started.
    """
    for job in jobs:
        if job.get('name') != 'hourly':
            continue
        steps = {s.get('name'): s for s in job.get('steps', [])}
        compute = steps.get('Recompute fifty members every hour and publish each verified run', {})
        handoff = steps.get('Hand off to the next bounded cloud session', {})
        if (compute.get('status') == 'completed' and compute.get('conclusion') == 'success'
                and (handoff.get('status') == 'in_progress'
                     or handoff.get('status') == 'completed' and handoff.get('conclusion') == 'success')):
            return True
    return False


def workflow_busy(runs, current_id, finished_sessions=()):
    ignored = {str(current_id), *(str(i) for i in finished_sessions)}
    return any(str(r['id']) not in ignored and r['status'] != 'completed'
               and r.get('path','').split('@')[0].split('/')[-1] == WORKFLOW for r in runs)


def cloud_gate(chain=False):
    repo = os.environ['GITHUB_REPOSITORY']
    runs = json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/actions/runs?per_page=100'], text=True))['workflow_runs']
    finished_sessions = []
    if not chain:
        for r in runs:
            if (str(r['id']) != os.environ['GITHUB_RUN_ID'] and r['status'] == 'in_progress'
                    and r.get('path','').split('@')[0].split('/')[-1] == WORKFLOW):
                jobs = json.loads(subprocess.check_output(
                    ['gh','api',f"repos/{repo}/actions/runs/{r['id']}/jobs?per_page=100"], text=True))['jobs']
                if handoff_ready(jobs):
                    finished_sessions.append(r['id'])
    free = not workflow_busy(runs, os.environ['GITHUB_RUN_ID'], finished_sessions)
    if chain:
        if free:
            subprocess.run(['gh','workflow','run',WORKFLOW,'--repo',repo,'--ref','main'], check=True)
        print('Queued next bounded hourly session:', free)
    else:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f'run={str(free).lower()}\n')
        print('No existing hourly session:', free)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--cache', type=Path)
    ap.add_argument('--session-minutes', type=int, default=0)
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--chain', action='store_true')
    a = ap.parse_args()
    if a.gate or a.chain:
        return cloud_gate(a.chain)
    if not a.output or not a.cache:
        ap.error('--output and --cache are required')
    if a.session_minutes:
        if not 1 <= a.session_minutes <= 300:
            ap.error('A bounded session must be 1..300 minutes')
        return run_session(a.output.resolve(), a.cache.resolve(), a.session_minutes)
    run_once(a.output.resolve(), a.cache.resolve())


if __name__ == '__main__':
    main()
