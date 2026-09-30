"""Append explicitly requested six-hour storm issues to a pinned history plan.

Old issue definitions and weather bundles are immutable. Only missing named-storm
ticks are added, with a separate causal weather bundle when the local atlas covers
them. This changes display/backfill priority, not model weights or evaluation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

HOUR = 3600 * 10**9


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def storm_ticks(entry, storm, times):
    """Use all in-domain six-hour starts, without future-label/skill filtering."""
    points = sorted(storm['points'], key=lambda p: p['time'])
    previous_by_time = {p['time']: p for p in points}
    if len(previous_by_time) != len(points):
        raise ValueError('Duplicate observed timestamps')
    result = []
    for p in points:
        tick = int(np.datetime64(p['time'].replace('Z', ''), 'ns').astype('int64'))
        if tick % (6 * HOUR) or not (0 < p['lat'] < 60 and 100 < p['lon'] < 180):
            continue
        ai = int(np.searchsorted(times, tick))
        wanted = tick + np.arange(-8, 1) * 6 * HOUR
        local = ai >= 8 and ai < len(times) and np.array_equal(times[ai-8:ai+1], wanted)
        prior_time = str(np.datetime64(tick-6*HOUR, 'ns').astype('datetime64[s]')) + 'Z'
        prior = previous_by_time.get(prior_time)
        motion = [0., 0.] if prior is None else [
            (p['lon']-prior['lon'])*111.2*np.cos(np.deg2rad(p['lat'])),
            (p['lat']-prior['lat'])*111.2]
        result.append({
            'id': 'auto-tick-'+entry['id']+'-'+p['time'].replace('-', '').replace(':', '')[:13],
            'storm_id': entry['id'], 'name': entry['name'], 'season': entry['season'],
            'issue_time_utc': p['time'], 'lat': p['lat'], 'lon': p['lon'],
            'pressure_hpa': p.get('pressure_hpa'), 'wind_kt': p.get('wind_kt'),
            'motion': motion, 'atlas': ai if local else None,
            'weather_source': 'verified-local-atlas' if local else 'NOAA-NCEP-Reanalysis-1-remote',
            # These future points are display/scoring labels, never inference inputs.
            'observed': [v for v in points if tick <= int(np.datetime64(
                v['time'].replace('Z', ''), 'ns').astype('int64')) <= tick+120*HOUR]})
    if not result:
        raise ValueError('No eligible Western Pacific six-hour issues: '+entry['id'])
    return result


def merge_plan(previous, requested, source_sha256):
    """Preserve each old row exactly, append new IDs, and prioritize by request."""
    # The pinned queue is large: share immutable old rows rather than duplicate
    # their display labels in memory. Only metadata and new rows are writable.
    plan = dict(previous)
    plan['bundles'] = dict(previous['bundles'])
    plan['coverage'] = [dict(r) for r in previous['coverage']]
    rows = {r['id']: r for r in plan['queue']}
    if len(rows) != len(plan['queue']):
        raise ValueError('Duplicate original issue IDs')
    details = []
    for entry, candidates in requested:
        added = []
        for row in candidates:
            if row['id'] not in rows:
                rows[row['id']] = copy.deepcopy(row)
                added.append(row['id'])
        details.append({'storm_id': entry['id'], 'name': entry['name'],
                        'season': entry['season'], 'six_hour_issues': len(candidates),
                        'new_issue_ids': added})
    priority = {entry['id']: i for i, (entry, _) in enumerate(requested)}
    original_order = {r['id']: i for i, r in enumerate(previous['queue'])}
    plan['queue'] = sorted(rows.values(), key=lambda r: (
        priority.get(r['storm_id'], len(priority)),
        r['issue_time_utc'] if r['storm_id'] in priority else '',
        original_order.get(r['id'], len(original_order))))
    old = {r['id']: r for r in previous['queue']}
    if any(rows[k] != v for k, v in old.items()):
        raise ValueError('Previously planned issue was modified')
    for coverage in plan['coverage']:
        match = next((d for d in details if d['storm_id'] == coverage['id']), None)
        if match:
            coverage.update(status='queued_requested_six_hour_ticks',
                            issue_count=sum(r['storm_id'] == coverage['id'] for r in rows.values()))
    plan['requested_storms'] = details
    plan['parent_manifest_sha256'] = source_sha256
    plan['selection'] += (' Explicitly requested Wayne (1986), Herb (1996), and Nari (2001) '
                          'are prioritized, including Wayne six-hour starts before 1996. '
                          'Other pre-1996 all-tick coverage is not expanded.')
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--site', type=Path, required=True)
    ap.add_argument('--previous', type=Path, required=True)
    ap.add_argument('--previous-sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--storm', action='append', required=True)
    args = ap.parse_args()
    if sha(args.previous) != args.previous_sha256:
        raise ValueError('Wrong pinned parent manifest')
    previous = json.loads(args.previous.read_text())
    catalog = json.loads((args.site/'public/data/history/catalog.json').read_text())
    entries = {r[catalog['fields'].index('id')]: dict(zip(catalog['fields'], r))
               for r in catalog['rows']}
    with np.load(args.project/'track_build/basin_all_int8.npz', allow_pickle=False) as atlas:
        times = atlas['time'].astype('int64')
        scale, offset = atlas['scale'].copy(), atlas['offset'].copy()
    requested = []
    existing = {r['id'] for r in previous['queue']}
    bundles = {}
    args.output.mkdir(parents=True, exist_ok=True)
    slp = q = None
    for sid in args.storm:
        entry = entries[sid]
        if entry['season'] < 1970 or 'WP' not in entry['basins']:
            raise ValueError('Unsupported requested storm')
        year = json.loads((args.site/f"public/data/history/{entry['season']}.json").read_text())
        ticks = storm_ticks(entry, year[sid], times)
        local = [r for r in ticks if r['id'] not in existing and r['atlas'] is not None]
        if local:
            if slp is None:
                for name, filename in [('basin', 'track_build/basin_all_int8.npz'),
                                       ('slp', 'track_build/basin_slp_atlas_float16.npy')]:
                    if sha(args.project/filename) != previous['source_sha256'][name]:
                        raise ValueError('Local source changed: '+name)
                slp = np.load(args.project/'track_build/basin_slp_atlas_float16.npy', mmap_mode='r')
                q = np.load(args.project/'data/v164_reuse/basin_q_verified.npy', mmap_mode='r')
            indices = np.unique(np.concatenate([np.arange(r['atlas']-8, r['atlas']+1) for r in local]))
            key = 'requested-'+sid
            path = args.output/f'weather-{key}.npz'
            np.savez_compressed(path, time=times[indices], q=np.asarray(q[indices]),
                                slp=np.asarray(slp[indices]), scale=scale, offset=offset)
            bundles[key] = {'file': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size}
            for row in local:
                row['bundle_key'] = key
        requested.append((entry, ticks))
    plan = merge_plan(previous, requested, args.previous_sha256)
    plan['bundles'].update(bundles)
    path = args.output/'manifest.json'
    path.write_text(json.dumps(plan, separators=(',', ':'), allow_nan=False))
    print(json.dumps({'manifest': str(path), 'manifest_sha256': sha(path),
                      'total_issues': len(plan['queue']), 'requested': plan['requested_storms'],
                      'new_bundles': bundles, 'all_original_rows_preserved': True}))


if __name__ == '__main__':
    main()
