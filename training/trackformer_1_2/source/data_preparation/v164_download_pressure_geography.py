"""Mac-only, resumable MSLP patches and geography. Not a full trainer dataset."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import v164_prepare_data as prep


def make_plan(track):
    if prep.trainer.digest(track) != prep.TRACK_SHA:
        raise ValueError('Unexpected track archive')
    rng = np.random.default_rng(164)
    with np.load(track, allow_pickle=False) as z:
        times, ids = z['base_time'], z['storm_id'].astype(str)
        lat, lon = z['base_lat'], z['base_lon'] % 360
    first = {}
    for sid, ns in zip(ids, times):
        first[sid] = min(first.get(sid, 9999), prep.year(ns))
    rows = []
    for split, lo, hi, count in [('train', 2000, 2021, 800), ('validation', 2022, 2023, 100), ('test', 2024, 2025, 100)]:
        groups = defaultdict(list)
        for i, (sid, ns) in enumerate(zip(ids, times)):
            ns = int(ns)
            if not (lo <= first[sid] <= hi and lo <= prep.year(ns-8*prep.STEP)
                    and prep.year(ns+20*prep.STEP) <= hi and ns % prep.STEP == 0
                    and np.isfinite(lat[i]) and np.isfinite(lon[i])
                    and 0 < lat[i] < 60 and 100 < lon[i] < 180):
                continue
            groups[sid].append(i)
        for group in groups.values():
            rng.shuffle(group)
        storms = sorted(groups)
        rng.shuffle(storms)
        chosen = []
        while len(chosen) < count:
            before = len(chosen)
            for sid in storms:
                if groups[sid]:
                    chosen.append(groups[sid].pop())
                    if len(chosen) == count:
                        break
            if len(chosen) == before:
                raise ValueError(f'Insufficient eligible {split} windows')
        for i in chosen:
            rows.append({'storm_id': str(ids[i]), 'issue_ns': int(times[i]), 'split': split,
                         'center_lat': float(np.round(lat[i]*4)/4), 'center_lon': float(np.round(lon[i]*4)/4),
                         'center_source': 'retrospective issue-time best-track observation; no future positions',
                         'track_archive_row': int(i)})
    plan = {'schema': 'v164-pressure-geography-v1', 'dataset_role': 'partial_inputs_not_training_ready',
            'source': prep.SOURCE, 'track_sha256': prep.TRACK_SHA, 'rows': rows,
            'lead_hours': list(range(-48, 121, 6)), 'shape': [len(rows), 29, 121, 121],
            'units': 'hPa', 'resolution_degrees': .25, 'fixed_issue_time_crops': True,
            'forecast_inputs': 'lead_hours <= 0 only', 'targets': 'lead_hours > 0 only',
            'split_policy': 'existing 2000-2021/2022-2023/2024-2025 policy, pending separately requested changes',
            'hindcast_not_operational_availability_claim': True}
    plan['sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def patch_indices(row):
    lat = row['center_lat'] + 15 - np.arange(121)*.25
    lon = row['center_lon'] - 15 + np.arange(121)*.25
    return np.rint((90-lat)*4).astype(int), np.rint(lon*4).astype(int) % 1440


def patch_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run(root, track, workers, max_frames=None):
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root/'download.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    db = None
    try:
        plan = make_plan(track)
        path = root/'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Existing download plan differs; refusing overwrite')
        prep.trainer.atomic_json(path, plan)
        shape = tuple(plan['shape'])
        output = root/'pressure_hpa.npy'
        a = np.lib.format.open_memmap(output, mode='r+' if output.exists() else 'w+', dtype='float32', shape=shape)
        if a.shape != shape or a.dtype != np.float32:
            raise ValueError('Existing array shape/dtype differs')
        indices = [patch_indices(r) for r in plan['rows']]
        requests = defaultdict(list)
        for i, row in enumerate(plan['rows']):
            for j in range(29):
                requests[row['issue_ns'] + (j-8)*prep.STEP].append((i, j))
        db = sqlite3.connect(root/'progress.sqlite')
        db.execute('CREATE TABLE IF NOT EXISTS patches (i INTEGER, j INTEGER, sha TEXT, PRIMARY KEY(i,j))')
        verified = set()
        for i, j, sha in db.execute('SELECT i,j,sha FROM patches'):
            if patch_hash(a[i,j]) == sha:
                verified.add((i,j))
        pending = [(ns, slots) for ns, slots in sorted(requests.items()) if any(s not in verified for s in slots)]
        print(json.dumps({'windows': len(plan['rows']), 'pressure_gib': a.nbytes/1024**3,
                          'unique_times': len(requests), 'remaining_times': len(pending),
                          'reused_verified_patches': len(verified), 'workers': workers}), flush=True)
        source = prep.Era5Source()
        geography = root/'geography.npz'
        if not geography.exists():
            ns = plan['rows'][0]['issue_ns']
            land = source.read('land_sea_mask', ns)
            height = source.read('geopotential_at_surface', ns)/9.80665
            if not np.isfinite(land).all() or not np.isfinite(height).all():
                raise ValueError('Invalid geography')
            with (root/'geography.pending').open('wb') as f:
                np.savez_compressed(f, latitude=source.lat, longitude=source.lon,
                                    land_fraction=land, elevation_m=height)
            (root/'geography.pending').replace(geography)
        with np.load(geography, allow_pickle=False) as z:
            if z['land_fraction'].shape != (721,1440) or z['elevation_m'].shape != (721,1440):
                raise ValueError('Geography grid mismatch')
        prep.trainer.atomic_json(root/'geography_manifest.json', {'sha256': prep.trainer.digest(geography),
            'source': prep.SOURCE, 'size': geography.stat().st_size, 'units': {'elevation_m': 'm', 'land_fraction': '1'}})
        def fetch(item):
            ns, slots = item
            pressure = source.read('mean_sea_level_pressure', ns)/100
            if not np.isfinite(pressure).all() or pressure.min() < 800 or pressure.max() > 1120:
                raise ValueError(f'Invalid MSLP at {ns}')
            return ns, [(i,j,pressure[indices[i][0][:,None],indices[i][1][None,:]]) for i,j in slots if (i,j) not in verified]
        started = time.monotonic()
        initial = len(requests)-len(pending)
        selected = pending if max_frames is None else pending[:max_frames]
        # Process bounded groups: completed patches are committed before requesting more.
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(selected), workers):
                for ns, patches in pool.map(fetch, selected[offset:offset+workers]):
                    for i,j,values in patches:
                        a[i,j] = values
                    a.flush()
                    for i,j,_ in patches:
                        db.execute('INSERT OR REPLACE INTO patches VALUES (?,?,?)', (i,j,patch_hash(a[i,j])))
                        verified.add((i,j))
                    db.commit()
                    done += 1
                    elapsed = time.monotonic()-started
                    status = {'completed_times': initial+done, 'total_times': len(requests),
                              'verified_patches': len(verified), 'total_patches': shape[0]*shape[1],
                              'elapsed_s_this_run': round(elapsed,1),
                              'eta_s': round((len(pending)-done)*elapsed/done),
                              'last_time': str(np.datetime64(ns,'ns')), 'status': 'downloading'}
                    prep.trainer.atomic_json(root/'progress.json', status)
                    print(json.dumps(status), flush=True)
        if len(verified) == shape[0]*shape[1]:
            prep.trainer.atomic_json(root/'completion.json', {'status': 'pressure_and_geography_complete',
                'training_ready': False, 'plan_sha256': plan['sha256'],
                'pressure_sha256': prep.trainer.digest(output), 'geography_sha256': prep.trainer.digest(geography)})
            print('Pressure and geography verified complete. Other training inputs still required.', flush=True)
    finally:
        if db is not None:
            db.close()
        (root/'download.lock').unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('/Volumes/D/typhoon_predict/data/v164_pressure_geography'))
    p.add_argument('--track', type=Path, default=Path('/Volumes/D/typhoon_predict/track_build/track_windows_v13.npz'))
    p.add_argument('--workers', type=int, choices=range(1,9), default=4)
    p.add_argument('--max-frames', type=int)
    args = p.parse_args()
    run(args.output, args.track, args.workers, args.max_frames)
