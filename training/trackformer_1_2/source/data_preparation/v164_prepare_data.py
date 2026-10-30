"""Prepare a resumable, explicitly labelled v164 data pilot from public ERA5.

This downloads native fields. It does not upsample the old coarse archive,
start training, or put verification labels into model inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import v164_global_regional_train as trainer

SOURCE = 'https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
TRACK_SHA = 'cb8d7291aa8140124ca522feb448a1d44124b018efc4ba1d54824b60a4f73451'
SURFACE = ['mean_sea_level_pressure', 'surface_pressure', '10m_u_component_of_wind',
           '10m_v_component_of_wind', '2m_temperature', 'sea_surface_temperature']
UPPER = ['u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity', 'geopotential']
STEP = 6 * trainer.HOUR


def year(ns):
    return int(np.datetime64(int(ns), 'ns').astype('datetime64[Y]').astype(int)) + 1970


def build_plan(track_path, counts=(3, 1, 1), oversample=1):
    if trainer.digest(track_path) != TRACK_SHA:
        raise ValueError('Shared track archive checksum mismatch')
    with np.load(track_path, allow_pickle=False) as z:
        tracks = {k: z[k] for k in ('storm_id', 'base_time', 'base_lat', 'base_lon', 'n_leads', 'target', 'target_mask')}
        issue_intensity = z['track'][:, -1, 4:6]*z['track_std'][4:6] + z['track_mean'][4:6]
    sid, times = tracks['storm_id'], tracks['base_time']
    first_year = {}
    for storm, ns in zip(sid, times):
        first_year[str(storm)] = min(first_year.get(str(storm), 9999), year(ns))
    lookup = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(sid, times))}
    rows = []
    rng = np.random.default_rng(164)
    for split, bounds, count in zip(('train', 'validation', 'test'), ((2000, 2021), (2022, 2023), (2024, 2025)), counts):
        candidates = {}
        for i, (storm, ns) in enumerate(zip(sid, times)):
            storm, ns = str(storm), int(ns)
            lat, lon = float(tracks['base_lat'][i]), float(tracks['base_lon'][i]) % 360
            if not (bounds[0] <= first_year[storm] <= bounds[1] and bounds[0] <= year(ns-8*STEP)
                    and year(ns+20*STEP) <= bounds[1] and ns % STEP == 0 and tracks['n_leads'][i] >= 20
                    and issue_intensity[i, 0] >= 90 and 850 < issue_intensity[i, 1] <= 960
                    and 0 < lat < 40 and 100 < lon < 180 and '1979' not in storm and storm.upper() != 'TIP'):
                continue
            future = [lookup.get((storm, ns + lead*STEP)) for lead in range(1, 21)]
            if sum(j is not None for j in future) < 16:
                continue
            candidates.setdefault(storm, []).append((i, future))
        storms = sorted(candidates)
        if len(storms) < count:
            raise ValueError(f'Not enough eligible {split} storms: {len(storms)} < {count}')
        for storm in rng.choice(storms, size=min(len(storms), count*oversample), replace=False):
            choices = candidates[storm]
            i, future = min(choices, key=lambda item: float(issue_intensity[item[0], 1]))
            p, vmax = tracks['target'][i, :, 3], tracks['target'][i, :, 2]
            pm = tracks['target_mask'][i, :, 3] & np.isfinite(p) & (p > 800) & (p < 1100)
            vm = tracks['target_mask'][i, :, 2] & np.isfinite(vmax) & (vmax >= 0) & (vmax < 250)
            rows.append({'issue_ns': int(times[i]), 'split': split, 'storm_id': str(storm),
                         'verification_issue_center': [float(tracks['base_lat'][i]), float(tracks['base_lon'][i]) % 360],
                         'target_centers': [[float(tracks['base_lat'][j]), float(tracks['base_lon'][j]) % 360]
                                            if j is not None else [0., 0.] for j in future],
                         'route_mask': [int(j is not None) for j in future],
                         'target_pressure_hpa': np.where(pm, p, 0).tolist(), 'pressure_mask': pm.astype(int).tolist(),
                         'target_vmax_kt': np.where(vm, vmax, 0).tolist(), 'vmax_mask': vm.astype(int).tolist()})
    # An unconditional atmospheric sample, not a claim that no cyclone exists.
    rows.append({'issue_ns': int(np.datetime64('2021-01-15T00', 'ns').astype('int64')), 'split': 'train',
                 'storm_id': None, 'sample_kind': 'unconditional_atmosphere'})
    stamps = sorted({row['issue_ns'] + i*STEP for row in rows for i in range(-8, 21)})
    indices = {t: i for i, t in enumerate(stamps)}
    for row in rows:
        row['issue_index'] = indices[row['issue_ns']]
    result = {'schema': 'v164-preparation-plan-v1', 'dataset_role': 'pipeline_pilot_not_full_training',
              'source': SOURCE, 'track_sha256': TRACK_SHA, 'rows': rows, 'time': stamps,
              'estimated_field_bytes': len(stamps)*31*(181*360+241*321)*4,
              'network_warning': 'ARCO upper-air chunks contain all 37 levels and the globe; transfer can greatly exceed saved size.',
              'sampling': 'mature intense issue-time storms for association testing; not representative benchmark sampling',
              'counts': list(counts)}
    result['plan_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def resolve_associations(plan, source):
    selected, rejected = [], []
    for split, required in zip(('train', 'validation', 'test'), plan['counts']):
        accepted = 0
        for row in plan['rows']:
            if row['split'] != split or not row.get('storm_id'):
                continue
            print('Checking issue-time analysis association:', split, row['storm_id'], flush=True)
            pressure = source.domains(source.read('mean_sea_level_pressure', row['issue_ns'])/100)[1]
            center = analysis_center(pressure[None], source.rl, source.ro)
            truth = np.asarray(row['verification_issue_center'])
            distance = float(np.linalg.norm((center-truth)*[111.2, 111.2*np.cos(np.deg2rad(truth[0]))]))
            if distance > 400:
                rejected.append({'storm_id': row['storm_id'], 'issue_ns': row['issue_ns'], 'distance_km': distance})
                continue
            selected.append(dict(row))
            accepted += 1
            if accepted == required:
                break
        if accepted < required:
            raise ValueError(f'Insufficient unambiguous issue-time associations for {split}; no dataset was published')
    selected.extend(dict(r) for r in plan['rows'] if not r.get('storm_id'))
    times = sorted({r['issue_ns']+i*STEP for r in selected for i in range(-8, 21)})
    lookup = {t: i for i, t in enumerate(times)}
    for row in selected:
        row['issue_index'] = lookup[row['issue_ns']]
    result = dict(plan, rows=selected, time=times, rejected_issue_associations=rejected,
                  estimated_field_bytes=len(times)*31*(181*360+241*321)*4)
    result.pop('plan_sha256', None)
    result['plan_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


class Era5Source:
    def __init__(self):
        import xarray as xr
        self.ds = xr.open_zarr(SOURCE, consolidated=True, chunks=None)
        missing = set(SURFACE + UPPER + ['land_sea_mask', 'geopotential_at_surface']) - set(self.ds)
        if missing:
            raise ValueError(f'ERA5 missing variables: {sorted(missing)}')
        if not set(trainer.LEVELS).issubset(set(self.ds.level.values.tolist())):
            raise ValueError('ERA5 does not supply all required pressure levels')
        self.lat, self.lon = self.ds.latitude.values, self.ds.longitude.values
        if not np.allclose(self.lat, np.linspace(90, -90, 721)) or not np.allclose(self.lon, np.arange(1440)*.25):
            raise ValueError('Unexpected ERA5 native grid; no implicit regridding')
        self.gl, self.go = self.lat[::4], self.lon[::4]
        self.ys = np.flatnonzero((self.lat >= 0) & (self.lat <= 60))
        self.xs = np.flatnonzero((self.lon >= 100) & (self.lon <= 180))
        self.rl, self.ro = self.lat[self.ys], self.lon[self.xs]

    def read(self, name, ns, levels=None):
        date = np.datetime64(int(ns), 'ns')
        stop = np.datetime64(self.ds.attrs['valid_time_stop'])
        if date > stop or date < np.datetime64(self.ds.attrs['valid_time_start']):
            raise ValueError(f'{date} outside final ERA5 availability; no ERA5T substitution')
        a = self.ds[name].sel(time=date)
        if levels is not None:
            a = a.sel(level=list(levels)).transpose('level', 'latitude', 'longitude')
        else:
            a = a.transpose('latitude', 'longitude')
        for attempt in range(3):
            try:
                return np.asarray(a.values, dtype='float32')
            except (OSError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def domains(self, a):
        return a[..., ::4, ::4], a[..., self.ys[:, None], self.xs[None, :]]

    def frame(self, ns):
        out = {name: np.empty((31, len(lat), len(lon)), dtype='float32')
               for name, lat, lon in [('global', self.gl, self.go), ('regional', self.rl, self.ro)]}
        for i, name in enumerate(SURFACE):
            print('  reading', name, flush=True)
            a = self.read(name, ns)
            if i < 2:
                a = a / 100.0
            if i == 5:
                a = np.where((a >= 260) & (a <= 330), a, np.nan)
            for domain, values in zip(out, self.domains(a)):
                out[domain][i] = values
        for j, name in enumerate(UPPER):
            print('  reading', name, 'levels', trainer.LEVELS, flush=True)
            a = self.read(name, ns, trainer.LEVELS)
            for domain, values in zip(out, self.domains(a)):
                for level in range(5):
                    out[domain][6+level*5+j] = values[level]
        for name, values in out.items():
            validate_frame(values, name)
        return out

    def static(self, ns):
        land = np.clip(self.read('land_sea_mask', ns), 0, 1)
        altitude = self.read('geopotential_at_surface', ns)/9.80665
        result = {}
        for i, (name, lat, lon) in enumerate([('global', self.gl, self.go), ('regional', self.rl, self.ro)]):
            yy, xx = np.meshgrid(lat, lon, indexing='ij')
            result[name] = np.stack([yy/90, xx/180-1, self.domains(land)[i], self.domains(altitude)[i]/8000]).astype('float32')
        return result


def validate_frame(a, name):
    if not np.isfinite(a[[i for i in range(31) if i != 5]]).all():
        raise ValueError(f'Nonfinite {name} atmospheric field; refusing zero-fill')
    if not (800 < a[0].min() <= a[0].max() < 1120):
        raise ValueError(f'{name} MSLP range/unit failure')
    if not (150 < a[4].min() <= a[4].max() < 350):
        raise ValueError(f'{name} temperature range/unit failure')


def analysis_center(field, lat, lon, previous=None):
    pressure = field[0].copy()
    yy, xx = np.meshgrid(lat, lon, indexing='ij')
    allowed = (yy > 0) & (yy < 40)
    if previous is not None:
        distance = ((yy-previous[0])*111.2)**2 + ((xx-previous[1])*111.2*np.cos(np.deg2rad(previous[0])))**2
        allowed &= distance < 600**2
    index = np.argmin(np.where(allowed, pressure, np.inf))
    return np.array([yy.flat[index], xx.flat[index]], dtype='float32')


def materialize(plan, root, source, max_gb=4, smoke=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan_path, progress_path = root/'preparation_plan.json', root/'preparation_progress.json'
    if plan_path.exists() and json.loads(plan_path.read_text())['plan_sha256'] != plan['plan_sha256']:
        raise ValueError('Existing preparation uses a different plan; choose a separate dataset directory')
    needed = plan['estimated_field_bytes']
    if needed > max_gb*1e9:
        raise ValueError(f'Plan needs {needed/1e9:.2f} GB; exceeds explicit {max_gb} GB limit')
    existing = sum(p.stat().st_size for p in root.glob('*.npy'))
    if shutil.disk_usage(root).free < max(0, needed-existing) + (128 if smoke else 512)*1024**2:
        raise OSError('Insufficient storage for data and safety margin')
    trainer.atomic_json(plan_path, plan)
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'frames': {}}
    if (root/'manifest.json').exists():
        trainer.load_manifest(root/'manifest.json', smoke=smoke)
        print('Prepared dataset already verified.', flush=True)
        return
    arrays = {}
    for name, lat, lon in [('global', source.gl, source.go), ('regional', source.rl, source.ro)]:
        p = root/(name+'.npy')
        shape = (len(plan['time']), 31, len(lat), len(lon))
        arrays[name] = np.lib.format.open_memmap(p, mode='r+' if p.exists() else 'w+', dtype='float32', shape=shape)
        if arrays[name].shape != shape:
            raise ValueError('Partial data shape mismatch')
    for i, ns in enumerate(plan['time']):
        key = str(ns)
        def frame_digest():
            h = hashlib.sha256()
            for a in arrays.values():
                h.update(np.ascontiguousarray(a[i]).tobytes())
            return h.hexdigest()
        if key in progress['frames'] and frame_digest() == progress['frames'][key]:
            continue
        print(f'ERA5 frame {i+1}/{len(plan["time"])}: {np.datetime64(ns, "ns")} (31 channels)', flush=True)
        frame = source.frame(ns)
        for name, a in arrays.items():
            a[i] = frame[name]
            a.flush()
        progress['frames'][key] = frame_digest()
        progress.update(completed=len(progress['frames']), total=len(plan['time']), last_completed_utc=str(np.datetime64(ns, 'ns')))
        trainer.atomic_json(progress_path, progress)
    np.save(root/'time.npy', np.asarray(plan['time'], dtype='int64'))
    for name, a in source.static(plan['time'][0]).items():
        np.save(root/(name+'_static.npy'), a)
    sums, squares, counts = np.zeros(31), np.zeros(31), np.zeros(31)
    for i, ns in enumerate(plan['time']):
        if not 2000 <= year(ns) <= 2021:
            continue
        for a in arrays.values():
            values = np.asarray(a[i], dtype='float64').reshape(31, -1)
            sums += np.nansum(values, axis=1)
            squares += np.nansum(values**2, axis=1)
            counts += np.isfinite(values).sum(axis=1)
    if (counts == 0).any():
        raise ValueError('Missing training-only normalization data')
    mean = sums/counts
    std = np.maximum(np.sqrt(np.maximum(squares/counts-mean**2, 0)), 1e-6)
    rows = []
    for original in plan['rows']:
        row = dict(original)
        i = row['issue_index']
        # Association is derived solely from issue/past ERA5, never best-track coordinates.
        center = analysis_center(arrays['regional'][i], source.rl, source.ro)
        past = analysis_center(arrays['regional'][i-1], source.rl, source.ro, center)
        row['center'] = center.tolist()
        row['motion_km_per_6h'] = [float((center[1]-past[1])*111.2*np.cos(np.deg2rad(center[0]))), float((center[0]-past[0])*111.2)]
        truth = row.pop('verification_issue_center', None)
        if truth is not None:
            dist = np.linalg.norm((center-np.array(truth))*[111.2, 111.2*np.cos(np.deg2rad(truth[0]))])
            row['issue_association_error_km'] = float(dist)
            if dist > 400:
                row['route_mask'] = row['pressure_mask'] = row['vmax_mask'] = [0]*20
        rows.append(row)
    (root/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    m = {'schema': 'v164-atmosphere-v1', 'synthetic': bool(smoke), 'dataset_role': plan['dataset_role'],
         'channels': trainer.CHANNELS, 'units': trainer.UNITS,
         'global_lat': source.gl.tolist(), 'global_lon': source.go.tolist(),
         'regional_lat': source.rl.tolist(), 'regional_lon': source.ro.tolist(),
         'normalization': {'fit_split': 'train', 'end_year': 2021, 'mean': mean.tolist(), 'std': std.tolist()},
         'future_weather_as_inputs': False, 'official_forecasts_as_inputs': False,
         'inputs_are_analysis_valid_time_hindcasts_not_operational_release_time': True,
         'source': SOURCE, 'track_archive_sha256': plan['track_sha256'], 'files': {}}
    for name in ('global', 'regional', 'global_static', 'regional_static', 'time', 'samples'):
        p = root/(name+('.jsonl' if name == 'samples' else '.npy'))
        m['files'][name] = {'path': p.name, 'sha256': trainer.digest(p)}
    for split in ('train', 'validation', 'test'):
        if not any(row['split'] == split and any(row.get('route_mask', [])) for row in rows):
            raise ValueError(f'No valid analysis-associated storm labels in {split}. Pilot is insufficient; do not train it.')
    # Validate a temporary manifest before publishing the ready marker.
    candidate = root/'manifest.pending.json'
    trainer.atomic_json(candidate, m)
    trainer.load_manifest(candidate, smoke=smoke)
    candidate.replace(root/'manifest.json')
    print('PILOT DATA READY. This is not the full training corpus or evidence of forecast skill.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--max-gb', type=float, default=4)
    p.add_argument('--cases', type=int, nargs=3, default=[3, 1, 1], metavar=('TRAIN', 'VAL', 'TEST'))
    args = p.parse_args()
    if min(args.cases) < 1:
        p.error('Each split needs at least one case')
    plan = build_plan(args.track, args.cases, oversample=4 if args.prepare else 1)
    if args.prepare:
        args.output.mkdir(parents=True, exist_ok=True)
        lock = args.output/'preparation.lock'
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        try:
            source = Era5Source()
            previous = args.output/'preparation_plan.json'
            if previous.exists():
                plan = json.loads(previous.read_text())
                if plan['track_sha256'] != TRACK_SHA or plan['counts'] != args.cases or plan['source'] != SOURCE:
                    raise ValueError('Existing preparation configuration differs; choose a new output directory')
                saved_hash = plan['plan_sha256']
                payload = {k: v for k, v in plan.items() if k != 'plan_sha256'}
                if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != saved_hash:
                    raise ValueError('Existing preparation plan checksum mismatch')
            else:
                plan = resolve_associations(plan, source)
            print(json.dumps({k: v for k, v in plan.items() if k not in ('rows', 'time')}, indent=2), flush=True)
            materialize(plan, args.output, source, args.max_gb)
        finally:
            lock.unlink(missing_ok=True)
    else:
        print(json.dumps({k: v for k, v in plan.items() if k not in ('rows', 'time')}, indent=2), flush=True)
        print('Plan only. Add --prepare to download. No training is started.', flush=True)


if __name__ == '__main__':
    main()
