"""Prepare honest 50-member historical video data using the frozen 1.2 on MPS.

Reads existing verified local analyses only; no downloads or model fitting.
Future best-track coordinates/pressures are attached AFTER model inference.
Each calendar-selected example is development evidence, not a holdout test.
"""
import argparse
import csv
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

import automatic_forecasts as auto
from ensemble_forecast import array_hash, run_ensemble, mean_outputs
from build_fung_wong_video import distances

CASES = {
    'soudelor': ('SOUDELOR', '2015211N13162', '2015-08-05T00:00:00Z'),
    'mangkhut': ('MANGKHUT', '2018250N12170', '2018-09-11T00:00:00Z'),
    'meranti': ('MERANTI', '2016253N13144', '2016-09-10T00:00:00Z'),
}
SOURCE_HASHES = {
    'track_build/basin_all_int8.npz': '71e5f1bb46c44e5936c506a3f8af021c1fe94c238ed030c05c66e8f342374b6d',
    'track_build/basin_slp_atlas_float16.npy': '9f6a4ec7b2f6beadfa1075af9176d2a653e39a824d504d989e5e56e19c571521',
    'output/automatic-forecast-cache/geography.npz': 'f5471471da4a1e15f85fbfb2529065806281357c8e50cb272fcd2b02e8553b96',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def exact_history_indices(times, issue):
    wanted = auto.ns(issue) + np.arange(-8, 1, dtype='int64') * 6 * auto.HOUR
    indices = np.searchsorted(times, wanted)
    if np.any(indices >= len(times)) or not np.array_equal(times[indices], wanted):
        raise ValueError('Missing exact causal six-hour analysis; no nearest-date fill')
    return indices


def number(row, key):
    value = row.get(key, '').strip()
    return float(value) if value else None


def issue_row(rows, name, sid, issue):
    now = auto.parse(issue)
    stamp = now.strftime('%Y-%m-%d %H:%M:%S')
    p = rows[stamp]
    lat, lon = float(p['LAT']), float(p['LON'])
    prior = rows.get((now-timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'))
    motion = [0., 0.] if prior is None else [
        (lon-float(prior['LON']))*111.2*np.cos(np.deg2rad(lat)),
        (lat-float(prior['LAT']))*111.2]
    # Construct an allowlisted input row. No future display labels are attached.
    return {'name': name, 'storm_id': sid, 'issue_time_utc': issue,
            'lat': lat, 'lon': lon, 'motion': motion,
            'pressure_hpa': number(p, 'TOKYO_PRES'), 'wind_kt': number(p, 'TOKYO_WIND')}


def observations(rows, issue):
    route, pressure = [], []
    now = auto.parse(issue)
    for k in range(21):
        row = rows.get((now+timedelta(hours=k*6)).strftime('%Y-%m-%d %H:%M:%S'))
        route.append([float(row['LAT']), float(row['LON'])] if row else [np.nan, np.nan])
        value = number(row, 'TOKYO_PRES') if row else None
        pressure.append(value if value is not None and 800 < value < 1100 else np.nan)
    return np.asarray(route), np.asarray(pressure[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--storm', choices=list(CASES), action='append', required=True)
    ap.add_argument('--chunk', type=int, default=5)
    args = ap.parse_args()
    if not str(args.output.resolve()).startswith('/Volumes/D/'):
        raise ValueError('New artifacts must stay on /Volumes/D')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS GPU required; no silent CPU fallback')
    args.output.mkdir(parents=True, exist_ok=True)
    contract_meta = json.loads((auto.MODEL/'manifest.json').read_text())
    contract = contract_meta['data_contract']
    if contract_meta['public_version'] != '1.2' or contract_meta['source_checkpoint_sha256'] != auto.CHECKPOINT:
        raise ValueError('Wrong release identity')
    for filename, expected in SOURCE_HASHES.items():
        if sha(args.project/filename) != expected:
            raise ValueError('Local analysis/geography identity mismatch: '+filename)
    for name, expected in contract_meta['source_module_sha256'].items():
        if sha(auto.MODEL/name) != expected:
            raise ValueError('Frozen release code changed: '+name)
    weights = args.project/'output/automatic-forecast-cache/weights.pt'
    if sha(weights) != contract_meta['inference_weights_sha256']:
        raise ValueError('Wrong frozen model weights')
    with np.load(args.project/'track_build/basin_all_int8.npz', allow_pickle=False) as z:
        times = z['time'].astype('int64')
        scale, offset = z['scale'].copy(), z['offset'].copy()
        if list(z['channels']) != contract['channels'][1:]:
            raise ValueError('Channel order mismatch')
        if not np.array_equal(z['lat'], contract['global_lat']) or not np.array_equal(z['lon'], contract['global_lon']):
            raise ValueError('Analysis grid mismatch')
        # The mmap is only a decoded-cache optimization. Compare all requested
        # subsets to the hash-verified original encoded source before inference.
        q_mmap = np.load(args.project/'data/v164_reuse/basin_q_verified.npy', mmap_mode='r')
        q_original = z['q']
        for slug in args.storm:
            idx = exact_history_indices(times, CASES[slug][2])
            if not np.array_equal(q_mmap[idx], q_original[idx]):
                raise ValueError('Decoded weather cache differs from original')
        del q_original
    slp = np.load(args.project/'track_build/basin_slp_atlas_float16.npy', mmap_mode='r')
    csvpath = args.project/'data/ibtracs/ibtracs.WP.list.v04r01.csv'
    csvhash = sha(csvpath)
    with csvpath.open() as stream:
        all_rows = [r for r in csv.DictReader(stream) if r['SID'] in {CASES[s][1] for s in args.storm}]
    coastpath = args.project/'trackformer-weatherlab-site/public/data/history/coastlines.json'
    coastlines = json.loads(coastpath.read_text())
    geo = np.load(args.project/'output/automatic-forecast-cache/geography.npz', allow_pickle=False)
    torch.set_num_threads(4)
    model = auto.CoreForecaster(contract).eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    for slug in args.storm:
        dest = args.output/slug
        # A finished issue is immutable, even when rerunning the command.
        if (dest/'verification.json').exists():
            receipt = json.loads((dest/'verification.json').read_text())
            for filename, expected in receipt['files_sha256'].items():
                if sha(dest/filename) != expected:
                    raise ValueError('Completed artifact changed: '+filename)
            print(json.dumps({'storm': slug, 'reused_verified_complete': True}), flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        name, sid, issue = CASES[slug]
        rows = {r['ISO_TIME']: r for r in all_rows if r['SID'] == sid}
        if len(rows) != sum(r['SID'] == sid for r in all_rows):
            raise ValueError('Duplicate storm observation timestamp')
        row = issue_row(rows, name, sid, issue)
        idx = exact_history_indices(times, issue)
        weather = np.concatenate((np.asarray(slp[idx], dtype='float32')[:, None],
            np.asarray(q_mmap[idx], dtype='float32')*scale[None, :, None, None]+offset[None, :, None, None]), axis=1)
        x = auto.inputs(weather, times[idx], row, contract, geo)
        if not np.array_equal(x['center'][0].numpy(), np.asarray([row['lat'], row['lon']], dtype='float32')):
            raise ValueError('+0 observation alignment failed')
        np.savez_compressed(dest/'weather-history.npz', physical_weather=weather, time_ns=times[idx])
        np.savez_compressed(dest/'model-inputs.npz', **{k: v.numpy() for k, v in x.items()}, history_time_ns=times[idx])
        print(json.dumps({'storm': slug, 'starting': issue, 'device': 'mps', 'members': 50}), flush=True)
        output, policy = run_ensemble(model, x, contract, 'mps', args.chunk,
            lambda done, total, elapsed: print(json.dumps({'storm': slug, 'members_done': done,
                'members': total, 'elapsed_seconds': elapsed}), flush=True))
        means = mean_outputs(output)
        for key in ('center', 'pressure', 'basin', 'regional'):
            if not np.allclose(means[key], output[key].mean(axis=0, dtype=np.float64), rtol=0, atol=1e-4):
                raise ValueError('Mean aggregation failed: '+key)
        np.savez_compressed(dest/'ensemble-members.npz', **output, seeds=np.asarray(policy['seeds']))
        # Attach future labels only after all forecast members have finished.
        obs, truth_pressure = observations(rows, issue)
        origin = np.array([row['lat'], row['lon']])
        anchor = np.round(origin*4)/4
        lat = anchor[0]+np.linspace(15, -15, 121)
        lon = anchor[1]+np.linspace(-15, 15, 121)
        route = np.vstack([origin, means['center']])
        errors = distances(means['center'], obs[1:])
        pressure_mask = np.isfinite(truth_pressure)
        npz = dest/f'{slug}_video.npz'
        np.savez_compressed(npz, forecast_lat_lon=route, observed_lat_lon=obs,
            central_pressure_hpa=means['pressure'], observed_pressure_hpa=truth_pressure,
            regional_pressure_hpa=means['regional'], basin_pressure_hpa=means['basin'],
            latitude=lat, longitude=lon, track_error_km=errors,
            regional_valid_mask=(lat[:, None] >= 0)&(lat[:, None] <= 60)&(lon[None, :] >= 100)&(lon[None, :] <= 180),
            valid_member_counts=output['track_valid'].sum(axis=0))
        rings = [r for r in coastlines if any(100 <= p[0] <= 180 and 0 <= p[1] <= 60 for p in r)]
        (dest/f'{slug}_coastlines.json').write_text(json.dumps(rings, separators=(',', ':'))+'\n')
        local_source = {name: value for name, value in SOURCE_HASHES.items()}
        meta = {'model': 'Trackformer 1.2', 'storm': name, 'storm_id': sid, 'issue_time_utc': issue,
            'members': 50, 'device': 'mps', 'checkpoint_sha256': auto.CHECKPOINT,
            'inference_weights_sha256': contract_meta['inference_weights_sha256'],
            'source_module_sha256': contract_meta['source_module_sha256'],
            'data_sha256': sha(npz), 'ensemble_policy': policy,
            'leads_hours': list(range(6, 121, 6)), 'display_bounds': [100, 180, 0, 60],
            'source_inputs': {'source': 'Existing exact NOAA NCEP Reanalysis-1 basin atlas; retrospective analyses.',
                'channels': contract['channels'], 'units': contract['units'],
                'analysis_times_utc': [str(np.datetime64(int(t), 'ns').astype('datetime64[s]'))+'Z' for t in times[idx]],
                'source_sha256': local_source, 'subset_sha256': sha(dest/'weather-history.npz'),
                'physical_weather_sha256': array_hash(weather),
                'normalized_input_sha256': sha(dest/'model-inputs.npz'),
                'issue_observation': row,
                'motion_note': 'Only current and exact previous six-hour observed coordinates; east/north km per six hours.',
                'causal': True, 'future_labels_used_for_inference': False},
            'observed_csv_sha256': csvhash,
            'observed_route_source': 'Exact matching-timestamp IBTrACS LAT/LON; display verification only.',
            'observed_pressure_source': 'IBTrACS TOKYO_PRES JMA best track, not JMA forecasts; missing labels remain missing.',
            'field_source': 'Equal-weight physical means of all 50 basin and fixed-issue regional fields; no directly averaged moving grids.',
            'native_history_available': False,
            'grid_note': '0.25-degree regional reconstruction sampling, not native effective resolution. Basin sampling is 2.5 degrees. Coarse basin only outside the fixed regional patch.',
            'regional_bounds': [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
            'first_forecast_lead_outside_regional_hours': next((6*(i+1) for i, point in enumerate(means['center'])
                if not (lon.min() <= point[1] <= lon.max() and lat.min() <= point[0] <= lat.max())), None),
            'metrics': {'mean_track_error_km': float(np.nanmean(errors)),
                'track_error_120h_km': float(errors[-1]) if np.isfinite(errors[-1]) else None,
                'central_pressure_mae_hpa': float(np.abs(means['pressure'][pressure_mask]-truth_pressure[pressure_mask]).mean()) if pressure_mask.any() else None,
                'pressure_valid_leads': int(pressure_mask.sum()), 'position_valid_leads': int(np.isfinite(errors).sum())},
            'selection': 'Calendar issue chosen before running these forecasts, without screening errors. Historical development showcase; dates may overlap checkpoint fitting/validation years. Not an untouched test or evidence of no overfitting.'}
        auto.write(dest/f'{slug}_video.json', meta)
        names = ['weather-history.npz', 'model-inputs.npz', 'ensemble-members.npz',
                 f'{slug}_video.npz', f'{slug}_video.json', f'{slug}_coastlines.json']
        auto.write(dest/'verification.json', {'model': 'Trackformer 1.2', 'device': 'mps', 'members': 50,
            'distinct_inputs': len(set(policy['input_sha256'])), 'distinct_routes': len(set(policy['route_sha256'])),
            'distinct_basin_fields': len(set(policy['basin_field_sha256'])), 'route_points': 21,
            'pressure_frames': 20, 'physical_means_verified': True, 'observed_plus_zero_verified': True,
            'checkpoint_sha256': auto.CHECKPOINT, 'files_sha256': {n: sha(dest/n) for n in names}})
        print(json.dumps({'complete': slug, 'output': str(dest), 'metrics': meta['metrics']}), flush=True)


if __name__ == '__main__':
    main()
