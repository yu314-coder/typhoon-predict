"""Read-only audit of the released 1.2 dataset; no inference or cache refresh.

Requires the original checksum-pinned local dataset, not new downloads. Writes
only the requested small provenance receipt on /Volumes/D. Unlike the training
loader, this does not rewrite its verified basin mmap or initialize PyTorch.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOUR = 3_600_000_000_000
SPLITS = {'train': (2000, 2021, 13949, 800),
          'validation': (2022, 2023, 1041, 100),
          'test': (2024, 2025, 1195, 100)}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc(ns):
    return str(np.datetime64(int(ns), 'ns').astype('datetime64[s]')) + 'Z'


def audit(dataset_root):
    released = json.loads((ROOT / 'models/trackformer_1_2_field/manifest.json').read_text())
    path = dataset_root / 'manifest.json'
    if sha(path) != released['dataset_sha256']:
        raise ValueError('Dataset manifest differs from the released checkpoint')
    data = json.loads(path.read_text())
    if data['schema'] != 'v164-reuse-pressure-v1' or data['pilot']:
        raise ValueError('Exact full dataset required')
    files = {}
    for key, entry in data['files'].items():
        local = dataset_root / entry['path']
        actual = sha(local)
        if actual != entry['sha256']:
            raise ValueError('Dataset file checksum differs: ' + key)
        files[key] = dict(file=entry['path'], sha256=actual, bytes=local.stat().st_size)
    with np.load(dataset_root / files['track']['file'], allow_pickle=False) as z:
        times, storms = z['base_time'], z['storm_id']
        lat, lon = z['base_lat'], z['base_lon'] % 360
    with np.load(dataset_root / files['basin']['file'], allow_pickle=False) as z:
        atlas, channels = z['time'], z['channels'].tolist()
        if channels != released['data_contract']['channels'][1:]:
            raise ValueError('Basin channel order differs')
    if not np.all(np.diff(atlas) == 6 * HOUR):
        raise ValueError('Expected exact regular six-hour atlas')
    years = times.astype('datetime64[ns]').astype('datetime64[Y]').astype(int) + 1970
    first = {}
    for sid, year in zip(storms, years):
        first[str(sid)] = min(first.get(str(sid), 9999), int(year))
    first_years = np.array([first[str(sid)] for sid in storms])
    bounds = np.stack((times - 48 * HOUR, times + 120 * HOUR), axis=1)
    bound_years = bounds.astype('datetime64[ns]').astype('datetime64[Y]').astype(int) + 1970
    indices = np.searchsorted(atlas, times)
    eligible = ((indices >= 8) & (indices + 20 < len(atlas)) &
                (atlas[np.clip(indices, 0, len(atlas) - 1)] == times) &
                np.isfinite(lat) & np.isfinite(lon) &
                (lat > 0) & (lat < 60) & (lon > 100) & (lon < 180))
    plan = json.loads((dataset_root / files['plan']['file']).read_text())
    patches = Counter(row['split'] for row in plan['rows'])
    if plan['shape'] != [1000, 29, 121, 121] or plan['lead_hours'] != list(range(-48, 121, 6)):
        raise ValueError('Native pressure package shape/time mismatch')
    split_report, split_ids = {}, {}
    for name, (low, high, expected, native) in SPLITS.items():
        mask = (eligible & (first_years >= low) & (first_years <= high) &
                (bound_years[:, 0] >= low) & (bound_years[:, 1] <= high))
        if int(mask.sum()) != expected or patches[name] != native:
            raise ValueError('Split counts differ: ' + name)
        split_ids[name] = set(storms[mask])
        rows = set(np.flatnonzero(mask).tolist())
        for patch in (r for r in plan['rows'] if r['split'] == name):
            i = patch['track_archive_row']
            if i not in rows or patch['issue_ns'] != int(times[i]) or patch['storm_id'] != str(storms[i]):
                raise ValueError('Native patch is not in the matching original split')
        split_report[name] = dict(years=[low, high], windows=expected,
            storms=len(split_ids[name]), native_pressure_windows=native,
            first_issue_utc=utc(times[mask].min()), last_issue_utc=utc(times[mask].max()),
            earliest_history_utc=utc(bounds[mask, 0].min()),
            latest_target_utc=utc(bounds[mask, 1].max()))
    if any(split_ids[a] & split_ids[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('A storm appears in multiple partitions')
    norm = released['data_contract']['normalization']
    if norm['fit_split'] != 'train' or norm['end_year'] != 2021:
        raise ValueError('Released normalization cutoff differs')
    return dict(schema='trackformer-1.2-training-data-audit-v1', status='verified',
        public_version='1.2', source_checkpoint_version=released['source_checkpoint_version'],
        source_checkpoint_sha256=released['source_checkpoint_sha256'],
        inference_weights_sha256=released['inference_weights_sha256'],
        dataset_manifest_sha256=released['dataset_sha256'], verified_archive_files=files,
        sources=dict(track='NOAA IBTrACS-derived track_windows_v13',
            basin='NOAA PSL NCEP/NCAR Reanalysis 1',
            native_pressure=plan['source'], geography=plan['source'],
            raw_ibtracs_release='Not recorded in the selected dataset manifest; derived archive hash is authoritative'),
        channels=released['data_contract']['channels'], normalization_fit_years=[2000, 2021],
        partitions=split_report, whole_storm_splits=True, boundary_crossing_windows_excluded=True,
        history_leads_hours=list(range(-48, 1, 6)), target_leads_hours=list(range(6, 121, 6)),
        upstream_cache_coverage=dict(
            track=dict(first_issue_utc=utc(times.min()), last_issue_utc=utc(times.max())),
            basin=dict(first_analysis_utc=utc(atlas.min()), last_analysis_utc=utc(atlas.max()))),
        cutoff_interpretation='Fitting through 2021; validation through 2023; original test through 2025. Newer cache/live records are not gradient fitting.',
        inference_performed=False, model_weights_modified=False, training_data_modified=False,
        audit_scope='All six derived archive file hashes, release manifest identity and loader-equivalent split/patch membership. Not a new raw-download or operational-availability audit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path('/Volumes/D')):
        raise ValueError('Diagnostic output must remain on /Volumes/D')
    report = audit(args.dataset_root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'partitions': report['partitions'], 'output': str(output)}))
