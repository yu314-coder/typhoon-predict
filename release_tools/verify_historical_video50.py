"""Check video identity, actual distinct member means, causal inputs and +0."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import automatic_forecasts as auto
from ensemble_forecast import array_hash
from forecast_historical_video50_mac import CASES, sha


def verify(forecasts, repo):
    report = {}
    for slug, (name, sid, issue) in CASES.items():
        saved = forecasts/slug
        data = repo/'evaluation/release_data'
        meta = json.loads((data/f'{slug}_video.json').read_text())
        original = json.loads((saved/'verification.json').read_text())
        for filename, expected in original['files_sha256'].items():
            if sha(saved/filename) != expected:
                raise ValueError('Original inference artifact changed: '+filename)
        assert meta['checkpoint_sha256'] == auto.CHECKPOINT
        assert (meta['storm'], meta['storm_id'], meta['issue_time_utc']) == (name, sid, issue)
        assert meta['members'] == 50 and meta['leads_hours'] == list(range(6, 121, 6))
        assert sha(data/f'{slug}_video.npz') == meta['data_sha256']
        with np.load(saved/'model-inputs.npz', allow_pickle=False) as x:
            np.testing.assert_array_equal(x['history_time_ns'], auto.ns(issue)+np.arange(-8, 1)*6*auto.HOUR)
            assert not x['detail_available'].any()
            origin = x['center'][0].copy()
        counts, deltas = {}, {}
        with np.load(saved/'ensemble-members.npz', allow_pickle=False) as members, np.load(data/f'{slug}_video.npz', allow_pickle=False) as mean:
            np.testing.assert_array_equal(members['seeds'], np.arange(2043, 2093))
            np.testing.assert_allclose(mean['forecast_lat_lon'][0], origin, atol=1e-5)
            np.testing.assert_allclose(mean['observed_lat_lon'][0], origin, atol=1e-5)
            for raw_key, mean_key in [('center', 'forecast_lat_lon'), ('pressure', 'central_pressure_hpa'),
                                      ('basin', 'basin_pressure_hpa'), ('regional', 'regional_pressure_hpa')]:
                raw = members[raw_key]
                assert raw.shape[0] == 50 and raw.shape[1] == 20 and np.isfinite(raw).all()
                if raw_key != 'center':
                    assert raw.min() >= 800 and raw.max() <= 1100
                actual = mean[mean_key][1:] if raw_key == 'center' else mean[mean_key]
                expected = raw.mean(axis=0, dtype=np.float64).astype('float32')
                np.testing.assert_array_equal(actual, expected)
                counts[raw_key] = len({array_hash(v) for v in raw})
                assert counts[raw_key] == 50
                deltas[raw_key] = float(np.max(np.abs(actual-expected)))
        movie = repo/meta['video']['file']
        if sha(movie) != meta['video']['sha256']:
            raise ValueError('Video file SHA mismatch')
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,nb_frames:format=duration', '-of', 'json', str(movie)]))
        stream = probe['streams'][0]
        assert (stream['codec_name'], stream['width'], stream['height'], int(stream['nb_frames'])) == ('h264', 1600, 1000, 690)
        assert abs(float(probe['format']['duration'])-23) < .1
        report[slug] = {'storm_id': sid, 'issue_time_utc': issue, 'model': 'Trackformer 1.2',
            'checkpoint_sha256': meta['checkpoint_sha256'], 'members': 50,
            'distinct_inputs': len(set(meta['ensemble_policy']['input_sha256'])),
            'distinct_member_outputs': counts, 'max_mean_difference': deltas,
            'causal_nine_analyses_verified': True, 'observed_plus_zero_alignment_verified': True,
            'future_observations_not_in_model_packet': True, 'masked_native_history_verified': True,
            'native_history_available': False, 'route_points': 21, 'pressure_states': 20,
            'mp4_duration_seconds': float(probe['format']['duration']), 'mp4_dimensions': [1600, 1000],
            'files_sha256': {'mean_fields': meta['data_sha256'], 'mp4': meta['video']['sha256'],
                'individual_members': sha(saved/'ensemble-members.npz'), 'model_inputs': sha(saved/'model-inputs.npz')},
            'evaluation_limit': 'Calendar-selected historical development examples, not untouched holdouts.'}
        assert report[slug]['distinct_inputs'] == 50
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--forecasts', type=Path, required=True)
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--receipt', type=Path, required=True)
    args = ap.parse_args()
    if not str(args.receipt.resolve()).startswith('/Volumes/D/'):
        raise ValueError('Verification receipt must stay on /Volumes/D')
    report = verify(args.forecasts, args.repo)
    auto.write(args.receipt, {'verified': True, 'storms': report})
    print(json.dumps({'verified': list(report), 'members_per_storm': 50, 'mean_differences': 0,
                      'receipt': str(args.receipt)}))
