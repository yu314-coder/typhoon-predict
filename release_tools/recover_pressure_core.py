"""Recover omitted geographic pressure exports, without changing the release.

Replay checksum-verified saved causal tensors. Capture the model's own tapered
moving anomaly and use exactly the regional output equation (basin + anomaly).
Never insert a scalar pressure, shift toward a route, or use verification truth.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

MODEL = Path(__file__).resolve().parents[1] / 'models/trackformer_1_2_field'
sys.path.insert(0, str(MODEL))
from model import CoreForecaster
from ensemble_forecast import run_ensemble, mean_outputs

CHECKPOINT = 'f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0'
HOUR = 3600 * 10**9

def ns(stamp):
    return int(np.datetime64(stamp.replace('Z',''), 'ns').astype('int64'))

def parse(stamp):
    return datetime.fromisoformat(stamp.replace('Z', '+00:00'))

def utc(stamp):
    return stamp.astimezone(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class CaptureModel(CoreForecaster):
    def __init__(self, contract):
        super().__init__(contract)
        self.blocks = []

    def capture(self, state, initial=False):
        lat, lon = self.coordinates(state['origin'])
        return {
            'anomaly': (state['anomaly'] * (1 if initial else self.taper) * self.std[:, :1])[:, 0].detach().cpu().numpy(),
            'latitude': lat[:, :, 0].detach().cpu().numpy(),
            'longitude': lon[:, 0, :].detach().cpu().numpy(),
            'basin': (state['g'][:, 0] * self.std[0, 0, 0, 0] + self.mean[0, 0, 0, 0]).detach().cpu().numpy(),
        }

    def initial(self, inputs):
        state = super().initial(inputs)
        self.blocks.append([self.capture(state, initial=True)])
        return state

    def step(self, state):
        state, output = super().step(state)
        self.blocks[-1].append(self.capture(state))
        return state, output


def reconstruct(block, latitude, longitude, contract):
    """Member reconstruction on the same geographic lattice BEFORE averaging."""
    yy, xx = np.meshgrid(latitude, longitude, indexing='ij')
    points = np.stack([yy, xx], -1)
    basin_mean = block['basin'].mean(0, dtype=np.float64)
    field = RegularGridInterpolator(
        (np.asarray(contract['global_lat'])[::-1], contract['global_lon']),
        basin_mean[::-1], bounds_error=True)(points)
    for anomaly, lat, lon in zip(block['anomaly'], block['latitude'], block['longitude']):
        # Outside the model's 640-km frame the tapered anomaly is exactly zero.
        # The edge taper is already in the network equation, not a display fit.
        rows = np.flatnonzero((latitude >= lat[-1]) & (latitude <= lat[0]))
        cols = np.flatnonzero((longitude >= lon[0]) & (longitude <= lon[-1]))
        if not len(rows) or not len(cols):
            continue
        sub = points[np.ix_(rows, cols)]
        term = RegularGridInterpolator((lat[::-1], lon), anomaly[::-1],
                                      bounds_error=True)(sub)
        field[np.ix_(rows, cols)] += term / len(block['anomaly'])
    if not np.isfinite(field).all() or field.min() < 800 or field.max() > 1100:
        raise ValueError('Nonfinite/unphysical geographic reconstruction')
    return field.astype('float32')


def local_patch(block, contract):
    lat = block['latitude']; lon = block['longitude']
    north = min(60, np.ceil(lat.max() * 10) / 10)
    south = max(0, np.floor(lat.min() * 10) / 10)
    west = max(100, np.floor(lon.min() * 10) / 10)
    east = min(180, np.ceil(lon.max() * 10) / 10)
    ys = np.round(np.arange(north, south - .01, -.1), 6)
    xs = np.round(np.arange(west, east + .01, .1), 6)
    return dict(latitude=ys.tolist(), longitude=xs.tolist(),
                pressure_hpa=np.round(reconstruct(block, ys, xs, contract), 2).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--saved', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--stem', required=True)
    ap.add_argument('--members', type=int, choices=[1, 50], default=50)
    ap.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    ap.add_argument('--reference', type=Path, help='Original one-member forecast JSON')
    args = ap.parse_args()
    if not str(args.output.resolve()).startswith('/Volumes/D/'):
        raise ValueError('Artifacts must stay on /Volumes/D')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((MODEL / 'manifest.json').read_text())
    contract = manifest['data_contract']
    if manifest['public_version'] != '1.2' or manifest['source_checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Wrong release')
    for filename, expected in manifest['source_module_sha256'].items():
        if sha(MODEL / filename) != expected:
            raise ValueError('Frozen model source changed')
    weights = Path('/Volumes/D/typhoon_predict/output/automatic-forecast-cache/weights.pt')
    if sha(weights) != manifest['inference_weights_sha256']:
        raise ValueError('Frozen weights changed')
    receipt = json.loads((args.saved / 'verification.json').read_text())
    for filename, expected in receipt['files_sha256'].items():
        if sha(args.saved / filename) != expected:
            raise ValueError('Original saved artifact changed: ' + filename)
    meta = json.loads((args.saved / f'{args.stem}_video.json').read_text())
    with np.load(args.saved / 'model-inputs.npz', allow_pickle=False) as z:
        times = z['history_time_ns']
        allowed = {'global_history','regional_history','global_static','regional_static',
                   'detail_available','center','motion','issue_intensity','issue_mask'}
        if set(z.files) != allowed | {'history_time_ns'}:
            raise ValueError('Unexpected input tensors')
        inputs = {k: torch.from_numpy(z[k].copy()) for k in allowed}
    if times[-1] != ns(meta['issue_time_utc']) or not np.all(np.diff(times) == 6 * HOUR):
        raise ValueError('Not exact causal history')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS required')
    torch.set_num_threads(4)
    model = CaptureModel(contract).eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    if args.members == 50:
        output, policy = run_ensemble(model, inputs, contract, args.device, chunk=5,
            progress=lambda done, total, elapsed: print(json.dumps({'members': done, 'seconds': elapsed}), flush=True))
        with np.load(args.saved / 'ensemble-members.npz', allow_pickle=False) as z:
            differences = {k: float(np.max(np.abs(output[k].astype(float)-z[k])))
                           for k in ('center','pressure','basin','regional')}
        if differences['center'] > .0001 or max(differences[k] for k in ('pressure','basin','regional')) > .01:
            raise ValueError('Replay differs from original forecasts: ' + str(differences))
        means = mean_outputs(output)
    else:
        if not args.reference:
            raise ValueError('A one-member reference forecast is required')
        reference = json.loads(args.reference.read_text())
        if (reference['members'] != 1 or reference['checkpoint_sha256'] != CHECKPOINT
                or reference['storm_id'] != meta['storm_id']
                or ns(reference['issue_time_utc']) != times[-1]):
            raise ValueError('Wrong reference forecast identity')
        input_hash = hashlib.sha256(b''.join(inputs[k].numpy().tobytes() for k in sorted(inputs))).hexdigest()
        if input_hash != reference['input_tensor_sha256']:
            raise ValueError('Original issue input hash differs')
        model = model.to(args.device)
        with torch.inference_mode():
            state = model.initial({k: v.to(args.device) for k, v in inputs.items()})
            predictions = []
            for _ in range(20):
                state, pred = model.step(state)
                predictions.append(pred)
        center = np.stack([p['center'][0].cpu().numpy() for p in predictions])
        pressure = np.array([float(p['pressure'][0].cpu()) for p in predictions])
        expected = np.array([[p['lat'], p['lon'], p['pressure_hpa']] for p in reference['route'][1:]])
        differences = dict(center=float(np.max(np.abs(center-expected[:,:2]))),
                           pressure=float(np.max(np.abs(pressure-expected[:,2]))))
        if differences['center'] > .001 or differences['pressure'] > .05:
            raise ValueError('Replay does not reproduce the original cloud issue: ' + str(differences))
        means = dict(center=center, pressure=pressure)
        policy = {'member_count': 1, 'method': 'unperturbed deterministic release replay'}
    blocks = [{key: np.concatenate([chunk[i][key] for chunk in model.blocks])
               for key in model.blocks[0][i]} for i in range(21)]
    common_lat = np.linspace(60, 0, 241)
    common_lon = np.linspace(100, 180, 321)
    fields = np.stack([reconstruct(b, common_lat, common_lon, contract) for b in blocks[1:]])
    # Verify the export equation against the saved fixed regional reconstruction.
    equation_error = None
    if args.members == 50:
        rs = inputs['regional_static'][0].numpy()
        lat, lon = rs[0,:,0]*90, (rs[1,0,:]+1)*180
        # Only compare coordinates within the model basin; outside is border padding.
        ri = np.flatnonzero((lat >= 0) & (lat <= 60)); ci = np.flatnonzero((lon >= 100) & (lon <= 180))
        reconstructed = np.stack([reconstruct(b, lat[ri], lon[ci], contract) for b in blocks[1:]])
        equation_error = float(np.max(np.abs(reconstructed-means['regional'][:,ri][:,:,ci])))
        if equation_error > .02:
            raise ValueError('Geographic export does not match the network regional equation: ' + str(equation_error))
    raw = {}
    for key in ('anomaly','latitude','longitude','basin'):
        raw[key] = np.stack([b[key] for b in blocks], axis=1)
    np.savez_compressed(args.output / 'captured-core-members.npz', **raw)
    np.savez_compressed(args.output / 'common-pressure.npz', pressure_hpa=fields,
        latitude=common_lat, longitude=common_lon,
        initial_pressure_hpa=reconstruct(blocks[0], common_lat, common_lon, contract))
    source = dict(model='Trackformer 1.2', checkpoint_sha256=CHECKPOINT,
                  inference_weights_sha256=manifest['inference_weights_sha256'],
                  storm_id=meta['storm_id'], issue_time_utc=meta['issue_time_utc'], members=args.members,
                  input_file_sha256=sha(args.saved/'model-inputs.npz'),
                  original_replay_max_difference=differences, regional_equation_max_difference_hpa=equation_error,
                  ensemble_policy=policy,
                  method='Original model equation: bilinear basin + georeferenced model tapered moving anomaly; physical member fields registered before equal-weight mean.',
                  scalar_pressure_inserted=False, route_or_truth_alignment=False,
                  native_detail_history_available=False, model_core_information_spacing_km=20,
                  sampling_note='0.1-degree core export and 0.25-degree basin reconstruction are sampling, not additional observations.')
    patches = dict(**source, issue=local_patch(blocks[0], contract),
                   frames=[dict(**local_patch(b, contract), lead_hours=i*6,
                        valid_time_utc=utc(parse(meta['issue_time_utc']) + timedelta(hours=i*6)))
                        for i,b in enumerate(blocks[1:],1)])
    (args.output/'core-reconstruction.json').write_text(json.dumps(patches,separators=(',',':'),allow_nan=False)+'\n')
    (args.output/'verification.json').write_text(json.dumps(dict(**source,
        files_sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}),indent=2)+'\n')
    print(json.dumps({'complete':str(args.output), 'replay_difference':differences,
                      'regional_equation_error_hpa':equation_error}),flush=True)


if __name__ == '__main__':
    main()
