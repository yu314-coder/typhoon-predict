"""Frozen-release 50-member input-perturbation ensemble, not latent sampling.

Matches benchmark_ensemble50_field.py: seeds 2043..2092, normalized history
noise 0.025/0.015, 5/9-cell smoothing. No perturbation of origin, static
geography, missing-data flags, motion, or issue intensity. No future inputs.
"""
import hashlib
import time
import numpy as np
import torch
from torch.nn import functional as F

SEED_BASE = 2043
MEMBERS = 50


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def smooth_noise(raw, kernel):
    tensor = torch.from_numpy(raw.astype('float32', copy=False))
    shape = tensor.shape
    tensor = F.avg_pool2d(tensor.reshape(-1, 1, *shape[-2:]), kernel,
                          stride=1, padding=kernel // 2).reshape(shape)
    tensor = tensor - tensor.mean(dim=(-2, -1), keepdim=True)
    return (tensor / tensor.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)).numpy()


def member_inputs(base, member_ids):
    if base['center'].shape[0] != 1:
        raise ValueError('Live ensemble expects one storm issue')
    result = {k: np.repeat(v, len(member_ids), axis=0) for k, v in base.items()}
    for index, member in enumerate(member_ids):
        rng = np.random.default_rng(SEED_BASE + member)
        # Include the singleton case dimension to match the saved benchmark.
        g = rng.standard_normal((1, 9, 8, 25, 33), dtype=np.float32)
        r = rng.standard_normal((1, 9, 1, 121, 121), dtype=np.float32)
        result['global_history'][index] += smooth_noise(g, 5)[0] * .025
        result['regional_history'][index] += smooth_noise(r, 9)[0] * .015
    return result


def analysis_motion(parts):
    """Use current JMA Analysis motion only; never a future forecast point."""
    part = next(p for p in parts if isinstance(p.get('part'), dict)
                and p['part'].get('en') == 'Analysis' and p.get('advancedHours') == 0)
    directions = ['北','北北東','北東','東北東','東','東南東','南東','南南東',
                  '南','南南西','南西','西南西','西','西北西','北西','北北西']
    try:
        speed = float(part.get('speed', {}).get('km/h'))
        bearing = directions.index(part.get('course')) * 22.5
    except (ValueError, TypeError):
        return [0., 0.], {'available': False, 'note': 'Current analysis motion missing; zero input.'}
    if not 0 <= speed <= 150:
        raise ValueError('Invalid current analysis motion')
    radians = np.deg2rad(bearing)
    return [float(6*speed*np.sin(radians)), float(6*speed*np.cos(radians))], {
        'available': True, 'source_part': 'Analysis', 'speed_km_h': speed,
        'bearing_degrees': bearing, 'input_units': 'east/north km per six hours'}


def run_ensemble(model, inputs, contract, device='cpu', chunk=5, progress=None):
    if not 1 <= chunk <= 10:
        raise ValueError('Use a bounded ensemble chunk of 1..10')
    base = {k: v.detach().cpu().numpy() for k, v in inputs.items()}
    arrays = {key: [] for key in ('center','pressure','basin','regional','track_valid','vmax')}
    input_hashes = []
    started = time.monotonic()
    scale, offset = contract['normalization']['std'][0], contract['normalization']['mean'][0]
    model = model.eval().to(device)
    for first in range(0, MEMBERS, chunk):
        ids = list(range(first, min(first+chunk, MEMBERS)))
        perturbed = member_inputs(base, ids)
        input_hashes += [hashlib.sha256(b''.join(perturbed[k][i].tobytes()
                            for k in sorted(perturbed))).hexdigest() for i in range(len(ids))]
        tensors = {k: torch.from_numpy(v.astype('float32', copy=False)).to(device)
                   for k, v in perturbed.items()}
        steps = {k: [] for k in arrays}
        with torch.inference_mode():
            state = model.initial(tensors)
            for lead in range(20):
                state, pred = model.step(state)
                values = {'center': pred['center'], 'pressure': pred['pressure'],
                          'basin': pred['global'][:,0]*scale+offset,
                          'regional': pred['regional'][:,0]*scale+offset,
                          'track_valid': pred['track_valid'], 'vmax': pred['vmax']}
                for key, value in values.items():
                    a = value.detach().cpu().numpy()
                    if not np.isfinite(a).all():
                        raise ValueError(f'Non-finite member {first} at +{(lead+1)*6}h: {key}')
                    if key in ('pressure','basin') and (a.min()<800 or a.max()>1100):
                        raise ValueError('Unphysical member pressure')
                    steps[key].append(a)
        for key in arrays:
            arrays[key].append(np.stack(steps[key], axis=1))
        if progress:
            progress(len(input_hashes), MEMBERS, round(time.monotonic()-started, 2))
    output = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    route_hashes = [array_hash(a) for a in output['center']]
    field_hashes = [array_hash(a) for a in output['basin']]
    if any(len(set(hashes)) != MEMBERS for hashes in (input_hashes, route_hashes, field_hashes)):
        raise ValueError('Ensemble contains repeated members')
    policy = {'method': 'seeded causal input-perturbation ensemble, not latent samples',
              'member_count': MEMBERS, 'member_ids': list(range(MEMBERS)),
              'seeds': list(range(SEED_BASE, SEED_BASE+MEMBERS)), 'weights': [1/MEMBERS]*MEMBERS,
              'global_noise_normalized_std': .025, 'regional_noise_normalized_std': .015,
              'global_smoothing_kernel': 5, 'regional_smoothing_kernel': 9,
              'native_detail_available': False,
              'missing_native_history_remains_masked': True,
              'input_sha256': input_hashes, 'route_sha256': route_hashes,
              'basin_field_sha256': field_hashes,
              'mean_policy': 'Equal-weight physical basin and fixed-issue regional grids; member-mean coordinates and central pressures separately. Moving-core grids are not directly averaged.',
              'inference_seconds': round(time.monotonic()-started, 3)}
    return output, policy


def mean_outputs(output):
    return {k: output[k].mean(axis=0, dtype=np.float64).astype('float32')
            for k in ('center','pressure','basin','regional','vmax')}
