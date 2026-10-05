"""Append-only export of the released model's learned vmax head, not a wind fit.

Only call export_wind after the existing causal replay audit has passed. This
does not derive wind from isobars or substitute observed future intensity.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone

import numpy as np
from immutable_basin_field import verify_basin_source

METHOD = 'released-auxiliary-vmax-export-v1'
CHECKPOINT = 'f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0'
LIMITS = {'route_degrees': .001, 'core_pressure_hpa': .05, 'basin_pressure_hpa': .006}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse(stamp):
    return datetime.fromisoformat(stamp.replace('Z', '+00:00'))


def limits(backend):
    if backend not in ('cpu', 'mps'):
        raise ValueError('Unknown wind replay backend')
    return dict(LIMITS, route_degrees=.002 if backend == 'mps' else .001)


def export_wind(reference, predictions, source_hashes, replay_difference, backend='cpu'):
    if reference['members'] != 1 or reference['checkpoint_sha256'] != CHECKPOINT or len(predictions) != 20:
        raise ValueError('Wind replay must retain the original single-member identity')
    tolerance = limits(backend)
    if set(replay_difference) != set(tolerance) or any(not np.isfinite(replay_difference[k]) or
            replay_difference[k] > tolerance[k] or replay_difference[k] < 0 for k in tolerance):
        raise ValueError('Unaudited wind replay')
    points = []
    for i, prediction in enumerate(predictions):
        original = reference['route'][i+1]
        wind = prediction['vmax'].detach().cpu().numpy()
        center = prediction['center'].detach().cpu().numpy()
        if wind.shape != (1,) or center.shape != (1, 2):
            raise ValueError('Wrong model wind member count')
        value = float(wind[0])
        inside = all(0 < lat < 60 and 100 < lon < 180
                     for lat, lon in (center[0], (original['lat'], original['lon'])))
        reason = 'available' if inside and np.isfinite(value) and 0 <= value < 250 else (
            'outside_model_domain' if not inside else 'invalid_model_wind')
        saved = original.get('wind_kt_auxiliary')
        if saved is not None and (not np.isfinite(value) or abs(value-saved) > .05):
            raise ValueError('Replay changed an already saved model wind')
        points.append(dict(lead_hours=original['lead_hours'], valid_time_utc=original['valid_time_utc'],
            lat=original['lat'], lon=original['lon'], wind_kt_auxiliary=value if reason == 'available' else None,
            wind_kt_auxiliary_valid=reason == 'available', valid_members=int(reason == 'available'),
            total_members=1, reason=reason))
    return dict(schema_version='1.0', model='Trackformer 1.2', method=METHOD,
        forecast_id=reference['id'], storm_id=reference['storm_id'],
        issue_time_utc=reference['issue_time_utc'], checkpoint_sha256=CHECKPOINT, members=1,
        input_tensor_sha256=reference['input_tensor_sha256'], units='kt',
        wind_averaging_period='unvalidated; not agency-matched 1-minute or 10-minute wind',
        experimental=True, pressure_derived=False, observed_future_wind_used=False,
        source_hashes=source_hashes, execution_backend=backend,
        replay_max_difference=replay_difference, replay_tolerance=tolerance,
        points=points, note='Actual released neural auxiliary vmax at +6 through +120 h. '
        'Same pinned causal inputs and checkpoint, checked against immutable route, pressure and basin outputs. '
        'No training or fitted correction. Original forecast files unchanged; unsupported values remain null.')


def verify_wind(document, output, planned, input_manifest_hash, weights_hash):
    ident = document['forecast_id']
    if (ident not in planned or document['members'] != 1 or document['checkpoint_sha256'] != CHECKPOINT
            or document['method'] != METHOD or document['units'] != 'kt'
            or document['experimental'] is not True or document['pressure_derived'] is not False
            or document['observed_future_wind_used'] is not False):
        raise ValueError('Wrong model wind identity or source')
    tolerance = limits(document['execution_backend'])
    diff = document['replay_max_difference']
    if document['replay_tolerance'] != tolerance or set(diff) != set(tolerance) or any(
            not np.isfinite(diff[k]) or not 0 <= diff[k] <= tolerance[k] for k in tolerance):
        raise ValueError('Wind replay audit failed')
    forecast = output/'forecasts'/f'{ident}.json'
    source = document['source_hashes']
    if (sha(forecast) != source['forecast_sha256']
            or input_manifest_hash != source['input_manifest_sha256'] or weights_hash != source['weights_sha256']):
        raise ValueError('Immutable wind source hashes changed')
    verify_basin_source(output, ident, source)
    reference = json.loads(forecast.read_text())
    row = planned[ident]
    if (reference['id'] != ident or reference['members'] != 1 or reference['checkpoint_sha256'] != CHECKPOINT
            or reference['storm_id'] != document['storm_id'] or row['storm_id'] != reference['storm_id']
            or parse(row['issue_time_utc']) != parse(document['issue_time_utc'])
            or reference['input_tensor_sha256'] != document['input_tensor_sha256']
            or parse(reference['issue_time_utc']) != parse(document['issue_time_utc'])):
        raise ValueError('Wrong wind causal input/plan identity')
    if len(document['points']) != 20 or len(reference['route']) != 21:
        raise ValueError('Incomplete wind forecast')
    for i, point in enumerate(document['points']):
        old = reference['route'][i+1]
        if (point['lead_hours'] != (i+1)*6 or point['lead_hours'] != old['lead_hours']
                or parse(point['valid_time_utc']) != parse(old['valid_time_utc'])
                or parse(point['valid_time_utc']) != parse(reference['issue_time_utc'])+timedelta(hours=(i+1)*6)
                or point['lat'] != old['lat'] or point['lon'] != old['lon'] or point['total_members'] != 1):
            raise ValueError('Wind lead/route identity mismatch')
        value = point['wind_kt_auxiliary']
        valid = point['wind_kt_auxiliary_valid']
        if not isinstance(valid, bool) or point['valid_members'] != int(valid):
            raise ValueError('Wind member mask mismatch')
        if valid:
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value)
                    or not 0 <= value < 250 or point['reason'] != 'available'
                    or not (0 < point['lat'] < 60 and 100 < point['lon'] < 180)):
                raise ValueError('Invalid model wind value')
            if old.get('wind_kt_auxiliary') is not None and abs(value-old['wind_kt_auxiliary']) > .05:
                raise ValueError('Saved wind differs')
        elif value is not None or point['reason'] not in ('outside_model_domain', 'invalid_model_wind'):
            raise ValueError('Missing wind was zero-filled')
