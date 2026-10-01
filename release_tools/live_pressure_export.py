"""Export the frozen model's physical moving cores on geographic coordinates.

Capture only: no change to forward equations, weights, trajectories or pressure
readouts. Ensemble anomalies are registered BEFORE their equal-weight mean.
"""
import base64
import hashlib
import numpy as np
from recover_pressure_core import CaptureModel, reconstruct

METHOD = 'model-geographic-common-grid-core-mean-v1'


def capturing_model(model, contract):
    result = CaptureModel(contract).eval()
    result.load_state_dict(model.state_dict(), strict=True)
    return result


def member_blocks(chunks, members):
    if not chunks or any(len(chunk) != 21 for chunk in chunks):
        raise ValueError('Missing captured +0 through +120 h states')
    blocks = [{key: np.concatenate([chunk[i][key] for chunk in chunks])
               for key in ('anomaly', 'latitude', 'longitude', 'basin')}
              for i in range(21)]
    if any(len(block['anomaly']) != members for block in blocks):
        raise ValueError('Wrong captured member count')
    return blocks


def encoded_mean(block, contract):
    # A shared geographic lattice, never index-wise/recentered native patches.
    # Bound memory and the public codec to 25,000 cells. .25 degrees is an export
    # sampling interval, NOT additional information beyond the native 20 km.
    for step in (.25, .5):
        north = min(60, np.ceil(block['latitude'].max()/step)*step)
        south = max(0, np.floor(block['latitude'].min()/step)*step)
        west = max(100, np.floor(block['longitude'].min()/step)*step)
        east = min(180, np.ceil(block['longitude'].max()/step)*step)
        lat = np.round(np.arange(north, south-step/10, -step), 6)
        lon = np.round(np.arange(west, east+step/10, step), 6)
        if len(lat)*len(lon) <= 25000:
            break
    if len(lat) < 2 or len(lon) < 2:
        return dict(available=False, reason='All moving cores outside basin support')
    field = reconstruct(block, lat, lon, contract)
    integers = np.rint((field.astype(float)-1000)*100).astype('int32').ravel()
    delta = np.diff(np.r_[0, integers])
    if delta.min() < -32768 or delta.max() > 32767:
        raise ValueError('Pressure storage delta overflow')
    return dict(available=True, latitude=lat.tolist(), longitude=lon.tolist(),
                sampling_degrees=step, native_information_km=20,
                pressure_encoding=dict(format='delta-int16-le-base64',
                    shape=list(field.shape), scale_hpa=.01, offset_hpa=1000),
                pressure_delta_base64=base64.b64encode(delta.astype('<i2').tobytes()).decode())


def core_export(chunks, forecast, contract):
    blocks = member_blocks(chunks, forecast['members'])
    route = forecast['route']
    if len(route) != 21:
        raise ValueError('Incomplete core valid times')
    hashes = [hashlib.sha256(b''.join(block[key][member].tobytes()
                for block in blocks for key in ('anomaly','latitude','longitude','basin'))).hexdigest()
              for member in range(forecast['members'])]
    if forecast['members'] == 50 and len(set(hashes)) != 50:
        raise ValueError('Repeated captured ensemble members')
    policy = dict(member_count=forecast['members'],
                  weights=[1/forecast['members']]*forecast['members'],
                  member_state_sha256=hashes,
                  registration='Each member basin plus tapered anomaly on actual geographic coordinates before averaging; no route shift',
                  scalar_pressure_is_field_minimum=False)
    return dict(schema_version='1.0', model='Trackformer 1.2',
        forecast_id=forecast['id'], storm_id=forecast['storm_id'],
        checkpoint_sha256=forecast['checkpoint_sha256'], members=forecast['members'],
        input_tensor_sha256=forecast['input_tensor_sha256'],
        issue_time_utc=forecast['issue_time_utc'], units='hPa', method=METHOD,
        scalar_pressure_inserted=False, route_or_truth_alignment=False,
        native_history_available=False, common_grid_policy=policy,
        issue={**encoded_mean(blocks[0], contract), 'valid_time_utc':route[0]['valid_time_utc'], 'lead_hours':0},
        frames=[{**encoded_mean(blocks[i+1], contract),
                 'valid_time_utc':route[i+1]['valid_time_utc'], 'lead_hours':(i+1)*6}
                for i in range(20)])
