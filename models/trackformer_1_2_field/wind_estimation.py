"""Uncalibrated pressure-gradient diagnostics, not new neural weights.

Diagnose each physical member before ensemble aggregation. These are circular,
ocean-assumption estimates, not resolved 10-m winds or official quadrant radii.
No observed future intensity/radii, fitted showcase corrections, or mean-field
substitutions are accepted. See WIND_ESTIMATION.md for assumptions/limitations.
"""
import numpy as np

VERSION = 'pressure-gradient-experimental-v1'
KEYS = ('maximum_wind_auxiliary_kt', 'pressure_wind_estimate_kt',
        'rmw_estimate_km', 'r34_estimate_km', 'r50_estimate_km', 'r64_estimate_km')
ASSUMPTIONS = dict(air_density_kg_m3=1.15, ocean_surface_reduction=0.8,
                   maximum_radius_km=480, azimuth_samples=72,
                   radial_smoothing='three-point pressure average',
                   geometry='azimuthal mean; circular equivalent radii, not quadrants',
                   wind_averaging_period='not calibrated to an agency averaging period',
                   land_friction_asymmetry_and_translation='not modelled',
                   calibration='none; fixed physical assumptions, not fitted to benchmarks')


def unavailable(reason, auxiliary_wind=None):
    result = {key: None for key in KEYS}
    if auxiliary_wind is not None and np.isfinite(auxiliary_wind) and 0 <= auxiliary_wind < 250:
        result[KEYS[0]] = float(auxiliary_wind)
    return dict(result, status=reason, experimental=True, method=VERSION)


def diagnose_member(pressure_hpa, latitude, longitude, center, auxiliary_wind=None,
                    *, source_spacing_km, source='model pressure reconstruction',
                    sampling_radius_limit_km=None):
    """Bilinear polar sampling on a rectilinear *physical member* pressure grid.

    source_spacing_km is underlying information spacing, not upsampled pixel
    spacing. Coarse (>50 km) fields cannot support these inner-core diagnostics.
    Missing rings, domain exits, inner unresolved peaks and outer non-crossings
    remain missing; the last sampled radius is never passed off as a crossing.
    """
    result = unavailable('unresolved', auxiliary_wind)
    result.update(source=source, source_spacing_km=float(source_spacing_km))
    p = np.asarray(pressure_hpa, dtype=float)
    lat, lon = np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
    c = np.asarray(center, dtype=float)
    if p.shape != (len(lat), len(lon)) or len(lat) < 3 or len(lon) < 3:
        raise ValueError('Expected a physical rectilinear pressure grid')
    if not np.isfinite(source_spacing_km) or source_spacing_km <= 0:
        raise ValueError('Underlying source spacing must be known')
    if not np.isfinite(c).all() or not (0 < c[0] < 60 and 100 < c[1] < 180):
        result['status'] = 'outside_model_domain'
        result[KEYS[0]] = None
        return result
    if source_spacing_km > 50:
        result['status'] = 'underlying_grid_too_coarse'
        return result
    if lat[1] < lat[0]: lat, p = lat[::-1], p[::-1]
    if lon[1] < lon[0]: lon, p = lon[::-1], p[:, ::-1]
    if not np.allclose(np.diff(lat), np.diff(lat)[0], rtol=1e-3, atol=1e-5) or not np.allclose(np.diff(lon), np.diff(lon)[0], rtol=1e-3, atol=1e-5):
        raise ValueError('Expected a regular latitude/longitude grid')
    step = max(20., source_spacing_km,
               float(np.diff(lat)[0])*111.2,
               float(np.diff(lon)[0])*111.2*np.cos(np.deg2rad(c[0])))
    if step > 50:
        result['status'] = 'sampling_grid_too_coarse'
        return result
    p = np.where(np.isfinite(p) & (p >= 800) & (p <= 1100), p, np.nan)
    radius_limit = ASSUMPTIONS['maximum_radius_km']
    if sampling_radius_limit_km is not None:
        if not np.isfinite(sampling_radius_limit_km) or sampling_radius_limit_km <= 0:
            raise ValueError('A positive supported sampling radius is required')
        radius_limit = min(radius_limit, sampling_radius_limit_km)
    result['supported_sampling_radius_km'] = float(radius_limit)
    radii = np.arange(0., radius_limit + .1, step)
    theta = np.arange(ASSUMPTIONS['azimuth_samples'])*2*np.pi/ASSUMPTIONS['azimuth_samples']
    ys = c[0] + radii[:, None]*np.sin(theta)/111.2
    xs = c[1] + radii[:, None]*np.cos(theta)/(111.2*np.cos(np.deg2rad(c[0])))
    fy, fx = (ys-lat[0])/(lat[1]-lat[0]), (xs-lon[0])/(lon[1]-lon[0])
    inside = (fy >= 0) & (fx >= 0) & (fy <= len(lat)-1) & (fx <= len(lon)-1)
    iy = np.clip(np.floor(fy).astype(int), 0, len(lat)-2)
    ix = np.clip(np.floor(fx).astype(int), 0, len(lon)-2)
    dy, dx = fy-iy, fx-ix
    samples = (p[iy, ix]*(1-dy)*(1-dx) + p[iy+1, ix]*dy*(1-dx) +
               p[iy, ix+1]*(1-dy)*dx + p[iy+1, ix+1]*dy*dx)
    usable = inside.all(1) & np.isfinite(samples).all(1)
    missing = np.flatnonzero(~usable)
    stop = int(missing[0]) if len(missing) else len(radii)
    if stop < 7:
        result['status'] = 'insufficient_complete_ring_coverage'
        return result
    radii, profile = radii[:stop], samples[:stop].mean(1)*100  # hPa -> Pa
    smooth = profile.copy()
    smooth[1:-1] = (profile[:-2] + profile[1:-1] + profile[2:])/3
    # Vg^2/r + |f| Vg = (1/rho) dp/dr. Estimate ocean surface-equivalent speed.
    derivative = np.gradient(smooth, radii*1000)
    f = abs(2*7.292115e-5*np.sin(np.deg2rad(c[0])))
    r = radii*1000
    rho = ASSUMPTIONS['air_density_kg_m3']
    reduction = ASSUMPTIONS['ocean_surface_reduction']
    velocity = (np.sqrt((f*r/2)**2 + r*np.maximum(derivative, 0)/rho) - f*r/2)*reduction
    knots = velocity/0.514444444444
    # Discard the endpoint derivative; require >=2 real cells for a peak.
    peak = int(np.argmax(knots[1:-1])) + 1
    result.update(radial_step_km=float(step), complete_ring_limit_km=float(radii[-2]))
    if peak < 2 or peak >= len(radii)-2 or knots[peak] < 1:
        result['status'] = 'maximum_wind_not_resolved'
        return result
    result.update(pressure_wind_estimate_kt=float(knots[peak]),
                  rmw_estimate_km=float(radii[peak]), status='experimental_estimate')
    for threshold in (34, 50, 64):
        key = f'r{threshold}_estimate_km'
        if knots[peak] < threshold:
            continue  # unresolved threshold, not a demonstrated zero-radius storm
        crossing = next((i for i in range(peak+1, len(radii)-1)
                         if knots[i] < threshold <= knots[i-1]), None)
        if crossing is not None:
            fraction = (threshold-knots[crossing-1])/(knots[crossing]-knots[crossing-1])
            result[key] = float(radii[crossing-1] + fraction*step)
    return result


def summarize_members(members):
    """Require all members for a labelled ensemble mean; disclose censoring."""
    if not members:
        raise ValueError('Actual member diagnostics are required')
    summary = dict(method=VERSION, experimental=True, members=len(members),
                   aggregation='diagnose each member, then equal-weight mean; never diagnose the mean pressure',
                   assumptions=ASSUMPTIONS, estimates={},
                   status_counts={status: sum(m['status'] == status for m in members)
                                  for status in sorted({m['status'] for m in members})})
    for key in KEYS:
        values = [m[key] for m in members if m[key] is not None and np.isfinite(m[key])]
        complete = len(values) == len(members)
        summary['estimates'][key] = dict(mean=float(np.mean(values)) if complete else None,
            p10=float(np.quantile(values, .1)) if complete else None,
            p90=float(np.quantile(values, .9)) if complete else None,
            valid_members=len(values), total_members=len(members))
    return summary


def diagnose_outputs(output, contract):
    """Extract physical moving-core members without touching weights or state."""
    array = lambda tensor: tensor.detach().cpu().numpy()
    center, wind = array(output['center']), array(output['vmax'])
    if not all(key in output for key in ('core', 'core_lat', 'core_lon')):
        return [unavailable('member_core_not_saved', value) for value in wind]
    pressure = array(output['core'])[:, 0]*contract['normalization']['std'][0]+contract['normalization']['mean'][0]
    if 'core_valid' in output:
        pressure = np.where(array(output['core_valid'])[:,0], pressure, np.nan)
    lat, lon = array(output['core_lat']), array(output['core_lon'])
    return [diagnose_member(pressure[i], lat[i,:,0], lon[i,0,:], center[i], wind[i],
                           source_spacing_km=20, source='physical 20-km learned moving-core reconstruction')
            for i in range(len(center))]
