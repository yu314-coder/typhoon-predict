"""Reusable, unvalidated pressure-driven wind/structure candidate.

Not a change to released neural weights or pressure fields. The initialization
uses ONLY the exact issue-time USA 1-minute wind and reported quadrant radii.
Future pressure deficits come from each genuine forecast member. An initialized
outer wind envelope is scaled by sqrt(deficit / issue deficit), motivated by
gradient-wind/Holland balance. RMW is explicitly PERSISTENCE, not a measured or
learned future RMW. Missing observations are never invented as reference data.

Nominal targets match USA radius units, thresholds and quadrant-max geometry,
but this candidate has no validated 1-minute/10-m wind calibration. Strict
official wind/radius skill scores MUST remain disabled.
"""
from __future__ import annotations

from datetime import datetime
import math

import numpy as np

VERSION = 'pressure-scaled-issue-structure-experimental-v1'
QUADRANTS = ('NE', 'SE', 'SW', 'NW')
THRESHOLDS = (34, 50, 64)
METRICS = ('wind_estimate_kt', 'rmw_persistence_km') + tuple(
    f'r{t}_{q}_estimate_km' for t in THRESHOLDS for q in QUADRANTS)
MAX_RADIUS_KM = 1500.0
REFERENCES = (
    'https://noaa-ocs-modeling.github.io/PaHM/html/models.html',
    'https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf',
)
DEFINITIONS = dict(
    units='km', quantity_kind='radius_not_diameter',
    thresholds_kt=list(THRESHOLDS), quadrant_order=list(QUADRANTS),
    quadrant_statistic='maximum_extent_within_quadrant',
    target_wind_averaging_seconds=60,
    validated_wind_averaging_seconds=None,
    validated_surface_height_m=None,
    rmw='issue-time reported RMW carried forward unchanged; persistence baseline',
    official_scoring_allowed=False,
)


def number(row, key, low=0.0, high=10000.0):
    try:
        value = float(row.get(key, '').strip())
    except (TypeError, ValueError, AttributeError):
        return None
    return value if math.isfinite(value) and low <= value <= high else None


def radius_km(row, key, *, rmw=False):
    """IBTrACS USA radii are nautical-mile radii, not diameters."""
    value = number(row, key, 0.001 if rmw else 0.0, 999.0)
    return None if value is None else value * 1.852


def ambient_pressure(field, latitude, longitude, center):
    """Use actual coarse-model cells for ambient pressure, not inner-core size.

    Median available cells 600..1000 km from the predicted center; require
    >=2 physical cells in each quadrant. Record sample count/coverage. Missing
    quadrants, out-of-domain centers and nonphysical values fail closed.
    """
    p = np.asarray(field, dtype=float)
    lat, lon, c = (np.asarray(a, dtype=float) for a in (latitude, longitude, center))
    if p.shape != (len(lat), len(lon)) or c.shape != (2,):
        raise ValueError('Pressure grid/coordinate shape mismatch')
    if not np.isfinite(c).all() or not (0 < c[0] < 60 and 100 < c[1] < 180):
        return dict(value_hpa=None, status='outside_model_domain', cells=0)
    yy, xx = np.meshgrid(lat, lon, indexing='ij')
    a, b = np.deg2rad(yy-c[0]), np.deg2rad(xx-c[1])
    h = np.sin(a/2)**2 + np.cos(np.deg2rad(yy))*np.cos(np.deg2rad(c[0]))*np.sin(b/2)**2
    distance = 2*6371.0088*np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    mask = (600 <= distance) & (distance <= 1000)
    physical = np.isfinite(p) & (800 <= p) & (p <= 1100)
    qmask = ((yy >= c[0]) & (xx >= c[1]), (yy < c[0]) & (xx >= c[1]),
             (yy < c[0]) & (xx < c[1]), (yy >= c[0]) & (xx < c[1]))
    counts = [int((mask & physical & q).sum()) for q in qmask]
    usable = mask & physical
    result = dict(cells=int(usable.sum()), candidate_cells=int(mask.sum()),
                  quadrant_cells=dict(zip(QUADRANTS, counts)),
                  annulus_km=[600, 1000], statistic='median_available_physical_cells')
    if min(counts) < 2:
        return dict(result, value_hpa=None, status='insufficient_ambient_coverage')
    return dict(result, value_hpa=float(np.median(p[usable])), status='available')


def initialize(row, issue_time_utc, ambient_hpa):
    """Allowlisted fields from ONE exact issue row; no future labels accepted."""
    issue = datetime.fromisoformat(issue_time_utc.replace('Z', '+00:00'))
    if issue.utcoffset() is None or issue.utcoffset().total_seconds() != 0:
        raise ValueError('Issue time must be UTC')
    if row.get('ISO_TIME') != issue.strftime('%Y-%m-%d %H:%M:%S'):
        raise ValueError('Initialization is not the exact issue-time report')
    # Model pressure is initialized from TOKYO_PRES in the released runner.
    # USA wind/radii are a NEW diagnostic input, not silently injected into the
    # neural issue_intensity tensor, and not obtained from future observations.
    wind = number(row, 'USA_WIND', 0.01, 249.0)
    pressure = number(row, 'TOKYO_PRES', 800.0, 1100.0)
    rmw = radius_km(row, 'USA_RMW', rmw=True)
    pn = float(ambient_hpa) if ambient_hpa is not None else None
    if pn is not None and not (math.isfinite(pn) and 800 <= pn <= 1100):
        raise ValueError('Invalid issue ambient pressure')
    radii = {f'r{t}_{q}': radius_km(row, f'USA_R{t}_{q}')
             for t in THRESHOLDS for q in QUADRANTS}
    deficit = pn-pressure if pn is not None and pressure is not None else None
    b = None
    if wind is not None and deficit is not None and deficit > 1:
        # Fixed literature-motivated shape range, not selected on these storms.
        b = float(np.clip(1.15*math.e*(wind*.514444444444/.8)**2/(deficit*100), 1, 2.5))
    return dict(method=VERSION, storm_id=row.get('SID'), issue_time_utc=issue_time_utc,
                input_wind_kt=wind, input_model_pressure_hpa=pressure,
                ambient_pressure_hpa=pn, pressure_deficit_hpa=deficit,
                rmw_persistence_km=rmw, native_quadrant_radii_km=radii,
                holland_inspired_tail_b=b, source_agency=row.get('USA_AGENCY', '').strip(),
                source_track_type=row.get('TRACK_TYPE', '').strip(),
                usa_reference_center=[number(row, 'USA_LAT', -90, 90), number(row, 'USA_LON', -180, 180)],
                model_origin=[number(row, 'LAT', -90, 90), number(row, 'LON', -180, 180)],
                definitions=DEFINITIONS,
                mixed_agency_initialization='USA 1-minute wind/radii; model TOKYO pressure. No conversion between wind periods.',
                calibration='none; initialization is issue assimilation, not future-target fitting')


def envelope(initial, quadrant):
    """Strictly decreasing outer envelope; expose inconsistent issue reports."""
    vmax, rmw = initial['input_wind_kt'], initial['rmw_persistence_km']
    if vmax is None or rmw is None:
        return None, ['missing_issue_wind_or_rmw']
    points = [(float(rmw), float(vmax))]
    candidates, warnings = [], []
    for t in reversed(THRESHOLDS):
        r = initial['native_quadrant_radii_km'][f'r{t}_{quadrant}']
        if r is None:
            continue
        if r == 0:
            # Reported absence is not a blank. A constant-peak quadrant envelope
            # cannot honor an absent lower isotach: fail closed for this quadrant.
            if t <= vmax:
                return None, [f'reported_zero_r{t}_incompatible_with_constant_peak_envelope']
            continue
        if t >= vmax or r <= rmw:
            warnings.append(f'inconsistent_initial_r{t}_{quadrant}_not_assimilated')
            continue
        candidates.append((r, float(t)))
    for radius, value in sorted(candidates):
        if radius <= points[-1][0] or value >= points[-1][1]:
            warnings.append('nonmonotonic_issue_isotach_not_assimilated')
        else:
            points.append((radius, value))
    return points, warnings


def outer_radius(points, initial_b, initial_threshold):
    """Solve the analytic outer envelope; do not fabricate a grid-edge crossing."""
    if initial_threshold <= 0:
        return None
    if initial_threshold > points[0][1]:
        return 0.0  # complete candidate envelope never reaches the threshold
    for (r0, v0), (r1, v1) in zip(points[:-1], points[1:]):
        if v1 <= initial_threshold <= v0:
            fraction = math.log(initial_threshold/v0)/math.log(v1/v0)
            return math.exp(math.log(r0) + fraction*math.log(r1/r0))
    r, v = points[-1]
    # Holland's far-field wind asymptote V~r^(-B/2); this is an explicit
    # parametric extrapolation, not a native resolved wind or pressure field.
    result = r*(v/initial_threshold)**(2/initial_b)
    return float(result) if result <= MAX_RADIUS_KM else None


def diagnose(initial, pressure_hpa, ambient, *, track_valid=True):
    result = {key: None for key in METRICS}
    result.update(method=VERSION, experimental=True, status='unavailable', warnings=[],
                  official_scoring_allowed=False, ambient=ambient)
    if not track_valid or ambient['status'] == 'outside_model_domain':
        result['status'] = 'outside_model_domain_or_invalid_track'
        return result
    wind, d0 = initial['input_wind_kt'], initial['pressure_deficit_hpa']
    if wind is None or d0 is None or d0 <= 1:
        result['status'] = 'missing_issue_wind_or_positive_pressure_deficit'
        return result
    if ambient['value_hpa'] is None or not math.isfinite(pressure_hpa) or not 800 <= pressure_hpa <= 1100:
        result['status'] = 'missing_physical_model_pressure'
        return result
    future_deficit = ambient['value_hpa']-pressure_hpa
    if future_deficit <= 0:
        result['status'] = 'model_has_no_positive_vortex_pressure_deficit'
        return result
    scale = math.sqrt(future_deficit/d0)
    predicted_wind = wind*scale
    if not math.isfinite(predicted_wind) or predicted_wind >= 250:
        result['status'] = 'wind_outside_candidate_physical_range'
        return result
    result.update(wind_estimate_kt=float(predicted_wind), status='experimental_issue_anchored_estimate',
                  pressure_deficit_hpa=float(future_deficit), pressure_scaling=float(scale),
                  rmw_persistence_km=initial['rmw_persistence_km'])
    for quadrant in QUADRANTS:
        points, warnings = envelope(initial, quadrant)
        result['warnings'] += warnings
        if points is None:
            continue  # wind remains usable even if structure was not observed
        for threshold in THRESHOLDS:
            radius = outer_radius(points, initial['holland_inspired_tail_b'], threshold/scale)
            result[f'r{threshold}_{quadrant}_estimate_km'] = radius
            if radius is None:
                result['warnings'].append(f'r{threshold}_{quadrant}_beyond_{MAX_RADIUS_KM:g}_km_candidate_range')
    return result


def summarize(members, *, expected_members):
    if len(members) != expected_members or expected_members < 1:
        raise ValueError('Actual member diagnostics do not match the claimed count')
    summary = dict(method=VERSION, members=len(members), definitions=DEFINITIONS,
                   aggregation='diagnose each physical forecast member then equal-weight mean',
                   estimates={}, official_scoring_allowed=False)
    for key in METRICS:
        values = [m[key] for m in members if m[key] is not None and math.isfinite(m[key])]
        complete = len(values) == expected_members
        summary['estimates'][key] = dict(
            mean=float(np.mean(values)) if complete else None,
            p10=float(np.quantile(values, .1)) if complete else None,
            p90=float(np.quantile(values, .9)) if complete else None,
            valid_members=len(values), total_members=expected_members)
    return summary


def official_radius_mae(*_args, **_kwargs):
    raise ValueError('Strict official scoring blocked: no validated surface-wind averaging-period calibration; RMW is persistence')
