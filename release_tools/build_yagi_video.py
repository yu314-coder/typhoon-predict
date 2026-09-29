"""Render the selected saved Yagi 50-member forecast, without model inference.

Requires NumPy, Matplotlib and ffmpeg; --source-root prepares the public subset.
Never shift the model field or route to improve alignment with observations.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'evaluation/release_data'
CHECKPOINT = 'f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def distances(a, b):
    a, b = np.deg2rad(a), np.deg2rad(b)
    d = b - a
    h = np.sin(d[:, 0]/2)**2 + np.cos(a[:, 0])*np.cos(b[:, 0])*np.sin(d[:, 1]/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def prepare(root):
    source = root / 'benchmark_ensemble50/v173_e4/v173_e4_causal_ensemble50.npz'
    report_path = source.parent / 'report.json'
    report = json.loads(report_path.read_text())
    assert report['checkpoint_sha256'] == CHECKPOINT
    assert report['ensemble_policy']['member_count'] == 50
    assert len(set(report['ensemble_policy']['member_ids'])) == 50
    case = json.loads((DATA/'cohort_270.json').read_text())['cases'][178]
    assert case['storm_id'] == '2024244N09137' and case['issue_time_utc'] == '2024-09-03T00:00:00Z'
    with np.load(source, allow_pickle=False) as z:
        assert int(z['source_rows'][178]) == case['source_row']
        center = np.array([z['base_lat'][178], z['base_lon'][178]], dtype=float)
        local = z['ensemble50_local'][178].astype(float)
        route = np.column_stack((center[0]+local[:, 1]/111.2,
            center[1]+local[:, 0]/(111.2*max(np.cos(np.deg2rad(center[0])), .2))))
        pressure = z['ensemble50_pressure_hpa'][178].copy()
        native = z['ensemble50_native_mslp_hpa'][178].copy()
        coarse = z['ensemble50_mslp_hpa'][178].copy()
    plans = json.loads((root/'data/v164_reuse/plan.json').read_text())['rows']
    assert not any(p['track_archive_row'] == case['source_row'] for p in plans)
    # No stored native-detail history for this issue. Match the evaluator's
    # quarter-degree issue anchor; output sampling is not native resolution.
    anchor = np.round(center*4)/4
    lat, lon = anchor[0]+np.linspace(15, -15, 121), anchor[1]+np.linspace(-15, 15, 121)
    csvpath = root/'data/ibtracs/ibtracs.WP.list.v04r01.csv'
    with csvpath.open() as stream:
        rows = {r['ISO_TIME']: r for r in csv.DictReader(stream) if r['SID'] == case['storm_id']}
    issue = datetime.fromisoformat(case['issue_time_utc'].replace('Z', '+00:00'))
    obs, obs_pressure = [], []
    for k in range(21):
        row = rows[(issue+timedelta(hours=6*k)).strftime('%Y-%m-%d %H:%M:%S')]
        obs.append([float(row['LAT']), float(row['LON'])])
        value = row['TOKYO_PRES'].strip()
        obs_pressure.append(float(value) if value else np.nan)
    obs = np.asarray(obs)
    assert np.allclose(obs[0], center, atol=.001)
    truth_pressure = np.array(obs_pressure[1:])
    errors = distances(route, obs[1:])
    mask = np.isfinite(truth_pressure) & (truth_pressure > 800) & (truth_pressure < 1100)
    path = DATA/'yagi_video.npz'
    np.savez_compressed(path, forecast_lat_lon=np.vstack([center, route]), observed_lat_lon=obs,
        central_pressure_hpa=pressure, observed_pressure_hpa=truth_pressure,
        regional_pressure_hpa=native, basin_pressure_hpa=coarse, latitude=lat, longitude=lon,
        track_error_km=errors, original_local_route_km=local)
    rings = json.loads((root/'trackformer-weatherlab-site/public/data/history/coastlines.json').read_text())
    rings = [r for r in rings if any(100 <= x <= 125 and 12 <= y <= 26 for x, y in r)]
    (DATA/'yagi_coastlines.json').write_text(json.dumps(rings, separators=(',', ':'))+'\n')
    meta = {'model': 'Trackformer 1.2', 'storm': 'YAGI', **case, 'case_index': 178,
        'members': 50, 'ensemble_policy': report['ensemble_policy'], 'leads_hours': list(range(6, 121, 6)),
        'source_archive_sha256': sha(source), 'source_report_sha256': sha(report_path),
        'observed_csv_sha256': sha(csvpath), 'data_sha256': sha(path), 'checkpoint_sha256': CHECKPOINT,
        'observed_route_source': 'Exact six-hour IBTrACS LAT/LON for the same storm and valid timestamps.',
        'observed_pressure_source': 'IBTrACS TOKYO_PRES (JMA best-track central pressure), not an official forecast.',
        'route_projection': 'Saved mean local east/north km converted using issue latitude; no fitting, translation, rotation or scaling to observed route.',
        'field_source': 'Unmodified saved ensemble50_native_mslp_hpa and ensemble50_mslp_hpa on their common fixed grids.',
        'native_history_available': False, 'grid_note': 'Regional reconstruction sampled at 0.25 degrees; native-detail history unavailable for this case. Outside regional coverage use only saved 2.5-degree basin field.',
        'metrics': {'mean_track_error_km': float(errors.mean()), 'track_error_120h_km': float(errors[-1]),
            'maximum_track_error_km': float(errors.max()), 'within_200km_leads': int((errors<200).sum()),
            'central_pressure_mae_hpa': float(np.abs(pressure[mask]-truth_pressure[mask]).mean()),
            'pressure_valid_leads': int(mask.sum())},
        'selection': 'User-selected Yagi issue after inspecting geographic track alignment among saved development cases. Selected good route example, not representative performance or an untouched holdout. Pressure skill is assessed separately.'}
    (DATA/'yagi_video.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta['metrics'], indent=2), flush=True)


def build(work):
    work.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA/'yagi_video.json').read_text())
    assert sha(DATA/'yagi_video.npz') == meta['data_sha256']
    with np.load(DATA/'yagi_video.npz', allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    rings = json.loads((DATA/'yagi_coastlines.json').read_text())
    f, o, lat, lon = (a[k] for k in ('forecast_lat_lon', 'observed_lat_lon', 'latitude', 'longitude'))
    bounds = [102, 123, 14, 24]
    levels, contours = np.arange(976, 1017, 2), np.arange(976, 1017, 4)
    glat, glon, leads = np.linspace(60, 0, 25), np.linspace(100, 180, 33), np.arange(6, 121, 6)
    start = datetime.fromisoformat(meta['issue_time_utc'].replace('Z', '+00:00'))
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 15})
    for k, lead in enumerate(leads):
        fig = plt.figure(figsize=(16, 10), dpi=100, facecolor='#f8fafb')
        gs = fig.add_gridspec(2, 2, width_ratios=[3.9, 1.25], height_ratios=[4.5, 1.4],
            left=.055, right=.93, bottom=.155, top=.85, wspace=.25, hspace=.38)
        ax = fig.add_subplot(gs[0, :])
        ax.contourf(glon, glat, a['basin_pressure_hpa'][k], levels=levels, cmap='RdYlBu_r', extend='both')
        ax.contour(glon, glat, a['basin_pressure_hpa'][k], levels=contours, colors='#546971', linewidths=.45)
        im = ax.contourf(lon, lat, a['regional_pressure_hpa'][k], levels=levels, cmap='RdYlBu_r', extend='both')
        cs = ax.contour(lon, lat, a['regional_pressure_hpa'][k], levels=contours, colors='#354b58', linewidths=.65)
        ax.clabel(cs, levels=contours[::2], fmt='%d', fontsize=9)
        for ring in rings:
            xy = np.asarray(ring); ax.plot(xy[:, 0], xy[:, 1], c='#66716b', lw=.8)
        ax.add_patch(Rectangle((lon.min(), lat.min()), np.ptp(lon), np.ptp(lat), fill=False, edgecolor='#265b61', lw=1.2, ls=':'))
        for points, color, label, style in [(o, '#182f42', 'Observed best track', '--'), (f, '#b31565', 'Trackformer 1.2 mean of 50', '-')]:
            ax.plot(points[:, 1], points[:, 0], ls=style, c=color, alpha=.2, lw=1.6)
            ax.plot(points[:k+2, 1], points[:k+2, 0], ls=style, c=color, lw=2.8, label=label)
            ax.scatter(points[k+1, 1], points[k+1, 0], s=75, c=color, edgecolors='white', linewidths=1.8, zorder=6)
        ax.plot([f[k+1, 1], o[k+1, 1]], [f[k+1, 0], o[k+1, 0]], ':', c='#7a5a85', lw=1.4)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='Longitude °E', ylabel='Latitude °N')
        ax.set_aspect(1/np.cos(np.deg2rad(19))); ax.grid(alpha=.13)
        ax.legend(loc='lower left', fontsize=11, framealpha=.95)
        ax.text(.985, .035, 'Same map / same valid time\nNo route shifting or rescaling', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': 5})
        fig.colorbar(im, cax=fig.add_axes([.952, .40, .014, .40]), label='Model mean sea-level pressure (hPa)')
        curve = fig.add_subplot(gs[1, 0])
        curve.plot(leads, a['central_pressure_hpa'], c='#b31565', lw=2, label='Model')
        curve.plot(leads, a['observed_pressure_hpa'], '--', c='#182f42', lw=2, label='Observed (JMA best track)')
        curve.axvline(lead, c='#688e98', lw=1.5)
        curve.scatter([lead], [a['central_pressure_hpa'][k]], c='#b31565', s=40, zorder=4)
        curve.set(xlim=(6, 120), xlabel='Forecast lead (hours)', ylabel='Central pressure (hPa)', xticks=[6, 24, 48, 72, 96, 120])
        curve.grid(alpha=.2); curve.legend(loc='lower left', fontsize=10)
        info = fig.add_subplot(gs[1, 1]); info.axis('off')
        info.text(0, 1, f"Position error: {a['track_error_km'][k]:.0f} km\nModel: {a['central_pressure_hpa'][k]:.1f} hPa\nObserved: {a['observed_pressure_hpa'][k]:.0f} hPa\n\nSelected good-route example\nPressure intensity is too weak", va='top', fontsize=12, color='#244754')
        valid = start+timedelta(hours=int(lead))
        fig.suptitle(f'Trackformer 1.2  /  YAGI  /  +{lead:03d} h', x=.055, y=.97, ha='left', fontsize=24, fontweight='bold', color='#173947')
        fig.text(.055, .918, f'Issue: {start:%d %b %Y %H:%M} UTC     |     Valid: {valid:%d %b %Y %H:%M} UTC     |     50-member mean', fontsize=14, color='#385865')
        fig.text(.055, .067, 'Actual model pressure with 4 hPa isobars. Observations are verification only; selected example, not typical skill.', fontsize=11, color='#47636d')
        fig.text(.055, .040, 'Dotted boundary: fixed regional reconstruction (no native-detail history). Outside: coarse basin field only.', fontsize=10, color='#47636d')
        fig.savefig(work/f'frame_{k:03d}.png', dpi=100)
        if k == 9: fig.savefig(REPO/'evaluation/trackformer_1_2_yagi_video_poster.png', dpi=100)
        plt.close(fig)
    output = REPO/'docs/trackformer_1_2_yagi.mp4'
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-framerate', '1', '-i', str(work/'frame_%03d.png'),
        '-vf', 'fps=30,tpad=stop_mode=clone:stop_duration=3', '-c:v', 'h264_videotoolbox', '-b:v', '4M', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(output)], check=True)
    meta['video'] = {'file': 'docs/trackformer_1_2_yagi.mp4', 'duration_seconds': 23, 'forecast_states': 20,
        'codec': 'H.264', 'encoder': 'h264_videotoolbox', 'sha256': sha(output),
        'playback': 'Each six-hour state held for 1 second; final frame held 3 seconds extra. No interpolated forecast states.'}
    (DATA/'yagi_video.json').write_text(json.dumps(meta, indent=2)+'\n')
    print('Saved', output, output.stat().st_size, 'bytes', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path)
    p.add_argument('--work', type=Path, required=True)
    args = p.parse_args()
    if args.source_root: prepare(args.source_root)
    build(args.work)
