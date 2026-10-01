"""Fair, serialized continuation: never replace a pending live workflow."""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

WORKFLOWS = {'automatic-forecasts.yml', 'pressure-core-backfill.yml'}


def live_refresh_due(status, now):
    stamp = status.get('live_checked_at_utc')
    if not stamp:
        stamps = [r['issue_time_utc'] for r in status.get('live_issues', [])]
        stamp = max(stamps, default=None)
    if not stamp:
        return True
    checked = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    return (now-checked).total_seconds() >= 3*3600


def continuation_target(historical, core, wind, busy):
    if busy:
        return None  # GitHub concurrency replaces an existing pending job.
    if historical.get('historical_ready', 0) > 0 and historical.get('batch_succeeded', 0) > 0:
        return 'automatic-forecasts.yml'
    if core.get('continue_ready') or wind.get('continue_ready'):
        return 'pressure-core-backfill.yml'
    return None


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--check-live', action='store_true')
    args = ap.parse_args()
    historical = read(args.output/'status.json')
    if args.check_live:
        print(str(live_refresh_due(historical, datetime.now(timezone.utc))).lower())
        return
    repo = os.environ['GITHUB_REPOSITORY']
    runs = json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/actions/runs?per_page=100'], text=True))['workflow_runs']
    busy = [r for r in runs if str(r['id']) != os.environ['GITHUB_RUN_ID']
            and r['path'].split('/')[-1] in WORKFLOWS
            and r['status'] != 'completed']
    target = continuation_target(historical,
        read(args.output/'pressure-cores/status.json'), read(args.output/'model-wind/status.json'), busy)
    if not target:
        print('No duplicate continuation; pending runs:', [r['id'] for r in busy])
        return
    setting = 'historical_limit=500' if target == 'automatic-forecasts.yml' else 'limit=500'
    subprocess.run(['gh','workflow','run',target,'--repo',repo,'--ref','main','-f',setting],check=True)
    print('Queued one serialized continuation:', target)


if __name__ == '__main__':
    main()
