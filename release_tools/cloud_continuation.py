"""Fair, serialized continuation: never replace a pending live workflow."""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

WORKFLOWS = {'automatic-forecasts.yml', 'pressure-core-backfill.yml'}
LIVE_REFRESH_INTERVAL_SECONDS = 3600


def live_refresh_due(status, now):
    stamp = status.get('live_checked_at_utc')
    if not stamp:
        stamps = [r['issue_time_utc'] for r in status.get('live_issues', [])]
        stamp = max(stamps, default=None)
    if not stamp:
        return True
    checked = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    return (now-checked).total_seconds() >= LIVE_REFRESH_INTERVAL_SECONDS


def continuation_target(historical, core, wind, busy):
    if busy:
        return None  # GitHub concurrency replaces an existing pending job.
    if historical.get('historical_ready', 0) > 0 and historical.get('batch_succeeded', 0) > 0:
        return 'automatic-forecasts.yml'
    # A newly published historical issue is not yet included in old recovery
    # totals. Start its core/wind recovery rather than waiting six more hours.
    new_issues = historical.get('batch_succeeded', 0) > 0 and any(
        receipt.get('total', 0) < historical.get('historical_completed', 0)
        for receipt in (core, wind))
    if core.get('continue_ready') or wind.get('continue_ready') or new_issues:
        return 'pressure-core-backfill.yml'
    return None


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def recovery_batch_progress(core, wind, run_url):
    if not run_url or core.get('run_url') != run_url or wind.get('run_url') != run_url:
        raise ValueError('Missing current-run recovery status')
    attempted = max(core.get('batch_attempted', 0), wind.get('batch_attempted', 0))
    appended = core.get('batch_succeeded', 0) + wind.get('batch_succeeded', 0)
    remaining = max(core['total']-core['completed'], wind['total']-wind['completed'])
    if remaining and attempted > 0 and appended == 0:
        raise RuntimeError('Recovery stalled: all attempted replays failed; '
                           'partial verification receipts were published, no continuation queued')
    return dict(batch_attempted=attempted, appended_records=appended,
                remaining_issues=remaining, cooldown_only=remaining > 0 and attempted == 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--check-live', action='store_true')
    ap.add_argument('--check-recovery-progress', action='store_true')
    args = ap.parse_args()
    historical = read(args.output/'status.json')
    if args.check_live:
        print(str(live_refresh_due(historical, datetime.now(timezone.utc))).lower())
        return
    if args.check_recovery_progress:
        print(json.dumps(recovery_batch_progress(
            read(args.output/'pressure-cores/status.json'),
            read(args.output/'model-wind/status.json'), os.environ.get('RUN_URL'))))
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
