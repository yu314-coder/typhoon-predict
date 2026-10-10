"""Cheap cloud-only watchdog; inference still uses the existing serialized writer."""
import json
import os
import subprocess
from datetime import datetime, timezone

from cloud_continuation import WORKFLOWS, live_refresh_due
from backfill_queue import retry_is_cooling


def historical_work_due(status, now):
    remaining = status.get('historical_remaining')
    if remaining is None:
        remaining = max(0, status.get('historical_total', 0)-status.get('historical_completed', 0))
    if remaining <= 0:
        return False
    if status.get('historical_ready', 0) > 0:
        return True
    errors = {k: v for k, v in status.get('errors', {}).items()
              if k.startswith(('auto-hist-', 'auto-tick-'))}
    # Published ready/cooling counts are snapshots. Re-evaluate retry age so a
    # stopped continuation is recovered when its 24-hour cooldown has elapsed.
    return len(errors) < remaining or any(
        not retry_is_cooling(e, now, status.get('historical_input_version'))
        for e in errors.values())


def decision(status, runs, current_id, event, schedule, now):
    busy = [r['id'] for r in runs
            if str(r['id']) != str(current_id)
            and r.get('path', '').split('@')[0].split('/')[-1] in WORKFLOWS
            and r['status'] != 'completed']
    if busy:
        return False, 'Existing archive writer active or queued: ' + ', '.join(map(str, busy))
    if event != 'schedule':
        return True, 'Explicit workflow request; no competing writer'
    if schedule == '17 * * * *':
        return True, 'Primary hourly live check'
    if historical_work_due(status, now):
        return True, 'Resume missing planned history or expired retry cooldown'
    try:
        due = live_refresh_due(status, now)
        stamp = status.get('live_checked_at_utc')
        if stamp and datetime.fromisoformat(stamp.replace('Z', '+00:00')) > now:
            due = True  # An invalid future receipt cannot suppress updates.
    except (ValueError, TypeError, KeyError):
        due = True
    return due, 'Recover overdue hourly check' if due else 'Cloud checked within the last hour; no inference needed'


def main():
    repo = os.environ['GITHUB_REPOSITORY']
    runs = json.loads(subprocess.check_output(
        ['gh', 'api', f'repos/{repo}/actions/runs?per_page=100'], text=True))['workflow_runs']
    status = json.loads(subprocess.check_output(
        ['gh', 'api', f'repos/{repo}/contents/data/status.json?ref=forecast-data',
         '-H', 'Accept: application/vnd.github.raw+json'], text=True))
    run, reason = decision(status, runs, os.environ['GITHUB_RUN_ID'],
                           os.environ['GITHUB_EVENT_NAME'], os.environ.get('SCHEDULE', ''),
                           datetime.now(timezone.utc))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'run={str(run).lower()}\n')
    print(json.dumps({'run': run, 'reason': reason,
                      'last_cloud_check': status.get('live_checked_at_utc')}))


if __name__ == '__main__':
    main()
