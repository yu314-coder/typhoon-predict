"""Cheap cloud-only watchdog; inference still uses the existing serialized writer."""
import json
import os
import subprocess
from datetime import datetime, timezone

from cloud_continuation import WORKFLOWS, live_refresh_due


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
