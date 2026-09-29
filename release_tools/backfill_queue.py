"""Small, dependency-free continuation policy for the hosted archive."""
from datetime import datetime, timedelta


def retry_is_cooling(error, now, input_version=None):
    # A corrected input reader gets one immediate retry; new failures back off.
    # This must match both execution and status counts.
    if input_version is not None and error.get('input_version') != input_version:
        return False
    try:
        failed = datetime.fromisoformat(error['at'].replace('Z', '+00:00'))
        return now - failed < timedelta(hours=24)
    except (KeyError, TypeError, ValueError):
        return True


def queue_state(planned, done, errors, now, input_version=None):
    remaining = set(planned) - set(done)
    cooling = set()
    for ident in remaining:
        if ident not in errors:
            continue
        if retry_is_cooling(errors[ident], now, input_version):
            cooling.add(ident)
    return {'historical_remaining': len(remaining),
            'historical_ready': len(remaining - cooling),
            'historical_cooling_down': len(cooling)}


def should_continue(status, run_url, only_id=''):
    return (not only_id and bool(run_url) and status.get('run_url') == run_url
            and status.get('batch_succeeded', 0) > 0
            and status.get('historical_ready', 0) > 0)
