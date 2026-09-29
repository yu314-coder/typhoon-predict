"""Small, dependency-free continuation policy for the hosted archive."""
from datetime import datetime, timedelta


def queue_state(planned, done, errors, now):
    remaining = set(planned) - set(done)
    cooling = set()
    for ident in remaining:
        if ident not in errors:
            continue
        try:
            failed = datetime.fromisoformat(errors[ident]['at'].replace('Z', '+00:00'))
            if now - failed < timedelta(hours=24):
                cooling.add(ident)
        except (KeyError, TypeError, ValueError):
            # Unknown failure provenance is not permission for a tight retry loop.
            cooling.add(ident)
    return {'historical_remaining': len(remaining),
            'historical_ready': len(remaining - cooling),
            'historical_cooling_down': len(cooling)}


def should_continue(status, run_url, only_id=''):
    return (not only_id and bool(run_url) and status.get('run_url') == run_url
            and status.get('batch_succeeded', 0) > 0
            and status.get('historical_ready', 0) > 0)
