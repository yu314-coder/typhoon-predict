import unittest
from datetime import datetime, timezone
from live_schedule_gate import decision, historical_work_due


class LiveScheduleGateTests(unittest.TestCase):
    def check(self, status=None, runs=None, event='schedule', schedule='32,47,57 * * * *'):
        return decision(status or {}, runs or [], 10, event, schedule,
                        datetime(2026, 10, 3, 7, tzinfo=timezone.utc))[0]

    def test_watchdog_skips_recent_check_and_recovers_overdue_check(self):
        self.assertFalse(self.check({'live_checked_at_utc': '2026-10-03T06:30:00Z'}))
        self.assertTrue(self.check({'live_checked_at_utc': '2026-10-03T06:00:00Z'}))
        self.assertTrue(self.check({'live_checked_at_utc': '2026-10-03T05:30:00Z'}))

    def test_missing_bad_or_future_receipt_does_not_hide_a_stall(self):
        for stamp in [None, 'bad', '2026-10-03T09:00:00Z']:
            self.assertTrue(self.check({'live_checked_at_utc': stamp}))

    def test_history_resumes_while_live_is_fresh(self):
        fresh = {'live_checked_at_utc': '2026-10-03T06:55:00Z',
                 'historical_remaining': 1, 'historical_ready': 1}
        self.assertTrue(self.check(fresh))

    def test_expired_cooldowns_are_recomputed_from_receipts(self):
        status = {'historical_remaining': 1, 'historical_ready': 0,
                  'historical_input_version': 'v1',
                  'errors': {'auto-tick-test': {'at': '2026-10-02T06:00:00Z',
                                               'input_version': 'v1'}}}
        now = datetime(2026, 10, 3, 7, tzinfo=timezone.utc)
        self.assertTrue(historical_work_due(status, now))
        status['errors']['auto-tick-test']['at'] = '2026-10-03T06:00:00Z'
        self.assertFalse(historical_work_due(status, now))
        status['historical_remaining'] = 0
        self.assertFalse(historical_work_due(status, now))

    def test_hourly_primary_and_manual_request_still_check_fresh_jma(self):
        fresh = {'live_checked_at_utc': '2026-10-03T06:55:00Z'}
        self.assertTrue(self.check(fresh, schedule='17 * * * *'))
        self.assertTrue(self.check(fresh, event='workflow_dispatch'))

    def test_never_duplicate_either_writer_or_displace_pending_recovery(self):
        for name in ['automatic-forecasts.yml', 'pressure-core-backfill.yml']:
            for state in ['queued', 'in_progress', 'pending', 'waiting']:
                self.assertFalse(self.check(runs=[dict(id=11, path='.github/workflows/'+name, status=state)]))

    def test_own_run_and_completed_or_unrelated_runs_do_not_block(self):
        self.assertTrue(self.check(runs=[
            dict(id=10, path='.github/workflows/automatic-forecasts.yml', status='in_progress'),
            dict(id=11, path='.github/workflows/pressure-core-backfill.yml', status='completed'),
            dict(id=12, path='pages', status='queued')]))


if __name__ == '__main__':
    unittest.main()
