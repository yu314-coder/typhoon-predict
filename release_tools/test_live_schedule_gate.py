import unittest
from datetime import datetime, timezone
from live_schedule_gate import decision


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
