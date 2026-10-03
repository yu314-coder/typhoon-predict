import unittest
from datetime import datetime, timezone
from hourly_live import hourly_row, workflow_busy


class HourlyLiveTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(id='auto-live-TC2633-20261003T0900', storm_id='TC2633',
                        issue_time_utc='2026-10-03T09:00:00Z', lat=20., lon=146.1,
                        pressure_hpa=960., observed=[])

    def test_new_hour_runs_even_with_unchanged_jma_but_never_relabels_initialization(self):
        a = hourly_row(self.row, datetime(2026,10,3,10,23,tzinfo=timezone.utc))
        b = hourly_row(self.row, datetime(2026,10,3,11,2,tzinfo=timezone.utc))
        self.assertNotEqual(a['id'], b['id'])
        self.assertEqual(a['issue_time_utc'], b['issue_time_utc'])
        self.assertEqual(b['id'], self.row['id']+'-r20261003T1100')
        for key in self.row.keys()-{'id'}:
            self.assertEqual(a[key], self.row[key])

    def test_same_hour_is_idempotent(self):
        a = hourly_row(self.row, datetime(2026,10,3,10,1,tzinfo=timezone.utc))
        b = hourly_row(self.row, datetime(2026,10,3,10,59,tzinfo=timezone.utc))
        self.assertEqual(a, b)

    def test_no_future_initialization(self):
        with self.assertRaises(ValueError):
            hourly_row(self.row, datetime(2026,10,3,8,tzinfo=timezone.utc))

    def test_no_duplicate_session_or_interference_with_backfill(self):
        for state in ['in_progress', 'queued', 'pending', 'waiting']:
            self.assertTrue(workflow_busy([dict(id=2,status=state,path='.github/workflows/hourly-live.yml')], 1))
        self.assertFalse(workflow_busy([dict(id=1,status='in_progress',path='.github/workflows/hourly-live.yml')], 1))
        self.assertFalse(workflow_busy([dict(id=2,status='in_progress',path='.github/workflows/pressure-core-backfill.yml')], 1))


if __name__ == '__main__':
    unittest.main()
