import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from pathlib import Path
from hourly_live import hourly_row, workflow_busy, handoff_ready, run_session, CYCLE_TIMEOUT_SECONDS


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

    def test_short_remaining_session_hands_off_without_starting_inference(self):
        with patch('hourly_live.time.monotonic', side_effect=[0, 2750]), \
             patch('hourly_live.subprocess.run') as run, patch('hourly_live.git_publish') as publish:
            run_session(Path('/output/data'), Path('/cache'), 46)
        run.assert_not_called()
        publish.assert_not_called()

    def test_full_cycle_keeps_full_timeout_and_publishes_before_handoff(self):
        with patch('hourly_live.time.monotonic', side_effect=[0, 0, 3000, 3000]), \
             patch('hourly_live.subprocess.run') as run, patch('hourly_live.git_publish') as publish:
            run_session(Path('/output/data'), Path('/cache'), 50)
        self.assertEqual(run.call_args.kwargs['timeout'], CYCLE_TIMEOUT_SECONDS)
        publish.assert_called_once_with(Path('/output/data'))

    def test_no_duplicate_session_or_interference_with_backfill(self):
        for state in ['in_progress', 'queued', 'pending', 'waiting']:
            self.assertTrue(workflow_busy([dict(id=2,status=state,path='.github/workflows/hourly-live.yml')], 1))
        self.assertFalse(workflow_busy([dict(id=1,status='in_progress',path='.github/workflows/hourly-live.yml')], 1))
        self.assertFalse(workflow_busy([dict(id=2,status='in_progress',path='.github/workflows/pressure-core-backfill.yml')], 1))

    def test_parent_handoff_does_not_suppress_its_successor(self):
        jobs = [dict(name='hourly', steps=[
            dict(name='Recompute fifty members every hour and publish each verified run', status='completed', conclusion='success'),
            dict(name='Hand off to the next bounded cloud session', status='in_progress', conclusion=None),
        ])]
        self.assertTrue(handoff_ready(jobs))
        runs = [dict(id=2,status='in_progress',path='.github/workflows/hourly-live.yml')]
        self.assertTrue(workflow_busy(runs, 1))
        self.assertFalse(workflow_busy(runs, 1, finished_sessions=[2]))
        runs.append(dict(id=3,status='queued',path='.github/workflows/hourly-live.yml'))
        self.assertTrue(workflow_busy(runs, 1, finished_sessions=[2]))
        jobs[0]['steps'][1]['status'] = 'pending'
        self.assertFalse(handoff_ready(jobs))
        jobs[0]['steps'][1]['status'] = 'in_progress'
        jobs[0]['steps'][0]['conclusion'] = 'failure'
        self.assertFalse(handoff_ready(jobs))
        jobs[0]['steps'][0]['status'] = 'in_progress'
        jobs[0]['steps'][0]['conclusion'] = None
        self.assertFalse(handoff_ready(jobs))


if __name__ == '__main__':
    unittest.main()
