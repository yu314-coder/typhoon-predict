import unittest
from datetime import datetime, timezone
from backfill_queue import queue_state, should_continue, retry_is_cooling


class BackfillQueue(unittest.TestCase):
    def test_counts_and_cooldown(self):
        now = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
        errors = {'b': {'at': '2026-09-29T11:00:00Z'},
                  'c': {'at': '2026-09-28T11:00:00Z'},
                  'irrelevant': {'at': '2026-09-29T11:00:00Z'}}
        self.assertEqual(queue_state(['a', 'b', 'c', 'd'], ['a'], errors, now),
                         {'historical_remaining': 3, 'historical_ready': 2,
                          'historical_cooling_down': 1})

    def test_productive_batch_continues(self):
        self.assertTrue(should_continue({'run_url': 'run', 'batch_succeeded': 1,
                                         'historical_ready': 9}, 'run'))

    def test_completed_blocked_or_failed_batch_does_not_loop(self):
        for succeeded, ready in [(0, 9), (9, 0), (0, 0)]:
            self.assertFalse(should_continue({'run_url': 'run',
                             'batch_succeeded': succeeded, 'historical_ready': ready}, 'run'))

    def test_stale_status_and_targeted_retry_do_not_chain(self):
        status = {'run_url': 'old', 'batch_succeeded': 5, 'historical_ready': 9}
        self.assertFalse(should_continue(status, 'new'))
        self.assertFalse(should_continue(status, 'old', 'one-issue'))
        self.assertFalse(should_continue({}, ''))

    def test_bad_failure_timestamp_is_not_hot_retried(self):
        self.assertEqual(queue_state(['a'], [], {'a': {}}, datetime.now(timezone.utc))
                         ['historical_ready'], 0)

    def test_fixed_reader_retries_old_failures_once(self):
        now=datetime(2026,9,29,12,tzinfo=timezone.utc)
        old={'at':'2026-09-29T11:00:00Z'}
        new={**old,'input_version':'new-reader'}
        self.assertFalse(retry_is_cooling(old,now,'new-reader'))
        self.assertTrue(retry_is_cooling(new,now,'new-reader'))
        self.assertEqual(queue_state(['a','b'],[],{'a':old,'b':new},now,'new-reader'),
                         {'historical_remaining':2,'historical_ready':1,
                          'historical_cooling_down':1})


if __name__ == '__main__':
    unittest.main()
