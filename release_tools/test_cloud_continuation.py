import unittest
from cloud_continuation import continuation_target


class ContinuationTests(unittest.TestCase):
    def test_new_completed_issues_start_recovery_after_history_drains(self):
        history = {'historical_completed': 12, 'historical_ready': 0, 'batch_succeeded': 2}
        complete = {'total': 10, 'completed': 10, 'continue_ready': False}
        self.assertEqual(continuation_target(history, complete, complete, []), 'pressure-core-backfill.yml')
        self.assertIsNone(continuation_target(history, complete, complete, [99]))

    def test_complete_or_unproductive_batches_do_not_loop(self):
        history = {'historical_completed': 12, 'historical_ready': 0, 'batch_succeeded': 2}
        complete = {'total': 12, 'completed': 12}
        self.assertIsNone(continuation_target(history, complete, complete, []))
        self.assertIsNone(continuation_target({**history, 'batch_succeeded': 0}, {}, {}, []))

    def test_remaining_productive_history_keeps_priority(self):
        history = {'historical_completed': 12, 'historical_ready': 3, 'batch_succeeded': 2}
        self.assertEqual(continuation_target(history, {}, {}, []), 'automatic-forecasts.yml')


if __name__ == '__main__':
    unittest.main()
