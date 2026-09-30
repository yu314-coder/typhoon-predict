import unittest
import numpy as np
from extend_history_storms import HOUR, merge_plan, storm_ticks


class RequestedStormQueue(unittest.TestCase):
    def setUp(self):
        self.entry = {'id': '1986228N19120', 'name': 'Wayne', 'season': 1986}
        self.start = int(np.datetime64('1986-08-16T00:00:00', 'ns').astype('int64'))
        self.times = self.start + np.arange(-8, 4)*6*HOUR
        self.storm = {'points': [
            {'time': '1986-08-16T00:00:00Z', 'lat': 20., 'lon': 120., 'pressure_hpa': 995.},
            {'time': '1986-08-16T03:00:00Z', 'lat': 20., 'lon': 120.5},
            {'time': '1986-08-16T06:00:00Z', 'lat': 20., 'lon': 121.},
            {'time': '1986-08-16T12:00:00Z', 'lat': 20., 'lon': 99.}]}

    def test_all_six_hour_domain_starts_even_without_future_truth(self):
        rows = storm_ticks(self.entry, self.storm, self.times)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['atlas'], 8)
        self.assertEqual(rows[1]['atlas'], 9)
        self.assertEqual(rows[1]['motion'][1], 0)
        self.assertGreater(rows[1]['motion'][0], 0)
        self.assertIsNone(rows[1]['pressure_hpa'])

    def test_gapped_local_history_uses_remote_not_nearest_time(self):
        times = self.times.copy(); times[2] += HOUR
        rows = storm_ticks(self.entry, self.storm, times)
        self.assertIsNone(rows[0]['atlas'])
        self.assertEqual(rows[0]['weather_source'], 'NOAA-NCEP-Reanalysis-1-remote')

    def test_preserves_old_rows_bundles_and_prioritizes_without_duplicates(self):
        candidates = storm_ticks(self.entry, self.storm, self.times)
        original = dict(candidates[0], motion=[42., 13.], observed=[])
        other = dict(original, id='old-other', storm_id='other')
        previous = {'queue': [other, original], 'bundles': {'1986': {'sha256': 'frozen'}},
                    'coverage': [{'id': self.entry['id'], 'status': 'existing_verified_forecast'}],
                    'selection': 'Original'}
        result = merge_plan(previous, [(self.entry, candidates)], 'parent')
        self.assertEqual(result['queue'][:2][0], original)
        self.assertEqual(len(result['queue']), 3)
        self.assertEqual(result['queue'][2], other)
        self.assertEqual(result['bundles'], previous['bundles'])
        self.assertEqual(result['requested_storms'][0]['new_issue_ids'], [candidates[1]['id']])
        again = merge_plan(result, [(self.entry, candidates)], 'parent')
        self.assertEqual(again['queue'], result['queue'])
        self.assertEqual(again['requested_storms'][0]['new_issue_ids'], [])

    def test_duplicate_observed_timestamp_is_rejected(self):
        self.storm['points'].append(self.storm['points'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            storm_ticks(self.entry, self.storm, self.times)


if __name__ == '__main__':
    unittest.main()
