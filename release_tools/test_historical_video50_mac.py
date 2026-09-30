import unittest
import numpy as np

import automatic_forecasts as auto
from forecast_historical_video50_mac import exact_history_indices, issue_row, observations


class HistoricalVideoInputs(unittest.TestCase):
    def setUp(self):
        self.issue = '2016-09-10T00:00:00Z'
        self.end = auto.ns(self.issue)
        self.times = self.end + np.arange(-10, 3, dtype='int64')*6*auto.HOUR
        self.rows = {
            '2016-09-09 18:00:00': {'LAT': '14.1', 'LON': '140.5', 'TOKYO_PRES': '1006', 'TOKYO_WIND': ''},
            '2016-09-10 00:00:00': {'LAT': '14.5', 'LON': '140.0', 'TOKYO_PRES': '1004', 'TOKYO_WIND': ' '},
            '2016-09-10 06:00:00': {'LAT': '14.8', 'LON': '139.0', 'TOKYO_PRES': '1000', 'TOKYO_WIND': '35'},
        }

    def test_exact_history_ends_at_issue(self):
        indices = exact_history_indices(self.times, self.issue)
        np.testing.assert_array_equal(self.times[indices], self.end+np.arange(-8, 1)*6*auto.HOUR)

    def test_missing_time_cannot_use_future_or_nearest(self):
        with self.assertRaises(ValueError):
            exact_history_indices(np.delete(self.times, 6), self.issue)

    def test_future_labels_do_not_change_inputs(self):
        initial = issue_row(self.rows, 'MERANTI', 'test', self.issue)
        self.rows['2016-09-10 06:00:00'].update(LAT='50', LON='179', TOKYO_PRES='850')
        self.assertEqual(initial, issue_row(self.rows, 'MERANTI', 'test', self.issue))
        self.assertNotIn('observed', initial)

    def test_missing_wind_remains_missing(self):
        self.assertIsNone(issue_row(self.rows, 'MERANTI', 'test', self.issue)['wind_kt'])

    def test_observations_are_exact_and_missing_stays_missing(self):
        route, pressure = observations(self.rows, self.issue)
        np.testing.assert_allclose(route[0], [14.5, 140.0])
        self.assertEqual(pressure[0], 1000)
        self.assertTrue(np.isnan(route[2:]).all())
        self.assertTrue(np.isnan(pressure[1:]).all())


if __name__ == '__main__':
    unittest.main()
