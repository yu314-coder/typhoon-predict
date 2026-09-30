import unittest
from benchmark_intensity_v12_v11 import aggregate, common_errors, initial_available


class IntensityProtocolTests(unittest.TestCase):
    def test_initial_missing_pressure_is_not_filled(self):
        self.assertFalse(initial_available([40, 0]))
        self.assertFalse(initial_available([float('nan'), 990]))
        self.assertTrue(initial_available([0, 1000]))

    def test_common_masks_retain_zero_wind(self):
        result = common_errors([0, 2, 3], [1, float('nan'), 4], [0, 1, None], low=0, high=249)
        self.assertEqual(result, {'1.1': [0., None, None], '1.2': [1., None, None]})

    def test_equal_storm_not_equal_issue(self):
        rows = []
        for sid, old, new in [('a', 10, 4), ('a', 10, 4), ('b', 20, 12)]:
            rows.append({'storm_id': sid, 'errors': {'p': {'1.1': [old]*20, '1.2': [new]*20}}})
        report = aggregate(rows, 'p')
        self.assertEqual(report['models']['1.1']['equal_storm_mae'], 15)
        self.assertEqual(report['models']['1.2']['equal_storm_mae'], 8)
        self.assertEqual(report['valid_daily_issues'], 3)

    def test_missing_leads_remain_unavailable(self):
        report = aggregate([{'storm_id': 'a', 'errors': {'p': {'1.1': [1]+[None]*19, '1.2': [2]+[None]*19}}}], 'p')
        self.assertEqual(report['models']['1.1']['equal_storm_mae_by_lead'], [1]+[None]*19)
        self.assertEqual(report['paired_issue_leads_by_lead'], [1]+[0]*19)


if __name__ == '__main__':
    unittest.main()
