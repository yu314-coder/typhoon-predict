"""Score definitions and coverage guards, with no model inference."""
import unittest
from plot_intensity_benchmark import aggregate_daily, curve_scores


class CurveTests(unittest.TestCase):
    def test_bias_is_not_shape_accuracy(self):
        scores, coverage = curve_scores({'1.1': [1,2,3,4,5,6], '1.2': [11,12,13,14,15,16]},
                                        [1,2,3,4,5,6], low=0, high=200)
        self.assertEqual(scores['1.2']['mae'], 10)
        self.assertAlmostEqual(scores['1.2']['shape_error'], 0)
        self.assertEqual(scores['1.2']['trend_error'], 0)
        self.assertEqual(coverage['valid_adjacent_steps'], 5)

    def test_missing_never_zero_or_jump(self):
        scores, coverage = curve_scores({'1.1': [1,2,3,4,5,6], '1.2': [1,None,3,4,None,6]},
                                        [1,2,3,4,5,6], low=0, high=200)
        self.assertEqual(coverage['valid_leads'], 4)
        self.assertEqual(coverage['valid_adjacent_steps'], 1)
        self.assertIsNone(scores['1.1']['shape_error'])
        self.assertEqual(scores['1.1']['mae'], scores['1.2']['mae'])

    def test_flat_not_perfect_shape(self):
        scores, _ = curve_scores({'1.1': [1]*6, '1.2': [1,2,3,4,5,6]}, [1]*6, low=0, high=200)
        self.assertIsNone(scores['1.1']['shape_error'])
        self.assertIsNone(scores['1.2']['shape_error'])
        self.assertEqual(scores['1.1']['trend_error'], 0)
        self.assertEqual(scores['1.2']['trend_error'], 1)

    def test_opposite_curves(self):
        scores, _ = curve_scores({'1.1': [1,2,3,4,5,6], '1.2': [6,5,4,3,2,1]}, [1,2,3,4,5,6], low=0, high=200)
        self.assertAlmostEqual(scores['1.2']['shape_error'], 1)
        self.assertEqual(scores['1.2']['trend_error'], 1)

    def test_daily_then_equal_storm(self):
        records = [{'storm_id':'long','models':{'1.1':{'mae':0},'1.2':{'mae':10}}} for _ in range(10)]
        records += [{'storm_id':'short','models':{'1.1':{'mae':100},'1.2':{'mae':100}}}]
        scores = aggregate_daily(records)['mae']
        self.assertEqual(scores['models']['1.1']['equal_storm_mean'], 50)
        self.assertEqual(scores['models']['1.2']['equal_storm_mean'], 55)
        self.assertEqual(scores['valid_daily_issues'], 11)
        self.assertEqual(scores['valid_storms'], 2)


if __name__ == '__main__':
    unittest.main()
