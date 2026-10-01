"""Small offline checks; never import the weather model or run inference."""
import ast
from pathlib import Path
import unittest
import numpy as np
from deepmind_daily_benchmark import aggregate, curve_similarity, local, pressure_mask, route_metrics


class DeepMindBenchmarkTest(unittest.TestCase):
    def test_identical_routes_have_zero_error_and_matching_shape(self):
        truth = np.stack([np.arange(1, 21)*3, np.arange(1, 21)*4], -1)
        metrics = route_metrics({k: truth.copy() for k in ('1.1', '1.2', 'deepmind')}, truth)
        for result in metrics.values():
            self.assertAlmostEqual(result['mean_track_error_km'], 0)
            self.assertAlmostEqual(result['direction_error_deg'], 0)
            self.assertAlmostEqual(result['shape_similarity'], 1)

    def test_direction_mask_is_common_to_all_three_models(self):
        path = np.stack([np.arange(1, 21)*10, np.zeros(20)], -1)
        stationary = np.zeros((20, 2))
        results = route_metrics({'1.1': path, '1.2': path, 'deepmind': stationary}, path)
        self.assertTrue(all(r['direction_valid_steps'] == 0 and r['direction_error_deg'] is None for r in results.values()))

    def test_curve_shape_is_not_pressure_level_error(self):
        truth = np.arange(20, dtype=float)+950
        self.assertAlmostEqual(curve_similarity(truth+20, truth), 1)
        self.assertIsNone(curve_similarity(np.ones(20), truth))
        self.assertIsNone(curve_similarity(truth[:5], truth[:5]))

    def test_missing_and_invalid_pressure_never_become_zero_errors(self):
        truth = np.full(20, 950.)
        one, two, candidate = truth.copy(), truth.copy(), truth.copy()
        one[0], two[1], candidate[2] = np.nan, 0, 700
        mask = pressure_mask(truth, {'1.1': one, '1.2': two, 'deepmind': candidate})
        self.assertEqual(mask.sum(), 17)
        self.assertFalse(mask[:3].any())
        with self.assertRaises(ValueError):
            pressure_mask(truth, {'model': truth[:19]})

    def test_longitude_wrap_keeps_geographic_alignment(self):
        result = local(np.array([[0., 1.]]), 0., 359.)
        self.assertAlmostEqual(result[0, 0], 222.4)

    def test_empty_coverage_is_null_not_a_zero_score(self):
        results = aggregate({})
        self.assertIsNone(results['route']['deepmind']['mean_track_error_km']['value'])
        self.assertIsNone(results['pressure']['JMA']['models']['deepmind']['mae_hpa']['value'])

    def test_worker_never_reads_scoring_labels(self):
        source = Path(__file__).with_name('deepmind_daily_benchmark.py').read_text()
        worker = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        segment = ast.get_source_segment(source, worker)
        self.assertNotIn('truth_local', segment)
        self.assertNotIn('intensity-v12-v11', segment)
        self.assertNotIn('np.load', segment)
        self.assertIn('targets*np.nan', segment)


if __name__ == '__main__':
    unittest.main()
