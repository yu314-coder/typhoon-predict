"""Offline temporal/identity/coverage checks; no model or network imports."""
import hashlib
import json
import unittest

from build_deepmind_period_comparison import FOLDER, MODELS, ROOT, period_key, summarize
from import_deepmind_release_results import same
from plot_model_announcement import period_metrics


class DeepMindPeriodComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((FOLDER / 'period_comparison.json').read_text())
        cls.protocol = json.loads((FOLDER / 'protocol.json').read_text())
        cls.original = json.loads((FOLDER / 'benchmark.json').read_text())

    def test_strict_utc_boundary_not_2024_onward(self):
        self.assertEqual(period_key('2023-12-31T18:00:00Z'), 'before_2024')
        self.assertEqual(period_key('2024-01-01T00:00:00Z'), 'year_2024')
        self.assertEqual(period_key('2024-12-31T18:00:00Z'), 'year_2024')
        self.assertEqual(period_key('2025-01-01T00:00:00Z'), 'after_2024')
        with self.assertRaises(ValueError):
            period_key('2024-12-31T23:00:00+08:00')

    def test_counts_and_proportions_are_exact_partition(self):
        expected = {'before_2024': (1336, 230), 'year_2024': (76, 22), 'after_2024': (61, 18)}
        all_ids = []
        by_id = {c['case_index']: c for c in self.protocol['cases']}
        for key, (days, storms) in expected.items():
            period = self.report['periods'][key]
            self.assertEqual(period['daily_issues'], days)
            self.assertEqual(period['storms'], storms)
            self.assertAlmostEqual(period['daily_issue_percentage'], days / 1473 * 100)
            self.assertAlmostEqual(period['storm_percentage'], storms / 270 * 100)
            self.assertEqual(len(period['case_indices']), days)
            self.assertEqual(len({by_id[i]['storm_id'] for i in period['case_indices']}), storms)
            for index in period['case_indices']:
                self.assertEqual(period_key(by_id[index]['issue_time_utc']), key)
            all_ids.extend(period['case_indices'])
        self.assertEqual(sorted(all_ids), list(range(1473)))

    def test_total_preserves_all_original_scores(self):
        same(self.report['periods']['total']['scores'],
             self.original['aggregate_equal_complete_storm'], 'original totals')

    def test_pressure_coverage_never_becomes_track_coverage(self):
        total = self.report['periods']['total']
        post = self.report['periods']['after_2024']
        self.assertEqual(total['scores']['pressure']['JMA']['daily_cases'], 134)
        self.assertEqual(total['scores']['pressure']['JMA']['common_lead_points'], 2637)
        self.assertEqual(post['scores']['pressure']['JMA']['daily_cases'], 59)
        self.assertEqual(post['scores']['pressure']['JMA']['common_lead_points'], 1174)
        self.assertEqual(period_metrics(post)['mean_track_error_km']['coverage']['daily_issues'], 61)
        self.assertEqual(period_metrics(post)['pressure_JMA_hpa']['coverage']['daily_issues'], 59)
        missing = self.report['periods']['before_2024']['scores']['pressure']['JMA']
        for model in MODELS:
            self.assertIsNone(missing['models'][model]['mae_hpa']['value'])
        self.assertIsNone(period_metrics(self.report['periods']['before_2024'])['pressure_JMA_hpa']['values']['deepmind'])

    def test_empty_coverage_is_null_not_zero(self):
        empty = summarize([], 1473, 270)
        self.assertIsNone(empty['scores']['route']['deepmind']['mean_track_error_km']['value'])
        self.assertIsNone(empty['scores']['pressure']['JMA']['models']['deepmind']['mae_hpa']['value'])

    def test_equal_storm_not_pooled_daily_average(self):
        def record(storm, index, score):
            values = dict(mean_track_error_km=score, track_error_120h_km=score,
                          direction_error_deg=score, shape_similarity=.5,
                          track_error_by_lead_km=[score] * 20)
            return {'storm_id': storm, 'case_index': index, 'issue_time_utc': '2025-01-01T00:00:00Z',
                    'route_metrics': {m: values for m in MODELS}, 'pressure_metrics': {}}
        rows = [record('a', 0, 10), record('a', 1, 10), record('a', 2, 10), record('b', 3, 30)]
        self.assertEqual(summarize(rows, 4, 2)['scores']['route']['deepmind']['mean_track_error_km']['value'], 20)

    def test_same_checkpoint_and_immutable_source_hashes(self):
        self.assertTrue(self.report['single_checkpoint_across_all_periods'])
        self.assertEqual(self.report['model']['model_checkpoint'], 'WeatherNextCyclones_Mini_<2024')
        self.assertEqual(self.report['model']['trained_through'], 2023)
        self.assertEqual(self.report['model']['software_version'], '0.3.0')
        for name in ('protocol.json', 'verification.json', 'case-manifest.json', 'benchmark.json'):
            self.assertEqual(hashlib.sha256((FOLDER / name).read_bytes()).hexdigest(), self.report['source_hashes'][name])
        for key in ('new_inference_performed', 'model_weights_changed', 'forecast_arrays_modified', 'raw_inputs_reaudited'):
            self.assertFalse(self.report[key])
        self.assertEqual(self.report['missing_results_scored_as_zero'], 0)

    def test_chart_receipts_match_exact_data_and_bytes(self):
        for name, key in (('model_1_2_benchmark', 'total'), ('model_1_2_after_2024_benchmark', 'after_2024')):
            base = ROOT / 'evaluation/released_daily' / name
            receipt = json.loads(base.with_suffix('.json').read_text())
            self.assertEqual(receipt['period'], key)
            self.assertEqual(receipt['deepmind_model'], self.report['model'])
            self.assertEqual(receipt['source_sha256'], hashlib.sha256((FOLDER / 'period_comparison.json').read_bytes()).hexdigest())
            self.assertEqual(receipt['chart_sha256'], hashlib.sha256(base.with_suffix('.png').read_bytes()).hexdigest())
            expected = period_metrics(self.report['periods'][key])
            self.assertEqual(receipt['values'], {k: v['values'] for k, v in expected.items()})
            self.assertFalse(receipt['zero_fill'])


if __name__ == '__main__':
    unittest.main()
