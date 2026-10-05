"""Offline publication regressions: no inference, network, or model imports."""
import copy
import hashlib
import json
import unittest

from import_deepmind_release_results import ROOT, merge_snapshot, same, validate_identity


class DeepMindReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT/'evaluation/deepmind_daily'
        cls.protocol = json.loads((folder/'protocol.json').read_text())
        cls.receipt = json.loads((folder/'verification.json').read_text())
        cls.export = json.loads((folder/'benchmark.json').read_text())
        cls.base = json.loads((ROOT/'evaluation/released_daily/released_daily_benchmark.json').read_text())

    def test_exact_completed_model_and_cohort(self):
        validate_identity(self.protocol, self.receipt, self.export)

    def test_partial_run_cannot_be_published(self):
        receipt = copy.deepcopy(self.receipt)
        receipt['verified_daily_cases'] -= 1
        with self.assertRaisesRegex(ValueError, 'Incomplete coverage'):
            validate_identity(self.protocol, receipt, self.export)

    def test_missing_and_duplicate_ids_are_rejected(self):
        receipt = copy.deepcopy(self.receipt)
        receipt['case_indices'][-1] = receipt['case_indices'][-2]
        with self.assertRaisesRegex(ValueError, 'duplicate case IDs'):
            validate_identity(self.protocol, receipt, self.export)

    def test_checkpoint_substitution_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol['checkpoint_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'official checkpoint'):
            validate_identity(protocol, self.receipt, self.export)

    def test_cpu_results_cannot_be_relabelled_gpu(self):
        receipt = copy.deepcopy(self.receipt)
        receipt['cpu_predictions_reused_as_cuda'] = True
        with self.assertRaisesRegex(ValueError, 'CPU results relabelled'):
            validate_identity(self.protocol, receipt, self.export)

    def test_unavailable_does_not_equal_zero(self):
        with self.assertRaises(ValueError):
            same(0, None, 'unavailable')
        with self.assertRaises(ValueError):
            same(None, 0, 'unavailable')

    def test_original_scores_cannot_change_on_import(self):
        base = copy.deepcopy(self.base)
        next(m for m in base['metrics'] if m['key'] == 'mean_track_error_km')['values']['1.2'] = 0
        with self.assertRaisesRegex(ValueError, 'Original score changed'):
            merge_snapshot(base, self.export, self.receipt, {})

    def test_mean_of_fifty_and_mini_identity_remain_separate(self):
        models = {m['key']: m for m in self.base['models']}
        self.assertEqual(models['v12']['member_count'], 50)
        self.assertEqual(models['deepmind']['member_count'], 1)
        self.assertEqual(models['deepmind']['software_version'], '0.3.0')
        self.assertEqual(models['deepmind']['model_checkpoint'], 'WeatherNextCyclones_Mini_<2024')

    def test_audit_matches_imported_bytes_and_chart_snapshot(self):
        folder = ROOT/'evaluation/deepmind_daily'
        audit = json.loads((folder/'publication_audit.json').read_text())
        for name, expected in audit['imported_files_sha256'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256((ROOT/'evaluation/released_daily/released_daily_benchmark.json').read_bytes()).hexdigest(),
                         audit['snapshot_sha256'])
        self.assertFalse(audit['raw_era5_states_in_return_zip'])
        self.assertEqual(audit['missing_results_scored_as_zero'], 0)

    def test_all_published_values_are_actual_completed_export(self):
        rows = {m['key']: m for m in self.base['metrics']}
        for metric in self.export['metrics']:
            self.assertEqual(rows[metric['key']]['values']['deepmind'], metric['values']['deepmind'])
        self.assertNotIn('deepmind', rows['frechet_distance_km']['values'])
        self.assertNotIn('deepmind', rows['path_similarity']['values'])


if __name__ == '__main__':
    unittest.main()
