import os
import unittest
from unittest.mock import patch

import probe_archive_replay as probe


class ReadOnlyReplayProbe(unittest.TestCase):
    def test_fixed_bounded_cases_and_profiles(self):
        self.assertEqual(len(probe.CASES), 4)
        self.assertEqual(len(set(probe.CASES)), 4)
        self.assertEqual(len(probe.PROFILES), 8)
        self.assertIn('original-reader', probe.PROFILES)
        self.assertIn('capture-default', probe.PROFILES)
        self.assertTrue(all(name.startswith(('auto-tick-', 'auto-hist-')) for name in probe.CASES))

    def test_profile_environment_cannot_redirect_data_or_weights(self):
        allowed = {'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ATEN_CPU_CAPABILITY',
                   'DNNL_MAX_CPU_ISA', 'MKL_ENABLE_INSTRUCTIONS', 'MKL_CBWR'}
        self.assertTrue(all(set(settings) <= allowed for settings in probe.PROFILES.values()))

    def test_refuses_inference_next_to_local_gpu_training(self):
        with patch.dict(os.environ, {'GITHUB_ACTIONS':'false'}), \
             patch('sys.argv', ['probe', '--output', '/unused', '--cache', '/unused']):
            with self.assertRaisesRegex(RuntimeError, 'GitHub only'):
                probe.main()

    def test_requires_a_pinned_archive_before_opening_assets(self):
        with patch.dict(os.environ, {'GITHUB_ACTIONS':'true'}), \
             patch('sys.argv', ['probe', '--output', '/unused', '--cache', '/unused', '--archive-sha', 'forecast-data']):
            with self.assertRaisesRegex(ValueError, 'pinned immutable archive'):
                probe.main()


if __name__ == '__main__':
    unittest.main()
