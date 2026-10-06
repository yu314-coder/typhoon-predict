"""Portable regression checks for complete shared README publication."""
import json
import re
import unittest
from sync_public_model_cards import CURRENT_FIGURES, SYNC_FILES, MEDIA_REVISION, ROOT, media_url, render_card


class PublicModelCardsTest(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / 'README.md').read_text()
        self.header = '---\nlicense: mit\ntags:\n- research\ncustom_preserved: yes\n---\n'
        self.original = self.header + '\nOld model card\n'

    def test_full_card_and_remote_metadata_are_preserved(self):
        rendered = render_card(self.original, self.source)
        self.assertTrue(rendered.startswith(self.header))
        for text in ('# Introducing Trackformer 1.2', '1,473 days / 270 storms',
                     '134 days / 40 storms', '13.53', '12.84',
                     'WeatherNext software v0.3.0', '10.20 hPa', '258.7 km',
                     'no native wind-radius forecast head'):
            self.assertIn(text, rendered)
        self.assertNotIn('README revision', rendered)
        self.assertNotIn('Corrected pressure forecast', rendered)
        self.assertNotIn('Old model card', rendered)
        self.assertNotIn('scores are **pending**', rendered)
        self.assertIn('Mini, not the full-sized', rendered)
        self.assertIn('recent-only track results favour Mini', rendered)

    def test_one_pinned_featured_player(self):
        rendered = render_card(self.original, self.source)
        self.assertEqual(rendered.count('<video '), 1)
        self.assertEqual(rendered.count('src="' + media_url('mangkhut') + '"'), 1)
        self.assertLess(rendered.index('src="' + media_url('mangkhut') + '"'), rendered.index('## What improves'))
        self.assertIsNone(re.search(r'/resolve/main/docs/trackformer_[^"\s)]+\.mp4', rendered))
        self.assertIn(MEDIA_REVISION, rendered)
        self.assertIn('docs/showcase_archive.md', rendered)

    def test_github_animates_gif_while_hf_preserves_native_mp4(self):
        self.assertIn('](docs/trackformer_1_2_mangkhut.gif)](', self.source)
        rendered = render_card(self.original, self.source)
        self.assertNotIn('](docs/trackformer_1_2_mangkhut.gif)', rendered)
        self.assertEqual(rendered.count('src="' + media_url('mangkhut') + '"'), 1)
        self.assertIn('docs/trackformer_1_2_mangkhut.gif', SYNC_FILES)

    def test_no_github_relative_links_survive(self):
        rendered = render_card(self.original, self.source)
        for url in re.findall(r'\]\(([^)\s]+)\)', rendered):
            self.assertTrue(url.startswith(('https:', 'http:', '#', 'mailto:')), url)

    def test_missing_anchor_fails_instead_of_partial_stale_update(self):
        with self.assertRaisesRegex(ValueError, 'featured-film'):
            render_card(self.original, self.source.replace('## See the forecast:', '## Wrong heading:'))

    def test_missing_metadata_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'YAML metadata'):
            render_card('No metadata', self.source)

    def test_relative_paths_cannot_escape_public_repository(self):
        with self.assertRaisesRegex(ValueError, 'relative public link'):
            render_card(self.original, self.source + '\n[No secrets](../../private-file.txt)\n')

    def test_new_figures_are_pinned_without_changing_weights_or_paper(self):
        revision = 'a' * 40
        rendered = render_card(self.original, self.source, revision)
        for text in ('41.0%', '95% interval includes no improvement', 'same frozen starts',
                     'Missing labels', 'certified untouched holdout', '50-member mean'):
            self.assertIn(text, rendered)
        for path in CURRENT_FIGURES:
            self.assertIn('/resolve/' + revision + '/' + path, rendered)
            self.assertIn(path, SYNC_FILES)
        for path in ('docs/deepmind_daily_benchmark.md', 'docs/showcase_archive.md',
                     'evaluation/released_daily/released_daily_benchmark.json',
                     'evaluation/released_daily/released_daily_verification.json',
                     'evaluation/deepmind_daily/publication_audit.json',
                     'evaluation/deepmind_daily/verification.json'):
            self.assertIn(path, SYNC_FILES)
        self.assertNotIn('models/trackformer_1_2_field/weights.pt', SYNC_FILES)
        self.assertNotIn('paper/trackformer.tex', SYNC_FILES)
        self.assertNotIn('paper/trackformer.pdf', SYNC_FILES)
        for path in SYNC_FILES:
            self.assertTrue((ROOT / path).is_file(), path)

    def test_linked_current_docs_cannot_call_completed_run_pending(self):
        for name in ('README.md', 'models/trackformer_1_2_field/README.md',
                     'evaluation/README.md', 'docs/deepmind_daily_benchmark.md',
                     'docs/daily_storm_benchmark.md', 'docs/trackformer_1_2_evaluation.md',
                     'RELEASE_NOTES_TRACKFORMER_1_2.md'):
            with self.subTest(path=name):
                source = (ROOT / name).read_text()
                self.assertIn('Mini', source)
                self.assertIn('1,473', source)
                self.assertIn('270', source)
                for stale in ('results remain pending', 'run is in progress',
                              'scores are **pending**', 'deepmind comparison has not been run'):
                    self.assertNotIn(stale, source.lower())
                if name != 'README.md':
                    self.assertIn(name, SYNC_FILES)

    def assert_metric_row(self, source, label, values, places):
        rows = [line for line in source.splitlines() if line.startswith('| ')
                and line.split('|')[1].strip().startswith(label)]
        self.assertEqual(len(rows), 1, label)
        cells = [cell.strip() for cell in rows[0].strip().strip('|').split('|')]
        self.assertEqual(len(cells), 5, label)
        for index, key in enumerate(('1.1', '1.2', 'deepmind'), 1):
            match = re.search(r'\d+(?:\.\d+)?', cells[index].replace('**', '').replace(',', ''))
            self.assertIsNotNone(match, (label, key))
            self.assertEqual(match.group(), f'{values[key]:.{places}f}', (label, key))

    def test_readme_and_release_tables_match_completed_metrics_not_estimates(self):
        exported = json.loads((ROOT / 'evaluation/deepmind_daily/benchmark.json').read_text())
        receipt = json.loads((ROOT / 'evaluation/deepmind_daily/verification.json').read_text())
        metrics = {row['key']: row['values'] for row in exported['metrics']}
        self.assertEqual(exported['status'], 'complete_verified')
        self.assertEqual(receipt['verified_daily_cases'], 1473)
        self.assertEqual(receipt['verified_storms'], 270)
        specs = (
            ('Mean track error, +6 to +120 h', 'mean_track_error_km', 1),
            ('Six-hour track-direction error', 'direction_error_deg', 2),
            ('Central-pressure MAE · JMA', 'pressure_JMA_hpa', 2),
            ('Centred route-shape similarity', 'shape_similarity', 4),
            ('Pressure-curve similarity · JMA', 'pressure_JMA_curve_similarity', 4),
        )
        for source in (self.source, render_card(self.original, self.source)):
            for label, key, places in specs:
                with self.subTest(label=label):
                    self.assert_metric_row(source, label, metrics[key], places)
        release = (ROOT / 'RELEASE_NOTES_TRACKFORMER_1_2.md').read_text()
        for label, key, places in (
            ('Mean track error', 'mean_track_error_km', 1),
            ('Direction error', 'direction_error_deg', 2),
            ('Central-pressure MAE against JMA', 'pressure_JMA_hpa', 2),
        ):
            self.assert_metric_row(release, label, metrics[key], places)

    def test_recent_group_is_not_confused_with_combined_cohort(self):
        report = json.loads((ROOT / 'evaluation/deepmind_daily/period_comparison.json').read_text())
        post = report['periods']['after_2024']
        self.assertEqual(post['daily_issues'], 61)
        self.assertEqual(post['storms'], 18)
        self.assertEqual(post['years'], [2025, 2026])
        section = self.source.split('### After 2024', 1)[1]
        for label, key, places in (('Track position MAE', 'mean_track_error_km', 1),
                                   ('Track-direction error', 'direction_error_deg', 2),
                                   ('Route-shape similarity', 'shape_similarity', 4)):
            values = {model: post['scores']['route'][model][key]['value']
                      for model in ('1.1', '1.2', 'deepmind')}
            self.assert_metric_row(section, label, values, places)
        pressure = post['scores']['pressure']['JMA']['models']
        for label, key, places in (('JMA central-pressure MAE', 'mae_hpa', 2),
                                   ('JMA pressure-curve similarity', 'curve_similarity', 4)):
            self.assert_metric_row(section, label,
                                   {m: pressure[m][key]['value'] for m in ('1.1', '1.2', 'deepmind')}, places)

    def test_checkpoint_identity_and_proportions_precede_separate_charts(self):
        for source in (self.source, render_card(self.original, self.source)):
            self.assertLess(source.index('**DeepMind checkpoint:'), source.index('model_1_2_benchmark.png'))
            self.assertEqual(source.count('model_1_2_benchmark.png'), 1)
            self.assertEqual(source.count('model_1_2_after_2024_benchmark.png'), 1)
            for text in ('90.7%', '85.2%', '5.2%', '8.1%', '4.1%', '6.7%',
                         'Calendar 2024', '59 days / 18 storms / 1,174',
                         'not training-data composition', 'same Mini `<2024` checkpoint',
                         '100% Mini `<2024`; 0% other checkpoints'):
                self.assertIn(text, source)

    def test_evaluation_index_and_worker_handoff_metadata_are_explained(self):
        index = (ROOT / 'evaluation/README.md').read_text()
        self.assertIn('completed three-model comparison', index)
        self.assertNotIn('270 issue times from 90 storms', index)
        self.assertNotIn('results are not implied', index)
        self.assertIn('evaluation/README.md', SYNC_FILES)
        notes = (ROOT / 'docs/deepmind_daily_benchmark.md').read_text()
        self.assertIn('retained byte-for-byte', notes)
        self.assertIn('handoff before import, not the current benchmark status', notes)


if __name__ == '__main__':
    unittest.main()
