"""Portable regression checks for complete shared README publication."""
import json
import re
import unittest
from sync_public_model_cards import (CURRENT_FIGURES, DOC_SYNC_FILES, REPORT_FIGURES,
    SYNC_FILES, MEDIA_REVISION, ROOT, media_url, render_card, sha)


class PublicModelCardsTest(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / 'README.md').read_text()
        self.header = '---\nlicense: mit\ntags:\n- research\ncustom_preserved: yes\n---\n'
        self.original = self.header + '\nOld model card\n'

    def test_full_card_and_remote_metadata_are_preserved(self):
        rendered = render_card(self.original, self.source)
        self.assertTrue(rendered.startswith(self.header))
        for text in ('# TrackFormer 1.2', '1,473 daily starts / 270 storms',
                     '134 starts / 40 storms', '2,637', '12.84',
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
        self.assertLess(rendered.index('src="' + media_url('mangkhut') + '"'), rendered.index('## Results in the research report'))
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
        for text in ('same valid time', 'common pressure-scoring conditions',
                     'Missing labels', 'certified untouched holdout', '50-member mean'):
            self.assertIn(text, rendered)
        for path in (*REPORT_FIGURES, 'docs/trackformer_1_2_architecture.svg'):
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

    def assert_metric_row(self, source, label, values, places, models=('1.1', '1.2', 'deepmind')):
        rows = [line for line in source.splitlines() if line.startswith('| ')
                and line.split('|')[1].strip().startswith(label)]
        self.assertEqual(len(rows), 1, label)
        cells = [cell.strip() for cell in rows[0].strip().strip('|').split('|')]
        self.assertEqual(len(cells), len(models) + 2, label)
        for index, key in enumerate(models, 1):
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
            ('Mean track error, +6 to +120 h', 'mean_track_error_km', 2),
            ('Six-hour track-direction error', 'direction_error_deg', 2),
            ('Central-pressure MAE · JMA', 'pressure_JMA_hpa', 2),
        )
        for source in (self.source, render_card(self.original, self.source)):
            for label, key, places in specs:
                with self.subTest(label=label):
                    self.assert_metric_row(source, label, metrics[key], places, ('1.2', 'deepmind'))
            days = json.loads((ROOT / 'docs/report_figures/day_1_to_5.json').read_text())
            self.assert_metric_row(source, 'Day 5 track error',
                {m: days['track_error_km'][m][-1] for m in ('1.2', 'deepmind')}, 2, ('1.2', 'deepmind'))
        pressure = next(row for row in exported['metrics'] if row['key'] == 'pressure_JMA_hpa')
        self.assertEqual(pressure['common_lead_points'], 2637)
        for coverage in pressure['coverage'].values():
            self.assertEqual((coverage['storms'], coverage['daily_cases']), (40, 134))
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
        for model in ('1.2', 'deepmind'):
            value = post['scores']['route'][model]['mean_track_error_km']['value']
            self.assertIn(f'{value:.1f} km', self.source)
        for text in ('61 starts / 18 storms', 'Calendar 2024 is excluded', '1980–1999',
                     'Different inputs and member counts', 'development comparison'):
            self.assertIn(text, self.source)

    def test_report_scope_and_checkpoint_identity_precede_chart(self):
        for source in (self.source, render_card(self.original, self.source)):
            self.assertLess(source.index('**DeepMind checkpoint:'), source.index('report_figures/day_1_to_5.png'))
            self.assertEqual(source.count('report_figures/day_1_to_5.png'), 1)
            self.assertNotIn('1.1', source)
            for text in ('24, 48, 72, 96 and 120 h', 'not averages over each day',
                         'subset of those same 270 storms', 'not 50 trained networks',
                         'not proven optimal', 'no rainfall forecast head or humidity input'):
                self.assertIn(text, source)

    def test_pressure_subset_is_not_relabelled_as_missing_observations(self):
        for source in (self.source, render_card(self.original, self.source)):
            self.assertIn('does **not** mean the remaining 230 storms have no pressure observations', source)
            self.assertIn('original pipeline\'s issue-time intensity eligibility', source)
        chinese = (ROOT / 'docs/research_report_zh.md').read_text()
        for text in ('270個颱風中符合共同評分條件的40個颱風', '共134次起報', '2,637筆共同有效紀錄',
                     '不代表其餘 230 個颱風都沒有氣壓觀測'):
            self.assertIn(text, chinese)

    def test_training_years_sources_and_receipt_match_release(self):
        manifest = json.loads((ROOT / 'models/trackformer_1_2_field/manifest.json').read_text())
        audit = json.loads((ROOT / 'evaluation/training_data/trackformer_1_2_provenance.json').read_text())
        self.assertEqual(audit['status'], 'verified')
        for key in ('source_checkpoint_sha256', 'inference_weights_sha256'):
            self.assertEqual(audit[key], manifest[key])
        self.assertEqual(audit['dataset_manifest_sha256'], manifest['dataset_sha256'])
        self.assertEqual(audit['normalization_fit_years'], [2000, 2021])
        self.assertEqual(len(audit['verified_archive_files']), 6)
        self.assertTrue(audit['whole_storm_splits'])
        for name, years, count, native in (('train', [2000, 2021], 13949, 800),
                                           ('validation', [2022, 2023], 1041, 100),
                                           ('test', [2024, 2025], 1195, 100)):
            row = audit['partitions'][name]
            self.assertEqual((row['years'], row['windows'], row['native_pressure_windows']), (years, count, native))
        for key in ('inference_performed', 'model_weights_modified', 'training_data_modified'):
            self.assertFalse(audit[key])
        for source in (self.source, render_card(self.original, self.source)):
            for text in ('## Training data and year cutoffs', '2000–2021', '2022–2023', '2024–2025',
                         'not training through 2026', 'IBTrACS', 'NCEP/NCAR Reanalysis 1',
                         'ARCO-ERA5', 'land fraction', '13,949', '1,041', '1,195'):
                self.assertIn(text, source)
        for path in ('docs/trackformer_1_2_training_data.md',
                     'evaluation/training_data/trackformer_1_2_provenance.json', 'release_tools/audit_training_data.py'):
            self.assertIn(path, SYNC_FILES)

    def test_evaluation_index_and_worker_handoff_metadata_are_explained(self):
        index = (ROOT / 'evaluation/README.md').read_text()
        self.assertIn('completed three-model comparison', index)
        self.assertNotIn('270 issue times from 90 storms', index)
        self.assertNotIn('results are not implied', index)
        self.assertIn('evaluation/README.md', SYNC_FILES)
        notes = (ROOT / 'docs/deepmind_daily_benchmark.md').read_text()
        self.assertIn('retained byte-for-byte', notes)
        self.assertIn('handoff before import, not the current benchmark status', notes)

    def test_report_assets_match_verified_copies(self):
        import struct
        folder = ROOT / 'docs/report_figures'
        receipt = json.loads((folder / 'provenance.json').read_text())
        self.assertFalse(receipt['model_weights_changed'])
        self.assertFalse(receipt['model_or_truth_arrays_changed'])
        self.assertFalse(receipt['coordinates_or_times_shifted'])
        self.assertFalse(receipt['model_radius_head_available'])
        self.assertEqual(receipt['forecast_members_per_case'], 50)
        self.assertEqual(len(REPORT_FIGURES), 10)
        self.assertEqual(len(receipt['cases']), 3)
        for name, expected in receipt['files_sha256'].items():
            self.assertEqual(sha(folder / name), expected, name)
        for name in REPORT_FIGURES:
            data = (ROOT / name).read_bytes()
            self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
            width, height = struct.unpack('>II', data[16:24])
            self.assertGreater(width, 600)
            self.assertGreater(height, 250)
        days = json.loads((folder / 'day_1_to_5.json').read_text())
        self.assertEqual(days['lead_hours'], [24, 48, 72, 96, 120])
        self.assertEqual((days['route_storms'], days['route_issues']), (270, 1473))
        self.assertEqual(days['pressure_common_storms_by_day'], [40, 40, 40, 40, 39])
        self.assertEqual(days['pressure_common_issues_by_day'], [131, 133, 134, 133, 127])
        for values in (days['track_error_km'], days['central_pressure_mae_hpa']):
            self.assertEqual(set(values), {'1.2', 'deepmind'})
            self.assertTrue(all(len(series) == 5 for series in values.values()))

    def test_docs_only_publication_cannot_upload_weights_code_or_raw_forecasts(self):
        self.assertEqual(len(DOC_SYNC_FILES), len(set(DOC_SYNC_FILES)))
        for name in DOC_SYNC_FILES:
            self.assertIn(name, SYNC_FILES)
            self.assertTrue((ROOT / name).is_file(), name)
            self.assertTrue(name.endswith(('.md', '.png', '.svg', '.gif', '.json')), name)
            if name.endswith('.json'):
                self.assertIn(name, ('docs/report_figures/provenance.json', 'docs/report_figures/day_1_to_5.json'))
        self.assertTrue(set(REPORT_FIGURES) <= set(DOC_SYNC_FILES))

    def test_frozen_sources_and_weights_are_not_redefined_by_documentation(self):
        manifest = json.loads((ROOT / 'models/trackformer_1_2_field/manifest.json').read_text())
        provenance = json.loads((ROOT / 'docs/report_figures/provenance.json').read_text())
        for key in ('source_checkpoint_sha256', 'inference_weights_sha256'):
            self.assertEqual(provenance[key], manifest[key])
        for name, expected in manifest['source_module_sha256'].items():
            self.assertEqual(sha(ROOT / 'models/trackformer_1_2_field' / name), expected)

    def test_equipment_and_observed_radii_have_matching_scope(self):
        chinese = (ROOT / 'docs/research_report_zh.md').read_text()
        for text in ('Apple M4', '24 GB', 'NVIDIA GeForce RTX 3070', 'Metal', 'CUDA',
                     '13,949', '1,041', '1,195', '同一組模型參數', '不是訓練 50 個模型',
                     '未證明 50 組是最佳數量', '發布的 1.2 不包含降雨預測頭或水氣輸入'):
            self.assertIn(text, chinese)
        for name in REPORT_FIGURES:
            self.assertIn(name.removeprefix('docs/'), chinese)
        self.assertIn('not model radius predictions', self.source)
        self.assertIn('ordinary home computer', self.source)

    def test_report_document_relative_links_resolve(self):
        for name in ('README.md', 'docs/research_report_zh.md', 'docs/trackformer_1_2_evaluation.md',
                     'docs/trackformer_1_2_training_data.md', 'models/trackformer_1_2_field/README.md'):
            path = ROOT / name
            for url in re.findall(r'\]\(([^)\s]+)\)', path.read_text()):
                if url.startswith(('https:', 'http:', '#', 'mailto:')):
                    continue
                resolved = (path.parent / url.split('#', 1)[0]).resolve()
                self.assertTrue(resolved.is_relative_to(ROOT) and resolved.exists(), (name, url))


if __name__ == '__main__':
    unittest.main()
