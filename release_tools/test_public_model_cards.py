"""Portable regression checks for complete shared README publication."""
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
        for text in ('README revision: 1 October 2026',
                     'Daily-issue benchmark: 1,473 forecasts',
                     'API contract **1.1**', 'still partial', '25.03 kt'):
            self.assertIn(text, rendered)
        self.assertIn('N/A — no radius head', rendered)
        self.assertNotIn('Old model card', rendered)

    def test_exactly_four_pinned_players_and_corrected_primary(self):
        rendered = render_card(self.original, self.source)
        self.assertEqual(rendered.count('<video '), 4)
        for stem in ('fung_wong', 'soudelor', 'mangkhut', 'meranti'):
            self.assertEqual(rendered.count('src="' + media_url(stem) + '"'), 1)
        self.assertLess(rendered.index('src="' + media_url('mangkhut') + '"'), rendered.index('## What changed'))
        self.assertNotIn('/resolve/main/docs/trackformer_', rendered)
        self.assertIn(MEDIA_REVISION, rendered)
        self.assertIn('Original fixed-patch film', rendered)

    def test_no_github_relative_links_survive(self):
        rendered = render_card(self.original, self.source)
        for url in re.findall(r'\]\(([^)\s]+)\)', rendered):
            self.assertTrue(url.startswith(('https:', 'http:', '#', 'mailto:')), url)

    def test_missing_anchor_fails_instead_of_partial_stale_update(self):
        with self.assertRaisesRegex(ValueError, 'corrected-film'):
            render_card(self.original, self.source.replace('## Corrected pressure forecast', '## Wrong heading'))

    def test_missing_metadata_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'YAML metadata'):
            render_card('No metadata', self.source)

    def test_relative_paths_cannot_escape_public_repository(self):
        with self.assertRaisesRegex(ValueError, 'relative public link'):
            render_card(self.original, self.source + '\n[No secrets](../../private-file.txt)\n')

    def test_updated_pressure_graph_and_paper_are_published_together(self):
        revision = 'a' * 40
        rendered = render_card(self.original, self.source, revision)
        for text in ('Pressure graph similarity', '0.7074', '0.7118', '0.7237', '0.6637', '131 days', 'time-warped', 'slight mean improvement', '13.53 to 12.84'):
            self.assertIn(text, rendered)
        for path in CURRENT_FIGURES:
            self.assertIn('/resolve/' + revision + '/' + path, rendered)
            self.assertIn(path, SYNC_FILES)
        for path in ('paper/trackformer.tex', 'paper/trackformer.pdf',
                     'evaluation/released_daily/released_daily_benchmark.json',
                     'evaluation/released_daily/released_daily_verification.json'):
            self.assertIn(path, SYNC_FILES)
        self.assertNotIn('models/trackformer_1_2_field/weights.pt', SYNC_FILES)
        for path in SYNC_FILES:
            self.assertTrue((ROOT / path).is_file(), path)


if __name__ == '__main__':
    unittest.main()
