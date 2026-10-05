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
        for text in ('# Introducing Trackformer 1.2', '1,473 days / 270 storms',
                     '134 days / 40 storms', '13.53', '12.84',
                     'WeatherNext software v0.3.0', '10.20 hPa', '288.5 km',
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


if __name__ == '__main__':
    unittest.main()
