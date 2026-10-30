"""Portable integrity checks for the compact public documentation layout."""
import hashlib
import json
from pathlib import Path
import re
import unittest
import zipfile

from sync_public_model_cards import (ROOT, SYNC_FILES, DOC_SYNC_FILES, REPORT_FIGURES,
    MEDIA_REVISION, media_url, render_card)


class PublicModelCardsTest(unittest.TestCase):
    def setUp(self):
        self.source=(ROOT/'README.md').read_text()
        self.reference=(ROOT/'docs/REFERENCE.md').read_text()
        self.header='---\nlicense: mit\ncustom_preserved: yes\n---\n'
        self.index=json.loads((ROOT/'docs/archive/documentation-manifest.json').read_text())

    def test_three_current_markdown_entry_points(self):
        actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*.md') if '.git' not in p.parts}
        self.assertEqual(actual,set(self.index['final_markdown_paths']))
        self.assertLess(len(self.source.encode()),8000)

    def test_all_removed_documents_survive_exactly_in_archive(self):
        path=ROOT/self.index['archive_path']
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),self.index['archive_sha256'])
        with zipfile.ZipFile(path) as z:
            for entry in self.index['original_docs']:
                self.assertEqual(hashlib.sha256(z.read(entry['path'])).hexdigest(),entry['sha256'])
        for name in self.index['removed_paths']:
            self.assertFalse((ROOT/name).exists(),name)

    def test_pdf_remains_canonical_and_exact(self):
        source=json.loads((ROOT/'training/trackformer_1_2/SOURCE_MANIFEST.json').read_text())
        pdf=source['documentation_pdf']
        data=(ROOT/pdf['path']).read_bytes()
        self.assertTrue(data.startswith(b'%PDF-'))
        self.assertEqual(hashlib.sha256(data).hexdigest(),pdf['sha256'])
        self.assertEqual(pdf['pages'],106)
        self.assertIn(pdf['path'],self.source)

    def test_training_source_ledger_and_archived_document_links(self):
        m=json.loads((ROOT/'training/trackformer_1_2/SOURCE_MANIFEST.json').read_text())
        for entry in m['files']:
            self.assertEqual(hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest(),entry['sha256'])
        with zipfile.ZipFile(ROOT/self.index['archive_path']) as z:
            for entry in m['archived_documentation']:
                self.assertEqual(hashlib.sha256(z.read(entry['archive_member'])).hexdigest(),entry['sha256'])

    def test_sync_does_not_recreate_retired_files(self):
        retired=set(self.index['removed_paths'])
        self.assertFalse(retired.intersection(SYNC_FILES))
        self.assertFalse(retired.intersection(DOC_SYNC_FILES))
        self.assertEqual(len(SYNC_FILES),len(set(SYNC_FILES)))
        self.assertEqual(len(DOC_SYNC_FILES),len(set(DOC_SYNC_FILES)))
        for path in SYNC_FILES:
            self.assertTrue((ROOT/path).is_file(),path)
        self.assertNotIn('models/trackformer_1_2_field/weights.pt',SYNC_FILES)
        self.assertNotIn('paper/trackformer.pdf',SYNC_FILES)

    def test_hf_keeps_one_metadata_header_and_one_native_player(self):
        rendered=render_card(self.header+'\nOld card',self.source)
        self.assertTrue(rendered.startswith(self.header))
        self.assertEqual(rendered.count('license: mit'),1)
        self.assertEqual(rendered.count('<video '),1)
        self.assertEqual(rendered.count('src="'+media_url('mangkhut')+'"'),1)
        self.assertIn(MEDIA_REVISION,rendered)
        self.assertNotIn('Old card',rendered)
        self.assertNotIn('/resolve/main/docs/trackformer_1_2_mangkhut.mp4',rendered)
        for url in re.findall(r'\]\(([^)\s]+)\)',rendered):
            self.assertTrue(url.startswith(('https:','http:','#','mailto:')),url)

    def test_bad_metadata_or_missing_featured_section_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'YAML metadata'):
            render_card('No metadata',self.source)
        with self.assertRaisesRegex(ValueError,'featured-film'):
            render_card(self.header,self.source.replace('## See the forecast:','## Removed:'))
        with self.assertRaisesRegex(ValueError,'relative public link'):
            render_card(self.header,self.source+'\n[Private](../../private.txt)\n')

    def test_current_relative_links_and_fragments_resolve(self):
        for name in self.index['final_markdown_paths']:
            path=ROOT/name
            for url in re.findall(r'\]\(([^)\s]+)\)',path.read_text()):
                if url.startswith(('https:','http:','mailto:')):continue
                target,_,fragment=url.partition('#')
                dest=(path.parent/target).resolve() if target else path
                self.assertTrue(dest.is_relative_to(ROOT) and dest.exists(),(name,url))
                if fragment and dest.suffix=='.md':
                    text=dest.read_text()
                    anchors=set(re.findall(r'<a id="([^"]+)"',text))
                    for heading in re.findall(r'^#+\s+(.+)$',text,re.M):
                        anchors.add(re.sub(r'[^\w\- ]','',heading.lower()).replace(' ','-'))
                    self.assertIn(fragment,anchors,(name,url))

    def test_summary_numbers_match_actual_common_support(self):
        exported=json.loads((ROOT/'evaluation/deepmind_daily/benchmark.json').read_text())
        receipt=json.loads((ROOT/'evaluation/deepmind_daily/verification.json').read_text())
        self.assertEqual(exported['status'],'complete_verified')
        self.assertEqual((receipt['verified_daily_cases'],receipt['verified_storms']),(1473,270))
        metrics={row['key']:row['values'] for row in exported['metrics']}
        for label,key in [('Mean track error','mean_track_error_km'),('Six-hour track-direction','direction_error_deg'),('Central-pressure MAE','pressure_JMA_hpa')]:
            row=next(x for x in self.source.splitlines() if x.startswith('| '+label))
            for model in ('1.2','deepmind'):self.assertIn(f'{metrics[key][model]:.2f}',row)
        pressure=next(x for x in exported['metrics'] if x['key']=='pressure_JMA_hpa')
        self.assertEqual(pressure['common_lead_points'],2637)
        for coverage in pressure['coverage'].values():
            self.assertEqual((coverage['storms'],coverage['daily_cases']),(40,134))
        recent=json.loads((ROOT/'evaluation/deepmind_daily/period_comparison.json').read_text())['periods']['after_2024']
        self.assertEqual((recent['daily_issues'],recent['storms']),(61,18))
        for model in ('1.2','deepmind'):
            self.assertIn(f"{recent['scores']['route'][model]['mean_track_error_km']['value']:.1f} km",self.source)

    def test_scientific_identity_and_split_provenance_unchanged(self):
        m=json.loads((ROOT/'models/trackformer_1_2_field/manifest.json').read_text())
        audit=json.loads((ROOT/'evaluation/training_data/trackformer_1_2_provenance.json').read_text())
        self.assertEqual(audit['dataset_manifest_sha256'],m['dataset_sha256'])
        for key in ['source_checkpoint_sha256','inference_weights_sha256']:
            self.assertEqual(audit[key],m[key])
        for name,expected in m['source_module_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/'models/trackformer_1_2_field'/name).read_bytes()).hexdigest(),expected)
        for name,count,native in [('train',13949,800),('validation',1041,100),('test',1195,100)]:
            p=audit['partitions'][name]
            self.assertEqual((p['windows'],p['native_pressure_windows']),(count,native))
        self.assertTrue(audit['whole_storm_splits'])
        for phrase in ['no rainfall head','humidity input','not an operational warning service','batch 16','batch 10','4,384']:
            self.assertIn(phrase,self.source)

    def test_figure_provenance_and_all_report_bytes_stay_unchanged(self):
        folder=ROOT/'docs/report_figures'
        receipt=json.loads((folder/'provenance.json').read_text())
        for key in ['model_weights_changed','model_or_truth_arrays_changed','coordinates_or_times_shifted','model_radius_head_available']:
            self.assertFalse(receipt[key])
        self.assertEqual(receipt['forecast_members_per_case'],50)
        self.assertEqual(len(REPORT_FIGURES),10)
        for name,expected in receipt['files_sha256'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),expected)
        days=json.loads((folder/'day_1_to_5.json').read_text())
        self.assertEqual(days['lead_hours'],[24,48,72,96,120])
        self.assertEqual((days['route_storms'],days['route_issues']),(270,1473))
        row=next(x for x in self.source.splitlines() if x.startswith('| Day 5'))
        for model in ['1.2','deepmind']:self.assertIn(f"{days['track_error_km'][model][-1]:.2f}",row)


if __name__=='__main__':unittest.main()
