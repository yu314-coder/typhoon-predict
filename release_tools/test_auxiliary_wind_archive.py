import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from auxiliary_wind_archive import CHECKPOINT, LIMITS, export_wind, sha, verify_wind
from immutable_basin_field import read_basin_field


class ActualModelWind(unittest.TestCase):
    def setUp(self):
        issue = datetime(2025, 11, 6, tzinfo=timezone.utc)
        route = [dict(lat=20., lon=125., pressure_hpa=960., lead_hours=i*6,
            valid_time_utc=(issue+timedelta(hours=i*6)).isoformat().replace('+00:00', 'Z')) for i in range(21)]
        self.reference = dict(id='auto-tick-test', storm_id='WP', members=1, checkpoint_sha256=CHECKPOINT,
            input_tensor_sha256='a'*64, issue_time_utc=route[0]['valid_time_utc'], route=route)
        self.predictions = [dict(center=torch.tensor([[20., 125.]]), vmax=torch.tensor([40.+i])) for i in range(20)]
        self.hashes = dict(forecast_sha256='b'*64, basin_field_gzip_sha256='c'*64,
            input_manifest_sha256='d'*64, weights_sha256='e'*64)
        self.diff = {key: 0. for key in LIMITS}

    def document(self):
        return export_wind(self.reference, self.predictions, self.hashes, self.diff)

    def test_actual_head_outputs_and_exact_future_leads(self):
        old = copy.deepcopy(self.reference)
        document = self.document()
        self.assertEqual([p['wind_kt_auxiliary'] for p in document['points']], list(range(40, 60)))
        self.assertEqual([p['lead_hours'] for p in document['points']], list(range(6, 121, 6)))
        self.assertEqual(self.reference, old)
        self.assertFalse(document['pressure_derived'])
        self.assertFalse(document['observed_future_wind_used'])

    def test_original_member_identity_is_not_relabelled_fifty(self):
        self.reference['members'] = 50
        with self.assertRaisesRegex(ValueError, 'single-member'):
            self.document()

    def test_missing_or_extra_members_fail(self):
        self.predictions[0]['vmax'] = torch.tensor([40., 40.])
        with self.assertRaisesRegex(ValueError, 'member count'):
            self.document()

    def test_failed_immutable_replay_cannot_export_wind(self):
        self.diff['route_degrees'] = .0011
        with self.assertRaisesRegex(ValueError, 'Unaudited'):
            self.document()
        self.diff['route_degrees'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Unaudited'):
            self.document()

    def test_actual_zero_wind_is_retained_but_invalid_is_null(self):
        self.predictions[0]['vmax'] = torch.tensor([0.])
        self.predictions[1]['vmax'] = torch.tensor([float('nan')])
        self.predictions[2]['vmax'] = torch.tensor([300.])
        points = self.document()['points']
        self.assertEqual(points[0]['wind_kt_auxiliary'], 0.)
        self.assertTrue(points[0]['wind_kt_auxiliary_valid'])
        for point in points[1:3]:
            self.assertIsNone(point['wind_kt_auxiliary'])
            self.assertEqual(point['valid_members'], 0)
            self.assertEqual(point['reason'], 'invalid_model_wind')

    def test_domain_exit_is_a_gap_not_observed_fill(self):
        self.predictions[4]['center'] = torch.tensor([[20., 181.]])
        point = self.document()['points'][4]
        self.assertIsNone(point['wind_kt_auxiliary'])
        self.assertEqual(point['reason'], 'outside_model_domain')

    def test_already_saved_wind_must_match(self):
        self.reference['route'][1]['wind_kt_auxiliary'] = 50.
        with self.assertRaisesRegex(ValueError, 'already saved'):
            self.document()

    def test_sidecar_audit_rejects_wrong_times_masks_and_source_files(self):
        temp_base = Path('/Volumes/D/typhoon_predict/output') if Path('/Volumes/D').exists() else Path(os.environ.get('RUNNER_TEMP', tempfile.gettempdir()))
        with tempfile.TemporaryDirectory(dir=temp_base) as tmp:
            root = Path(tmp)
            (root/'forecasts').mkdir(); (root/'fields').mkdir()
            f = root/'forecasts/auto-tick-test.json'; b = root/'fields/auto-tick-test.json.gz'
            f.write_text(json.dumps(self.reference)); b.write_bytes(b'immutable basin bytes')
            self.hashes.update(forecast_sha256=sha(f), basin_field_gzip_sha256=sha(b))
            planned = {'auto-tick-test': {'storm_id': 'WP', 'issue_time_utc': self.reference['issue_time_utc']}}
            verify = lambda d: verify_wind(d, root, planned, 'd'*64, 'e'*64)
            document = self.document(); verify(document)
            changed = copy.deepcopy(document); changed['points'][0]['valid_time_utc'] = self.reference['issue_time_utc']
            with self.assertRaisesRegex(ValueError, 'lead/route'):
                verify(changed)
            changed = copy.deepcopy(document); changed['points'][0]['wind_kt_auxiliary_valid'] = False
            changed['points'][0]['valid_members'] = 0
            with self.assertRaisesRegex(ValueError, 'zero-filled'):
                verify(changed)
            changed = copy.deepcopy(document); changed['input_tensor_sha256'] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'causal input'):
                verify(changed)
            b.write_bytes(b'changed basin')
            with self.assertRaisesRegex(ValueError, 'hashes changed'):
                verify(document)

    def test_plain_original_field_is_verified_without_rewriting_it(self):
        temp_base = Path('/Volumes/D/typhoon_predict/output') if Path('/Volumes/D').exists() else Path(os.environ['RUNNER_TEMP'])
        with tempfile.TemporaryDirectory(dir=temp_base) as tmp:
            root = Path(tmp)
            (root/'forecasts').mkdir(); (root/'fields').mkdir()
            f = root/'forecasts/auto-tick-test.json'; b = root/'fields/auto-tick-test.json'
            f.write_text(json.dumps(self.reference)); b.write_text('{"pressure_hpa":[[1000.0]]}')
            before = (f.read_bytes(),b.read_bytes())
            _, field_hashes = read_basin_field(root,self.reference['id'])
            self.hashes.pop('basin_field_gzip_sha256')
            self.hashes.update(forecast_sha256=sha(f),**field_hashes)
            planned = {'auto-tick-test': {'storm_id':'WP','issue_time_utc':self.reference['issue_time_utc']}}
            verify_wind(self.document(),root,planned,'d'*64,'e'*64)
            self.assertEqual((f.read_bytes(),b.read_bytes()),before)
            self.assertFalse((root/'fields/auto-tick-test.json.gz').exists())


if __name__ == '__main__':
    unittest.main()
