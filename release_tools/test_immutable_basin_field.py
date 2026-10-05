import copy
import gzip
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from immutable_basin_field import basin_path, read_basin_field, verify_basin_source


class OriginalBasinStorage(unittest.TestCase):
    def setUp(self):
        base = Path('/Volumes/D/typhoon_predict/output') if Path('/Volumes/D').exists() else Path(os.environ['RUNNER_TEMP'])
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'fields').mkdir()
        self.ident = 'auto-hist-1970050N07151'
        self.raw = b'{"forecast_id":"auto-hist-1970050N07151","pressure_hpa":[[1000.01]]}\n'

    def save(self, compressed):
        path = self.root/'fields'/f'{self.ident}.json{ ".gz" if compressed else ""}'
        path.write_bytes(gzip.compress(self.raw, mtime=0) if compressed else self.raw)
        return path

    def test_plain_and_gzip_roundtrip_and_hash_actual_immutable_file(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                path = self.save(compressed)
                before = path.read_bytes()
                document, source = read_basin_field(self.root, self.ident)
                self.assertEqual(document, json.loads(self.raw))
                self.assertEqual(source['basin_field_sha256'], hashlib.sha256(before).hexdigest())
                self.assertEqual(source['basin_field_file'], path.name)
                self.assertEqual('basin_field_gzip_sha256' in source, compressed)
                self.assertEqual(verify_basin_source(self.root, self.ident, source), path)
                self.assertEqual(path.read_bytes(), before)

    def test_prefer_existing_gzip_like_original_archive_auditor(self):
        self.save(False)
        compressed = self.save(True)
        self.assertEqual(basin_path(self.root, self.ident), compressed)

    def test_original_gzip_only_receipts_stay_valid(self):
        path = self.save(True)
        source = {'basin_field_gzip_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.assertEqual(verify_basin_source(self.root, self.ident, source), path)

    def test_changed_plain_source_is_rejected(self):
        path = self.save(False)
        _, source = read_basin_field(self.root, self.ident)
        path.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'hashes changed'):
            verify_basin_source(self.root, self.ident, source)

    def test_missing_source_cannot_silently_skip_or_fall_back_to_new_storage(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Missing immutable'):
            read_basin_field(self.root, self.ident)
        self.save(False)
        with self.assertRaisesRegex(ValueError, 'hashes changed'):
            verify_basin_source(self.root, self.ident, {'basin_field_gzip_sha256': hashlib.sha256(self.raw).hexdigest()})

    def test_declared_format_filename_and_legacy_hash_must_agree(self):
        self.save(False)
        _, source = read_basin_field(self.root, self.ident)
        for field, value in [('basin_field_file', '../other.json'),
                             ('basin_field_format', 'invalid'),
                             ('basin_field_gzip_sha256', source['basin_field_sha256'])]:
            with self.subTest(field=field):
                altered = copy.deepcopy(source)
                altered[field] = value
                with self.assertRaises(ValueError):
                    verify_basin_source(self.root, self.ident, altered)


if __name__ == '__main__':
    unittest.main()
