"""Transport fixtures only, never weather observations or forecast results."""
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import v1312_cloud_sources as t


class Response:
    def __init__(self, body, code=200, headers=None):
        self.body, self.status, self.headers = io.BytesIO(body), code, headers or {}
    def read(self, n=-1):
        return self.body.read(n)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class TransportTests(unittest.TestCase):
    def test_transport_packing_preserves_full_hour_coverage(self):
        hours = list(range(81678))  # index fixture, not observations
        packed = list(t.parts(hours, dict(initial_batch_hours=64, batch_hours=1024)))
        self.assertEqual(len(packed[0]), 64)
        self.assertEqual([x for part in packed for x in part], hours)
        self.assertEqual(len(packed), 81)

    def test_fixed_exact_source_path(self):
        url, path = t.source_location(957549600)
        self.assertEqual(url, t.ARCHIVE + "/2000/05/05/CMORPH_V1.0_ADJ_8km-30min_2000050518.nc")
        self.assertTrue(path.startswith("native/2000/05/05/"))

    def test_rejects_outside_upload_destinations(self):
        for uri in ("http://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=x",
                    "https://evil.example/upload/drive/v3/files?uploadType=resumable&upload_id=x",
                    "https://www.googleapis.com/drive/v3/files?uploadType=resumable&upload_id=x",
                    "https://user@www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=x"):
            with self.assertRaises(ValueError):
                t.check_session(uri)

    def test_native_unchanged_bytes_and_full_sha(self):
        fixture = b"\x89HDF\r\n\x1a\nTRANSPORT_FIXTURE_NOT_OBSERVATIONS"
        with tempfile.TemporaryDirectory() as directory, patch.object(t.OPENER, "open",
                return_value=Response(fixture, headers={"Content-Length": str(len(fixture))})):
            path, receipt = t.fetch_hour(957549600, directory)
            self.assertEqual(path.read_bytes(), fixture)
            self.assertEqual(receipt["sha256"], hashlib.sha256(fixture).hexdigest())
            self.assertTrue(receipt["scientific_geometry_interval_mask_audit_pending"])

    def test_truncated_source_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(t.OPENER, "open",
                return_value=Response(b"abc", headers={"Content-Length": "20"})):
            with self.assertRaises(ValueError):
                t.fetch_hour(957549600, directory)

    def test_no_oauth_and_parent_checksum_enforcement(self):
        cap = dict(name="fixture.bin", parent_id="specific-folder",
            session="https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=fixture")
        data = b"TRANSPORT_FIXTURE_NOT_OBSERVATIONS"
        metadata = dict(id="fixture-id", name=cap["name"], parents=[cap["parent_id"]],
            size=str(len(data)), md5Checksum=hashlib.md5(data).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(data)
            def send(request, timeout):
                self.assertNotIn("Authorization", request.headers)
                self.assertEqual(request.full_url, cap["session"])
                return Response(json.dumps(metadata).encode())
            with patch.object(t.OPENER, "open", side_effect=send):
                self.assertTrue(t.upload(path, cap)["server_checksum_verified"])
            metadata["parents"] = ["outside-folder"]
            with patch.object(t.OPENER, "open", side_effect=send), self.assertRaises(ValueError):
                t.upload(path, cap)

    def test_resume_follows_server_byte_range(self):
        cap = dict(name="fixture.bin", parent_id="specific-folder",
            session="https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=fixture")
        data = b"12345678"
        meta = dict(id="f", name=cap["name"], parents=[cap["parent_id"]],
            size="8", md5Checksum=hashlib.md5(data).hexdigest())
        requests = []
        def send(req, timeout):
            requests.append(req)
            if len(requests) == 1:
                raise urllib.error.URLError("FIXTURE")
            if len(requests) == 2:
                raise urllib.error.HTTPError(cap["session"], 308, "Resume", {"Range": "bytes=0-3"}, io.BytesIO(b""))
            self.assertEqual(req.data, b"5678")
            return Response(json.dumps(meta).encode())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(data)
            with patch.object(t.OPENER, "open", side_effect=send), patch.object(t.time, "sleep"):
                t.upload(path, cap)
        self.assertEqual(len(requests), 3)


if __name__ == "__main__":
    unittest.main()
