"""Lossless public ARCO ERA5 transport for fresh TrackFormer 1.3.1.\n\nAdapted from the project weather range reader; no old model code or weights.

No new weather source, interpolation, training, package installation or missing
value replacement. Retrieve generation-pinned compressed blocks intersecting
the requested latitude strips and pressure levels. Reserved anonymous buffer
space is not weather: only fully fetched blocks can be passed to getitem.
"""
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import mmap
import re
import ssl
import struct
import threading
import time
from urllib.parse import urlsplit

import numpy as np
import numcodecs.blosc
from numcodecs import Blosc

import source_contract as original

SURFACE = ["mean_sea_level_pressure", "surface_pressure", "10m_u_component_of_wind",
           "10m_v_component_of_wind", "2m_temperature", "sea_surface_temperature"]
UPPER = ["u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity", "geopotential"]
LEVELS = [850, 700, 500, 300, 200]
_codec_lock = threading.Lock()
_lib = ctypes.CDLL(numcodecs.blosc.__file__)
_getitem = _lib.blosc_getitem
_getitem.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_getitem.restype = ctypes.c_int
_validate = _lib.blosc_cbuffer_validate
_validate.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_validate.restype = ctypes.c_int


@dataclass(frozen=True)
class ObjectIdentity:
    total: int
    generation: str
    etag: str


class HTTPReader:
    """Persistent per-thread public HTTPS connections, bounded finite retries."""
    def __init__(self):
        self.local = threading.local()
        self.lock = threading.Lock()
        self.bytes = 0
        self.requests = 0
        self.context = ssl.create_default_context()

    def get(self, url, interval=None, expected=None, maximum=16 * 1024**2):
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname != "storage.googleapis.com" or parts.query:
            raise ValueError("Only the grounded public GCS source is permitted")
        path = parts.path
        headers = {"Accept-Encoding": "identity", "User-Agent": "Trackformer-lossless-range-preparation/1"}
        if interval is not None:
            lo, hi = interval
            if not 0 <= lo < hi:
                raise ValueError("Invalid half-open byte range")
            headers["Range"] = f"bytes={lo}-{hi-1}"
            maximum = hi - lo
        if expected is not None:
            path += "?generation=" + expected.generation
            headers["If-Match"] = expected.etag
        for attempt in range(3):
            response = None
            try:
                connection = getattr(self.local, "connection", None)
                if connection is None:
                    connection = http.client.HTTPSConnection(parts.hostname, timeout=45, context=self.context)
                    self.local.connection = connection
                connection.request("GET", path, headers=headers)
                response = connection.getresponse()
                metadata = {k.lower(): v for k, v in response.getheaders()}
                if response.status in (429, 500, 502, 503, 504):
                    raise OSError(f"Retryable public-source HTTP {response.status}")
                required_status = 206 if interval is not None else 200
                if response.status != required_status:
                    raise ValueError(f"Source HTTP {response.status}, expected {required_status}")
                length = int(metadata.get("content-length", -1))
                if not 0 <= length <= maximum or (interval is not None and length != maximum):
                    raise ValueError("Source ignored range or supplied invalid length")
                data = response.read(maximum + 1)
                if len(data) != length:
                    raise OSError("Truncated public-source response")
                if metadata.get("content-encoding", "identity") != "identity":
                    raise ValueError("Unexpected HTTP content encoding")
                total = length
                if interval is not None:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", metadata.get("content-range", ""))
                    if match is None or (int(match[1]), int(match[2]) + 1) != interval:
                        raise ValueError("Wrong Content-Range")
                    total = int(match[3])
                generation, etag = metadata.get("x-goog-generation", ""), metadata.get("etag", "")
                if not generation.isdigit() or not etag:
                    raise ValueError("Missing immutable public-source object identity")
                observed = ObjectIdentity(total, generation, etag)
                if expected is not None and observed != expected:
                    raise ValueError("Source object changed between partial reads")
                with self.lock:
                    self.bytes += len(data)
                    self.requests += 1
                return data, observed
            except (OSError, http.client.HTTPException):
                connection = getattr(self.local, "connection", None)
                if connection is not None:
                    connection.close()
                    self.local.connection = None
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
            finally:
                if response is not None:
                    response.close()
        raise RuntimeError("Unreachable retry state")

    def metadata(self, url):
        raw, identity = self.get(url, maximum=2 * 1024**2)
        return json.loads(raw), dict(generation=identity.generation, etag=identity.etag,
                                    sha256=hashlib.sha256(raw).hexdigest())


def merged_ranges(intervals):
    result = []
    for lo, hi in sorted(set(intervals)):
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((lo, hi))
    return result


def read_spans(remote, url, spans, expected_items):
    """Read only declared contiguous float32 spans with full-block coverage.

    Blosc may physically reorder blocks during multithreaded compression. The
    offset table is sorted by physical offset only to find encoded block ends;
    logical block indices and requested item offsets remain unchanged.
    """
    if not spans or any(not 0 <= s < s + n <= expected_items for s, n in spans):
        raise ValueError("Invalid requested items")
    header, object_id = remote.get(url, (0, 16))
    version, _, flags, typesize, nbytes, blocksize, cbytes = struct.unpack("<BBBBIII", header)
    if (version != 2 or typesize != 4 or nbytes != expected_items * 4 or
            cbytes != object_id.total or not 16 <= cbytes <= nbytes + 16 or
            not 0 < blocksize <= nbytes or blocksize % 4 or flags & 8):
        raise ValueError("Unexpected or unsafe native Blosc header")
    downloaded = 16
    hashes = [{"start": 0, "end": 16, "sha256": hashlib.sha256(header).hexdigest()}]
    if flags & 2:
        if cbytes != nbytes + 16:
            raise ValueError("Invalid memcpy chunk")
        values = []
        for start, count in spans:
            lo, hi = 16 + start * 4, 16 + (start + count) * 4
            raw, _ = remote.get(url, (lo, hi), object_id)
            downloaded += len(raw)
            hashes.append(dict(start=lo, end=hi, sha256=hashlib.sha256(raw).hexdigest()))
            values.append(np.frombuffer(raw, "<f4").copy())
    else:
        nblocks = math.ceil(nbytes / blocksize)
        table_end = 16 + nblocks * 4
        if table_end >= cbytes or nblocks > 65536:
            raise ValueError("Unsafe block table")
        table, _ = remote.get(url, (16, table_end), object_id)
        downloaded += len(table)
        hashes.append(dict(start=16, end=table_end, sha256=hashlib.sha256(table).hexdigest()))
        offsets = np.frombuffer(table, "<u4").astype(np.int64)
        if len(set(offsets.tolist())) != nblocks or (offsets < table_end).any() or (offsets >= cbytes).any():
            raise ValueError("Invalid compressed block offsets")
        physical = sorted(offsets.tolist())
        ends = dict(zip(physical, physical[1:] + [cbytes]))
        needed = set()
        for start, count in spans:
            needed.update(range(start * 4 // blocksize, ((start + count) * 4 - 1) // blocksize + 1))
        intervals = merged_ranges([(int(offsets[i]), ends[int(offsets[i])]) for i in needed])
        # Sparse anonymous reservation avoids touching the unrequested 37-level
        # source payload. Padding is NEVER decoded/exported as meteorology.
        with mmap.mmap(-1, cbytes) as encoded:
            encoded[:16], encoded[16:table_end] = header, table
            for lo, hi in intervals:
                raw, _ = remote.get(url, (lo, hi), object_id)
                encoded[lo:hi] = raw
                downloaded += len(raw)
                hashes.append(dict(start=lo, end=hi, sha256=hashlib.sha256(raw).hexdigest()))
            fetched = set()
            for index in needed:
                lo, hi = int(offsets[index]), ends[int(offsets[index])]
                if any(a <= lo and hi <= b for a, b in intervals):
                    fetched.add(index)
            if fetched != needed:
                raise ValueError("A requested native block was not fetched")
            anchor = ctypes.c_char.from_buffer(encoded)
            address = ctypes.addressof(anchor)
            declared = ctypes.c_size_t()
            values = []
            try:
                with _codec_lock:
                    if _validate(address, cbytes, ctypes.byref(declared)) != 0 or declared.value != nbytes:
                        raise ValueError("Native codec buffer safety validation failed")
                    for start, count in spans:
                        destination = np.empty(count, dtype="<f4")
                        copied = _getitem(address, start, count, destination.ctypes.data)
                        if copied != count * 4:
                            raise ValueError("Incomplete native block decompression")
                        values.append(destination)
            finally:
                # Release the exported buffer even if codec validation fails.
                del anchor
    proof = dict(url=url, generation=object_id.generation, etag=object_id.etag,
                 full_object_bytes=cbytes, downloaded_bytes=downloaded,
                 uncompressed_bytes=nbytes, requested_item_spans=[list(v) for v in spans],
                 requested_blocks_complete=True, partial_range_sha256=hashes,
                 full_source_object_sha256_verified=False)
    return values, proof


class RangeSource:
    def __init__(self, network_workers=16):
        if not 1 <= network_workers <= 96:
            raise ValueError("Network workers must be bounded at 1..96")
        self.http = HTTPReader()
        self.pool = ThreadPoolExecutor(max_workers=network_workers)
        self.root, self.metadata_receipts = original.SOURCE, {}
        self.attributes, proof = self.http.metadata(self.root + "/.zattrs")
        self.metadata_receipts[".zattrs"] = proof
        self.arrays = {}
        for name in SURFACE + UPPER + ["latitude", "longitude", "level", "time"]:
            metadata, p = self.http.metadata(f"{self.root}/{name}/.zarray")
            attrs, a = self.http.metadata(f"{self.root}/{name}/.zattrs")
            if (metadata["zarr_format"] != 2 or metadata["order"] != "C" or metadata["filters"] is not None
                    or metadata["compressor"]["id"] != "blosc" or
                    any(k in attrs for k in ("scale_factor", "add_offset", "_FillValue"))):
                raise ValueError("Changed native source metadata")
            self.arrays[name] = (metadata, attrs)
            self.metadata_receipts[name] = dict(array=p, attributes=a)
        self.lat = self.coordinate("latitude")
        self.lon = self.coordinate("longitude")
        self.levels = self.coordinate("level").tolist()
        np.testing.assert_array_equal(self.lat, np.linspace(90, -90, 721))
        np.testing.assert_array_equal(self.lon, np.arange(1440) * .25)
        self.ys = np.flatnonzero((self.lat >= 0) & (self.lat <= 60))
        self.xs = np.flatnonzero((self.lon >= 100) & (self.lon <= 180))
        self.rl, self.ro = self.lat[self.ys], self.lon[self.xs]
        self.indices = [self.levels.index(v) for v in LEVELS]
        self.time_cache, self.time_lock = {}, threading.Lock()
        time_meta, time_attrs = self.arrays["time"]
        if (time_attrs["units"] != "hours since 1900-01-01 00:00:00" or
                time_attrs["calendar"] != "proleptic_gregorian" or time_meta["dtype"] != "<i8"):
            raise ValueError("Changed exact time coordinate")
        for name in SURFACE + UPPER:
            metadata, attrs = self.arrays[name]
            shape = [time_meta["shape"][0], 721, 1440]
            chunks = [1, 721, 1440]
            dimensions = ["time", "latitude", "longitude"]
            if name in UPPER:
                shape.insert(1, 37)
                chunks.insert(1, 37)
                dimensions.insert(1, "level")
            if (metadata["shape"] != shape or metadata["chunks"] != chunks or metadata["dtype"] != "<f4"
                    or attrs["_ARRAY_DIMENSIONS"] != dimensions):
                raise ValueError("Changed native field geometry/order")

    def coordinate(self, name, chunk=0):
        metadata, _ = self.arrays[name]
        raw, identity = self.http.get(f"{self.root}/{name}/{chunk}", maximum=16 * 1024**2)
        decoded = Blosc(**{k: v for k, v in metadata["compressor"].items() if k != "id"}).decode(raw)
        values = np.frombuffer(decoded, dtype=metadata["dtype"]).copy()
        self.metadata_receipts[f"{name}/{chunk}"] = dict(generation=identity.generation, etag=identity.etag,
                                                        sha256=hashlib.sha256(raw).hexdigest())
        return values

    def time_index(self, seconds):
        date = np.datetime64(int(seconds), "s")
        if not np.datetime64(self.attributes["valid_time_start"]) <= date <= np.datetime64(self.attributes["valid_time_stop"]):
            raise ValueError("Time outside final ERA5; no ERA5T substitute")
        origin = int(np.datetime64("1900-01-01", "s").astype(np.int64))
        if (seconds - origin) % 3600:
            raise ValueError("Non-native issue time")
        index = (int(seconds) - origin) // 3600
        metadata, _ = self.arrays["time"]
        if not 0 <= index < metadata["shape"][0]:
            raise ValueError("Time index outside source")
        chunk, offset = divmod(index, metadata["chunks"][0])
        with self.time_lock:
            if chunk not in self.time_cache:
                self.time_cache[chunk] = self.coordinate("time", chunk)
            if int(self.time_cache[chunk][offset]) != index:
                raise ValueError("Source valid-time coordinate mismatch")
        return index

    def variable(self, name, index):
        upper = name in UPPER
        levels = self.indices if upper else [0]
        plane, row_start, row_count = 721 * 1440, int(self.ys[0]) * 1440, len(self.ys) * 1440
        spans = [(level * plane + row_start, row_count) for level in levels]
        suffix = f"{index}.0.0.0" if upper else f"{index}.0.0"
        values, receipt = read_spans(self.http, f"{self.root}/{name}/{suffix}", spans, plane * (37 if upper else 1))
        regional = np.stack([v.reshape(len(self.ys), 1440)[:, self.xs] for v in values])
        return regional, receipt

    def frame(self, seconds):
        index = self.time_index(seconds)
        futures = {name: self.pool.submit(self.variable, name, index) for name in SURFACE + UPPER}
        output, receipts = np.empty((31, 241, 321), np.float32), []
        for i, name in enumerate(SURFACE):
            values, receipt = futures[name].result()
            if i < 2:
                values = values / 100.0
            if i == 5:
                values = np.where((values >= 260) & (values <= 330), values, np.nan)
            output[i] = values[0]
            receipts.append(receipt)
        for j, name in enumerate(UPPER):
            values, receipt = futures[name].result()
            for level in range(5):
                output[6 + level * 5 + j] = values[level]
            receipts.append(receipt)
        values, valid = original.verify_frame(output, self.rl, self.ro)
        return values, valid, receipts

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

