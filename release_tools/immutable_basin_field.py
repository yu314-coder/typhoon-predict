"""Read and verify original basin exports without rewriting their storage.

The original archive contains both plain JSON and lossless gzip JSON. Hash the
actual source file, not a recompressed or reserialized substitute. Existing
gzip-only sidecar receipts remain valid.
"""
import gzip
import hashlib
import json


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def basin_path(output, ident):
    for suffix in ('.json.gz', '.json'):
        path = output/'fields'/f'{ident}{suffix}'
        if path.is_file():
            return path
    raise FileNotFoundError(f'Missing immutable basin field: {ident}')


def read_basin_field(output, ident):
    path = basin_path(output, ident)
    raw = path.read_bytes()
    compressed = path.name.endswith('.json.gz')
    document = json.loads(gzip.decompress(raw) if compressed else raw)
    hashes = dict(basin_field_file=path.name,
                  basin_field_format='json.gz' if compressed else 'json',
                  basin_field_sha256=hashlib.sha256(raw).hexdigest())
    if compressed:
        hashes['basin_field_gzip_sha256'] = hashes['basin_field_sha256']
    return document, hashes


def verify_basin_source(output, ident, source):
    if 'basin_field_sha256' in source:
        fmt = source.get('basin_field_format')
        if fmt not in ('json', 'json.gz') or source.get('basin_field_file') != f'{ident}.{fmt}':
            raise ValueError('Invalid immutable basin source file/format')
        path = output/'fields'/f'{ident}.{fmt}'
        expected = source['basin_field_sha256']
        if 'basin_field_gzip_sha256' in source and (
                fmt != 'json.gz' or source['basin_field_gzip_sha256'] != expected):
            raise ValueError('Conflicting immutable basin hashes changed')
    else:
        # Existing receipts pinned the compressed file explicitly. Never fall
        # back to a plain file when that original compressed source is missing.
        path = output/'fields'/f'{ident}.json.gz'
        expected = source['basin_field_gzip_sha256']
    if not path.is_file() or sha(path) != expected:
        raise ValueError('Immutable basin field hashes changed')
    return path
