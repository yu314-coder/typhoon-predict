"""Fetch exact public training inputs/checkpoints, verifying every SHA-256.

Uses Python's standard library. Does not train, install packages, or use Drive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import urllib.parse
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def download(entry, root, base, prefix):
    relative = Path(entry['path']).relative_to(prefix)
    path = root / relative
    if path.exists():
        if path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']:
            print('Verified existing', relative, flush=True)
            return
        raise ValueError('Existing file differs; preserve/move it before retrying: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.downloading')
    url = base + urllib.parse.quote(entry['path'], safe='/')
    print('Downloading', relative, entry['bytes'], 'bytes', flush=True)
    try:
        h = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as out:
            for block in iter(lambda: response.read(8 * 1024**2), b''):
                out.write(block)
                h.update(block)
                size += len(block)
            out.flush()
            os.fsync(out.fileno())
        if size != entry['bytes'] or h.hexdigest() != entry['sha256']:
            raise ValueError('Downloaded content differs: ' + str(relative))
        temp.replace(path)
    except Exception:
        if temp.is_file():
            temp.unlink()
        raise
    print('Verified', relative, flush=True)


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, default=here)
    p.add_argument('--revision', default='main', help='Use the immutable HF revision in PUBLICATION.md for exact publication')
    p.add_argument('--group', choices=['all', 'data', 'checkpoints'], default='all')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    manifest = json.loads((here / 'assets.json').read_text())
    base = 'https://huggingface.co/' + manifest['repository'] + '/resolve/' + urllib.parse.quote(a.revision, safe='') + '/'
    for entry in manifest['files']:
        if a.group != 'all' and entry['group'] != a.group:
            continue
        if a.verify_only:
            path = a.destination / Path(entry['path']).relative_to('training/trackformer_1_2')
            if not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                raise ValueError('Missing or different asset: ' + str(path))
            print('Verified', path, flush=True)
        else:
            download(entry, a.destination, base, 'training/trackformer_1_2')


if __name__ == '__main__':
    main()
