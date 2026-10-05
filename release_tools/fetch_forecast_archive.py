"""Bounded retry of the large immutable archive fetch; no inference or reset."""
import argparse
import subprocess
import time
from pathlib import Path


def fetch_archive(checkout):
    command = ['git', '-c', 'http.version=HTTP/1.1', 'fetch', '--depth=1', 'origin', 'forecast-data']
    for attempt in range(3):
        result = subprocess.run(command, cwd=checkout)
        if result.returncode == 0:
            return
        if attempt < 2:
            delay = 15*(attempt+1)
            print(f'Archive fetch failed; retry {attempt+2}/3 in {delay}s', flush=True)
            time.sleep(delay)
    raise subprocess.CalledProcessError(result.returncode, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    args = parser.parse_args()
    fetch_archive(args.checkout)


if __name__ == '__main__':
    main()
