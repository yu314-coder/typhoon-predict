"""Small, exact f000 analysis subsets from the public NOAA GFS archive.

Only eight indexed GRIB messages are fetched, not full global forecast files.
The caller additionally validates the GRIB timestamps, levels and physical units.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'
CHANNELS = [('PRMSL', 'mean sea level'), ('HGT', '500 mb'),
            ('UGRD', '850 mb'), ('VGRD', '850 mb'),
            ('UGRD', '500 mb'), ('VGRD', '500 mb'),
            ('UGRD', '200 mb'), ('VGRD', '200 mb')]


def analysis_url(cycle):
    if cycle.minute or cycle.second or cycle.microsecond or cycle.hour % 6:
        raise ValueError('GFS archive requires an exact six-hour analysis cycle')
    return (f'{BASE}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/'
            f'gfs.t{cycle:%H}z.pgrb2.1p00.f000')


def analysis_ranges(index, cycle):
    rows = [line.split(':') for line in index.strip().splitlines()]
    if not rows or any(len(row) < 6 for row in rows):
        raise ValueError('Malformed GFS archive index')
    offsets = [int(row[1]) for row in rows]
    if offsets[0] != 0 or any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError('Non-increasing GFS message offsets')
    found = {}
    for i, row in enumerate(rows):
        key = (row[3], row[4])
        if key not in CHANNELS:
            continue
        if row[2] != f'd={cycle:%Y%m%d%H}' or row[5] != 'anl':
            raise ValueError('Archive index must contain the exact f000 analysis')
        if key in found or i + 1 == len(rows):
            raise ValueError('Duplicate or unbounded GFS analysis message')
        found[key] = (offsets[i], offsets[i + 1] - 1)
    if set(found) != set(CHANNELS):
        raise ValueError('Missing GFS archive analysis channels')
    return [found[key] for key in CHANNELS]


def validate_range(response, start, end):
    expected = end - start + 1
    header = response.headers.get('Content-Range', '')
    if (response.status_code != 206 or
            not header.startswith(f'bytes {start}-{end}/') or
            len(response.content) != expected):
        raise ValueError('Server did not return the exact bounded GRIB byte range')
    b = response.content
    if (len(b) < 20 or b[:4] != b'GRIB' or b[7] != 2 or b[-4:] != b'7777'
            or int.from_bytes(b[8:16], 'big') != len(b)):
        raise ValueError('Incomplete or invalid archived GRIB2 message')
    return b


def download_analysis(cycle, cache, get):
    url = analysis_url(cycle)
    p = Path(cache) / f'archive-gfs-{cycle:%Y%m%d%H}.grib2'
    if not p.exists():
        ranges = analysis_ranges(get(url + '.idx').text, cycle)

        def fetch(bounds):
            start, end = bounds
            response = get(url, headers={'Range': f'bytes={start}-{end}'})
            return validate_range(response, start, end)

        with ThreadPoolExecutor(max_workers=4) as pool:
            b = b''.join(pool.map(fetch, ranges))
        p.parent.mkdir(parents=True, exist_ok=True)
        temp = p.with_suffix('.part')
        temp.write_bytes(b)
        temp.replace(p)
    return p, url
