"""Build a small README GIF of every genuine state in the published MP4."""
import hashlib
import json
from pathlib import Path
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'docs/trackformer_1_2_mangkhut.mp4'
DEST = ROOT/'docs/trackformer_1_2_mangkhut.gif'
SOURCE_SHA = '220e09eea71af88c9622131f4248be9b9914adae16785c5caee349b1e10abc20'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError('Source is not the verified twenty-state Mangkhut film')
    temporary = DEST.with_name('trackformer_1_2_mangkhut.build.gif')
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'warning',
        '-threads', '2', '-i', str(SOURCE), '-filter_complex_threads', '2',
        '-filter_complex', 'fps=1,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
        '-loop', '0', str(temporary)], check=True)
    hashes, durations = [], []
    with Image.open(temporary) as gif:
        size, loop = gif.size, gif.info.get('loop')
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info['duration'])
            hashes.append(hashlib.sha256(gif.convert('RGB').tobytes()).hexdigest())
    if len(hashes) != 20 or len(set(hashes)) != 20 or durations != [1000]*20 or loop != 0:
        raise ValueError('GIF does not preserve twenty distinct one-second forecast states')
    if temporary.stat().st_size > 10_000_000:
        raise ValueError('GIF is too large for the compact README preview')
    temporary.replace(DEST)
    receipt = {'status': 'verified', 'source_mp4_sha256': SOURCE_SHA,
        'gif_sha256': digest(DEST), 'size_bytes': DEST.stat().st_size, 'dimensions': list(size),
        'distinct_frames': len(set(hashes)), 'frame_durations_ms': durations,
        'duration_seconds': sum(durations)/1000, 'infinite_loop': True,
        'frame_rgb_sha256': hashes, 'preview_only': True,
        'forecast_arrays_modified': False, 'source_movie_modified': False,
        'note': 'One rendered frame per genuine forecast state; palette/downscale preview, no new inference or temporal interpolation. Full-quality MP4 remains available.'}
    DEST.with_suffix('.gif.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'frame_rgb_sha256'}, indent=2))


if __name__ == '__main__':
    main()
