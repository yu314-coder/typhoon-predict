"""Regression: never add an extra frozen forecast tail to new encodes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_fung_wong_video import encode_video


class VideoPlaybackTest(unittest.TestCase):
    def test_all_twenty_states_without_padding(self):
        with tempfile.TemporaryDirectory(dir='/Volumes/D/typhoon_predict/output') as name:
            work = Path(name)
            for i in range(20):
                (work/f'frame_{i:03d}.png').touch()
            video = work/'video.mp4'
            video.touch()
            probe = {'streams': [{'codec_name': 'h264', 'width': 1600,
                      'height': 1000, 'nb_frames': '600'}], 'format': {'duration': '20'}}
            with patch('build_fung_wong_video.subprocess.run') as run, patch(
                    'build_fung_wong_video.subprocess.check_output', return_value=json.dumps(probe).encode()):
                result = encode_video(work, video)
                command = run.call_args.args[0]
                self.assertNotIn('tpad', ' '.join(command))
                self.assertEqual(command[command.index('-frames:v')+1], '600')
                self.assertEqual(result['extra_final_hold_seconds'], 0)
                self.assertEqual(result['duration_seconds'], 20)

    def test_missing_state_is_not_repeated(self):
        with tempfile.TemporaryDirectory(dir='/Volumes/D/typhoon_predict/output') as name:
            with self.assertRaisesRegex(ValueError, 'Missing or unexpected'):
                encode_video(Path(name), Path(name)/'video.mp4')


if __name__ == '__main__':
    unittest.main()
