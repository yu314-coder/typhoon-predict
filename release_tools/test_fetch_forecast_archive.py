import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fetch_forecast_archive import fetch_archive


class BoundedArchiveFetch(unittest.TestCase):
    @patch('fetch_forecast_archive.time.sleep')
    @patch('fetch_forecast_archive.subprocess.run')
    def test_success_never_retries_or_resets_existing_files(self, run, sleep):
        run.return_value = Mock(returncode=0)
        fetch_archive(Path('/existing-archive'))
        run.assert_called_once_with(['git','-c','http.version=HTTP/1.1','fetch','--depth=1','origin','forecast-data'],cwd=Path('/existing-archive'))
        sleep.assert_not_called()

    @patch('fetch_forecast_archive.time.sleep')
    @patch('fetch_forecast_archive.subprocess.run')
    def test_transient_bad_pack_is_retried_with_backoff(self, run, sleep):
        run.side_effect = [Mock(returncode=128),Mock(returncode=0)]
        fetch_archive(Path('/existing-archive'))
        self.assertEqual(run.call_count,2)
        sleep.assert_called_once_with(15)

    @patch('fetch_forecast_archive.time.sleep')
    @patch('fetch_forecast_archive.subprocess.run')
    def test_persistent_failure_stops_after_three_attempts(self, run, sleep):
        run.return_value = Mock(returncode=128)
        with self.assertRaises(subprocess.CalledProcessError):
            fetch_archive(Path('/existing-archive'))
        self.assertEqual(run.call_count,3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list],[15,30])


if __name__ == '__main__':
    unittest.main()
