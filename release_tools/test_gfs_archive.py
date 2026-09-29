import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from gfs_archive import CHANNELS, analysis_ranges, analysis_url, validate_range


class ArchiveContract(unittest.TestCase):
    def setUp(self):
        self.cycle = datetime(2026, 4, 9, tzinfo=timezone.utc)
        self.index = '\n'.join(
            f'{i+1}:{i*100}:d=2026040900:{var}:{level}:anl:'
            for i, (var, level) in enumerate(CHANNELS + [('TMP', 'surface')]))

    def test_exact_eight_analysis_ranges(self):
        self.assertEqual(analysis_ranges(self.index, self.cycle),
                         [(i*100, i*100+99) for i in range(8)])
        self.assertTrue(analysis_url(self.cycle).endswith('.f000'))

    def test_wrong_date_or_forecast_step_rejected(self):
        for index in [self.index.replace('d=2026040900', 'd=2026040906'),
                      self.index.replace(':anl:', ':6 hour fcst:')]:
            with self.assertRaisesRegex(ValueError, 'exact f000'):
                analysis_ranges(index, self.cycle)

    def test_missing_or_duplicate_channel_rejected(self):
        missing = self.index.replace('PRMSL', 'TMP')
        duplicate = self.index.replace('TMP:surface', 'PRMSL:mean sea level')
        for index in [missing, duplicate]:
            with self.assertRaises(ValueError):analysis_ranges(index, self.cycle)

    def test_non_six_hour_cycle_rejected(self):
        with self.assertRaisesRegex(ValueError, 'six-hour'):
            analysis_url(self.cycle.replace(hour=3))

    def test_non_increasing_offsets_rejected(self):
        with self.assertRaisesRegex(ValueError, 'offsets'):
            analysis_ranges(self.index.replace(':100:', ':0:'), self.cycle)

    def response(self):
        b = b'GRIB' + bytes([0, 0, 0, 2]) + (20).to_bytes(8, 'big') + b'7777'
        return SimpleNamespace(status_code=206,
                               headers={'Content-Range':'bytes 100-119/2000'}, content=b)

    def test_exact_range_verified(self):
        response=self.response()
        self.assertEqual(validate_range(response,100,119),response.content)

    def test_ignored_range_or_wrong_range_rejected(self):
        response=self.response();response.status_code=200
        with self.assertRaisesRegex(ValueError,'exact bounded'):
            validate_range(response,100,119)
        with self.assertRaisesRegex(ValueError,'exact bounded'):
            validate_range(self.response(),0,19)

    def test_incomplete_grib_rejected(self):
        response=self.response();response.content=response.content[:-4]+b'0000'
        with self.assertRaisesRegex(ValueError,'invalid'):
            validate_range(response,100,119)


if __name__=='__main__':unittest.main()
