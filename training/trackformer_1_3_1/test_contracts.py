"""CPU-only scientific contract fixtures, not a trained-model qualification."""
import unittest
import numpy as np
from contracts import check_times, history_times, effective_mask, sample_pressure


class Contracts(unittest.TestCase):
    def test_future_or_missing_history_rejected(self):
        t = 1609459200
        check_times(t, history_times(t))
        for bad in (history_times(t)[1:], history_times(t)[::-1], history_times(t + 21600)):
            with self.assertRaises(ValueError):
                check_times(t, bad)

    def test_below_ground_and_negative_humidity_not_training_targets(self):
        a = np.ones((31, 241, 321), np.float32)
        a[1] = 1000
        a[1, 10, 10] = 750
        a[14, 0, 0] = -1e-6
        v = np.ones(a.shape, bool)
        before = a.copy()
        m = effective_mask(a, v)
        self.assertFalse(m[6:11, 10, 10].any())
        self.assertTrue(m[11:16, 10, 10].all())
        self.assertFalse(m[14, 0, 0])
        np.testing.assert_array_equal(before, a)
        self.assertTrue(v.all())

    def test_scalar_is_bilinear_value_of_same_physical_map(self):
        lat, lon = np.array([2., 1., 0.]), np.array([100., 101., 102.])
        p = 980 + lat[:, None] * 2 + lon[None, :] * .1
        m = np.ones(p.shape, bool)
        self.assertAlmostEqual(sample_pressure(p, m, lat, lon, (1.4, 100.7)), 992.87)
        self.assertAlmostEqual(sample_pressure(p, m, lat, lon, (0., 102.)), 990.2)
        self.assertIsNone(sample_pressure(p, m, lat, lon, (0., 103.)))

    def test_unsupported_pressure_is_not_fabricated(self):
        lat, lon = np.array([2., 1., 0.]), np.array([100., 101., 102.])
        p, m = np.full((3, 3), 980.), np.ones((3, 3), bool)
        m[0, 0] = False
        self.assertIsNone(sample_pressure(p, m, lat, lon, (1.5, 100.5)))
        self.assertEqual(sample_pressure(p, m, lat, lon, (1., 101.)), 980.)

    def test_moving_low_readout_moves_on_unchanged_geography(self):
        lat, lon = np.linspace(30, 10, 81), np.linspace(110, 140, 121)
        yy, xx = np.meshgrid(lat, lon, indexing="ij")
        for y, x in [(20., 120.), (22., 123.), (25., 126.)]:
            p = 1010 - 50 * np.exp(-((yy-y)**2+(xx-x)**2)/2)
            self.assertAlmostEqual(sample_pressure(p, np.ones(p.shape, bool), lat, lon, (y, x)), 960.)


if __name__ == "__main__":
    unittest.main()
