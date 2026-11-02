"""Public source transport contract; no data is fetched on import."""
import numpy as np
from contracts import CHANNELS, LATITUDE, LONGITUDE

SOURCE = 'https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'


def verify_frame(frame, latitude, longitude):
    a = np.asarray(frame, np.float32)
    if a.shape != (31, 241, 321):
        raise ValueError('Expected native quarter-degree 31-field WP frame')
    np.testing.assert_array_equal(latitude, LATITUDE)
    np.testing.assert_array_equal(longitude, LONGITUDE)
    v = np.isfinite(a)
    if not v[np.arange(31) != 5].all():
        raise ValueError('Incomplete atmospheric frame')
    if not 800 < a[0].min() <= a[0].max() < 1120:
        raise ValueError('MSLP must be hPa')
    if not 150 < a[4].min() <= a[4].max() < 350:
        raise ValueError('Temperature must be kelvin')
    return a, v
