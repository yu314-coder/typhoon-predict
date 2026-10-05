import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
import automatic_forecasts as a


class ExactNcepRetry(unittest.TestCase):
    def setUp(self):
        self.dates=np.arange(np.datetime64('1978-08-12T00','ns'),
            np.datetime64('1978-08-14T06','ns'),np.timedelta64(6,'h'))
        self.contract={'global_lat':[20.,10.], 'global_lon':[140.,150.]}
        self.specs=[('slp',None,0)]
        self.dataset=xr.Dataset({'slp':(('time','lat','lon'),
            np.full((9,2,2),100000.,dtype='float32'),{'units':'Pa'})},
            coords={'time':self.dates, 'lat':[20.,10.], 'lon':[140.,150.]})

    def test_reopens_identical_source_after_transient_index_error(self):
        with patch.object(xr,'open_dataset',side_effect=[KeyError("index 'time'"),self.dataset]) as reader, \
             patch.object(a.time,'sleep') as sleep:
            result=a.read_ncep_exact(xr,'same-source','slp',self.dates,self.contract,self.specs)
        self.assertEqual(reader.call_count,2)
        self.assertEqual(reader.call_args_list[0],reader.call_args_list[1])
        sleep.assert_called_once_with(1)
        np.testing.assert_array_equal(result[0],np.full((9,2,2),1000.,dtype='float32'))

    def test_missing_exact_timestamp_is_not_nearest_filled(self):
        missing=self.dataset.isel(time=slice(1,None))
        with patch.object(xr,'open_dataset',return_value=missing) as reader, patch.object(a.time,'sleep') as sleep:
            with self.assertRaises(KeyError):
                a.read_ncep_exact(xr,'same-source','slp',self.dates,self.contract,self.specs)
        self.assertEqual(reader.call_count,3)
        self.assertEqual(sleep.call_count,2)

    def test_unit_change_fails_without_substitution_or_retry(self):
        self.dataset.slp.attrs['units']='hPa'
        with patch.object(xr,'open_dataset',return_value=self.dataset) as reader, patch.object(a.time,'sleep') as sleep:
            with self.assertRaisesRegex(ValueError,'Unexpected SLP unit'):
                a.read_ncep_exact(xr,'same-source','slp',self.dates,self.contract,self.specs)
        self.assertEqual(reader.call_count,1)
        sleep.assert_not_called()

    def test_causal_selection_unchanged_by_retries(self):
        future=self.dataset.assign_coords(time=self.dates+np.timedelta64(6,'h'))
        with patch.object(xr,'open_dataset',return_value=future), patch.object(a.time,'sleep'):
            with self.assertRaises(KeyError):
                a.read_ncep_exact(xr,'same-source','slp',self.dates,self.contract,self.specs)


if __name__=='__main__':unittest.main()
