import json
import unittest
from unittest.mock import patch
from datetime import datetime,timezone,timedelta
import numpy as np
import automatic_forecasts as a

class CausalContract(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((a.MODEL/'manifest.json').read_text())['data_contract']
        self.row={'issue_time_utc':'2026-09-29T06:00:00Z','lat':20.,'lon':130.,'pressure_hpa':980.,'wind_kt':None}
        self.t=a.ns(self.row['issue_time_utc'])+np.arange(-8,1)*6*a.HOUR
        self.w=np.zeros((9,8,25,33),dtype='float32')
    def test_future_rejected(self):
        with self.assertRaisesRegex(ValueError,'causal'):a.inputs(self.w,self.t+6*a.HOUR,self.row,self.c,{})
    def test_gap_rejected(self):
        t=self.t.copy();t[2]+=a.HOUR
        with self.assertRaisesRegex(ValueError,'causal'):a.inputs(self.w,t,self.row,self.c,{})
    def test_stale_rejected(self):
        with self.assertRaisesRegex(ValueError,'causal'):a.inputs(self.w,self.t-18*a.HOUR,self.row,self.c,{})
    def test_nonfinite_rejected(self):
        self.w[0,0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Non-finite'):a.inputs(self.w,self.t,self.row,self.c,{})
    def test_other_basin_rejected(self):
        self.row['lon']=30
        with self.assertRaisesRegex(ValueError,'domain'):a.inputs(self.w,self.t,self.row,self.c,{})
    def test_epoch_is_1970_utc(self):
        self.assertEqual(a.ns('1970-01-01T00:00:00Z'),0)

    def test_pre_cutoff_remains_reanalysis(self):
        row={'issue_time_utc':'2026-03-17T18:00:00Z'}
        with patch.object(a,'ncep',return_value=(self.w,self.t)) as ncep:
            _,_,_,source=a.remote_history(row,None,self.c)
        ncep.assert_called_once_with(row,self.c)
        self.assertEqual(source['provider'],'NOAA NCEP Reanalysis 1')

    def test_later_dates_use_nine_distinct_causal_analyses(self):
        with patch.object(a,'download_analysis',return_value=(a.MODEL/'manifest.json','url')) as download, \
             patch.object(a,'decode_gfs',return_value=self.w[0]), \
             patch.object(a,'ncep',side_effect=AssertionError('R1 source ended')):
            weather,times,note,source=a.remote_history(self.row,None,self.c)
        self.assertEqual(weather.shape,(9,8,25,33))
        np.testing.assert_array_equal(times,self.t)
        self.assertEqual([call.args[0] for call in download.call_args_list],
                         [a.parse(self.row['issue_time_utc'])-timedelta(hours=6*(8-i)) for i in range(9)])
        self.assertTrue(source['experimental_transfer'])
        self.assertIn('retrospective',note)

if __name__=='__main__':unittest.main()
