import unittest
from datetime import timedelta
from verify_forecast_archive import CHECKPOINT,stamp,verify_issue


class ArchiveIntegrity(unittest.TestCase):
    def setUp(self):
        issue='2026-04-09T00:00:00Z';start=stamp(issue)
        self.row={'id':'case','storm_id':'storm','issue_time_utc':issue,'lat':8.7,'lon':152.1,'pressure_hpa':1000}
        self.forecast={'id':'case','storm_id':'storm','issue_time_utc':issue,'members':1,
            'checkpoint_sha256':CHECKPOINT,'input_tensor_sha256':'a'*64,
            'input_history_times_utc':[(start+timedelta(hours=i*6)).isoformat() for i in range(-8,1)],
            'route':[{'lat':8.7,'lon':152.1,'pressure_hpa':1000,'lead_hours':i*6,
                      'valid_time_utc':(start+timedelta(hours=i*6)).isoformat()} for i in range(21)]}
        self.field={'forecast_id':'case','members':1,'checkpoint_sha256':CHECKPOINT,'units':'hPa',
            'latitude':list(range(25)),'longitude':list(range(33)),
            'valid_times_utc':[(start+timedelta(hours=i*6)).isoformat() for i in range(1,21)],
            'pressure_hpa':[[[1000.0]*33 for _ in range(25)] for _ in range(20)]}

    def test_complete_causal_output(self):
        verify_issue(self.row,self.forecast,self.field)

    def test_wrong_start_rejected(self):
        self.forecast['route'][0]['lat']=9
        with self.assertRaisesRegex(ValueError,'align'):verify_issue(self.row,self.forecast,self.field)

    def test_truncated_pressure_rejected(self):
        self.field['pressure_hpa'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete'):verify_issue(self.row,self.forecast,self.field)

    def test_future_input_rejected(self):
        self.forecast['input_history_times_utc'][-1]='2026-04-09T06:00:00Z'
        with self.assertRaisesRegex(ValueError,'Non-causal'):verify_issue(self.row,self.forecast,self.field)

    def test_wrong_pressure_time_rejected(self):
        self.field['valid_times_utc'][0]='2026-04-09T12:00:00Z'
        with self.assertRaisesRegex(ValueError,'lead/time'):verify_issue(self.row,self.forecast,self.field)

    def test_nonfinite_pressure_rejected(self):
        self.field['pressure_hpa'][0][0][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'physical'):verify_issue(self.row,self.forecast,self.field)

    def test_wrong_release_rejected(self):
        self.field['checkpoint_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'identity'):verify_issue(self.row,self.forecast,self.field)


if __name__=='__main__':unittest.main()
