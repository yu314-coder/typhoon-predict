import copy
import unittest
from datetime import datetime,timezone
from automatic_forecasts import jma_analysis_row


class JmaAnalysis(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,9,29,19,tzinfo=timezone.utc)
        self.parts=[{'part':'title','name':{'en':'Surigae'},'issue':{'UTC':'2026-09-29T18:45:00Z'}},
                    {'part':{'en':'Analysis'},'advancedHours':0,'validtime':{'UTC':'2026-09-29T18:00:00Z'},
                     'position':{'deg':[29.4,136.7]},'pressure':'996',
                     'maximumWind':{'sustained':{'kt':'40','m/s':'20'}}},
                    {'part':{'en':'Forecast for 12 hours ahead'},'advancedHours':12,
                     'position':{'deg':[31,138.4]},'pressure':'900'}]
    def test_real_analysis_not_forecast_or_bulletin_time(self):
        row=jma_analysis_row('TC2632',self.parts,self.now)
        self.assertEqual((row['lat'],row['lon'],row['pressure_hpa'],row['wind_kt']),(29.4,136.7,996,40))
        self.assertEqual(row['issue_time_utc'],'2026-09-29T18:00:00Z')
    def test_future_or_stale_analysis_rejected(self):
        for stamp in ['2026-09-30T00:00:00Z','2026-09-28T00:00:00Z']:
            parts=copy.deepcopy(self.parts);parts[1]['validtime']['UTC']=stamp
            with self.assertRaisesRegex(ValueError,'stale or future'):jma_analysis_row('TC2632',parts,self.now)
    def test_forecast_mislabelled_analysis_rejected(self):
        self.parts[1]['advancedHours']=12
        with self.assertRaisesRegex(ValueError,'not a forecast'):jma_analysis_row('TC2632',self.parts,self.now)
    def test_missing_values_remain_missing(self):
        self.parts[1]['pressure']='-';self.parts[1]['maximumWind']['sustained']['kt']=None
        row=jma_analysis_row('TC2632',self.parts,self.now)
        self.assertIsNone(row['pressure_hpa']);self.assertIsNone(row['wind_kt'])
    def test_ambiguous_analysis_rejected(self):
        self.parts.append(copy.deepcopy(self.parts[1]))
        with self.assertRaisesRegex(ValueError,'exactly one'):jma_analysis_row('TC2632',self.parts,self.now)


if __name__=='__main__':unittest.main()
