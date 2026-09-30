import copy
import unittest

import numpy as np

from forecast_historical_video50_mac import requested_cases, exact_history_indices
from storm_structure_diagnostics import (
    initialize, radius_km, envelope, outer_radius, diagnose, summarize,
    ambient_pressure, official_radius_mae, THRESHOLDS, QUADRANTS, METRICS,
)


class StormStructureTests(unittest.TestCase):
    def row(self):
        row = dict(SID='2015211N13162', ISO_TIME='2015-08-05 00:00:00',
                   USA_WIND='100', TOKYO_PRES='950', USA_RMW='20',
                   LAT='20', LON='135', USA_LAT='20', USA_LON='135',
                   USA_AGENCY='jtwc_wp', TRACK_TYPE='main')
        for t, r in ((34, 140), (50, 80), (64, 50)):
            for i, q in enumerate(QUADRANTS):
                row[f'USA_R{t}_{q}'] = str(r+i*5)
        return row

    def initial(self, row=None):
        return initialize(row or self.row(), '2015-08-05T00:00:00Z', 1010)

    def member(self, cp=950, initial=None):
        return diagnose(initial or self.initial(), cp, {'value_hpa':1010, 'status':'available'})

    def test_nautical_miles_are_radius_not_diameter(self):
        self.assertAlmostEqual(radius_km({'USA_RMW':'12'}, 'USA_RMW', rmw=True), 22.224)
        self.assertEqual(radius_km({'USA_R34_NE':'0'}, 'USA_R34_NE'), 0)
        self.assertIsNone(radius_km({'USA_R34_NE':''}, 'USA_R34_NE'))
        self.assertIsNone(radius_km({'USA_RMW':'0'}, 'USA_RMW', rmw=True))

    def test_exact_issue_only_and_no_future_influence(self):
        before = self.initial()
        row = self.row()
        row.update(future_wind='999', future_r34='1', USA_EYE='700', TOKYO_R30_LONG='900')
        self.assertEqual(before, self.initial(row))
        row['ISO_TIME'] = '2015-08-05 06:00:00'
        with self.assertRaises(ValueError):
            self.initial(row)

    def test_native_quadrant_isotachs_initialized_without_rms_or_period_conversion(self):
        initial = self.initial()
        for q in QUADRANTS:
            points, warnings = envelope(initial, q)
            self.assertEqual(warnings, [])
            for t in THRESHOLDS:
                self.assertAlmostEqual(outer_radius(points, initial['holland_inspired_tail_b'], t),
                                       initial['native_quadrant_radii_km'][f'r{t}_{q}'])
        result = self.member()
        self.assertNotEqual(result['r34_NE_estimate_km'], result['r34_SW_estimate_km'])
        self.assertEqual(initial['definitions']['target_wind_averaging_seconds'], 60)
        self.assertIsNone(initial['definitions']['validated_wind_averaging_seconds'])

    def test_pressure_drives_future_and_does_not_change_rmw_persistence(self):
        initial = self.initial()
        before = copy.deepcopy(initial)
        weak, strong = self.member(980, initial), self.member(900, initial)
        self.assertGreater(strong['wind_estimate_kt'], weak['wind_estimate_kt'])
        self.assertGreater(strong['r34_NE_estimate_km'], weak['r34_NE_estimate_km'])
        self.assertEqual(strong['rmw_persistence_km'], weak['rmw_persistence_km'])
        self.assertEqual(initial, before)

    def test_missing_rmw_does_not_block_wind_or_invent_a_radius(self):
        row = self.row()
        row['USA_RMW'] = ''
        result = self.member(initial=self.initial(row))
        self.assertIsNotNone(result['wind_estimate_kt'])
        self.assertIsNone(result['rmw_persistence_km'])
        self.assertTrue(all(result[f'r{t}_{q}_estimate_km'] is None for t in THRESHOLDS for q in QUADRANTS))

    def test_below_threshold_is_candidate_zero_not_missing_reference(self):
        result = self.member(1005)
        self.assertLess(result['wind_estimate_kt'], 34)
        self.assertEqual(result['r34_NE_estimate_km'], 0)
        self.assertIsNone(outer_radius([(30, 100)], 1, 1))  # no last-edge substitute

    def test_missing_zero_or_inconsistent_initial_quadrant_is_visible(self):
        row = self.row()
        row['USA_R34_NE'] = '0'
        result = self.member(initial=self.initial(row))
        self.assertIsNone(result['r34_NE_estimate_km'])
        self.assertIsNotNone(result['r34_SE_estimate_km'])
        self.assertIn('reported_zero', result['warnings'][0])
        row['USA_R34_NE'] = '1'
        points, warnings = envelope(self.initial(row), 'NE')
        self.assertTrue(any('inconsistent' in w for w in warnings))
        self.assertTrue(points)

    def test_missing_pressure_invalid_tracks_do_not_create_wind(self):
        initial = self.initial()
        for cp, ambient, valid in ((np.nan, {'value_hpa':1010, 'status':'available'}, True),
                                   (950, {'value_hpa':None, 'status':'insufficient_ambient_coverage'}, True),
                                   (950, {'value_hpa':1010, 'status':'available'}, False)):
            result = diagnose(initial, cp, ambient, track_valid=valid)
            self.assertTrue(all(result[k] is None for k in METRICS))

    def test_true_member_mean_noncommutativity_and_partial_members(self):
        a, b = self.member(980), self.member(900)
        result = summarize([a,b], expected_members=2)['estimates']['wind_estimate_kt']
        self.assertAlmostEqual(result['mean'], (a['wind_estimate_kt']+b['wind_estimate_kt'])/2)
        self.assertNotAlmostEqual(result['mean'], self.member(940)['wind_estimate_kt'], places=3)
        broken = dict(a, wind_estimate_kt=None)
        result = summarize([a]*49+[broken], expected_members=50)['estimates']['wind_estimate_kt']
        self.assertIsNone(result['mean'])
        self.assertEqual(result['valid_members'],49)
        with self.assertRaises(ValueError):
            summarize([a,b], expected_members=50)

    def test_ambient_uses_physical_grid_cells_and_fails_closed_at_domain(self):
        lat, lon = np.linspace(60,0,25), np.linspace(100,180,33)
        p = np.full((25,33),1010.)
        before = p.copy()
        result = ambient_pressure(p,lat,lon,[20,135])
        self.assertEqual(result['value_hpa'],1010)
        self.assertGreaterEqual(min(result['quadrant_cells'].values()),2)
        self.assertIsNone(ambient_pressure(p,lat,lon,[20,181])['value_hpa'])
        self.assertIsNone(ambient_pressure(p*np.nan,lat,lon,[20,135])['value_hpa'])
        np.testing.assert_array_equal(p,before)

    def test_strict_skill_gate_not_bypassed_by_matching_units_or_targets(self):
        with self.assertRaisesRegex(ValueError,'calibration'):
            official_radius_mae(self.member(),self.row())

    def test_generic_forecast_case_validation_and_causal_exact_history(self):
        cases=requested_cases(None,[['nari','2001248N22135','2001-09-08T00:00:00Z']])
        self.assertEqual(len(cases),1)
        for slug, sid, issue in (('../bad','2001248N22135','2001-09-08T00:00:00Z'),
                                  ('nari','wrong','2001-09-08T00:00:00Z'),
                                  ('nari','2001248N22135','2001-09-08T03:00:00Z')):
            with self.assertRaises(ValueError):
                requested_cases(None,[[slug,sid,issue]])
        hour=3600*10**9
        end=np.datetime64('2001-09-08T00:00:00','ns').astype('int64')
        times=end+np.arange(-8,2)*6*hour
        indices=exact_history_indices(times,'2001-09-08T00:00:00Z')
        self.assertEqual(int(times[indices[-1]]),end)
        with self.assertRaises(ValueError):
            exact_history_indices(np.delete(times,4),'2001-09-08T00:00:00Z')


if __name__=='__main__':
    unittest.main()
