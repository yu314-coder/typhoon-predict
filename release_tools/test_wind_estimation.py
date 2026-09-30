import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models/trackformer_1_2_field'))
from wind_estimation import diagnose_member, diagnose_outputs, summarize_members, unavailable


class PressureWindDiagnostics(unittest.TestCase):
    def vortex(self, depth=60, extent=7):
        lat = 20 + np.linspace(extent, -extent, 141)
        lon = 135 + np.linspace(-extent, extent, 141)
        y, x = np.meshgrid(lat, lon, indexing='ij')
        radius = np.hypot((y-20)*111.2, (x-135)*111.2*np.cos(np.deg2rad(20)))
        p = 1010-depth*np.exp(-radius**2/(2*100**2))
        return diagnose_member(p, lat, lon, [20, 135], 80, source_spacing_km=20)

    def test_physical_vortex_has_finite_peak_and_outward_crossings(self):
        d = self.vortex()
        self.assertEqual(d['status'], 'experimental_estimate')
        self.assertGreater(d['pressure_wind_estimate_kt'], 64)
        self.assertGreater(d['r34_estimate_km'], d['r50_estimate_km'])
        self.assertGreater(d['r50_estimate_km'], d['r64_estimate_km'])
        self.assertGreater(d['r64_estimate_km'], d['rmw_estimate_km'])

    def test_constant_and_coarse_fields_do_not_invent_radius(self):
        lat, lon = np.linspace(30, 10, 25), np.linspace(125, 145, 33)
        for spacing in (20, 278):
            d = diagnose_member(np.ones((25, 33))*1000, lat, lon, [20, 135],
                                source_spacing_km=spacing)
            self.assertIsNone(d['r34_estimate_km'])
            self.assertIsNone(d['rmw_estimate_km'])

    def test_missing_grid_and_outside_domain_remain_missing(self):
        p = np.full((9, 9), np.nan)
        lat, lon = np.linspace(21, 19, 9), np.linspace(134, 136, 9)
        self.assertEqual(diagnose_member(p,lat,lon,[20,135],source_spacing_km=20)['status'],
                         'insufficient_complete_ring_coverage')
        self.assertEqual(diagnose_member(p,lat,lon,[20,181],source_spacing_km=20)['status'],
                         'outside_model_domain')

    def test_member_mean_is_not_diagnostic_of_mean_pressure(self):
        a, b = self.vortex(40), self.vortex(80)
        summary = summarize_members([a,b])['estimates']['pressure_wind_estimate_kt']
        self.assertAlmostEqual(summary['mean'], (a['pressure_wind_estimate_kt']+b['pressure_wind_estimate_kt'])/2)
        self.assertNotAlmostEqual(summary['mean'], self.vortex(60)['pressure_wind_estimate_kt'],places=3)

    def test_partial_members_do_not_get_a_fifty_member_mean(self):
        summary = summarize_members([self.vortex() for _ in range(49)]+[unavailable('missing')])
        r = summary['estimates']['r34_estimate_km']
        self.assertEqual(r['valid_members'],49)
        self.assertEqual(r['total_members'],50)
        self.assertIsNone(r['mean'])

    def test_float32_core_coordinates_masks_and_inputs_are_preserved(self):
        class ArrayTensor:
            def __init__(self, data): self.data = np.asarray(data)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.data

        lat = (20 + np.linspace(-640, 640, 65)/111.2).astype(np.float32)
        lon = (135 + np.linspace(-640, 640, 65)/(111.2*np.cos(np.deg2rad(20)))).astype(np.float32)
        y, x = np.meshgrid(lat, lon, indexing='ij')
        r = np.hypot((y-20)*111.2, (x-135)*111.2*np.cos(np.deg2rad(20)))
        p = (1010-60*np.exp(-r*r/(2*100**2))).astype(np.float32)
        output = {key:ArrayTensor(value) for key,value in {
            'center': [[20,135]], 'vmax':[80], 'core':p[None,None],
            'core_lat':y[None], 'core_lon':x[None],
            'core_valid':np.ones((1,1,65,65), dtype=bool)}.items()}
        before = p.copy()
        contract = {'normalization': {'std':[1], 'mean':[0]}}
        d = diagnose_outputs(output, contract)[0]
        self.assertEqual(d['status'], 'experimental_estimate')
        np.testing.assert_array_equal(output['core'].data[0,0], before)
        output['core_valid'].data[:] = False
        missing = diagnose_outputs(output, contract)[0]
        self.assertIsNone(missing['rmw_estimate_km'])
        self.assertEqual(missing['maximum_wind_auxiliary_kt'],80)

    def test_out_of_domain_auxiliary_and_upsampled_coarse_radii_are_not_exposed(self):
        lat, lon = np.linspace(30,10,25), np.linspace(125,145,33)
        p = np.ones((25,33))*1000
        d = diagnose_member(p,lat,lon,[20,181],80,source_spacing_km=20)
        self.assertIsNone(d['maximum_wind_auxiliary_kt'])
        d = diagnose_member(p,lat,lon,[20,135],80,source_spacing_km=20)
        self.assertEqual(d['status'],'sampling_grid_too_coarse')
        self.assertIsNone(d['rmw_estimate_km'])


if __name__ == '__main__': unittest.main()
