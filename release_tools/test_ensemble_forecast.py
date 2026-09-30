import unittest
import numpy as np
import torch
from ensemble_forecast import member_inputs, mean_outputs, analysis_motion, run_ensemble


class EnsemblePolicy(unittest.TestCase):
    def test_seeded_distinct_inputs_preserve_issue_and_missing_flags(self):
        base={'global_history':np.zeros((1,9,8,25,33),dtype='float32'),
              'regional_history':np.zeros((1,9,1,121,121),dtype='float32'),
              'center':np.array([[20,130]],dtype='float32'),
              'detail_available':np.zeros((1,1),dtype='float32')}
        one=member_inputs(base,[0,1]);again=member_inputs(base,[0,1])
        np.testing.assert_array_equal(one['global_history'],again['global_history'])
        self.assertFalse(np.array_equal(one['global_history'][0],one['global_history'][1]))
        np.testing.assert_array_equal(one['center'],np.repeat(base['center'],2,axis=0))
        self.assertFalse(one['detail_available'].any())
        std=torch.from_numpy(one['global_history']).std(dim=(-2,-1)).numpy()
        np.testing.assert_allclose(std,.025,atol=1e-6)
        self.assertFalse(base['global_history'].any())

    def test_mean_is_of_member_outputs_not_the_field_minimum(self):
        output={k:np.array([[[2.,8.]],[[6.,4.]]],dtype='float32')
                for k in ('center','pressure','basin','regional','vmax')}
        mean=mean_outputs(output)
        np.testing.assert_allclose(mean['pressure'],[[4.,6.]])
        self.assertNotEqual(float(mean['pressure'][0,0]),float(mean['basin'].min())+1)

    def test_motion_is_only_the_current_analysis(self):
        parts=[{'part':{'en':'Forecast'},'advancedHours':12,'course':'南','speed':{'km/h':100}},
               {'part':{'en':'Analysis'},'advancedHours':0,'course':'北東','speed':{'km/h':15}}]
        motion,note=analysis_motion(parts)
        np.testing.assert_allclose(motion,[63.6396103,63.6396103])
        self.assertEqual(note['source_part'],'Analysis')

    def test_new_wind_export_keeps_fifty_distinct_members_and_initial_fields(self):
        class FakeModel:
            def eval(self): return self
            def to(self, device): return self
            def initial(self, x):
                return {'g':x['global_history'][:,-1], 'r':x['regional_history'][:,-1],
                        'center':x['center'].clone()}
            def step(self, s):
                g = s['g'] + .001
                shift = g[:,0,1,1]
                center = s['center'] + torch.stack((shift,shift),1)*.01
                p = 1000+shift
                result = {'global':g, 'regional':s['r'], 'center':center, 'pressure':p,
                          'vmax':80+shift, 'track_valid':torch.ones(len(center),dtype=torch.bool)}
                return dict(s,g=g,center=center), result

        inputs = {k:torch.from_numpy(v) for k,v in {
            'global_history':np.zeros((1,9,8,25,33),dtype='float32'),
            'regional_history':np.zeros((1,9,1,121,121),dtype='float32'),
            'center':np.array([[20,135]],dtype='float32'),
            'detail_available':np.zeros((1,1),dtype='float32')}.items()}
        contract = {'normalization': {'std':[1]*8, 'mean':[1000]+[0]*7}}
        output, policy = run_ensemble(FakeModel(),inputs,contract,chunk=10)
        self.assertEqual(output['initial_basin'].shape,(50,25,33))
        self.assertEqual(len(set(policy['input_sha256'])),50)
        self.assertEqual(len(set(policy['route_sha256'])),50)
        self.assertEqual(len(set(policy['basin_field_sha256'])),50)
        self.assertEqual(len(policy['wind_estimation_by_lead']),20)
        for lead,summary in enumerate(policy['wind_estimation_by_lead']):
            self.assertEqual(summary['members'],50)
            wind=summary['estimates']['maximum_wind_auxiliary_kt']
            self.assertEqual(wind['valid_members'],50)
            self.assertAlmostEqual(wind['mean'],float(output['vmax'][:,lead].mean(dtype=np.float64)))
            self.assertIsNone(summary['estimates']['r34_estimate_km']['mean'])
        self.assertFalse(inputs['global_history'].any())


if __name__=='__main__':unittest.main()
