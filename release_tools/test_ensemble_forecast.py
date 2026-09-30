import unittest
import numpy as np
import torch
from ensemble_forecast import member_inputs, mean_outputs, analysis_motion


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


if __name__=='__main__':unittest.main()
