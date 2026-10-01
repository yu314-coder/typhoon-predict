import unittest
import base64
import copy
import torch
import automatic_forecasts as a
import numpy as np
from pressure_core_backfill import encoded_grid,verify_replay
from recover_pressure_core import reconstruct
from verify_pressure_core_archive import verify_grid

class GeographicCoreExport(unittest.TestCase):
    def setUp(self):
        self.contract={'global_lat':[60,30,0],'global_lon':[100,140,180]}
        self.block={'basin':np.full((2,3,3),1000.,dtype='float32'),
            'latitude':np.array([[25,20,15],[25,20,15]]),
            'longitude':np.array([[120,125,130],[130,135,140]]),
            'anomaly':np.array([[[0,0,0],[0,-40,0],[0,0,0]],[[0,0,0],[0,-20,0],[0,0,0]]],dtype='float32')}
    def test_register_members_before_mean(self):
        result=reconstruct(self.block,np.array([25,20,15]),np.array([120,125,130,135,140]),self.contract)
        self.assertEqual(float(result[1,1]),980.)
        self.assertEqual(float(result[1,3]),990.)
        self.assertEqual(float(result[1,2]),1000.)
        # Index-wise mean would instead invent a single -30-hPa low.
    def test_encoded_grid_is_physical_and_round_trips(self):
        block={k:v[:1] for k,v in self.block.items()}
        result=encoded_grid(block,self.contract);verify_grid(result)
        q=np.frombuffer(base64.b64decode(result['pressure_delta_base64']),dtype='<i2').astype('int32').cumsum()/100+1000
        self.assertEqual(float(q.reshape(3,3)[1,1]),960.)
    def test_no_core_shift_or_scalar_pressure_input(self):
        expected=reconstruct(self.block,np.array([25,20,15]),np.array([120,125,130]),self.contract)
        self.block['pressure']=np.array([900.,850.])
        actual=reconstruct(self.block,np.array([25,20,15]),np.array([120,125,130]),self.contract)
        np.testing.assert_array_equal(actual,expected)
    def test_outside_actual_core_keeps_basin(self):
        result=reconstruct(self.block,np.array([55,50]),np.array([160,165]),self.contract)
        np.testing.assert_array_equal(result,np.full((2,2),1000.))

class ImmutableReplayChecks(unittest.TestCase):
    def setUp(self):
        self.inputs={'center':torch.tensor([[20.,125.]])}
        identity=a.digest(b''.join(self.inputs[k].numpy().tobytes() for k in sorted(self.inputs)))
        stamp='2025-11-07T00:00:00Z'
        from datetime import timedelta
        route=[dict(lat=20.,lon=125.,pressure_hpa=960.,lead_hours=i*6,
            valid_time_utc=a.utc(a.parse(stamp)+timedelta(hours=i*6))) for i in range(21)]
        self.reference=dict(id='auto-tick-test',members=1,checkpoint_sha256=a.CHECKPOINT,
            input_tensor_sha256=identity,issue_time_utc=stamp,route=route)
        self.predictions=[dict(center=self.inputs['center'],pressure=torch.tensor([960.])) for _ in range(20)]
        self.blocks=[{'basin':np.full((1,3,3),1000.)} for _ in range(21)]
        self.field=dict(forecast_id='auto-tick-test',members=1,checkpoint_sha256=a.CHECKPOINT,
            valid_times_utc=[r['valid_time_utc'] for r in route[1:]],pressure_hpa=np.full((20,3,3),1000.))
    def test_unchanged_original_outputs_pass(self):
        self.assertEqual(verify_replay(self.reference,self.inputs,self.predictions,self.blocks,self.field),
            dict(route_degrees=0.,core_pressure_hpa=0.,basin_pressure_hpa=0.))
    def test_changed_causal_inputs_fail(self):
        changed={'center':torch.tensor([[21.,125.]])}
        with self.assertRaisesRegex(ValueError,'input tensor hash'):
            verify_replay(self.reference,changed,self.predictions,self.blocks,self.field)
    def test_changed_scalar_does_not_get_patched(self):
        altered=copy.deepcopy(self.reference);altered['route'][5]['pressure_hpa']=940.
        with self.assertRaisesRegex(ValueError,'does not reproduce'):
            verify_replay(altered,self.inputs,self.predictions,self.blocks,self.field)
    def test_missing_lead_is_not_accepted_as_complete(self):
        with self.assertRaisesRegex(ValueError,'lead count'):
            verify_replay(self.reference,self.inputs,self.predictions[:-1],self.blocks,self.field)
    def test_cross_backend_tolerance_is_bounded_and_cpu_stays_stricter(self):
        predictions=copy.deepcopy(self.predictions)
        predictions[19]['center']=torch.tensor([[20.0015,125.]])
        with self.assertRaisesRegex(ValueError,'does not reproduce'):
            verify_replay(self.reference,self.inputs,predictions,self.blocks,self.field)
        self.assertLess(verify_replay(self.reference,self.inputs,predictions,self.blocks,self.field,backend='mps')['route_degrees'],.002)
        predictions[19]['center']=torch.tensor([[20.003,125.]])
        with self.assertRaisesRegex(ValueError,'does not reproduce'):
            verify_replay(self.reference,self.inputs,predictions,self.blocks,self.field,backend='mps')

if __name__=='__main__':unittest.main()
