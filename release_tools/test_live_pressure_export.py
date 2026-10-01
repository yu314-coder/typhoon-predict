import base64
import copy
import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
from live_pressure_export import encoded_mean, member_blocks, core_export
from cloud_continuation import continuation_target, live_refresh_due


class LiveCoreExport(unittest.TestCase):
    def setUp(self):
        self.contract = {'global_lat':[60,30,0], 'global_lon':[100,140,180]}
        self.block = {'basin':np.full((2,3,3),1000.,dtype='float32'),
            'latitude':np.array([[25,20,15],[25,20,15]]),
            'longitude':np.array([[120,125,130],[130,135,140]]),
            'anomaly':np.array([[[0,0,0],[0,-40,0],[0,0,0]],
                                [[0,0,0],[0,-20,0],[0,0,0]]],dtype='float32')}

    def decode(self, value):
        return (np.frombuffer(base64.b64decode(value['pressure_delta_base64']),
            dtype='<i2').astype('int32').cumsum()/100+1000).reshape(value['pressure_encoding']['shape'])

    def test_actual_geographic_ensemble_is_not_recentered_or_scalar_fit(self):
        value = encoded_mean(self.block, self.contract)
        field = self.decode(value)
        i = value['latitude'].index(20)
        self.assertEqual(field[i,value['longitude'].index(125)],980)
        self.assertEqual(field[i,value['longitude'].index(135)],990)
        self.assertEqual(field[i,value['longitude'].index(130)],1000)
        changed = copy.deepcopy(self.block); changed['pressure'] = np.array([850,800])
        self.assertEqual(encoded_mean(changed,self.contract),value)

    def test_export_has_a_bounded_common_grid(self):
        block = copy.deepcopy(self.block)
        block['latitude'] = np.array([[60,30,0],[60,30,0]])
        block['longitude'] = np.array([[100,140,180],[100,140,180]])
        value = encoded_mean(block,self.contract)
        self.assertEqual(value['sampling_degrees'],.5)
        self.assertLessEqual(np.prod(value['pressure_encoding']['shape']),25000)

    def test_chunk_assembly_keeps_all_states_and_members(self):
        chunks = [[{k:v[:1] for k,v in self.block.items()} for _ in range(21)],
                  [{k:v[1:] for k,v in self.block.items()} for _ in range(21)]]
        blocks = member_blocks(chunks,2)
        self.assertEqual(len(blocks),21)
        np.testing.assert_array_equal(blocks[7]['longitude'],self.block['longitude'])
        with self.assertRaises(ValueError):member_blocks(chunks,50)
        with self.assertRaises(ValueError):member_blocks([chunks[0][:-1]],1)

    def test_fifty_export_has_twenty_exact_leads_and_distinct_captured_members(self):
        from datetime import timedelta
        block={k:np.repeat(v[:1],50,axis=0) for k,v in self.block.items()}
        block['anomaly'][:,1,1]-=np.arange(50)*.01
        issue=datetime(2026,10,1,9,tzinfo=timezone.utc)
        forecast=dict(id='live50-demo',storm_id='TC2633',members=50,
            checkpoint_sha256='a'*64,input_tensor_sha256='b'*64,
            issue_time_utc=issue.isoformat(),route=[dict(lead_hours=i*6,
                valid_time_utc=(issue+timedelta(hours=i*6)).isoformat()) for i in range(21)])
        exported=core_export([[block for _ in range(21)]],forecast,self.contract)
        self.assertEqual(len(exported['frames']),20)
        self.assertEqual(len(set(exported['common_grid_policy']['member_state_sha256'])),50)
        self.assertEqual(exported['frames'][-1]['lead_hours'],120)
        self.assertFalse(exported['scalar_pressure_inserted'])
        self.assertFalse(exported['route_or_truth_alignment'])


class FairContinuation(unittest.TestCase):
    def test_never_replace_a_pending_live_run(self):
        self.assertIsNone(continuation_target({'historical_ready':50,'batch_succeeded':1},
            {'continue_ready':True},{},[{'workflow':'automatic-forecasts'}]))

    def test_live_job_hands_back_to_core_wind_chain_after_original_plan_complete(self):
        self.assertEqual(continuation_target({'historical_ready':0},
            {'continue_ready':True},{},[]),'pressure-core-backfill.yml')
        self.assertIsNone(continuation_target({'historical_ready':0},{},{},[]))

    def test_fresh_live_check_is_not_repeated_in_every_core_batch(self):
        now = datetime(2026,10,1,10,tzinfo=timezone.utc)
        self.assertFalse(live_refresh_due({'live_checked_at_utc':'2026-10-01T09:00:00Z'},now))
        self.assertTrue(live_refresh_due({'live_issues':[{'issue_time_utc':'2026-09-30T21:00:00Z'}]},now))


if __name__ == '__main__':unittest.main()
