import unittest
from unittest.mock import patch, Mock
import base64
import copy
import json
import os
import tempfile
from pathlib import Path
import torch
import automatic_forecasts as a
import numpy as np
from pressure_core_backfill import (METHOD,REPLAY_VERSION,ReplayMismatch,recovery_is_cooling,
    replay_profile,verified_replay,core_batch_queue,core_queue_order,encoded_grid,verify_replay,sha)
from recover_pressure_core import reconstruct
from verify_pressure_core_archive import verify_grid,verify_issue as verify_core_issue
from immutable_basin_field import read_basin_field

class CoreQueuePriority(unittest.TestCase):
    def test_urgent_batch_publishes_without_waiting_for_unrelated_500_issues(self):
        rows=[{'id':'old','storm_id':'1996001'},{'id':'fung','storm_id':'2025308N09144'}]
        self.assertEqual(core_batch_queue(rows,set(),set()),[rows[1]])
        self.assertEqual(core_batch_queue(rows,{'fung'},set()),[rows[1],rows[0]])
    def test_cooling_priority_issue_does_not_stall_runnable_older_exports(self):
        rows=[{'id':'old','storm_id':'1996001'},{'id':'fung','storm_id':'2025308N09144'}]
        self.assertEqual(core_batch_queue(rows,set(),{'fung'}),[rows[1],rows[0]])
    def test_all_fung_wong_issues_are_prioritized_without_mutating_plan(self):
        rows = [
            {'id':'auto-tick-1996001-19960101T0000','storm_id':'1996001'},
            {'id':'auto-tick-2018250N12170-20180911T0000','storm_id':'2018250N12170'},
            {'id':'auto-tick-2025308N09144-20251107T1200','storm_id':'2025308N09144'},
            {'id':'auto-tick-2025308N09144-20251106T0000','storm_id':'2025308N09144'},
        ]
        original = copy.deepcopy(rows)
        self.assertEqual([r['id'] for r in sorted(rows,key=core_queue_order)],
                         [rows[3]['id'],rows[2]['id'],rows[1]['id'],rows[0]['id']])
        self.assertEqual(rows,original)

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
    def test_nonfinite_prediction_cannot_be_marked_as_zero_difference(self):
        for key in ('center','pressure'):
            predictions=copy.deepcopy(self.predictions)
            predictions[19][key]=torch.full_like(predictions[19][key],float('nan'))
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                verify_replay(self.reference,self.inputs,predictions,self.blocks,self.field)
        blocks=copy.deepcopy(self.blocks);blocks[19]['basin'][0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            verify_replay(self.reference,self.inputs,self.predictions,blocks,self.field)
    def test_core_auditor_accepts_plain_original_without_changing_archive(self):
        temp_base = Path('/Volumes/D/typhoon_predict/output') if Path('/Volumes/D').exists() else Path(os.environ['RUNNER_TEMP'])
        with tempfile.TemporaryDirectory(dir=temp_base) as tmp:
            root = Path(tmp)
            (root/'forecasts').mkdir(); (root/'fields').mkdir()
            reference = dict(self.reference,storm_id='WP')
            f=root/'forecasts/auto-tick-test.json'; b=root/'fields/auto-tick-test.json'
            f.write_text(json.dumps(reference)); b.write_text('{"pressure_hpa":[[1000.0]]}')
            before=(f.read_bytes(),b.read_bytes())
            _, hashes=read_basin_field(root,reference['id'])
            block={'basin':np.full((1,3,3),1000.,dtype='float32'),
                   'latitude':np.array([[25,20,15]]),'longitude':np.array([[120,125,130]]),
                   'anomaly':np.zeros((1,3,3),dtype='float32')}
            grid=encoded_grid(block,{'global_lat':[60,30,0],'global_lon':[100,140,180]})
            document=dict(forecast_id=reference['id'],storm_id='WP',members=1,checkpoint_sha256=a.CHECKPOINT,
                method=METHOD,scalar_pressure_inserted=False,route_or_truth_alignment=False,
                replay_max_difference=dict(route_degrees=0.,core_pressure_hpa=0.,basin_pressure_hpa=0.),
                source_hashes=dict(forecast_sha256=sha(f),**hashes),input_tensor_sha256=reference['input_tensor_sha256'],
                issue_time_utc=reference['issue_time_utc'],issue=grid,
                frames=[dict(grid,lead_hours=p['lead_hours'],valid_time_utc=p['valid_time_utc']) for p in reference['route'][1:]])
            verify_core_issue(document,root,{reference['id']})
            self.assertEqual((f.read_bytes(),b.read_bytes()),before)
            self.assertFalse((root/'fields/auto-tick-test.json.gz').exists())
    def test_cross_backend_tolerance_is_bounded_and_cpu_stays_stricter(self):
        predictions=copy.deepcopy(self.predictions)
        predictions[19]['center']=torch.tensor([[20.0015,125.]])
        with self.assertRaisesRegex(ValueError,'does not reproduce'):
            verify_replay(self.reference,self.inputs,predictions,self.blocks,self.field)
        self.assertLess(verify_replay(self.reference,self.inputs,predictions,self.blocks,self.field,backend='mps')['route_degrees'],.002)
        predictions[19]['center']=torch.tensor([[20.003,125.]])
        with self.assertRaisesRegex(ValueError,'does not reproduce'):
            verify_replay(self.reference,self.inputs,predictions,self.blocks,self.field,backend='mps')

class BoundedCpuReplay(unittest.TestCase):
    def test_first_full_match_records_profile_without_replaying(self):
        output = ({'immutable':'core'}, {'immutable':'wind'})
        export = Mock(return_value=output)
        result, wind = verified_replay(export)
        export.assert_called_once_with()
        self.assertEqual(result['execution_profile'],'native')
        self.assertEqual(wind['execution_profile'],'native')
        self.assertEqual(result['immutable'],'core')
        self.assertEqual(wind['immutable'],'wind')
        self.assertEqual(result['replay_version'],REPLAY_VERSION)

    def test_only_completed_same_input_mismatch_may_retry(self):
        mismatch = ReplayMismatch({'route_degrees':.01})
        export = Mock(side_effect=[mismatch, ({}, {})])
        original = torch.backends.mha.get_fastpath_enabled()
        result, wind = verified_replay(export)
        self.assertEqual(export.call_count,2)
        self.assertEqual(result['execution_profile'],'unfused-attention')
        self.assertFalse(result['replay_runtime']['attention_fastpath'])
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(),original)
        self.assertEqual(wind['replay_runtime'],result['replay_runtime'])

    def test_input_source_and_identity_failures_never_try_another_profile(self):
        for message in ('Original causal input tensor hash mismatch',
                        'Wrong immutable forecast identity','Nonfinite/unphysical geographic reconstruction'):
            export = Mock(side_effect=ValueError(message))
            with self.assertRaisesRegex(ValueError,message):
                verified_replay(export)
            self.assertEqual(export.call_count,1)

    def test_all_profiles_rejected_bounded_no_nearest_match(self):
        export = Mock(side_effect=ReplayMismatch({'route_degrees':.002}))
        original = torch.backends.mha.get_fastpath_enabled()
        with self.assertRaises(ReplayMismatch):
            verified_replay(export)
        self.assertEqual(export.call_count,3)
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(),original)

    def test_mps_does_not_run_cpu_fallbacks(self):
        export = Mock(side_effect=ReplayMismatch({'route_degrees':.003}))
        with self.assertRaises(ReplayMismatch):
            verified_replay(export,backend='mps')
        self.assertEqual(export.call_count,1)

    def test_profile_is_restored_even_after_source_exception(self):
        original = torch.backends.mha.get_fastpath_enabled()
        with self.assertRaisesRegex(ValueError,'source'):
            with replay_profile('sdpa-math'):
                self.assertFalse(torch.backends.mha.get_fastpath_enabled())
                raise ValueError('source')
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(),original)

    def test_repaired_version_gets_one_retry_then_six_hour_cooldown(self):
        from datetime import timedelta
        now=a.parse('2026-10-05T11:00:00Z')
        old={'at':a.utc(now),'error':'failure'}
        self.assertFalse(recovery_is_cooling(old,now))
        current=dict(old,replay_version=REPLAY_VERSION)
        self.assertTrue(recovery_is_cooling(current,now))
        self.assertFalse(recovery_is_cooling(current,now+timedelta(hours=6)))
        self.assertTrue(recovery_is_cooling({'replay_version':REPLAY_VERSION},now))

if __name__=='__main__':unittest.main()
