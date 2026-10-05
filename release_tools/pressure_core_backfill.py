"""Append omitted pressure-core exports to an immutable forecast archive.

Replay existing, checksum-pinned causal issue inputs on CPU. A field is exported
only if the input hash and every original route, scalar and basin field match.
The old forecasts/ and fields/ files are NEVER written by this job.
"""
import argparse
import base64
import contextlib
import gzip
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import automatic_forecasts as a
from recover_pressure_core import CaptureModel, reconstruct
from auxiliary_wind_archive import export_wind, METHOD as WIND_METHOD
from immutable_basin_field import read_basin_field

METHOD = 'model-geographic-moving-core-export-v1'
REPLAY_VERSION = '2026-10-05-strict-cpu-profiles-v2'


class ReplayMismatch(ValueError):
    """Only a completed same-input rollout may try a bounded CPU math fallback."""
    def __init__(self, difference):
        self.difference = difference
        super().__init__('Replay does not reproduce immutable outputs: '+str(difference))


def recovery_is_cooling(error, now):
    # A tested corrected execution/reader gets one retry. Its new failures keep
    # the original six-hour cooldown; never dispatch an unproductive tight loop.
    if error.get('replay_version') != REPLAY_VERSION:
        return False
    try:
        return now-a.parse(error['at']) < timedelta(hours=6)
    except (KeyError, TypeError, ValueError):
        return True


@contextlib.contextmanager
def replay_profile(profile):
    """Change only CPU attention arithmetic, restoring flags after each issue."""
    if profile not in ('native', 'unfused-attention', 'sdpa-math'):
        raise ValueError('Unsupported bounded CPU replay profile')
    fastpath = torch.backends.mha.get_fastpath_enabled()
    context = contextlib.nullcontext()
    if profile != 'native':
        torch.backends.mha.set_fastpath_enabled(False)
    if profile == 'sdpa-math':
        from torch.nn.attention import sdpa_kernel, SDPBackend
        context = sdpa_kernel(SDPBackend.MATH)
    try:
        with context:
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def verified_replay(export, *, backend='cpu'):
    """Accept the first FULL immutable replay match, never the smallest error.

    No fallback for identity/hash/source/lead/physical failures. No route shifts,
    scalar insertions, dtype changes, new inputs or wider acceptance limits.
    """
    profiles = ('native', 'unfused-attention', 'sdpa-math') if backend == 'cpu' else ('native',)
    for index, profile in enumerate(profiles):
        try:
            with replay_profile(profile):
                result, wind = export()
                runtime = dict(torch_version=torch.__version__, threads=torch.get_num_threads(),
                    attention_fastpath=torch.backends.mha.get_fastpath_enabled(),
                    cpu_capability=torch.backends.cpu.get_cpu_capability() if backend=='cpu' else None)
                for document in (result, wind):
                    document['execution_profile'] = profile
                    document['replay_version'] = REPLAY_VERSION
                    document['replay_runtime'] = runtime
                return result, wind
        except ReplayMismatch as error:
            print(json.dumps({'replay_profile_rejected':profile, 'difference':error.difference}),flush=True)
            if index == len(profiles)-1:
                raise

def core_queue_order(row):
    """Prioritize reported storms without changing any planned issue or output."""
    priority = ('2025308N09144', '2018250N12170')
    storm = row['storm_id']
    return (priority.index(storm) if storm in priority else len(priority), row['id'])

def core_batch_queue(rows, completed, cooling):
    """Publish the reported storm promptly, then resume normal bounded batches."""
    ordered = sorted(rows, key=core_queue_order)
    urgent = [r for r in ordered if r['storm_id']=='2025308N09144'
              and r['id'] not in completed and r['id'] not in cooling]
    return urgent or ordered

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def encoded_grid(block, contract):
    """Retain the actual 20-km model lattice; do not enlarge cloud storage by upsampling."""
    if len(block['anomaly']) != 1:
        raise ValueError('Historical core backfill requires one original member')
    lat, lon = block['latitude'][0], block['longitude'][0]
    lat = lat[(lat >= 0) & (lat <= 60)]; lon = lon[(lon >= 100) & (lon <= 180)]
    if len(lat)<2 or len(lon)<2:
        return {'available':False,'reason':'Moving model frame outside basin support'}
    field = reconstruct(block, lat, lon, contract)
    integers = np.rint((field.astype(float)-1000)*100).astype('int32').ravel()
    delta = np.diff(np.r_[0, integers])
    if delta.min() < -32768 or delta.max() > 32767:
        raise ValueError('Pressure storage delta overflow')
    return dict(available=True, latitude=lat.tolist(),longitude=lon.tolist(),
        pressure_encoding={'format':'delta-int16-le-base64','shape':list(field.shape),'scale_hpa':.01,'offset_hpa':1000},
        pressure_delta_base64=base64.b64encode(delta.astype('<i2').tobytes()).decode())

def verify_replay(reference, inputs, predictions, blocks, source_field, *, backend='cpu'):
    identity=a.digest(b''.join(inputs[k].detach().cpu().numpy().tobytes() for k in sorted(inputs)))
    if identity!=reference['input_tensor_sha256']:
        raise ValueError('Original causal input tensor hash mismatch')
    if reference['members']!=1 or reference['checkpoint_sha256']!=a.CHECKPOINT or len(reference['route'])!=21:
        raise ValueError('Wrong immutable forecast identity')
    if (len(predictions)!=20 or len(blocks)!=21
            or source_field['forecast_id']!=reference['id']
            or source_field['checkpoint_sha256']!=a.CHECKPOINT
            or source_field['members']!=1
            or len(source_field['pressure_hpa'])!=20
            or source_field['valid_times_utc']!=[p['valid_time_utc'] for p in reference['route'][1:]]):
        raise ValueError('Wrong immutable basin field identity or lead count')
    initial=reference['route'][0]
    if (initial['lead_hours']!=0 or a.ns(initial['valid_time_utc'])!=a.ns(reference['issue_time_utc'])
            or not np.allclose(inputs['center'][0].detach().cpu().numpy(),[initial['lat'],initial['lon']],atol=1e-4,rtol=0)):
        raise ValueError('Original issue-time position does not align')
    maximum={'route_degrees':0.,'core_pressure_hpa':0.,'basin_pressure_hpa':0.}
    for i,pred in enumerate(predictions):
        old=reference['route'][i+1]
        if a.ns(old['valid_time_utc']) != a.ns(reference['issue_time_utc'])+(i+1)*6*a.HOUR or old['lead_hours']!=(i+1)*6:
            raise ValueError('Original lead time mismatch')
        center=pred['center'][0].detach().cpu().numpy()
        pressure=float(pred['pressure'][0])
        basin=blocks[i+1]['basin'][0]
        original_basin=np.asarray(source_field['pressure_hpa'][i])
        if (center.shape!=(2,) or basin.shape!=original_basin.shape
                or not np.isfinite(center).all() or not np.isfinite(pressure)
                or not np.isfinite(basin).all() or not np.isfinite(original_basin).all()
                or not np.isfinite([old['lat'],old['lon'],old['pressure_hpa']]).all()):
            raise ValueError('Nonfinite or malformed immutable replay output')
        maximum['route_degrees']=max(maximum['route_degrees'],float(np.max(np.abs(center-[old['lat'],old['lon']]))))
        maximum['core_pressure_hpa']=max(maximum['core_pressure_hpa'],abs(pressure-old['pressure_hpa']))
        maximum['basin_pressure_hpa']=max(maximum['basin_pressure_hpa'],float(np.max(np.abs(basin-original_basin))))
    # A cross-backend MPS float32 rollout may accumulate sub-lattice sampling
    # differences. 0.002 degrees is at most 223 m, ~1% of the 20-km core grid.
    # Cloud replay retains its original stricter pinned-CPU threshold.
    route_limit=.002 if backend=='mps' else .001
    if maximum['route_degrees']>route_limit or maximum['core_pressure_hpa']>.05 or maximum['basin_pressure_hpa']>.006:
        raise ReplayMismatch(maximum)
    return maximum

def export_core(model, contract, inputs, row, reference, source_field, source_hashes, *, backend='cpu', include_wind=False):
    if (reference['id']!=row['id'] or reference['storm_id']!=row['storm_id']
            or a.ns(reference['issue_time_utc'])!=a.ns(row['issue_time_utc'])):
        raise ValueError('Pinned plan and original issue identity mismatch')
    model.blocks=[]
    with torch.inference_mode():
        state=model.initial(inputs);predictions=[]
        for _ in range(20):
            state,pred=model.step(state);predictions.append(pred)
    blocks=model.blocks[0]
    difference=verify_replay(reference,inputs,predictions,blocks,source_field,backend=backend)
    issue=encoded_grid(blocks[0],contract)
    frames=[dict(**encoded_grid(block,contract),lead_hours=(i+1)*6,
        valid_time_utc=reference['route'][i+1]['valid_time_utc']) for i,block in enumerate(blocks[1:])]
    result = dict(schema_version='1.0',model='Trackformer 1.2',forecast_id=reference['id'],
        storm_id=row['storm_id'],issue_time_utc=row['issue_time_utc'],members=1,
        checkpoint_sha256=a.CHECKPOINT,input_tensor_sha256=reference['input_tensor_sha256'],
        method=METHOD,units='hPa',rounding_hpa=.01,model_core_information_spacing_km=20,
        native_detail_history_available=False,scalar_pressure_inserted=False,route_or_truth_alignment=False,
        note='Original basin + original geographically registered tapered moving anomaly; no scalar insertion, route shift, or verification inputs. Internal reconstruction, not resolved observations.',
        execution_backend=backend,source_hashes=source_hashes,replay_max_difference=difference,
        replay_tolerance={'route_degrees':.002 if backend=='mps' else .001,'core_pressure_hpa':.05,'basin_pressure_hpa':.006},issue=issue,frames=frames)
    # The same audited rollout already produced vmax. Never run a second model
    # or diagnose a mean pressure grid to obtain the released learned wind head.
    if include_wind:
        return result, export_wind(reference,predictions,source_hashes,difference,backend)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=500);ap.add_argument('--minutes',type=int,default=40)
    ap.add_argument('--device',choices=['cpu','mps'],default='cpu',help='Cloud keeps pinned CPU; MPS is optional for local reproduction checks')
    ap.add_argument('--only-id');args=ap.parse_args()
    start=time.monotonic();torch.set_num_threads(2)
    config=json.loads((a.ROOT/'release_tools/history_inputs.json').read_text());base=config['base_url']
    plan=json.loads(a.asset(args.cache,'input-manifest.json',base+'/manifest.json',config['manifest_sha256']).read_text())
    manifest=json.loads((a.MODEL/'manifest.json').read_text());contract=manifest['data_contract']
    if manifest['source_checkpoint_sha256']!=a.CHECKPOINT:raise ValueError('Wrong checkpoint')
    for filename,expected in manifest['source_module_sha256'].items():
        if sha(a.MODEL/filename)!=expected:raise ValueError('Frozen source changed')
    weight=a.asset(args.cache,'weights.pt',a.HF+'/models/trackformer_1_2_field/weights.pt',manifest['inference_weights_sha256'])
    geo=np.load(a.asset(args.cache,'geography.npz',base+'/geography.npz',plan['geography']['sha256']),allow_pickle=False)
    model=CaptureModel(contract).eval();model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS is unavailable')
    model=model.to(args.device)
    dest=args.output/'pressure-cores';dest.mkdir(parents=True,exist_ok=True)
    old=json.loads((dest/'status.json').read_text()) if (dest/'status.json').exists() else {}
    wind_dest=args.output/'model-wind';wind_dest.mkdir(parents=True,exist_ok=True)
    wind_old=json.loads((wind_dest/'status.json').read_text()) if (wind_dest/'status.json').exists() else {}
    errors={**old.get('errors',{}),**wind_old.get('errors',{})};attempted=0;success=0;core_success=0;wind_success=0
    batch_now=datetime.now(timezone.utc)
    core_completed={p.name[:-8] for p in dest.glob('*.json.gz')}
    wind_completed={p.stem for p in wind_dest.glob('auto-*.json')}
    completed=core_completed & wind_completed
    cooling={ident for ident,e in errors.items() if recovery_is_cooling(e,batch_now)}
    queue=core_batch_queue(plan['queue'],completed,cooling)
    for row in queue:
        if attempted>=args.limit or time.monotonic()-start>args.minutes*60:break
        ident=row['id'];target=dest/f'{ident}.json.gz';wind_target=wind_dest/f'{ident}.json'
        if args.only_id and ident!=args.only_id:continue
        if target.exists() and wind_target.exists():continue
        original=args.output/'forecasts'/f'{ident}.json'
        now=datetime.now(timezone.utc)
        if ident in errors and recovery_is_cooling(errors[ident],now):continue
        attempted+=1
        try:
            reference=json.loads(original.read_text());field,field_hashes=read_basin_field(args.output,ident)
            if row['atlas'] is not None:
                bundle=plan['bundles'][row.get('bundle_key',str(row['season']))]
                with np.load(a.asset(args.cache,bundle['file'],base+'/'+bundle['file'],bundle['sha256']),allow_pickle=False) as z:
                    wanted=a.ns(row['issue_time_utc'])+np.arange(-8,1)*6*a.HOUR;idx=np.searchsorted(z['time'],wanted)
                    if np.any(idx>=len(z['time'])) or not np.array_equal(z['time'][idx],wanted):raise ValueError('Missing exact causal bundled history')
                    weather=np.concatenate([z['slp'][idx,None].astype('float32'),z['q'][idx].astype('float32')*z['scale'][None,:,None,None]+z['offset'][None,:,None,None]],axis=1);times=wanted
            else:weather,times,_,_=a.remote_history(row,args.cache,contract)
            inputs={k:v.to(args.device) for k,v in a.inputs(weather,times,row,contract,geo).items()}
            source_hashes = {'forecast_sha256':sha(original),**field_hashes,
                'input_manifest_sha256':config['manifest_sha256'],'weights_sha256':manifest['inference_weights_sha256']}
            result,wind=verified_replay(lambda: export_core(model,contract,inputs,row,reference,field,
                source_hashes,backend=args.device,include_wind=True),backend=args.device)
            if not target.exists():
                blob=gzip.compress(json.dumps(result,separators=(',',':'),allow_nan=False).encode(),mtime=0)
                tmp=target.with_suffix('.tmp');tmp.write_bytes(blob);tmp.replace(target);core_success+=1
                print(json.dumps({'core_complete':ident,'bytes':len(blob),'replay':result['replay_max_difference']}),flush=True)
            if not wind_target.exists():
                a.write(wind_target,wind);wind_success+=1
                print(json.dumps({'wind_complete':ident,'resolved_leads':sum(p['valid_members'] for p in wind['points'])}),flush=True)
            errors.pop(ident,None);success+=1
        except Exception as e:
            errors[ident]={'at':a.utc(now),'error':str(e)[:500],'replay_version':REPLAY_VERSION}
            print(json.dumps({'core_failed':ident,'error':str(e)[:500]}),flush=True)
    planned={r['id'] for r in plan['queue']};done={p.name[:-8] for p in dest.glob('*.json.gz')}&planned
    now=datetime.now(timezone.utc)
    wind_done={p.stem for p in wind_dest.glob('auto-*.json')}&planned
    core_cooling={i for i in errors if i not in done and recovery_is_cooling(errors[i],now)}
    wind_cooling={i for i in errors if i not in wind_done and recovery_is_cooling(errors[i],now)}
    joint_cooling=core_cooling|wind_cooling
    continue_ready=success>0 and bool(planned-(done&wind_done)-joint_cooling) and not args.only_id
    status=dict(model='Trackformer 1.2',checkpoint_sha256=a.CHECKPOINT,members=1,method=METHOD,
        total=len(planned),completed=len(done),ready=len(planned-done-core_cooling),cooling_down=len(core_cooling),
        batch_attempted=attempted,batch_succeeded=core_success,errors=errors,updated_at_utc=a.utc(now),
        run_url=a.os.environ.get('RUN_URL'),input_manifest_sha256=config['manifest_sha256'],replay_version=REPLAY_VERSION,
        original_forecasts_modified=False,original_basin_fields_modified=False,
        state='complete' if done==planned else 'backfilling',
        continue_ready=continue_ready)
    a.write(dest/'status.json',status);print(json.dumps({k:v for k,v in status.items() if k!='errors'}),flush=True)
    wind_status=dict(model='Trackformer 1.2',checkpoint_sha256=a.CHECKPOINT,members=1,method=WIND_METHOD,
        total=len(planned),completed=len(wind_done),ready=len(planned-wind_done-wind_cooling),
        cooling_down=len(wind_cooling),batch_attempted=attempted,batch_succeeded=wind_success,
        errors={i:e for i,e in errors.items() if i not in wind_done},updated_at_utc=a.utc(now),
        run_url=a.os.environ.get('RUN_URL'),input_manifest_sha256=config['manifest_sha256'],replay_version=REPLAY_VERSION,
        original_forecasts_modified=False,original_basin_fields_modified=False,
        original_core_fields_modified=False,wind_averaging_period='unvalidated',
        state='complete' if wind_done==planned else 'backfilling',continue_ready=continue_ready)
    a.write(wind_dest/'status.json',wind_status);print(json.dumps({k:v for k,v in wind_status.items() if k!='errors'}),flush=True)

if __name__=='__main__':main()
