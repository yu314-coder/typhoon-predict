"""Append omitted pressure-core exports to an immutable forecast archive.

Replay existing, checksum-pinned causal issue inputs on CPU. A field is exported
only if the input hash and every original route, scalar and basin field match.
The old forecasts/ and fields/ files are NEVER written by this job.
"""
import argparse
import base64
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

METHOD = 'model-geographic-moving-core-export-v1'

def core_queue_order(row):
    """Prioritize reported storms without changing any planned issue or output."""
    priority = ('2025308N09144', '2018250N12170')
    storm = row['storm_id']
    return (priority.index(storm) if storm in priority else len(priority), row['id'])

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
        maximum['route_degrees']=max(maximum['route_degrees'],float(np.max(np.abs(pred['center'][0].detach().cpu().numpy()-[old['lat'],old['lon']]))))
        maximum['core_pressure_hpa']=max(maximum['core_pressure_hpa'],abs(float(pred['pressure'][0])-old['pressure_hpa']))
        maximum['basin_pressure_hpa']=max(maximum['basin_pressure_hpa'],float(np.max(np.abs(blocks[i+1]['basin'][0]-source_field['pressure_hpa'][i]))))
    # A cross-backend MPS float32 rollout may accumulate sub-lattice sampling
    # differences. 0.002 degrees is at most 223 m, ~1% of the 20-km core grid.
    # Cloud replay retains its original stricter pinned-CPU threshold.
    route_limit=.002 if backend=='mps' else .001
    if maximum['route_degrees']>route_limit or maximum['core_pressure_hpa']>.05 or maximum['basin_pressure_hpa']>.006:
        raise ValueError('Replay does not reproduce immutable outputs: '+str(maximum))
    return maximum

def export_core(model, contract, inputs, row, reference, source_field, source_hashes, *, backend='cpu'):
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
    return dict(schema_version='1.0',model='Trackformer 1.2',forecast_id=reference['id'],
        storm_id=row['storm_id'],issue_time_utc=row['issue_time_utc'],members=1,
        checkpoint_sha256=a.CHECKPOINT,input_tensor_sha256=reference['input_tensor_sha256'],
        method=METHOD,units='hPa',rounding_hpa=.01,model_core_information_spacing_km=20,
        native_detail_history_available=False,scalar_pressure_inserted=False,route_or_truth_alignment=False,
        note='Original basin + original geographically registered tapered moving anomaly; no scalar insertion, route shift, or verification inputs. Internal reconstruction, not resolved observations.',
        execution_backend=backend,source_hashes=source_hashes,replay_max_difference=difference,
        replay_tolerance={'route_degrees':.002 if backend=='mps' else .001,'core_pressure_hpa':.05,'basin_pressure_hpa':.006},issue=issue,frames=frames)

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
    errors=old.get('errors',{});attempted=0;success=0
    queue=sorted(plan['queue'],key=core_queue_order)
    for row in queue:
        if attempted>=args.limit or time.monotonic()-start>args.minutes*60:break
        ident=row['id'];target=dest/f'{ident}.json.gz'
        if args.only_id and ident!=args.only_id:continue
        if target.exists():continue
        original=args.output/'forecasts'/f'{ident}.json';original_field=args.output/'fields'/f'{ident}.json.gz'
        if not original.exists() or not original_field.exists():continue
        now=datetime.now(timezone.utc)
        if ident in errors and now-a.parse(errors[ident]['at'])<timedelta(hours=6):continue
        attempted+=1
        try:
            reference=json.loads(original.read_text());field=json.loads(gzip.decompress(original_field.read_bytes()))
            if row['atlas'] is not None:
                bundle=plan['bundles'][row.get('bundle_key',str(row['season']))]
                with np.load(a.asset(args.cache,bundle['file'],base+'/'+bundle['file'],bundle['sha256']),allow_pickle=False) as z:
                    wanted=a.ns(row['issue_time_utc'])+np.arange(-8,1)*6*a.HOUR;idx=np.searchsorted(z['time'],wanted)
                    if np.any(idx>=len(z['time'])) or not np.array_equal(z['time'][idx],wanted):raise ValueError('Missing exact causal bundled history')
                    weather=np.concatenate([z['slp'][idx,None].astype('float32'),z['q'][idx].astype('float32')*z['scale'][None,:,None,None]+z['offset'][None,:,None,None]],axis=1);times=wanted
            else:weather,times,_,_=a.remote_history(row,args.cache,contract)
            inputs={k:v.to(args.device) for k,v in a.inputs(weather,times,row,contract,geo).items()}
            result=export_core(model,contract,inputs,row,reference,field,
                {'forecast_sha256':sha(original),'basin_field_gzip_sha256':sha(original_field),
                 'input_manifest_sha256':config['manifest_sha256'],'weights_sha256':manifest['inference_weights_sha256']},backend=args.device)
            blob=gzip.compress(json.dumps(result,separators=(',',':'),allow_nan=False).encode(),mtime=0)
            tmp=target.with_suffix('.tmp');tmp.write_bytes(blob);tmp.replace(target)
            errors.pop(ident,None);success+=1
            print(json.dumps({'core_complete':ident,'bytes':len(blob),'replay':result['replay_max_difference']}),flush=True)
        except Exception as e:
            errors[ident]={'at':a.utc(now),'error':str(e)[:500]}
            print(json.dumps({'core_failed':ident,'error':str(e)[:500]}),flush=True)
    planned={r['id'] for r in plan['queue']};done={p.name[:-8] for p in dest.glob('*.json.gz')}&planned
    now=datetime.now(timezone.utc)
    cooling={i for i in errors if i not in done and now-a.parse(errors[i]['at'])<timedelta(hours=6)}
    status=dict(model='Trackformer 1.2',checkpoint_sha256=a.CHECKPOINT,members=1,method=METHOD,
        total=len(planned),completed=len(done),ready=len(planned-done-cooling),cooling_down=len(cooling),
        batch_attempted=attempted,batch_succeeded=success,errors=errors,updated_at_utc=a.utc(now),
        run_url=a.os.environ.get('RUN_URL'),input_manifest_sha256=config['manifest_sha256'],
        original_forecasts_modified=False,original_basin_fields_modified=False,
        state='complete' if done==planned else 'backfilling',
        continue_ready=success>0 and len(planned-done-cooling)>0 and not args.only_id)
    a.write(dest/'status.json',status);print(json.dumps({k:v for k,v in status.items() if k!='errors'}),flush=True)

if __name__=='__main__':main()
