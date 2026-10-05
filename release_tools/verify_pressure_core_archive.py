"""Audit appended core exports against the immutable source forecasts."""
import argparse
import base64
import gzip
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from pressure_core_backfill import METHOD
from recover_pressure_core import CHECKPOINT,parse,utc
from immutable_basin_field import verify_basin_source

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify_grid(p):
    if p.get('available') is False:
        if p.get('reason')!='Moving model frame outside basin support':raise ValueError('Unknown missing-core reason')
        return
    e=p['pressure_encoding'];shape=e['shape']
    if e['format']!='delta-int16-le-base64' or e['scale_hpa']!=.01 or e['offset_hpa']!=1000 or shape!=[len(p['latitude']),len(p['longitude'])]:raise ValueError('Core grid encoding mismatch')
    if len(shape)!=2 or min(shape)<2 or max(shape)>65:raise ValueError('Not the native model core lattice')
    if not np.all(np.diff(p['latitude'])<0) or not np.all(np.diff(p['longitude'])>0):raise ValueError('Core geographic axes reversed')
    lat,lon=np.asarray(p['latitude']),np.asarray(p['longitude'])
    if lat.min()<0 or lat.max()>60 or lon.min()<100 or lon.max()>180:raise ValueError('Unsupported core coverage')
    raw=base64.b64decode(p['pressure_delta_base64'],validate=True)
    if len(raw)!=np.prod(shape)*2:raise ValueError('Truncated core pressure')
    values=np.frombuffer(raw,dtype='<i2').astype('int32').cumsum()/100+1000
    if not np.isfinite(values).all() or values.min()<800 or values.max()>1100:raise ValueError('Invalid physical core field')

def verify_issue(p,output,planned):
    ident=p['forecast_id']
    if ident not in planned or p['members']!=1 or p['checkpoint_sha256']!=CHECKPOINT or p['method']!=METHOD:raise ValueError('Wrong planned issue identity')
    if p['scalar_pressure_inserted'] is not False or p['route_or_truth_alignment'] is not False:raise ValueError('Artificial correction detected')
    backend=p.get('execution_backend','cpu')
    if backend not in ['cpu','mps']:raise ValueError('Unknown replay backend')
    limits={'route_degrees':.002 if backend=='mps' else .001,'core_pressure_hpa':.05,'basin_pressure_hpa':.006}
    if p.get('replay_tolerance',limits)!=limits or any(p['replay_max_difference'][k]>v for k,v in limits.items()):raise ValueError('Replay exceeded declared numeric tolerances')
    f=output/'forecasts'/f'{ident}.json'
    if sha(f)!=p['source_hashes']['forecast_sha256']:raise ValueError('Immutable forecast/field changed')
    verify_basin_source(output,ident,p['source_hashes'])
    reference=json.loads(f.read_text())
    if reference['input_tensor_sha256']!=p['input_tensor_sha256'] or reference['storm_id']!=p['storm_id'] or parse(reference['issue_time_utc'])!=parse(p['issue_time_utc']):raise ValueError('Causal source identity mismatch')
    if len(p['frames'])!=20:raise ValueError('Missing core lead')
    for i,frame in enumerate(p['frames']):
        original=reference['route'][i+1]
        if frame['lead_hours']!=original['lead_hours'] or parse(frame['valid_time_utc'])!=parse(original['valid_time_utc']):raise ValueError('Wrong core valid time')
        verify_grid(frame)
    verify_grid(p['issue'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'release_tools/history_inputs.json').read_text())
    manifest=json.loads((root/'models/trackformer_1_2_field/manifest.json').read_text())
    manifest_hash=sha(args.manifest)
    if manifest_hash!=config['manifest_sha256']:raise ValueError('Unpinned core backfill plan')
    plan=json.loads(args.manifest.read_text());planned={r['id'] for r in plan['queue']}
    dest=args.output/'pressure-cores';done=[]
    for filename in sorted(dest.glob('*.json.gz')):
        p=json.loads(gzip.decompress(filename.read_bytes()));verify_issue(p,args.output,planned)
        if (p['source_hashes']['input_manifest_sha256']!=manifest_hash
                or p['source_hashes']['weights_sha256']!=manifest['inference_weights_sha256']):raise ValueError('Wrong frozen input or weights identity')
        done.append(p['forecast_id'])
    status=json.loads((dest/'status.json').read_text())
    if len(set(done))!=len(done) or len(done)!=status['completed'] or status['total']!=len(planned):raise ValueError('Core published-count mismatch')
    receipt=dict(method=METHOD,checkpoint_sha256=CHECKPOINT,verified=len(done),planned=len(planned),
        state='complete' if set(done)==planned else 'partial',input_manifest_sha256=sha(args.manifest),
        source_archive_preserved=True,verified_at_utc=utc(datetime.now(timezone.utc)))
    (dest/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
