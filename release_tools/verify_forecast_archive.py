"""Verify saved outputs against the pinned issue queue, without model inference."""
import argparse
import gzip
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

CHECKPOINT='f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0'


def stamp(value):
    return datetime.fromisoformat(value.replace('Z','+00:00'))


def require(ok,message):
    if not ok:raise ValueError(message)


def verify_issue(row,forecast,field):
    issue=stamp(row['issue_time_utc']);ident=row['id']
    require(forecast['id']==ident and field['forecast_id']==ident,'Issue identity mismatch')
    require(forecast['storm_id']==row['storm_id'],'Storm identity mismatch')
    for doc in [forecast,field]:
        require(doc['checkpoint_sha256']==CHECKPOINT and doc['members']==1,'Release/member identity mismatch')
    require(stamp(forecast['issue_time_utc'])==issue,'Wrong initialization time')
    route=forecast['route'];require(len(route)==21,'Incomplete +120h route')
    require(route[0]['lat']==row['lat'] and route[0]['lon']==row['lon'] and
            route[0].get('pressure_hpa')==row.get('pressure_hpa'),'+0 does not align with issue observations')
    for i,point in enumerate(route):
        require(point['lead_hours']==i*6 and stamp(point['valid_time_utc'])==issue+timedelta(hours=i*6),'Wrong route lead/time')
        require(all(math.isfinite(point[k]) for k in ['lat','lon']),'Non-finite route')
        if i:require(math.isfinite(point['pressure_hpa']) and 800<point['pressure_hpa']<1100,'Invalid central pressure')
    times=[stamp(t) for t in forecast['input_history_times_utc']]
    require(len(times)==9 and times[-1]<=issue and issue-times[-1]<=timedelta(hours=12) and
            all(b-a==timedelta(hours=6) for a,b in zip(times,times[1:])), 'Non-causal or incomplete analysis history')
    require(len(forecast['input_tensor_sha256'])==64,'Missing input provenance hash')
    require(field['units']=='hPa' and len(field['latitude'])==25 and len(field['longitude'])==33,'Wrong pressure grid/units')
    values=field['pressure_hpa'];require(len(values)==20 and len(field['valid_times_utc'])==20,'Incomplete pressure sequence')
    for i,grid in enumerate(values):
        require(stamp(field['valid_times_utc'][i])==issue+timedelta(hours=(i+1)*6),'Wrong pressure lead/time')
        require(len(grid)==25 and all(len(line)==33 for line in grid),'Wrong pressure array shape')
        require(all(math.isfinite(p) and 800<=p<=1100 for line in grid for p in line),'Invalid physical pressure field')
    source=forecast.get('input_weather_source') or {}
    if source.get('provider')=='NOAA GFS archive':
        require(source.get('experimental_transfer') is True,'Missing GFS-transfer disclosure')
        analyses=source.get('analyses',[])
        require([stamp(s['valid_time_utc']) for s in analyses]==times,'Source timestamp mismatch')
        require(all(s['url'].endswith('.f000') and len(s['subset_sha256'])==64 for s in analyses),'Missing f000 source provenance')


def verify_archive(output,plan,require_complete=False):
    planned={r['id']:r for r in plan['queue']}
    require(len(planned)==len(plan['queue']),'Duplicate planned issue IDs')
    catalog=json.loads((output/'catalog.json').read_text())
    available={i['id'] for s in catalog['storms'] for i in s['issues']}&set(planned)
    saved={p.stem for p in (output/'forecasts').glob('*.json')}&set(planned)
    require(available==saved,'Published catalogue does not match saved issues')
    for ident in sorted(saved):
        forecast=json.loads((output/'forecasts'/f'{ident}.json').read_text())
        compressed=output/'fields'/f'{ident}.json.gz'
        plain=output/'fields'/f'{ident}.json'
        raw=gzip.decompress(compressed.read_bytes()) if compressed.exists() else plain.read_bytes()
        try:verify_issue(planned[ident],forecast,json.loads(raw))
        except (ValueError,KeyError,TypeError) as error:
            raise ValueError(f'{ident}: {error}') from error
    status=json.loads((output/'status.json').read_text())
    require(status['historical_completed']==len(saved) and status['historical_total']==len(planned),'Status count mismatch')
    missing=set(planned)-saved
    if require_complete:require(not missing,'Planned forecast issues remain missing')
    return {'schema_version':'1.0','verified':len(saved),'planned':len(planned),
            'remaining':len(missing),'complete':not missing,'checkpoint_sha256':CHECKPOINT,
            'checks':['issue +0 alignment','causal nine-analysis history','21 route points',
                      '20 finite pressure fields','exact +6 through +120h times','catalogue coverage'],
            'run_url':status.get('run_url')}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--require-complete',action='store_true')
    args=ap.parse_args()
    report=verify_archive(args.output,json.loads(args.manifest.read_text()),args.require_complete)
    temp=args.output/'verification.json.tmp';temp.write_text(json.dumps(report,separators=(',',':')))
    temp.replace(args.output/'verification.json')
    print(json.dumps(report))


if __name__=='__main__':main()
