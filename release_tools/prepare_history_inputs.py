"""Build a causal, input-only historical queue from the existing local atlas.

One first issue per older storm, or every six-hour issue in requested recent years.
No future-label or forecast-error filtering.
Original scientific archives are read-only. Output is published as research data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('project',type=Path);ap.add_argument('site',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--ticks-from-year',type=int,default=9999);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    atlas=np.load(a.project/'track_build/basin_all_int8.npz',allow_pickle=False)
    slp=np.load(a.project/'track_build/basin_slp_atlas_float16.npy',mmap_mode='r')
    q=np.load(a.project/'data/v164_reuse/basin_q_verified.npy',mmap_mode='r')
    times=atlas['time'].astype('int64');hour=3600*10**9
    cat=json.loads((a.site/'public/data/history/catalog.json').read_text())
    existing=json.loads((a.site/'public/data/forecast-history/v1/catalog.json').read_text())
    saved={s['id'] for s in existing['storms']}
    entries=[dict(zip(cat['fields'],r)) for r in cat['rows']]
    years={};queue=[];coverage=[]
    for e in entries:
        sid=e['id'];year=e['season']
        if year<1970:
            coverage.append({'id':sid,'status':'before_requested_1970_start'});continue
        if 'WP' not in e['basins']:
            coverage.append({'id':sid,'status':'outside_model_domain'});continue
        if sid in saved and year<a.ticks_from_year:
            coverage.append({'id':sid,'status':'existing_verified_forecast'});continue
        if year not in years:years[year]=json.loads((a.site/f'public/data/history/{year}.json').read_text())
        storm=years[year].get(sid);chosen=[]
        if storm:
            points=sorted(storm['points'],key=lambda p:p['time'])
            for p in points:
                ns=int(np.datetime64(p['time'].replace('Z',''),'ns').astype('int64'));ai=int(np.searchsorted(times,ns))
                if not(0<p['lat']<60 and 100<p['lon']<180) or ns%(6*hour):continue
                local=ai>=8 and ai<len(times) and np.array_equal(times[ai-8:ai+1],ns+np.arange(-8,1)*6*hour)
                previous=next((v for v in points if v['time']==str(np.datetime64(ns-6*hour,'ns').astype('datetime64[s]'))+'Z'),None)
                motion=[0.,0.] if previous is None else [(p['lon']-previous['lon'])*111.2*np.cos(np.deg2rad(p['lat'])),(p['lat']-previous['lat'])*111.2]
                row={'id':('auto-tick-'+sid+'-'+p['time'].replace('-','').replace(':','')[:13]) if year>=a.ticks_from_year else 'auto-hist-'+sid,'storm_id':sid,'name':e['name'],'season':year,'issue_time_utc':p['time'],
                    'lat':p['lat'],'lon':p['lon'],'pressure_hpa':p.get('pressure_hpa'),'wind_kt':p.get('wind_kt'),
                    'motion':motion,'atlas':ai if local else None,'weather_source':'verified-local-atlas' if local else 'NOAA-NCEP-Reanalysis-1-remote',
                    'observed':[v for v in points if ns<=int(np.datetime64(v['time'].replace('Z',''),'ns').astype('int64'))<=ns+120*hour]}
                chosen.append(row)
                if year<a.ticks_from_year:break
        if chosen:
            queue.extend(chosen);coverage.append({'id':sid,'status':'queued_six_hour_ticks' if year>=a.ticks_from_year else 'queued_first_issue','issue_count':len(chosen)})
        else:coverage.append({'id':sid,'status':'no_matching_48h_weather_history'})
    # Recent storms first; no quality-based selection. Group weather by year for bounded downloads.
    queue.sort(key=lambda r:(0 if r['storm_id']=='2025308N09144' else 1 if r['season']>=a.ticks_from_year else 2,-r['season'],r['storm_id'],r['issue_time_utc']))
    bundles={}
    for year in sorted({r['season'] for r in queue if r['atlas'] is not None}):
        rows=[r for r in queue if r['season']==year and r['atlas'] is not None];indices=np.unique(np.concatenate([np.arange(r['atlas']-8,r['atlas']+1) for r in rows]))
        file=a.output/f'weather-{year}.npz'
        np.savez_compressed(file,time=times[indices],q=np.asarray(q[indices]),slp=np.asarray(slp[indices]),scale=atlas['scale'],offset=atlas['offset'])
        bundles[str(year)]={'file':file.name,'sha256':sha(file),'bytes':file.stat().st_size}
    geo=a.project/'data/v164_pressure_geography/geography.npz'
    (a.output/'geography.npz').write_bytes(geo.read_bytes())
    manifest={'schema':'trackformer-1.2-input-queue-v1','selection':f'Every in-domain six-hour observation from {a.ticks_from_year} onward; first issue per uncovered older storm since 1970. No future truth requirement or error selection.','ticks_from_year':a.ticks_from_year,
        'weather_first':str(times[0].astype('datetime64[ns]')),'weather_last':str(times[-1].astype('datetime64[ns]')),
        'source_sha256':{name:sha(a.project/path) for name,path in [('basin','track_build/basin_all_int8.npz'),('slp','track_build/basin_slp_atlas_float16.npy')]},
        'geography':{'file':'geography.npz','sha256':sha(geo)},'bundles':bundles,'queue':queue,'coverage':coverage}
    (a.output/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'),allow_nan=False))
    print(json.dumps({'queued_storms':len({r['storm_id'] for r in queue}),'queued_issues':len(queue),'six_hour_issues':sum(r['season']>=a.ticks_from_year for r in queue),'weather_MiB':sum(b['bytes'] for b in bundles.values())/1048576,'coverage':{s:sum(r['status']==s for r in coverage) for s in sorted({r['status'] for r in coverage})}}))

if __name__=='__main__':main()
