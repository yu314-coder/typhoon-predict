"""Exact public 1.2 CPU forecasts for scheduled GitHub-hosted research jobs.

No future analyses/official forecast positions enter the model. One deterministic
member, explicitly distinct from the existing 50-member development benchmark.
The output Git branch is a resumable archive, never a source of model inputs.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlencode
import numpy as np
import requests
import torch
from scipy.interpolate import RegularGridInterpolator
from backfill_queue import queue_state, retry_is_cooling
from cloud_continuation import LIVE_REFRESH_INTERVAL_SECONDS
from gfs_archive import download_analysis
from ensemble_forecast import analysis_motion, run_ensemble, mean_outputs

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/trackformer_1_2_field'
sys.path.insert(0,str(MODEL))
from model import CoreForecaster
from wind_estimation import diagnose_outputs, summarize_members
CHECKPOINT='f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0'
HF='https://huggingface.co/euler314/typhoon-predict/resolve/f67f0b206876387c2aa6c9f8f19cb009b7a86091'
HOUR=3600*10**9
NCEP_LAST=datetime(2026,3,17,18,tzinfo=timezone.utc)
HISTORICAL_INPUT_VERSION='2026-09-30-exact-archived-gfs-v1'

def utc(d):return d.astimezone(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
def parse(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
def ns(s):return int(np.datetime64(s.replace('Z',''),'ns').astype('int64'))
def digest(b):return hashlib.sha256(b).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,separators=(',',':'),allow_nan=False));temp.replace(p)
def get(url,**kwargs):
    last=None
    for attempt in range(3):
        try:
            r=requests.get(url,timeout=(15,90),**kwargs);r.raise_for_status();return r
        except requests.RequestException as e:last=e;time.sleep(attempt+1)
    raise last
def asset(cache,name,url,expected):
    p=cache/name
    if p.exists() and digest(p.read_bytes())==expected:return p
    b=get(url).content
    if digest(b)!=expected:raise ValueError('Asset SHA-256 mismatch: '+name)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.part');tmp.write_bytes(b);tmp.replace(p);return p

def statics(contract,geo,lat,lon):
    def grid(y,x):
        yy,xx=np.meshgrid(y,x,indexing='ij');iy=np.rint((90-yy)*4).astype(int);ix=np.rint(xx*4).astype(int)%1440
        if np.any(iy<0) or np.any(iy>=geo['land_fraction'].shape[0]):raise ValueError('Geography outside latitude coverage')
        return np.stack([yy/90,xx/180-1,geo['land_fraction'][iy,ix],geo['elevation_m'][iy,ix]/8000]).astype('float32')
    return grid(contract['global_lat'],contract['global_lon']),grid(round(lat*4)/4+np.linspace(15,-15,121),round(lon*4)/4+np.linspace(-15,15,121))

def inputs(weather,times,row,contract,geo):
    issue=ns(row['issue_time_utc']);times=np.asarray(times,dtype='int64')
    if weather.shape!=(9,8,25,33) or times.shape!=(9,) or times[-1]>issue or issue-times[-1]>12*HOUR or not np.all(np.diff(times)==6*HOUR):
        raise ValueError('Nine exact causal six-hour analyses ending no later than issue time are required')
    if not np.isfinite(weather).all():raise ValueError('Non-finite weather')
    lat,lon=row['lat'],row['lon']
    if not (0<lat<60 and 100<lon<180):raise ValueError('Outside released Western Pacific model domain')
    gs,rs=statics(contract,geo,lat,lon)
    norm=contract['normalization'];mean=np.asarray(norm['mean'],dtype='float32');std=np.asarray(norm['std'],dtype='float32')
    raw=[row.get('wind_kt'),row.get('pressure_hpa')]
    mask=[v is not None and np.isfinite(v) and lo<v<hi for v,lo,hi in zip(raw,[-1,800],[250,1100])]
    v={'global_history':(weather-mean[None,:,None,None])/std[None,:,None,None],
       'regional_history':np.zeros((9,1,121,121),dtype='float32'),'detail_available':np.zeros(1,dtype='float32'),
       'global_static':gs,'regional_static':rs,'center':np.asarray([lat,lon]),'motion':np.asarray(row.get('motion',[0.,0.])),
       'issue_intensity':np.asarray([x if ok else 0 for x,ok in zip(raw,mask)]),'issue_mask':np.asarray(mask,dtype='float32')}
    return {k:torch.from_numpy(np.asarray(a,dtype='float32')[None].copy()) for k,a in v.items()}

def ncep(row,contract):
    """Read exact 4x-daily fields through issue time, including cross-year history."""
    import xarray as xr
    end=parse(row['issue_time_utc']);wanted=[end-timedelta(hours=6*(8-i)) for i in range(9)]
    fields=np.empty((9,8,25,33),dtype='float32')
    specs=[('slp',None,0),('hgt',500,1),('uwnd',850,2),('vwnd',850,3),('uwnd',500,4),('vwnd',500,5),('uwnd',200,6),('vwnd',200,7)]
    for year in sorted({t.year for t in wanted}):
        positions=[i for i,t in enumerate(wanted) if t.year==year];dates=np.asarray([wanted[i].replace(tzinfo=None) for i in positions],dtype='datetime64[ns]')
        for variable in ['slp','hgt','uwnd','vwnd']:
            group='surface' if variable=='slp' else 'pressure'
            url=f'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/{group}/{variable}.{year}.nc'
            with xr.open_dataset(url,engine='netcdf4') as ds:
                for var,level,ch in specs:
                    if var!=variable:continue
                    a=ds[var].sel(time=dates,lat=contract['global_lat'],lon=contract['global_lon'])
                    if level is not None:a=a.sel(level=level)
                    if not np.array_equal(a.time.values.astype('datetime64[ns]'),dates):raise ValueError('NCEP time mismatch')
                    values=np.asarray(a.values,dtype='float32')
                    if variable=='slp':
                        if a.attrs.get('units','').lower() not in ['pascals','pa']:raise ValueError('Unexpected SLP unit')
                        values=values/100
                    fields[positions,ch]=values
    if not np.isfinite(fields).all() or np.max(np.abs(fields))>100000:raise ValueError('Missing or invalid reanalysis')
    return fields,np.asarray([ns(utc(t)) for t in wanted])

def decode_gfs(p,cycle,contract):
    import eccodes as ec
    key=cycle.strftime('%Y%m%d%H')
    maps={};yy,xx=np.meshgrid(contract['global_lat'],contract['global_lon'],indexing='ij')
    with p.open('rb') as f:
        while (gid:=ec.codes_grib_new_from_file(f)) is not None:
            try:
                name=ec.codes_get(gid,'shortName');level=ec.codes_get(gid,'level');kind=ec.codes_get(gid,'typeOfLevel')
                if (ec.codes_get(gid,'step')!=0 or ec.codes_get(gid,'dataTime')!=cycle.hour*100
                        or str(ec.codes_get(gid,'dataDate'))!=cycle.strftime('%Y%m%d')):
                    raise ValueError('GFS must be the exact f000 analysis')
                target=0 if name in ['prmsl','msl'] and kind in ['meanSea','meanSeaLevel'] else ({('gh',500):1,('u',850):2,('v',850):3,('u',500):4,('v',500):5,('u',200):6,('v',200):7}.get((name,level)) if kind=='isobaricInhPa' else None)
                if target is None:continue
                if target in maps:raise ValueError('Duplicate GFS channel')
                units=ec.codes_get(gid,'units')
                expected={0:['Pa'],1:['gpm'],2:['m s**-1'],3:['m s**-1'],4:['m s**-1'],5:['m s**-1'],6:['m s**-1'],7:['m s**-1']}
                if units not in expected[target]:raise ValueError('Unexpected GFS physical unit: '+units)
                lat=ec.codes_get_array(gid,'latitudes');lon=ec.codes_get_array(gid,'longitudes');values=ec.codes_get_values(gid)
                latvals=np.unique(lat);lonvals=np.unique(lon);grid=np.full((len(latvals),len(lonvals)),np.nan)
                grid[np.searchsorted(latvals,lat),np.searchsorted(lonvals,lon)]=values
                result=RegularGridInterpolator((latvals,lonvals),grid,bounds_error=True)(np.stack([yy,xx],axis=-1))
                if not np.isfinite(result).all() or ec.codes_get(gid,'numberOfMissing'):
                    raise ValueError('Missing or non-finite GFS analysis values')
                maps[target]=result/100 if target==0 else result
            finally:ec.codes_release(gid)
    if set(maps)!=set(range(8)):raise ValueError('GFS channels missing: '+str(sorted(maps)))
    return np.stack([maps[i] for i in range(8)]).astype('float32')

def gfs(cycle,cache,contract):
    key=cycle.strftime('%Y%m%d%H');p=cache/f'gfs-{key}.grib2'
    if not p.exists():
        params={'file':f'gfs.t{key[8:]}z.pgrb2.1p00.f000','dir':f'/gfs.{key[:8]}/{key[8:]}/atmos',
                'lev_mean_sea_level':'on','lev_200_mb':'on','lev_500_mb':'on','lev_850_mb':'on',
                'var_PRMSL':'on','var_HGT':'on','var_UGRD':'on','var_VGRD':'on','subregion':'',
                'leftlon':'100','rightlon':'190','toplat':'60','bottomlat':'0'}
        b=get('https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?'+urlencode(params)).content
        if not b.startswith(b'GRIB') or len(b)>5000000:raise ValueError('Invalid NOAA GRIB response')
        temp=p.with_suffix('.part');temp.write_bytes(b);temp.replace(p)
    return decode_gfs(p,cycle,contract)

def remote_history(row,cache,contract):
    end=parse(row['issue_time_utc'])
    if end<=NCEP_LAST:
        weather,times=ncep(row,contract)
        return weather,times,'NOAA NCEP Reanalysis 1, exact six-hour history. Retrospective reconstruction; no operational-availability or independent-test claim.',{'provider':'NOAA NCEP Reanalysis 1'}
    dates=[end-timedelta(hours=6*(8-i)) for i in range(9)]
    weather=[];analyses=[]
    # Use a consistent source across the whole input window, never mix R1/GFS.
    for cycle in dates:
        p,url=download_analysis(cycle,cache,get)
        weather.append(decode_gfs(p,cycle,contract))
        analyses.append({'valid_time_utc':utc(cycle),'url':url,'subset_sha256':digest(p.read_bytes())})
    return (np.stack(weather),np.asarray([ns(utc(t)) for t in dates]),
            'Archived NOAA GFS f000 analyses at nine exact six-hour valid times. NCEP R1 ended March 17, 2026. Experimental GFS input transfer; retrospective reconstruction, not an operational-availability or independent-test claim.',
            {'provider':'NOAA GFS archive','experimental_transfer':True,'analyses':analyses})

def jma_analysis_row(sid,parts,now):
    """Only the current official analysis can initialize the neural forecast."""
    title=next(p for p in parts if p.get('part')=='title')
    analyses=[p for p in parts if isinstance(p.get('part'),dict) and p['part'].get('en')=='Analysis']
    if len(analyses)!=1:raise ValueError('JMA requires exactly one current analysis')
    p=analyses[0]
    if p.get('advancedHours')!=0:raise ValueError('JMA analysis must be +0, not a forecast')
    issue=p['validtime']['UTC'];valid=parse(issue);lat,lon=map(float,p['position']['deg'])
    if valid>now+timedelta(minutes=5) or now-valid>timedelta(hours=18):raise ValueError('JMA analysis is stale or future-dated')
    if not np.isfinite([lat,lon]).all() or not(-90<=lat<=90 and -180<=lon<=360):raise ValueError('Invalid JMA position')
    if not(0<lat<60 and 100<lon<180):return None
    def number(value,lo,hi):
        try:value=float(value)
        except (TypeError,ValueError):return None
        return value if np.isfinite(value) and lo<value<hi else None
    pressure=number(p.get('pressure'),800,1100)
    wind=number(p.get('maximumWind',{}).get('sustained',{}).get('kt'),-1,250)
    return {'id':'auto-live-'+sid+'-'+valid.strftime('%Y%m%dT%H%M'),'storm_id':sid,
            'name':title.get('name',{}).get('en','Unnamed'),'season':valid.year,'issue_time_utc':issue,
            'lat':lat,'lon':lon,'pressure_hpa':pressure,'wind_kt':wind,'motion':[0.,0.],'observed':[],
            'jma_analysis':{'source_url':f'https://www.jma.go.jp/bosai/typhoon/data/{sid}/specifications.json',
                            'bulletin_issue_utc':title.get('issue',{}).get('UTC'),
                            'analysis_valid_utc':issue,'maximum_wind_unit':'kt','motion_missing':True}}

def live_rows():
    root='https://www.jma.go.jp/bosai/typhoon/data'
    targets=get(root+'/targetTc.json').json();rows=[]
    for target in targets:
        sid=target.get('tropicalCyclone','')
        if not sid.startswith('TC') or not sid[2:].isdigit():continue
        parts=get(f'{root}/{sid}/specifications.json').json()
        row=jma_analysis_row(sid,parts,datetime.now(timezone.utc))
        if row is not None:
            row['ensemble_motion'],row['ensemble_motion_source']=analysis_motion(parts)
            rows.append(row)
    return rows

def infer(model,contract,geo,weather,times,row,out,kind,source,provenance=None):
    x=inputs(weather,times,row,contract,geo)
    live_core = kind.startswith('live-')
    if live_core:
        from live_pressure_export import capturing_model, core_export
        model = capturing_model(model, contract)
    with torch.inference_mode():
        state=model.initial(x);outputs=[]
        initial_pressure=(state['g'][0,0]*contract['normalization']['std'][0]+contract['normalization']['mean'][0]).numpy()
        for _ in range(20):state,o=model.step(state);outputs.append(o)
    route=[{'lat':row['lat'],'lon':row['lon'],'pressure_hpa':row.get('pressure_hpa'),'valid_time_utc':row['issue_time_utc'],'lead_hours':0}]
    fields=[];valid=[]
    for i,o in enumerate(outputs):
        center=o['center'][0].numpy();p=float(o['pressure'][0]);field=o['global'][0,0].numpy()*contract['normalization']['std'][0]+contract['normalization']['mean'][0]
        if not np.isfinite(center).all() or not np.isfinite(field).all() or not 800<p<1100 or field.min()<800 or field.max()>1100:raise ValueError('Model output failed physical/finite checks')
        stamp=utc(parse(row['issue_time_utc'])+timedelta(hours=(i+1)*6));valid.append(stamp);fields.append(np.round(field,2).tolist())
        route.append({'lat':float(center[0]),'lon':float(center[1]),'pressure_hpa':p,'valid_time_utc':stamp,'lead_hours':(i+1)*6,
                      'wind_kt_auxiliary':float(o['vmax'][0]),
                      'wind_estimation':summarize_members(diagnose_outputs(o,contract))})
    ident=row['id'];note=source+' Single deterministic member. Native detail unavailable and masked; issue-time wind/motion may be missing. No future weather or official forecast route is an input. Research only.'
    doc={'schema_version':'1.0','id':ident,'model':'Trackformer 1.2','checkpoint_sha256':CHECKPOINT,'storm_id':row['storm_id'],'name':row['name'],
         'season':row['season'],'issue_time_utc':row['issue_time_utc'],'members':1,'kind':kind,'route':route,'observed':row.get('observed',[]),
         'field_url':'/api/history/v1/fields/'+ident,'source_note':note,'pressure_note':'Model basin field; central pressure is the moving-core readout, not necessarily the basin minimum.',
         'input_history_times_utc':[str(np.datetime64(int(t),'ns').astype('datetime64[s]'))+'Z' for t in times],
         'input_tensor_sha256':digest(b''.join(x[k].numpy().tobytes() for k in sorted(x))),
         'input_weather_source':provenance,
         'issue_analysis_source':row.get('jma_analysis'),
         'generated_at_utc':utc(datetime.now(timezone.utc)),'run_url':os.environ.get('RUN_URL')}
    field={'model':'Trackformer 1.2','forecast_id':ident,'members':1,'checkpoint_sha256':CHECKPOINT,'latitude':contract['global_lat'],'longitude':contract['global_lon'],
           'valid_times_utc':valid,'units':'hPa','rounding_hpa':.01,'grid':'2.5 degree model basin','note':note,'pressure_hpa':fields,
           'issue_pressure_hpa':np.round(initial_pressure,2).tolist(),'issue_valid_time_utc':row['issue_time_utc'],
           'issue_pressure_note':'Causal model initial basin state, not a future forecast field.'}
    if live_core:
        field['core_reconstruction'] = core_export(model.blocks, doc, contract)
    # Lossless gzip keeps thousands of independent issue fields out of huge plain JSON blobs.
    field_path=out/'fields'/f'{ident}.json.gz';field_path.parent.mkdir(parents=True,exist_ok=True)
    temp=field_path.with_suffix('.tmp');temp.write_bytes(gzip.compress(json.dumps(field,separators=(',',':'),allow_nan=False).encode(),mtime=0));temp.replace(field_path)
    write(out/'forecasts'/f'{ident}.json',doc)
    print(json.dumps({'complete':ident,'forecast_points':len(route),'fields':len(fields)}),flush=True)
    return doc

def infer_live50(model,contract,geo,weather,times,row,out,source,provenance=None):
    """Append a distinct 50-member live issue; never overwrite the old one-member archive."""
    ident=row['id'].replace('auto-live-','auto-live50-',1)
    destination=out/'live50'
    if (destination/'forecasts'/f'{ident}.json').exists():
        return
    initial=dict(row,motion=row.get('ensemble_motion',row.get('motion',[0.,0.])))
    x=inputs(weather,times,initial,contract,geo)
    from live_pressure_export import capturing_model, core_export
    model = capturing_model(model, contract)
    output,policy=run_ensemble(model,x,contract,'cpu',chunk=2,
        progress=lambda done,total,elapsed:print(json.dumps({'live50_members':done,'total':total,'elapsed_seconds':elapsed}),flush=True))
    means=mean_outputs(output)
    route=[{'lat':row['lat'],'lon':row['lon'],'pressure_hpa':row.get('pressure_hpa'),
            'valid_time_utc':row['issue_time_utc'],'lead_hours':0}]
    valid=[]
    for i in range(20):
        stamp=utc(parse(row['issue_time_utc'])+timedelta(hours=(i+1)*6));valid.append(stamp)
        center=means['center'][i]
        route.append({'lat':float(center[0]),'lon':float(center[1]),'pressure_hpa':float(means['pressure'][i]),
                      'valid_time_utc':stamp,'lead_hours':(i+1)*6,'valid_member_count':int(output['track_valid'][:,i].sum()),
                      'wind_kt_auxiliary':float(means['vmax'][i]),'wind_estimation':policy['wind_estimation_by_lead'][i]})
    note=source+' Equal-weight mean of 50 distinct seeded input-perturbed forecasts, not latent samples or deterministic duplicates. Actual common-grid model basin pressure; native detail unavailable and masked. No future weather or official forecast route is an input. Experimental transfer; research only.'
    doc={'schema_version':'1.0','id':ident,'model':'Trackformer 1.2','checkpoint_sha256':CHECKPOINT,
         'storm_id':row['storm_id'],'name':row['name'],'season':row['season'],'issue_time_utc':row['issue_time_utc'],
         'members':50,'kind':'live-GFS-transfer-ensemble50','route':route,'ensemble_policy':policy,
         'source_note':note,'pressure_note':'Member-mean central pressure is read from each moving core; it is not the minimum of the mean basin field.',
         'input_history_times_utc':[str(np.datetime64(int(t),'ns').astype('datetime64[s]'))+'Z' for t in times],
         'input_tensor_sha256':digest(b''.join(x[k].numpy().tobytes() for k in sorted(x))),
         'input_weather_source':provenance,'issue_analysis_source':row.get('jma_analysis'),
         'motion_input':row.get('ensemble_motion_source'), 'generated_at_utc':utc(datetime.now(timezone.utc)),
         'run_url':os.environ.get('RUN_URL')}
    field={'model':'Trackformer 1.2','forecast_id':ident,'members':50,'checkpoint_sha256':CHECKPOINT,
           'latitude':contract['global_lat'],'longitude':contract['global_lon'],'valid_times_utc':valid,
           'units':'hPa','rounding_hpa':.01,'grid':'2.5 degree model basin; 50-member physical mean',
           'note':note,'pressure_hpa':np.round(means['basin'],2).tolist(),
           'issue_pressure_hpa':np.round(means['initial_basin'],2).tolist(),'issue_valid_time_utc':row['issue_time_utc'],
           'issue_pressure_note':'Physical mean of 50 causal model initial basin states, not a future forecast field.'}
    field['core_reconstruction'] = core_export(model.blocks, doc, contract)
    policy['core_mean_policy'] = field['core_reconstruction']['common_grid_policy']
    doc['pressure_note'] = ('Mean of 50 member central-pressure readouts, not the minimum of the mean map. '
        'The displayed physical map is each member basin plus geographically registered moving anomaly, averaged on a common grid.')
    folder=destination/'fields';folder.mkdir(parents=True,exist_ok=True)
    p=folder/f'{ident}.json.gz';temp=p.with_suffix('.tmp')
    temp.write_bytes(gzip.compress(json.dumps(field,separators=(',',':'),allow_nan=False).encode(),mtime=0));temp.replace(p)
    write(destination/'forecasts'/f'{ident}.json',doc)
    print(json.dumps({'complete_live50':ident,'distinct_inputs':50,'distinct_routes':50,'distinct_fields':50}),flush=True)

def live50_catalog(out):
    storms={}
    for p in sorted((out/'live50/forecasts').glob('*.json')):
        f=json.loads(p.read_text());s=storms.setdefault(f['storm_id'],{'id':f['storm_id'],'name':f['name'],'issues':[]})
        s['issues'].append({k:f[k] for k in ('id','issue_time_utc','members','kind')})
    for s in storms.values():s['issues'].sort(key=lambda i:i['issue_time_utc'],reverse=True)
    write(out/'live50/catalog.json',{'model':'Trackformer 1.2','members':50,'checkpoint_sha256':CHECKPOINT,
          'storms':list(storms.values()),'updated_at_utc':utc(datetime.now(timezone.utc))})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--historical-limit',type=int,default=80);ap.add_argument('--minutes',type=int,default=45);ap.add_argument('--skip-live',action='store_true');ap.add_argument('--only-id');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);a.cache.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.set_num_threads(2)
    config=json.loads((ROOT/'release_tools/history_inputs.json').read_text());source=config['base_url']
    plan=json.loads(asset(a.cache,'input-manifest.json',source+'/manifest.json',config['manifest_sha256']).read_text())
    meta=json.loads((MODEL/'manifest.json').read_text());assert meta['source_checkpoint_sha256']==CHECKPOINT
    for file,expected in meta['source_module_sha256'].items():assert digest((MODEL/file).read_bytes())==expected
    weight=asset(a.cache,'weights.pt',HF+'/models/trackformer_1_2_field/weights.pt',meta['inference_weights_sha256'])
    geo=np.load(asset(a.cache,'geography.npz',source+'/geography.npz',plan['geography']['sha256']),allow_pickle=False)
    contract=meta['data_contract'];model=CoreForecaster(contract).eval();model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    old=json.loads((a.output/'status.json').read_text()) if (a.output/'status.json').exists() else {}
    errors=old.get('errors',{});live=[];completed=0;succeeded=0
    planned_ids={row['id'] for row in plan['queue']}
    retired_errors=old.get('retired_source_errors',{})
    for ident in list(errors):
        if ident.startswith(('auto-hist-','auto-tick-')) and ident not in planned_ids:
            retired_errors[ident]=errors.pop(ident)
    if not a.skip_live:
        try:
            live=live_rows()
            for row in live:
                base_done=(a.output/'forecasts'/f"{row['id']}.json").exists()
                mean_done=(a.output/'live50/forecasts'/f"{row['id'].replace('auto-live-','auto-live50-',1)}.json").exists()
                if base_done and mean_done:continue
                try:
                    issue=parse(row['issue_time_utc']);now=datetime.now(timezone.utc);last=min(issue,now-timedelta(hours=3));last=last.replace(hour=last.hour//6*6,minute=0,second=0,microsecond=0)
                    weather=None
                    for lag in [0,6]:
                        end=last-timedelta(hours=lag)
                        if issue-end>timedelta(hours=12):continue
                        try:
                            dates=[end-timedelta(hours=6*(8-i)) for i in range(9)]
                            weather=np.stack([gfs(d,a.cache,contract) for d in dates]);break
                        except Exception as e:errors[row['id']]={'at':utc(now),'error':str(e)[:500]}
                    if weather is None:raise ValueError('Complete causal GFS history unavailable')
                    times=[ns(utc(d)) for d in dates]
                    note='Nine NOAA GFS f000 analyses; last valid '+utc(dates[-1])+'. Experimental GFS transfer.'
                    if not base_done:infer(model,contract,geo,weather,times,row,a.output,'live-GFS-transfer',note)
                    try:
                        provenance={'provider':'NOAA GFS f000 analysis','experimental_transfer':True,
                                    'analyses':[{'valid_time_utc':utc(d),'subset_sha256':digest((a.cache/f"gfs-{d.strftime('%Y%m%d%H')}.grib2").read_bytes()),
                                                 'source':'NOAA GFS f000; exact GRIB date, time, channels and units validated'} for d in dates]}
                        infer_live50(model,contract,geo,weather,times,row,a.output,note,provenance)
                        errors.pop('live50-'+row['storm_id'],None)
                    except Exception as e:
                        errors['live50-'+row['storm_id']]={'at':utc(datetime.now(timezone.utc)),'error':str(e)[:500]}
                    errors.pop(row['id'],None)
                except Exception as e:errors[row['id']]={'at':utc(datetime.now(timezone.utc)),'error':str(e)[:500]}
        except Exception as e:errors['live-feed']={'at':utc(datetime.now(timezone.utc)),'error':str(e)[:500]}
    for row in plan['queue']:
        if completed>=a.historical_limit or time.monotonic()-started>a.minutes*60:break
        ident=row['id']
        if a.only_id and ident!=a.only_id:continue
        if (a.output/'forecasts'/f'{ident}.json').exists():continue
        if (not a.only_id and ident in errors and
                retry_is_cooling(errors[ident],datetime.now(timezone.utc),HISTORICAL_INPUT_VERSION)):continue
        try:
            provenance={'provider':'Verified bundled NCEP Reanalysis 1'}
            if row['atlas'] is not None:
                b=plan['bundles'][row.get('bundle_key',str(row['season']))];z=np.load(asset(a.cache,b['file'],source+'/'+b['file'],b['sha256']),allow_pickle=False)
                wanted=ns(row['issue_time_utc'])+np.arange(-8,1)*6*HOUR;idx=np.searchsorted(z['time'],wanted)
                if not np.array_equal(z['time'][idx],wanted):raise ValueError('Bundled weather time mismatch')
                weather=np.concatenate([z['slp'][idx,None].astype('float32'),z['q'][idx].astype('float32')*z['scale'][None,:,None,None]+z['offset'][None,:,None,None]],axis=1);times=wanted
                source_note='Verified local NCEP atlas; exact causal history. Retrospective hindcast; may overlap fitting years, not an independent test.'
            else:weather,times,source_note,provenance=remote_history(row,a.cache,contract)
            infer(model,contract,geo,weather,times,row,a.output,'automatic-historical-hindcast',source_note,provenance);errors.pop(ident,None)
            succeeded+=1
        except Exception as e:errors[ident]={'at':utc(datetime.now(timezone.utc)),'error':str(e)[:500],'input_version':HISTORICAL_INPUT_VERSION};print(json.dumps({'failed':ident,'error':str(e)[:500]}),flush=True)
        completed+=1
    storms={}
    for p in sorted((a.output/'forecasts').glob('*.json')):
        f=json.loads(p.read_text());s=storms.setdefault(f['storm_id'],{'id':f['storm_id'],'name':f['name'],'season':f['season'],'issues':[]})
        s['issues'].append({k:f[k] for k in ['id','issue_time_utc','members','kind']})
    for s in storms.values():s['issues'].sort(key=lambda i:i['issue_time_utc'],reverse=True)
    planned={r['id'] for r in plan['queue']};ticks={r['id'] for r in plan['queue'] if r['id'].startswith('auto-tick-')}
    done={i['id'] for s in storms.values() for i in s['issues'] if i['id'] in planned}
    historical_done=len(done)
    status={'updated_at_utc':utc(datetime.now(timezone.utc)),'model':'Trackformer 1.2','members':1,'checkpoint_sha256':CHECKPOINT,'runner':'GitHub-hosted CPU; not the visitor or owner Mac',
            'live_checked_at_utc':utc(datetime.now(timezone.utc)) if not a.skip_live else old.get('live_checked_at_utc'),
            'live_refresh_interval_seconds':LIVE_REFRESH_INTERVAL_SECONDS,
            'live_refresh_policy':'Hourly cloud check; new inference only for a new valid JMA analysis. Existing issues are immutable. Scheduler/source delays are possible.',
            'live_ensemble_members':50,
            'historical_start_year':1970,'historical_total':len(plan['queue']),'historical_completed':historical_done,'errors':errors,
            'retired_source_errors':retired_errors,
            'historical_count_unit':'forecast issues','historical_storm_total':len({r['storm_id'] for r in plan['queue']}),
            'tick_start_year':plan.get('ticks_from_year'),'tick_hours':6,'tick_total':len(ticks),'tick_completed':len(done&ticks),
            'live_issues':[{'id':r['id'],'storm_id':r['storm_id'],'issue_time_utc':r['issue_time_utc'],'available':(a.output/'forecasts'/f"{r['id']}.json").exists()} for r in live],
            'run_url':os.environ.get('RUN_URL'),'elapsed_seconds':round(time.monotonic()-started,1),
            'batch_attempted':completed,'batch_succeeded':succeeded,
            'historical_input_version':HISTORICAL_INPUT_VERSION,
            'historical_pending_errors':len(set(errors)&(planned-done)),
            'requested_storms':[{**r,'planned':sum(p['storm_id']==r['storm_id'] for p in plan['queue']),
                                 'completed':sum(p['storm_id']==r['storm_id'] and p['id'] in done for p in plan['queue'])}
                                for r in plan.get('requested_storms',[])],
            **queue_state(planned,done,errors,datetime.now(timezone.utc),HISTORICAL_INPUT_VERSION)}
    live50_catalog(a.output)
    write(a.output/'catalog.json',{'schema_version':'1.0','model':'Trackformer 1.2','checkpoint_sha256':CHECKPOINT,'storms':list(storms.values()),'status':status})
    write(a.output/'status.json',status);write(a.output/'coverage.json',{'start_year':1970,'records':plan['coverage']})
    print(json.dumps({k:v for k,v in status.items() if k not in ['errors','retired_source_errors']}),flush=True)

if __name__=='__main__':main()
