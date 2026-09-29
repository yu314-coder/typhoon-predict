"""Build an offline interactive map from saved physical 50-member fields.

--source-root prepares the public data subset; otherwise use published subset.
No model inference, GPU allocation, training, or weather retrieval is performed.
"""
import argparse, base64, csv, hashlib, io, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
DATA=REPO/'evaluation/release_data'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def absolute(local,lat,lon):
    return np.stack([lat+local[...,1]/111.2,lon+local[...,0]/(111.2*max(np.cos(np.deg2rad(lat)),.2))],-1)

def prepare(root):
    archive=root/'benchmark_ensemble50/v173_e4/v173_e4_causal_ensemble50.npz'
    cases=json.loads((REPO/'evaluation/trackformer_1_2_showcase_selection.json').read_text())['all_case_metrics']
    patches={r['track_archive_row']:r for r in json.loads((root/'data/v164_reuse/plan.json').read_text())['rows']}
    with np.load(root/'track_build/track_windows_v13.npz') as w:
        target,mask=w['target'],w['target_mask']
        times,ids=w['base_time'],w['storm_id'].astype(str)
        lats,lons=w['base_lat'],w['base_lon']%360
    with np.load(archive) as z:
        routes=z['ensemble50_local'];truth=z['truth_local'];pressure=z['ensemble50_pressure_hpa']
        eligible=[]
        for c in cases:
            i=c['case_index'];r=c['source_row'];lat,lon=c['base_lat'],c['base_lon']
            patch=patches.get(r)
            anchor=np.array([patch['center_lat'],patch['center_lon']]) if patch else np.round(np.array([lat,lon])*4)/4
            p,t=absolute(routes[i],lat,lon),absolute(truth[i],lat,lon)
            points=np.vstack([[lat,lon],p,t])
            fits=(np.abs(points-anchor)<14).all() and (points[:,0]>0).all() and (points[:,0]<60).all() and (points[:,1]>100).all() and (points[:,1]<180).all()
            pm=mask[r,:20,3].astype(bool)&np.isfinite(target[r,:20,3])&(target[r,:20,3]>800)&(target[r,:20,3]<1100)
            pe=float(np.abs(pressure[i]-target[r,:20,3])[pm].mean()) if pm.any() else None
            m=c['models']['1.2']
            if fits and pm.sum()>=18 and pe<=15 and m['shape_similarity']>=.9 and m['direction_error_deg'] is not None and m['direction_error_deg']<=30:
                eligible.append((m['mean_track_error_km'],i,anchor,pe))
        _,i,anchor,pe=min(eligible,key=lambda x:(x[0],x[1]));c=cases[i];r=c['source_row']
        lookup={int(times[j]):int(j) for j in np.flatnonzero(ids==c['storm_id'])}
        issue=int(times[r]);future=[lookup[issue+k*6*3600*10**9] for k in range(21)]
        observed=np.stack([lats[future],lons[future]],-1)
        forecast=np.vstack([[c['base_lat'],c['base_lon']],absolute(routes[i],c['base_lat'],c['base_lon'])])
        assert (np.abs(np.vstack([observed,forecast])-anchor)<15).all()
        field=z['ensemble50_native_mslp_hpa'][i].copy()
        assert field.shape==(20,121,121) and np.isfinite(field).all()
        pm=mask[r,:20,3].astype(bool)&(target[r,:20,3]>800)&(target[r,:20,3]<1100)
        np.savez_compressed(DATA/'pressure_showcase.npz',pressure_hpa=field,latitude=anchor[0]+np.linspace(15,-15,121),longitude=anchor[1]+np.linspace(-15,15,121),forecast_lat_lon=forecast,observed_lat_lon=observed,central_pressure_hpa=pressure[i],observed_pressure_hpa=target[r,:20,3],pressure_mask=pm)
    names={}
    with (root/'data/ibtracs/ibtracs.WP.list.v04r01.csv').open() as f:
        for row in csv.DictReader(f):names[row['SID']]=row['NAME']
    rule='Among the saved 270 cases, require both routes within the fixed regional patch with 1-degree margin and valid basin coverage, shape >=0.90, heading error <=30 degrees, >=18 pressure labels and central-pressure MAE <=15 hPa; choose lowest mean track error. No field values altered.'
    meta={'public_model':'Trackformer 1.2','storm_name':names[c['storm_id']],'storm_id':c['storm_id'],'issue_time_utc':c['issue_time_utc'],'case_index':i,'source_row':r,'members':50,'member_policy':'50 smooth historical-input perturbations, not latent samples','selection_rule':rule,'eligible_cases':len(eligible),'selected_best_not_representative':True,'metrics':c['models']['1.2'],'central_pressure_mae_hpa':pe,'native_history_available':r in patches,'checkpoint_sha256':'f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0','archive_sha256':sha(archive),'data_sha256':sha(DATA/'pressure_showcase.npz'),'observed_route_source':'Exact six-hour issue-centre coordinates from the same storm in track_windows_v13, not a reconstruction from cumulative local steps. Retrospective best-track verification only.','field_source':'Unmodified saved ensemble50_native_mslp_hpa: model basin-plus-moving-core reconstruction on the fixed issue-anchored geographic grid. 0.25-degree output sampling is not native forecast resolution.','note':'Mean route and mean central pressure need not equal the minimum of the mean field.'}
    rings=json.loads((root/'trackformer-weatherlab-site/public/data/history/coastlines.json').read_text())
    rings=[ring for ring in rings if any(anchor[1]-15<=x<=anchor[1]+15 and max(0,anchor[0]-15)<=y<=anchor[0]+15 for x,y in ring)]
    (DATA/'pressure_showcase_coastlines.json').write_text(json.dumps(rings,separators=(',',':'))+'\n')
    (DATA/'pressure_showcase.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Selected',meta['storm_name'],meta['issue_time_utc'],'case',i,'track',meta['metrics']['mean_track_error_km'],'pressure',pe)

def build():
    meta=json.loads((DATA/'pressure_showcase.json').read_text())
    rings=json.loads((DATA/'pressure_showcase_coastlines.json').read_text())
    with np.load(DATA/'pressure_showcase.npz') as z: a={k:z[k] for k in z.files}
    assert sha(DATA/'pressure_showcase.npz')==meta['data_sha256']
    lat,lon=a['latitude'],a['longitude'];tracks=np.concatenate([a['forecast_lat_lon'],a['observed_lat_lon']])
    bounds=[max(100,float(lon.min()),float(tracks[:,1].min()-3)),min(180,float(lon.max()),float(tracks[:,1].max()+3)),max(0,float(lat.min()),float(tracks[:,0].min()-3)),min(60,float(lat.max()),float(tracks[:,0].max()+3))]
    ratio=(bounds[3]-bounds[2])/((bounds[1]-bounds[0])*np.cos(np.deg2rad((bounds[2]+bounds[3])/2)))
    fields=a['pressure_hpa'];valid=(lat[:,None]>=0)&(lat[:,None]<=60)&(lon[None,:]>=100)&(lon[None,:]<=180)
    lo=int(np.floor(fields[:,valid].min()/4)*4);hi=int(np.ceil(fields[:,valid].max()/4)*4)
    levels=np.arange(lo,hi+2,2);frames=[]
    for field in fields:
        fig=plt.figure(figsize=(9,9*ratio));ax=fig.add_axes([0,0,1,1])
        f=np.ma.masked_where(~valid,field)
        ax.contourf(lon,lat,f,levels=levels,cmap='RdYlBu_r',extend='both')
        contour=ax.contour(lon,lat,f,levels=np.arange(lo,hi+4,4),colors='#314656',linewidths=.65)
        ax.clabel(contour,levels=np.arange(lo,hi+4,8),fontsize=8,fmt='%d',inline_spacing=5)
        for ring in rings:
            xy=np.asarray(ring);ax.plot(xy[:,0],xy[:,1],c='#4f6164',lw=.8)
        ax.set(xlim=bounds[:2],ylim=bounds[2:]);ax.set_axis_off()
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=115);plt.close(fig)
        frames.append('data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())
    packet={'meta':meta,'bounds':bounds,'ratio':ratio,'range':[lo,hi],'frames':frames,'fields':fields.round(2).tolist(),'latitude':lat.tolist(),'longitude':lon.tolist(),'forecast':a['forecast_lat_lon'].tolist(),'observed':a['observed_lat_lon'].tolist(),'pressure':a['central_pressure_hpa'].tolist(),'observedPressure':[float(v) if ok else None for v,ok in zip(a['observed_pressure_hpa'],a['pressure_mask'])]}
    template=(REPO/'release_tools/pressure_showcase_template.html').read_text()
    html=template.replace('/*__DATA__*/',json.dumps(packet,separators=(',',':'),allow_nan=False))
    (REPO/'docs/trackformer_1_2_pressure.html').write_text(html)
    # A real, clickable README preview; it is not a fabricated map.
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained');k=11
    im=ax.contourf(lon,lat,np.ma.masked_where(~valid,fields[k]),levels=levels,cmap='RdYlBu_r')
    cs=ax.contour(lon,lat,np.ma.masked_where(~valid,fields[k]),levels=np.arange(lo,hi+4,4),colors='#314656',linewidths=.6);ax.clabel(cs,levels=np.arange(lo,hi+4,8),fmt='%d',fontsize=8)
    for ring in rings:
        xy=np.asarray(ring);ax.plot(xy[:,0],xy[:,1],c='#4f6164',lw=.8)
    for key,color,label,style in [('observed_lat_lon','#162437','Observed route','--'),('forecast_lat_lon','#b01467','1.2 mean of 50','-')]:
        p=a[key];ax.plot(p[:,1],p[:,0],style,c=color,lw=2.3,label=label);ax.scatter(p[k+1,1],p[k+1,0],c=color,s=50,edgecolors='white',zorder=6)
    ax.set(xlim=bounds[:2],ylim=bounds[2:],xlabel='Longitude °E',ylabel='Latitude °N',title='Trackformer 1.2 · PRAPIROON · +72 h\nModel pressure + forecast and observed routes')
    ax.set_aspect(1/np.cos(np.deg2rad((bounds[2]+bounds[3])/2)));ax.legend();fig.colorbar(im,ax=ax,label='Model mean sea-level pressure (hPa)',shrink=.75)
    fig.text(.5,-.015,'Click to explore +6 to +120 h · Selected best-performing example, not typical skill',ha='center',fontsize=10)
    fig.savefig(REPO/'evaluation/trackformer_1_2_interactive_pressure_preview.png',dpi=150,bbox_inches='tight');plt.close(fig)
    print('Built offline interactive map and README preview:',len(html),'bytes')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path);args=p.parse_args()
    if args.source_root:prepare(args.source_root)
    build()
