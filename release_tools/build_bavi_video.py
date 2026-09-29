"""Render saved Trackformer 1.2 Bavi means as an MP4; no new inference.

Requires numpy, matplotlib and ffmpeg. --source-root prepares the public subset.
The active GPU benchmark is not stopped or modified.
"""
import argparse,csv,hashlib,json,subprocess
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO=Path(__file__).resolve().parents[1]
DATA=REPO/'paper/release_data'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def prepare(root):
    source=root/'benchmark_ensemble50/v173_e4/showcase_four.npz'
    inputs=root/'runs/four_v166_maps/bavi/inputs.npz'
    with np.load(inputs) as z:
        lat,lon,center=z['latitude'],z['longitude'],z['center']
        times=z['time'];issue=int(z['observation_ns'])
        assert int(times[-1])<=issue and len(times)==9
    with np.load(source) as z:
        i=z['storms'].tolist().index('bavi');assert int(z['member_count'])==50
        raw=z['route'][i].astype('float64');p=z['pressure'][i];coarse=z['fields'][i];native=z['native'][i]
    # The old writer converted issue-centred local km back to longitude with
    # predicted latitude. Undo that projection and use the actual issue latitude,
    # recovering the member-mean geographic centres (to stored float precision).
    east=((raw[:,1]-center[1]+180)%360-180)*111.2*np.maximum(np.cos(np.deg2rad(raw[:,0])),.2)
    route=raw.copy();route[:,1]=center[1]+east/(111.2*max(np.cos(np.deg2rad(center[0])),.2))
    roundtrip=center[1]+(route[:,1]-center[1])*max(np.cos(np.deg2rad(center[0])),.2)/np.maximum(np.cos(np.deg2rad(raw[:,0])),.2)
    assert np.allclose(roundtrip,raw[:,1],atol=1e-6)
    rows={}
    csvpath=root/'data/ibtracs/ibtracs.WP.list.v04r01.csv'
    with csvpath.open() as f:
        for r in csv.DictReader(f):
            if r['SID']=='2026182N09163':rows[r['ISO_TIME']]=r
    t=datetime.fromtimestamp(issue/1e9,timezone.utc);obs=[];truthp=[]
    for k in range(21):
        r=rows[(t+timedelta(hours=6*k)).strftime('%Y-%m-%d %H:%M:%S')]
        obs.append([float(r['LAT']),float(r['LON'])]);value=r['USA_PRES'].strip()
        truthp.append(float(value) if value else np.nan)
    assert np.allclose(obs[0],center,atol=.001)
    path=DATA/'bavi_video.npz'
    np.savez_compressed(path,forecast_lat_lon=np.vstack([center,route]),original_archive_route=raw,observed_lat_lon=np.array(obs),central_pressure_hpa=p,observed_pressure_hpa=np.array(truthp)[1:],regional_pressure_hpa=native,basin_pressure_hpa=coarse,latitude=lat,longitude=lon)
    rings=json.loads((root/'trackformer-weatherlab-site/public/data/history/coastlines.json').read_text())
    rings=[r for r in rings if any(125<=x<=161 and 5<=y<=30 for x,y in r)]
    (DATA/'bavi_coastlines.json').write_text(json.dumps(rings,separators=(',',':'))+'\n')
    meta={'model':'Trackformer 1.2','storm':'BAVI','storm_id':'2026182N09163','issue_time_utc':t.isoformat().replace('+00:00','Z'),'members':50,'leads_hours':list(range(6,121,6)),
        'source_archive_sha256':sha(source),'inputs_sha256':sha(inputs),'observed_csv_sha256':sha(csvpath),'data_sha256':sha(path),
        'checkpoint_sha256':'f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0',
        'forecast_source':'Saved 50 smooth historical-input perturbation means from the released checkpoint; no new inference for this video.',
        'route_projection_correction':'Undo the archived predicted-latitude longitude conversion and restore the issue-latitude projection used by ensemble inference. Original route retained; round-trip verified.',
        'observed_route_source':'Exact six-hour LAT/LON from local IBTrACS WP archive, storm 2026182N09163.',
        'observed_pressure_source':'IBTrACS USA_PRES; not JMA/WMO pressure. Missing labels are masked.',
        'field_display':'Unchanged physical member-mean basin field, overlaid with the unchanged regional reconstruction only within its fixed coverage. No synthetic pressure extension or future observed pressure map.',
        'regional_bounds':[float(lon.min()),float(lon.max()),float(lat.min()),float(lat.max())],
        'qualification':'Requested Bavi development showcase, not an untouched holdout or a claim of typical skill.'}
    (DATA/'bavi_video.json').write_text(json.dumps(meta,indent=2)+'\n')

def build(work):
    work.mkdir(parents=True,exist_ok=True)
    meta=json.loads((DATA/'bavi_video.json').read_text());assert sha(DATA/'bavi_video.npz')==meta['data_sha256']
    with np.load(DATA/'bavi_video.npz') as z:a={k:z[k] for k in z.files}
    rings=json.loads((DATA/'bavi_coastlines.json').read_text())
    f,o=a['forecast_lat_lon'],a['observed_lat_lon'];lat,lon=a['latitude'],a['longitude']
    bounds=[127,159,6,27];levels=np.arange(932,1025,2);contours=np.arange(932,1025,4)
    glat=np.linspace(60,0,25);glon=np.linspace(100,180,33);lead=np.arange(6,121,6)
    start=datetime.fromisoformat(meta['issue_time_utc'].replace('Z','+00:00'))
    plt.rcParams.update({'font.size':12,'axes.titlesize':15})
    for k in range(20):
        fig=plt.figure(figsize=(16,10),dpi=100,facecolor='#f8fafb')
        gs=fig.add_gridspec(2,2,width_ratios=[3.9,1.25],height_ratios=[4.5,1.25],left=.055,right=.94,bottom=.10,top=.86,wspace=.25,hspace=.35)
        ax=fig.add_subplot(gs[0,:]);ax.set_facecolor('#f8fafb')
        ax.contourf(glon,glat,a['basin_pressure_hpa'][k],levels=levels,cmap='RdYlBu_r',extend='both')
        ax.contour(glon,glat,a['basin_pressure_hpa'][k],levels=contours,colors='#546971',linewidths=.45)
        im=ax.contourf(lon,lat,a['regional_pressure_hpa'][k],levels=levels,cmap='RdYlBu_r',extend='both')
        cs=ax.contour(lon,lat,a['regional_pressure_hpa'][k],levels=contours,colors='#354b58',linewidths=.65)
        ax.clabel(cs,levels=np.arange(936,1025,12),fmt='%d',fontsize=9)
        for ring in rings:
            xy=np.asarray(ring);ax.plot(xy[:,0],xy[:,1],c='#66716b',lw=.8)
        ax.add_patch(Rectangle((lon.min(),lat.min()),np.ptp(lon),np.ptp(lat),fill=False,edgecolor='#265b61',lw=1.2,ls=':'))
        for points,color,label,style in [(o,'#182f42','Observed best track','--'),(f,'#b31565','Trackformer 1.2 mean of 50','-')]:
            ax.plot(points[:,1],points[:,0],ls=style,c=color,alpha=.25,lw=1.6)
            ax.plot(points[:k+2,1],points[:k+2,0],ls=style,c=color,lw=2.8,label=label)
            ax.scatter(points[k+1,1],points[k+1,0],s=75,c=color,edgecolors='white',linewidths=1.8,zorder=6)
        ax.set(xlim=bounds[:2],ylim=bounds[2:],xlabel='Longitude °E',ylabel='Latitude °N');ax.grid(alpha=.13);ax.set_aspect(1/np.cos(np.deg2rad(16.5)))
        ax.legend(loc='lower left',fontsize=11,framealpha=.95)
        ax.text(.985,.03,'Dotted boundary: fixed regional reconstruction\nOutside: 2.5° basin pressure only',transform=ax.transAxes,ha='right',va='bottom',fontsize=10,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':5})
        cax=fig.add_axes([.952,.35,.014,.45]);fig.colorbar(im,cax=cax,label='Model mean sea-level pressure (hPa)')
        curve=fig.add_subplot(gs[1,0]);curve.plot(lead,a['central_pressure_hpa'],c='#b31565',lw=2,label='Model central pressure');curve.plot(lead,a['observed_pressure_hpa'],'--',c='#182f42',lw=2,label='Observed pressure (IBTrACS USA)')
        curve.axvline(lead[k],c='#688e98',lw=1.5);curve.scatter([lead[k]],[a['central_pressure_hpa'][k]],c='#b31565',s=40,zorder=4)
        curve.set(xlim=(6,120),xlabel='Forecast lead (hours)',ylabel='Central pressure (hPa)',xticks=[6,24,48,72,96,120]);curve.grid(alpha=.2);curve.legend(loc='upper right',fontsize=10,ncol=2)
        info=fig.add_subplot(gs[1,1]);info.axis('off');p=a['central_pressure_hpa'][k];op=a['observed_pressure_hpa'][k]
        inside=lon.min()<=f[k+1,1]<=lon.max() and lat.min()<=f[k+1,0]<=lat.max()
        msg=f'Model centre: {p:.1f} hPa\nObserved: {op:.0f} hPa\n\n'+('Centre within regional field' if inside else 'Centre outside regional patch\nDetailed core is not shown here')
        info.text(0,1,msg,va='top',fontsize=13,color='#244754')
        valid=start+timedelta(hours=int(lead[k]));fig.suptitle(f'Trackformer 1.2  /  BAVI  /  +{lead[k]:03d} h',x=.055,y=.97,ha='left',fontsize=24,fontweight='bold',color='#173947')
        fig.text(.055,.915,f'Issue: {start:%d %b %Y %H:%M} UTC     |     Valid: {valid:%d %b %Y %H:%M} UTC     |     50-member mean',fontsize=14,color='#385865')
        fig.text(.055,.035,'Actual model pressure fields; observed route is verification only. Mean central pressure is not the minimum of the mean map.',fontsize=11,color='#47636d')
        fig.savefig(work/f'frame_{k:03d}.png',dpi=100); 
        if k==9:fig.savefig(REPO/'paper/trackformer_1_2_bavi_video_poster.png',dpi=100)
        plt.close(fig)
    output=REPO/'docs/trackformer_1_2_bavi.mp4'
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','warning','-framerate','1','-i',str(work/'frame_%03d.png'),'-vf','fps=30,tpad=stop_mode=clone:stop_duration=3','-c:v','h264_videotoolbox','-b:v','4M','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(output)]
    subprocess.run(cmd,check=True)
    meta['video']={'frames':20,'duration_seconds':23,'codec':'H.264','encoder':'h264_videotoolbox','file':'docs/trackformer_1_2_bavi.mp4','sha256':sha(output),'playback':'1 second per six-hour lead; final frame held for 3 seconds; no interpolated forecast states'}
    (DATA/'bavi_video.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Saved',output,output.stat().st_size,'bytes')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path);p.add_argument('--work',type=Path,required=True);args=p.parse_args()
    if args.source_root:prepare(args.source_root)
    build(args.work)
