"""Draw any IBTrACS WP storm; optionally diagnose genuine saved 1.2 members.

This is a local renderer, not a website deployment or an archive-wide inference
job. Observational charts need no model forecast. Model charts require hashed
member archives and exact issue-time inputs, never a mean-field substitution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from storm_structure_diagnostics import (
    initialize, ambient_pressure, diagnose, summarize, number, radius_km,
    QUADRANTS, THRESHOLDS, METRICS, DEFINITIONS, VERSION, REFERENCES,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = 'f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0'
BLUE, RED, BLACK = '#2166b0', '#c33c47', '#203247'
QC = {'NE':'#2166b0', 'SE':'#cc6831', 'SW':'#008475', 'NW':'#9753a7'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def utc(value):
    return value.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_catalogue(path, selected):
    """Stream all storms into a coverage index; retain only requested rows."""
    catalogue, rows = {}, {sid:{} for sid in selected}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            sid = row['SID']
            if sid in ('', ' ') or not sid[:4].isdigit() or int(sid[:4]) < 1970:
                continue
            stamp = row['ISO_TIME']
            if not stamp or stamp == ' ':
                continue
            if sid not in catalogue:
                catalogue[sid] = dict(storm_id=sid, name=row['NAME'].strip(),
                    year=int(row['SEASON']), first_utc=stamp, last_utc=stamp,
                    reports=0, usa_wind_reports=0, jma_wind_reports=0, rmw_reports=0,
                    r34_full_quadrant_reports=0, r50_full_quadrant_reports=0,
                    r64_full_quadrant_reports=0, source_agencies=set(),
                    chart_supported=True, model_forecast_not_implied=True)
            item = catalogue[sid]
            item['first_utc'],item['last_utc'] = min(item['first_utc'],stamp), max(item['last_utc'],stamp)
            item['reports'] += 1
            item['usa_wind_reports'] += number(row,'USA_WIND',0,250) is not None
            item['jma_wind_reports'] += number(row,'TOKYO_WIND',0,250) is not None
            item['rmw_reports'] += radius_km(row,'USA_RMW',rmw=True) is not None
            item['source_agencies'].add(row['USA_AGENCY'].strip() or 'missing')
            for t in THRESHOLDS:
                item[f'r{t}_full_quadrant_reports'] += all(radius_km(row,f'USA_R{t}_{q}') is not None for q in QUADRANTS)
            if sid in rows:
                if stamp in rows[sid]:
                    raise ValueError('Duplicate selected storm UTC report: '+sid+' '+stamp)
                rows[sid][stamp] = row
    for item in catalogue.values():
        item['source_agencies'] = sorted(item['source_agencies'])
    return sorted(catalogue.values(), key=lambda item:item['first_utc'], reverse=True), rows


def observation_series(rows, times):
    records = [rows.get(t.strftime('%Y-%m-%d %H:%M:%S'),{}) for t in times]
    values = dict(
        pressure=np.array([number(r,'TOKYO_PRES',800,1100) for r in records],dtype=float),
        wind_usa=np.array([number(r,'USA_WIND',0,250) for r in records],dtype=float),
        wind_jma=np.array([number(r,'TOKYO_WIND',0,250) for r in records],dtype=float),
        rmw=np.array([radius_km(r,'USA_RMW',rmw=True) for r in records],dtype=float))
    for t in THRESHOLDS:
        for q in QUADRANTS:
            values[f'r{t}_{q}'] = np.array([radius_km(r,f'USA_R{t}_{q}') for r in records],dtype=float)
    return values


def load_forecast(folder, rows, contract, csv_hash):
    receipt_path = folder/'verification.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['checkpoint_sha256'] != CHECKPOINT or receipt['members'] != 50:
        raise ValueError('Requires verified released 1.2, actual 50 members')
    for filename, expected in receipt['files_sha256'].items():
        if Path(filename).name != filename or sha(folder/filename) != expected:
            raise ValueError('Immutable forecast file hash mismatch: '+filename)
    candidates = list(folder.glob('*_video.json'))
    if len(candidates) != 1:
        raise ValueError('Expected one exact forecast issue per folder')
    metadata = json.loads(candidates[0].read_text())
    if metadata['observed_csv_sha256'] != csv_hash:
        raise ValueError('IBTrACS file differs from the forecast receipt')
    sid, issue_text = metadata['storm_id'],metadata['issue_time_utc']
    issue = datetime.fromisoformat(issue_text.replace('Z','+00:00'))
    current = rows[sid].get(issue.strftime('%Y-%m-%d %H:%M:%S'))
    if current is None:
        raise ValueError('No exact issue-time initialization report')
    with np.load(folder/'ensemble-members.npz',allow_pickle=False) as z:
        members={k:z[k] for k in z.files}
    with np.load(folder/'weather-history.npz',allow_pickle=False) as z:
        weather,history_times=z['physical_weather'],z['time_ns']
    wanted=np.datetime64(issue_text.replace('Z',''),'ns').astype('int64')+np.arange(-8,1)*6*3600*10**9
    if weather.shape!=(9,8,25,33) or not np.array_equal(history_times,wanted):
        raise ValueError('Nine exact causal analyses required')
    if members['center'].shape!=(50,20,2) or members['basin'].shape!=(50,20,25,33):
        raise ValueError('Wrong actual physical member shapes')
    if members['pressure'].shape!=(50,20) or len(set(members['seeds'].tolist()))!=50:
        raise ValueError('Twenty outputs from fifty distinct seeds required')
    if len(set(metadata['ensemble_policy']['input_sha256']))!=50:
        raise ValueError('Repeated input members are not an ensemble')
    if (len(set(metadata['ensemble_policy']['route_sha256']))!=50
            or len(set(metadata['ensemble_policy']['basin_field_sha256']))!=50):
        raise ValueError('Repeated routes/fields are not fifty physical members')
    lat,lon=contract['global_lat'],contract['global_lon']
    origin=np.array([float(current['LAT']),float(current['LON'])])
    issue_pressure=number(current,'TOKYO_PRES',800,1100)
    recorded=metadata['source_inputs']['issue_observation']
    if recorded['pressure_hpa']!=issue_pressure or not np.allclose([recorded['lat'],recorded['lon']],origin,atol=1e-5,rtol=0):
        raise ValueError('Changed issue-time model inputs')
    with np.load(candidates[0].with_suffix('.npz'),allow_pickle=False) as video:
        if not np.allclose(video['forecast_lat_lon'][0],origin,atol=1e-5,rtol=0):
            raise ValueError('Saved forecast +0 does not match the observed issue origin')
        for member_key,video_key in (('pressure','central_pressure_hpa'),
                                     ('basin','basin_pressure_hpa'),
                                     ('regional','regional_pressure_hpa')):
            actual=members[member_key].mean(0,dtype=np.float64).astype('float32')
            if not np.array_equal(actual,video[video_key]):
                raise ValueError('Saved physical fifty-member mean mismatch: '+video_key)
        if not np.array_equal(members['center'].mean(0,dtype=np.float64).astype('float32'),video['forecast_lat_lon'][1:]):
            raise ValueError('Saved member-mean route mismatch')
    initial_ambient=ambient_pressure(weather[-1,0],lat,lon,origin)
    initial=initialize(current,issue_text,initial_ambient['value_hpa'])
    points,raw=[],[]
    for index in range(20):
        diagnostics=[]
        for member in range(50):
            ambient=ambient_pressure(members['basin'][member,index],lat,lon,members['center'][member,index])
            diagnostics.append(diagnose(initial,float(members['pressure'][member,index]),ambient,
                                       track_valid=bool(members['track_valid'][member,index])))
        summary=summarize(diagnostics,expected_members=50)
        points.append(dict(lead_hours=(index+1)*6,valid_time_utc=utc(issue+timedelta(hours=(index+1)*6)),summary=summary))
        raw.append(diagnostics)
    times=[issue+timedelta(hours=6*i) for i in range(21)]
    obs=observation_series(rows[sid],times)
    case=dict(slug=candidates[0].stem.removesuffix('_video'), name=metadata['storm'],
              sid=sid,issue=issue_text,x=np.arange(0,121,6),obs=obs,
              source_agencies=sorted({r['USA_AGENCY'].strip() or 'missing' for r in rows[sid].values()}),
              model_pressure=np.r_[issue_pressure,members['pressure'].mean(0,dtype=np.float64)],
              auxiliary_wind=np.r_[np.nan,members['vmax'].mean(0,dtype=np.float64)],
              initial=initial,points=points,model={},bands={},metadata=metadata,
              source_folder=str(folder))
    for key in METRICS:
        anchor=initial['input_wind_kt'] if key=='wind_estimate_kt' else (
            initial['rmw_persistence_km'] if key=='rmw_persistence_km' else None)
        # Do NOT hide that estimated absent isotachs at +0 are parametric. Native
        # issue reports appear in the solid official curves, not model anchors.
        case['model'][key]=np.array([anchor]+[p['summary']['estimates'][key]['mean'] for p in points],dtype=float)
        case['bands'][key]=np.array([[np.nan]+[p['summary']['estimates'][key][q] for p in points] for q in ('p10','p90')],dtype=float)
    digest={filename:sha(folder/filename) for filename in receipt['files_sha256']}
    report=dict(model='frozen Trackformer 1.2 pressure + separate experimental structure candidate',
                method=VERSION,storm_id=sid,issue_time_utc=issue_text,actual_members=50,
                checkpoint_sha256=CHECKPOINT,source_forecast_folder=str(folder),
                source_file_sha256=digest,observed_csv_sha256=csv_hash,
                future_labels_used_for_diagnostics=False,initial=initial,initial_ambient=initial_ambient,
                definitions=DEFINITIONS,points=points,member_diagnostics_by_lead=raw,
                official_wind_or_radius_skill_score=None,
                physical_50_member_means_verified=True, observed_plus_zero_alignment_verified=True,
                coverage={key:sum(p['summary']['estimates'][key]['mean'] is not None for p in points) for key in METRICS},
                warning='Retrospective development examples, not untouched tests. RMW is persistence; no validated wind-period calibration.')
    return case,report


def plot_environment(output):
    os.environ.setdefault('MPLCONFIGDIR',str(output/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.facecolor':'#fbfcfe',
        'grid.color':'#d9e1ea','axes.edgecolor':'#bbc8d5','savefig.facecolor':'white'})
    return plt


def line(ax,x,y,label,color,*,style='-',band=None):
    ax.plot(x,y,linestyle=style,color=color,lw=2,label=label)
    if band is not None:
        ax.fill_between(x,band[0],band[1],color=color,alpha=.12,lw=0)


def decorate(ax,x,ylabel):
    ax.set(xlabel='Hours after initialization' if max(x)<=120 else 'Hours from first report',
           ylabel=ylabel,xlim=(min(x),max(x) if max(x)>min(x) else min(x)+1))
    ax.grid(alpha=.55)
    ax.legend(fontsize=8,loc='best')
    if not any(np.isfinite(line.get_ydata()).any() for line in ax.lines):
        ax.set_yticks([])
        ax.set_ylim(0,1)
        legend=ax.get_legend()
        if legend is not None: legend.remove()
        ax.text(.5,.5,'No reported/supported data — not zero',ha='center',transform=ax.transAxes,color='#697c8e')


def draw_case(case,output,plt,*,observed_only=False):
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    x,obs=case['x'],case['obs']
    for ax in axes.ravel(): ax.set_axisbelow(True)
    line(axes[0,0],x,obs['pressure'],'JMA pressure',BLACK)
    line(axes[0,1],x,obs['wind_usa'],'USA wind · 1 min',BLACK)
    line(axes[0,1],x,obs['wind_jma'],'JMA wind · 10 min','#9aa7b5',style=':')
    line(axes[1,0],x,obs['rmw'],'Reported USA RMW',BLACK)
    if not observed_only:
        line(axes[0,0],x,case['model_pressure'],'1.2 · actual mean of 50',BLUE)
        line(axes[0,1],x,case['model']['wind_estimate_kt'],'Pressure-driven candidate · 50',RED,
             style='--',band=case['bands']['wind_estimate_kt'])
        line(axes[0,1],x,case['auxiliary_wind'],'Old auxiliary wind · uncalibrated','#3a9071',style=':')
        line(axes[1,0],x,case['model']['rmw_persistence_km'],'RMW persistence ONLY',RED,style='--')
    for t,ax in zip(THRESHOLDS,(axes[1,1],axes[2,0],axes[2,1])):
        for q in QUADRANTS:
            line(ax,x,obs[f'r{t}_{q}'],q+' reported',QC[q])
            if not observed_only:
                key=f'r{t}_{q}_estimate_km'
                line(ax,x,case['model'][key],q+' candidate',QC[q],style='--')
        ax.set_title(f'R{t}: quadrant maximum extent · radius, not diameter',fontsize=11)
        decorate(ax,x,'Radius (km)')
    for ax,title,ylabel in ((axes[0,0],'Central pressure','hPa'),
                            (axes[0,1],'Maximum sustained wind · periods kept separate','knots'),
                            (axes[1,0],'Reported USA RMW' if observed_only else 'RMW · future candidate is persistence','Radius (km)')):
        ax.set_title(title,fontsize=11)
        decorate(ax,x,ylabel)
    kind='Reported observations only' if observed_only else 'Experimental pressure-driven structure — not a validated wind forecast'
    fig.suptitle(f"{case['name']}  |  {case['issue'][:10]} UTC\n{kind}",fontsize=18,fontweight='bold',y=.995)
    source='Sources: '+', '.join(case['source_agencies'])
    footer=source+' · blank reports stay missing; native USA quadrants converted nm → km.'
    if not observed_only:
        footer+='\nSolid = reported reference; dashed = candidate. RMW is explicitly held constant. Wind/radius skill scores disabled.'
    fig.text(.06,.012,footer,fontsize=9,color='#52677c')
    fig.tight_layout(rect=(.01,.06,.99,.92),h_pad=2.1)
    filename=case['slug']+'_wind_radius.png'
    fig.savefig(output/filename,dpi=160)
    plt.close(fig)
    return filename


def draw_overview(cases,output,plt):
    cols=min(2,len(cases));rows=(len(cases)+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(13,4.5*rows),squeeze=False)
    for ax,case in zip(axes.ravel(),cases):
        x=case['x']
        line(ax,x,case['obs']['wind_usa'],'USA reference · 1 min',BLACK)
        line(ax,x,case['obs']['wind_jma'],'JMA reference · 10 min','#9aa7b5',style=':')
        line(ax,x,case['model']['wind_estimate_kt'],'Pressure-driven candidate · 50',RED,
             style='--',band=case['bands']['wind_estimate_kt'])
        line(ax,x,case['auxiliary_wind'],'Old auxiliary · uncalibrated','#3a9071',style=':')
        ax.set_title(case['name']+' · '+case['issue'][:10],loc='left',fontweight='bold')
        decorate(ax,x,'Wind (knots)')
    for ax in axes.ravel()[len(cases):]: ax.set_visible(False)
    fig.suptitle('All four storms draw wind curves',fontsize=21,fontweight='bold',y=.99)
    fig.text(.5,.94,'Separate experimental pressure-driven estimate; actual 50-member diagnostics. No 1-min calibration or skill claim.',
             ha='center',fontsize=10,color='#52677c')
    fig.text(.06,.012,'Fung-Wong: fresh Mac GPU replay, not the older video forecast. Other three: original saved video members.\n'
             'USA reports may be provisional (Fung-Wong: TCVitals). No future wind/radius observations enter the candidate.',fontsize=9,color='#52677c')
    fig.tight_layout(rect=(.01,.07,.99,.91),h_pad=2)
    fig.savefig(output/'four_storm_wind_candidate.png',dpi=160)
    plt.close(fig)


def observed_case(sid,rows):
    if not rows:
        raise ValueError('No IBTrACS reports for '+sid)
    stamps=sorted(rows)
    times=[datetime.strptime(t,'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc) for t in stamps]
    name=rows[stamps[0]]['NAME'].strip() or sid
    return dict(slug=sid+'_observed',name=name,sid=sid,issue=utc(times[0]),
        x=np.array([(t-times[0]).total_seconds()/3600 for t in times]),
        obs=observation_series(rows,times),
        source_agencies=sorted({r['USA_AGENCY'].strip() or 'missing' for r in rows.values()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ibtracs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--forecast-folder',type=Path,action='append',default=[])
    parser.add_argument('--observed-storm',action='append',default=[],help='Any exact WP IBTrACS SID; no inference')
    parser.add_argument('--catalogue-only',action='store_true')
    args=parser.parse_args()
    if not str(args.output.resolve()).startswith('/Volumes/D/'):
        raise ValueError('New artifacts/cache must stay on /Volumes/D')
    args.output.mkdir(parents=True,exist_ok=True)
    selected=set(args.observed_storm)
    for folder in args.forecast_folder:
        paths=list(folder.glob('*_video.json'))
        if len(paths)!=1: raise ValueError('Expected one forecast metadata file')
        selected.add(json.loads(paths[0].read_text())['storm_id'])
    catalogue,rows=read_catalogue(args.ibtracs,selected)
    write_json(args.output/'storm_catalogue.json',dict(scope='IBTrACS Western Pacific, 1970 onward',
        observations_only_not_forecast_coverage=True,storms=catalogue))
    csv_hash=sha(args.ibtracs)
    if args.catalogue_only:
        print(json.dumps(dict(storms=len(catalogue),output=str(args.output/'storm_catalogue.json'))))
        return
    contract=json.loads((ROOT/'models/trackformer_1_2_field/manifest.json').read_text())['data_contract']
    manifest=json.loads((ROOT/'models/trackformer_1_2_field/manifest.json').read_text())
    for filename,expected in manifest['source_module_sha256'].items():
        if sha(ROOT/'models/trackformer_1_2_field'/filename)!=expected:
            raise ValueError('Frozen neural model source changed: '+filename)
    plt=plot_environment(args.output)
    cases,images,reports=[],[],[]
    for folder in args.forecast_folder:
        case,report=load_forecast(folder,rows,contract,csv_hash)
        filename=draw_case(case,args.output,plt)
        report['chart']=filename
        write_json(args.output/(case['slug']+'_structure.json'),report)
        cases.append(case);images.append((case['name'],filename));reports.append(report)
        print(json.dumps(dict(storm=case['name'],wind_leads=report['coverage']['wind_estimate_kt'],
                              r34_quadrant_leads={q:report['coverage'][f'r34_{q}_estimate_km'] for q in QUADRANTS})),flush=True)
    for sid in args.observed_storm:
        case=observed_case(sid,rows[sid])
        filename=draw_case(case,args.output,plt,observed_only=True)
        images.append((case['name']+' · observations only',filename))
    if cases:
        draw_overview(cases,args.output,plt)
        images.insert(0,('Wind overview','four_storm_wind_candidate.png'))
    panels=''.join(f'<section><h2>{html.escape(name)}</h2><a href="{html.escape(filename)}"><img src="{html.escape(filename)}" alt="{html.escape(name)} plots"></a></section>' for name,filename in images)
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Storm wind and radius comparisons</title><style>body{font:16px system-ui;background:#edf2f7;color:#203247;max-width:1280px;margin:40px auto;padding:0 20px}section{background:white;margin:24px 0;padding:18px;border-radius:16px}img{width:100%;height:auto}a{color:#2166b0}p{line-height:1.6}</style>
    <h1>Storm wind &amp; radius plots</h1><p>Solid curves are agency reports. Dashed curves are an experimental pressure-driven candidate, not validated wind forecasts. RMW is persistence only. Blank reports remain missing; reported zero is not a missing value.</p>
    <p>Actual 50-member diagnostics, never a diagnostic of a mean pressure map. Released model weights and original videos are unchanged. Fung-Wong is a separately saved fresh replay. USA quadrants are 1-minute targets, JMA winds are 10-minute references; no cross-period conversion or strict radius skill score.</p>'''
    page+=panels+'<p><a href="storm_catalogue.json">All WP storms: data coverage catalogue</a> · <a href="verification.json">Verification receipt</a></p>'
    (args.output/'index.html').write_text(page)
    receipt=dict(method=VERSION,checkpoint_sha256=CHECKPOINT,storm_count=len(cases),
                 observed_only_storms=args.observed_storm,wp_catalogue_storms=len(catalogue),
                 observed_csv_sha256=csv_hash,actual_member_count=50 if cases else None,
                 future_wind_radius_labels_used_for_candidate=False,
                 neural_weights_changed=False,original_forecasts_overwritten=False,
                 official_wind_radius_scoring_allowed=False,references=list(REFERENCES),
                 charts=[filename for _,filename in images],
                 coverage={r['storm_id']:r['coverage'] for r in reports})
    receipt['source_code_sha256']={p.name:sha(p) for p in (Path(__file__),ROOT/'release_tools/storm_structure_diagnostics.py')}
    receipt['output_sha256']={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='verification.json'}
    write_json(args.output/'verification.json',receipt)
    print(json.dumps(dict(output=str(args.output/'index.html'),catalogue_storms=len(catalogue))),flush=True)


if __name__=='__main__':
    main()
