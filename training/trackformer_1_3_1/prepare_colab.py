"""Resumable public-source preparation; explicit command, never on import."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,shutil,time
import numpy as np
from contracts import CHANNELS,LATITUDE,LONGITUDE,sha256


def catalog():return json.loads(gzip.decompress((Path(__file__).parent/'catalog.json.gz').read_bytes()))


def select(n=64):
    d=catalog();groups={s:[r for r in d['records'] if r['split']==s] for s in ('train','development')}
    if not 2<=n<=len(groups['train']):raise ValueError('TRAIN starts must be2..11978')
    ranked=sorted(groups['train'],key=lambda r:hashlib.sha256(('131:'+r['id']).encode()).hexdigest())
    first=[];rest=[];seen=set()
    for r in ranked:
        if r['storm_id'] not in seen:first.append(r);seen.add(r['storm_id'])
        else:rest.append(r)
    train=(first+rest)[:n]
    dev=[]
    for sid in sorted({r['storm_id'] for r in groups['development']}):
        dev.append(min((r for r in groups['development'] if r['storm_id']==sid),key=lambda r:r['issue_seconds']))
    return d,dict(train=train,development=dev)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def cached(root,t,entry):
    p=root/'frames'/f'{t}.npz';r=p.with_suffix('.json')
    if not p.exists() or not r.exists():return False
    rec=json.loads(r.read_text())
    if sha256(p)!=rec['sha256'] or rec['values_sha256']!=entry['values_sha256']:
        raise ValueError('Changed cached frame: '+str(p))
    return True


def prepare(root,n,download=False,limit=None):
    d,cohort=select(n);root=Path(root);root.mkdir(parents=True,exist_ok=True)
    times=sorted({r['issue_seconds']+h*3600 for rows in cohort.values() for r in rows for h in range(-48,121,6)})
    missing=[t for t in times if not cached(root,t,d['weather'][str(t)])]
    estimate=dict(train_starts=n,development_starts=len(cohort['development']),unique_frames=len(times),
        missing_frames=len(missing),cached_native_GiB_estimate=sum(d['weather'][str(t)]['source_npz_bytes'] for t in times)/2**30,
        network_bytes_unknown=True,free_GiB=shutil.disk_usage(root).free/2**30)
    print(json.dumps(estimate,indent=2),flush=True)
    if not download:return estimate
    from arco_ranges import RangeSource
    (root/'frames').mkdir(exist_ok=True)
    selected=missing if limit is None else missing[:limit]
    source=RangeSource(network_workers=8);began=time.monotonic()
    try:
        for i,t in enumerate(selected):
            if shutil.disk_usage(root).free<2*1024**3:raise RuntimeError('Data preparation paused before another write: under2GiB free; existing files preserved')
            values,valid,proof=source.frame(t)
            h=hashlib.sha256(values.tobytes()).hexdigest()
            if h!=d['weather'][str(t)]['values_sha256']:raise ValueError('Upstream bytes differ from frozen local source frame')
            p=root/'frames'/f'{t}.npz';tmp=p.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f,values=values,valid=valid,latitude=LATITUDE,longitude=LONGITUDE,seconds=np.int64(t))
                f.flush();os.fsync(f.fileno())
            os.replace(tmp,p)
            save(p.with_suffix('.json'),dict(seconds=t,sha256=sha256(p),values_sha256=h,source_range_proofs=proof))
            rate=(time.monotonic()-began)/(i+1)
            print(json.dumps(dict(prepared=i+1,selected=len(selected),seconds_per_frame=rate,
                remaining_minutes=(len(missing)-i-1)*rate/60,transferred_bytes=source.http.bytes)),flush=True)
        elapsed=time.monotonic()-began
        save(root/'download-estimate.json',dict(measured_frames=len(selected),seconds=elapsed,
             remaining_minutes=(len(missing)-len(selected))*elapsed/max(1,len(selected))/60,bytes=source.http.bytes))
    finally:source.close()
    if limit is not None:return
    # Complete ready marker requires every hash-verified history and all20 targets.
    if not all(cached(root,t,d['weather'][str(t)]) for t in times):raise RuntimeError('Incomplete data; training is not ready')
    save(root/'cohort.json',cohort)
    save(root/'ready.json',dict(catalog_sha256=sha256(Path(__file__).parent/'catalog.json.gz'),
        cohort_sha256=sha256(root/'cohort.json'),train=n,development=len(cohort['development']),
        native_shape=[31,241,321],future_weather_as_inputs=False,all20_leads=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--train-starts',type=int,default=64)
    p.add_argument('--download',action='store_true');p.add_argument('--limit',type=int)
    a=p.parse_args();prepare(a.data,a.train_starts,a.download,a.limit)
