"""Reuse the completed 1.3.1 frame cache; no weather download or global audit."""
from pathlib import Path
import argparse,concurrent.futures,gzip,hashlib,json,os,shutil
from contracts import sha256


def catalog():return json.loads(gzip.decompress((Path(__file__).parent/'catalog.json.gz').read_bytes()))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def times(r):return {r['issue_seconds']+h*3600 for h in range(-48,121,6)}


def select_cached(d,old):
    available=set().union(*(times(r) for rows in old.values() for r in rows))
    dev=old['development'];dev_times=set().union(*(times(r) for r in dev))
    dev_storms={r['storm_id'] for r in dev}
    train=[r for r in d['records'] if r['split']=='train' and r['storm_id'] not in dev_storms
           and times(r)<=available and not times(r)&dev_times]
    train.sort(key=lambda r:hashlib.sha256(('132:'+r['id']).encode()).hexdigest())
    return dict(train=train,development=dev)


def prepare(source,dest,old_local=None):
    source,dest=Path(source),Path(dest);dest.mkdir(parents=True,exist_ok=True)
    ready=json.loads((source/'ready.json').read_text());old=json.loads((source/'cohort.json').read_text())
    if ready['catalog_sha256']!=sha256(Path(__file__).parent/'catalog.json.gz') or ready['cohort_sha256']!=sha256(source/'cohort.json'):
        raise ValueError('Existing cache cohort identity mismatch; original files left unchanged')
    d=catalog();cohort=select_cached(d,old)
    if len(cohort['train'])!=186 or len(cohort['development'])!=24:
        raise ValueError('Expected the completed64-start cache and its fixed24 development starts')
    if len({r['storm_id'] for r in cohort['train']})!=84:raise ValueError('Unexpected TRAIN storm set')
    required=sorted(set().union(*(times(r) for rows in cohort.values() for r in rows)))
    (dest/'frames').mkdir(exist_ok=True)
    names=[f'frames/{t}{suffix}' for t in required for suffix in ('.npz','.json')]
    total=0;copied=0
    # Reuse an existing VM staging directory where possible. Source receipts and
    # weather values are checked on first actual training read, not in a new audit.
    def stage(name):
        src=source/name;dst=dest/name;size=src.stat().st_size
        if old_local:
            local=Path(old_local)/name
            if local.exists() and local.stat().st_size==size:src=local
        if dst.exists() and dst.stat().st_size==size:return size,False
        tmp=dst.with_suffix(dst.suffix+'.tmp');shutil.copy2(src,tmp);os.replace(tmp,dst)
        return size,True
    needed=sum((source/n).stat().st_size for n in names if not (dest/n).exists())
    if shutil.disk_usage(dest).free<needed+1024**3:raise RuntimeError('Insufficient VM space for cached frames; no source files removed')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,(size,did_copy) in enumerate(pool.map(stage,names)):
            total+=size;copied+=int(did_copy)
            if (i+1)%200==0:print(json.dumps(dict(event='stage_existing_cache',files=i+1,total=len(names),copied=copied)),flush=True)
    save(dest/'cohort.json',cohort)
    result=dict(catalog_sha256=ready['catalog_sha256'],cohort_sha256=sha256(dest/'cohort.json'),train=186,
        training_storms=84,development=24,native_shape=[31,241,321],future_weather_as_inputs=False,all20_leads=True,
        disjoint_train_development_calendar_windows=True,source_cache_cohort_sha256=ready['cohort_sha256'],
        weather_downloaded=False,unique_frames=len(required),bytes_staged=total)
    save(dest/'ready.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--data',required=True);p.add_argument('--old-local')
    a=p.parse_args();prepare(a.source,a.data,a.old_local)
