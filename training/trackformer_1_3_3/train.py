"""CUDA direct-forecast pilot; all20 leads, exact shared-encoder gradients."""
from pathlib import Path
import argparse,fcntl,hashlib,json,math,os,random,time,traceback
import numpy as np
import torch
from contracts import LATITUDE,LONGITUDE,LEAD_HOURS,sha256
from data import Data
from model import TrackPressure,loss,FEATURE_KEYS
from prepare_colab import save

SOURCE_FILES=('contracts.py','source_contract.py','arco_ranges.py','prepare_colab.py','data.py','model.py','train.py','catalog.json.gz')


def identity():return {f:sha256(Path(__file__).parent/f) for f in SOURCE_FILES}


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'].cpu())
    if r['cuda']:torch.cuda.set_rng_state_all([s.cpu() for s in r['cuda']])


def save_checkpoint(path,model,opt,config,cursor):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng(),config=config,cursor=cursor),f)
        f.flush();os.fsync(f.fileno())
    digest=sha256(tmp);os.replace(tmp,path)
    save(path.with_suffix('.receipt.json'),dict(path=path.name,sha256=digest,cursor=cursor,code=config['code']))
    return digest


def load_checkpoint(path,model,opt,config):
    path=Path(path);receipt=json.loads(path.with_suffix('.receipt.json').read_text())
    if sha256(path)!=receipt['sha256']:raise ValueError('Checkpoint/receipt hash mismatch')
    # Only locally created, hash-verified trusted checkpoints are accepted.
    s=torch.load(path,map_location='cpu',weights_only=False)
    if s['config']!=config:raise ValueError('Resume configuration/source/data differs; use a new output directory')
    model.load_state_dict(s['model'],strict=True);opt.load_state_dict(s['optimizer']);restore_rng(s['rng'])
    return s['cursor']


def update(model,opt,data,records,horizon,device):
    model.train();opt.zero_grad(set_to_none=True);total=0.;parts={}
    for record in records:
        packet=data.history(record,device);encoded=model.initial(**packet)
        # Accumulate decoder gradients at feature leaves, then backpropagate
        # their sum through the history encoder once. Exact chain rule; no
        # truncated history/lead gradients and no20-lead activation retention.
        original=[encoded[key] for key in FEATURE_KEYS]
        leaves=[value.detach().requires_grad_(True) for value in original]
        state=dict(encoded,**dict(zip(FEATURE_KEYS,leaves)))
        for k in range(horizon):
            out=model.forecast(state,k)
            # Future observations are accessed only after producing this lead.
            target=data.label(record,k,device)
            value,detail=loss(model,out,**target,lat=packet['lat'],lon=packet['lon'])
            if not torch.isfinite(value):raise FloatingPointError('Non-finite objective; last committed checkpoint preserved')
            normalized=value/(len(records)*horizon)
            total+=float(normalized.detach())
            for key,v in detail.items():parts[key]=parts.get(key,0.)+float(v.detach())/(len(records)*horizon)
            normalized.backward()
        torch.autograd.backward(original,[value.grad if value.grad is not None else torch.zeros_like(value) for value in leaves])
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    opt.step()
    return dict(loss=total,gradient_norm=float(norm),**parts)


def distance(a,b):
    a,b=np.deg2rad(a),np.deg2rad(b)
    q=np.sin((a[0]-b[0])/2)**2+np.cos(a[0])*np.cos(b[0])*np.sin((a[1]-b[1])/2)**2
    return float(2*6371.0088*np.arcsin(np.sqrt(np.clip(q,0,1))))


def low_alignment(pred,center,lat,lon):
    yy,xx=np.meshgrid(lat,lon,indexing='ij')
    km2=((yy-center[0])*111.2)**2+((xx-center[1])*111.2*np.cos(np.deg2rad(center[0])))**2
    core=(km2<=400**2)&np.isfinite(pred)
    annulus=(km2>=400**2)&(km2<=600**2)&np.isfinite(pred)
    if not core.any():return dict(local_min_distance_km=None,local_low_depth_hPa=None)
    iy,ix=np.unravel_index(np.argmin(np.where(core,pred,np.inf)),pred.shape)
    return dict(local_min_distance_km=distance(center,[lat[iy],lon[ix]]),
        local_low_depth_hPa=float(np.mean(pred[annulus])-pred[iy,ix]) if annulus.any() else None)


@torch.inference_mode()
def validate(model,data,output,epoch,checkpoint_sha,device,config):
    dest=output/f'development-E{epoch}';dest.mkdir(exist_ok=True)
    receipt_path=dest/'receipt.json'
    if receipt_path.exists():
        rec=json.loads(receipt_path.read_text())
        if rec['checkpoint_sha256']!=checkpoint_sha or rec['config']!=config:raise ValueError('Validation identity mismatch')
        for name,h in rec['files'].items():
            if sha256(dest/name)!=h:raise ValueError('Changed saved validation')
        return rec
    model.eval();cases=[];files={};route=[];pressure=[];field_abs=field_sq=field_weight=0.;unsupported=0;possible=0
    persistence=[];constant_motion=[]
    alignments=[]
    perlead=[dict(track_km=[],central_pressure_error_hPa=[],field_abs=0.,field_sq=0.,field_weight=0.) for _ in LEAD_HOURS]
    for i,r in enumerate(data.cohort['development']):
        packet=data.history(r,device);state=model.initial(**packet)
        maps=[];centers=[];pressures=[];supported=[];truth=[];masks=[];truth_centers=[];truth_cp=[];trackmask=[];pmask=[]
        cm=dict(id=r['id'],storm_id=r['storm_id'],issue_seconds=r['issue_seconds'],leads=[])
        for k in range(20):
            state,out=model.step(state);y=data.label(r,k,device)
            pred=out['weather'][0,0].cpu().numpy();tr=y['target'][0,0].cpu().numpy();mask=y['valid'][0,0].cpu().numpy()
            cen=out['center'][0].cpu().numpy();cp=float(out['pressure'][0]);ok=bool(out['track_valid'][0])
            if not np.isfinite(pred).all() or not np.isfinite(cen).all() or not math.isfinite(cp):raise FloatingPointError('Non-finite forecast')
            truth_c=y['center'][0].cpu().numpy();cv=bool(y['center_valid'][0]);pv=bool(y['pressure_valid'][0]);tp=float(y['pressure'][0])
            err=distance(cen,truth_c) if cv and ok else None;pe=cp-tp if pv and ok else None
            if cv:possible+=1;unsupported+=int(not ok)
            if err is not None:route.append(err);perlead[k]['track_km'].append(err)
            if pe is not None:pressure.append(pe);perlead[k]['central_pressure_error_hPa'].append(pe)
            w=mask*np.cos(np.deg2rad(LATITUDE))[:,None];delta=pred-tr
            field_abs+=float((np.abs(delta)*w).sum(dtype='f8'));field_sq+=float((delta**2*w).sum(dtype='f8'));field_weight+=float(w.sum(dtype='f8'))
            perlead[k]['field_abs']+=float((np.abs(delta)*w).sum(dtype='f8'))
            perlead[k]['field_sq']+=float((delta**2*w).sum(dtype='f8'))
            perlead[k]['field_weight']+=float(w.sum(dtype='f8'))
            alignment=low_alignment(pred,cen,LATITUDE,LONGITUDE) if ok else dict(local_min_distance_km=None,local_low_depth_hPa=None)
            alignments.append(alignment)
            base=r['initial'];east,north=r['motion'];hours=(k+1)*6
            fixed=[base[0]+north*(k+1)/111.2,base[1]+east*(k+1)/(111.2*math.cos(math.radians(base[0])))]
            persist_error=distance(base[:2],truth_c) if cv else None
            motion_error=distance(fixed,truth_c) if cv else None
            if cv:persistence.append(persist_error);constant_motion.append(motion_error)
            cm['leads'].append(dict(lead_hours=hours,track_error_km=err,central_pressure_error_hPa=pe,track_supported=ok,
                truth_track_valid=cv,truth_pressure_valid=pv,persistence_km=persist_error,constant_motion_km=motion_error,**alignment))
            maps.append(pred);centers.append(cen);pressures.append(cp);supported.append(ok);truth.append(tr);masks.append(mask);truth_centers.append(truth_c);truth_cp.append(tp);trackmask.append(cv);pmask.append(pv)
        path=dest/f'case-{i:03d}.npz';tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,pressure_hPa=np.array(maps),track_lat_lon=np.array(centers),central_pressure_hPa=np.array(pressures,dtype='f4'),
                track_valid=np.array(supported),truth_pressure_hPa=np.array(truth),truth_valid=np.array(masks),truth_track_lat_lon=np.array(truth_centers),
                truth_central_pressure_hPa=np.array(truth_cp),truth_track_valid=np.array(trackmask),truth_central_pressure_valid=np.array(pmask),
                latitude=LATITUDE,longitude=LONGITUDE,lead_hours=np.array(LEAD_HOURS),issue_seconds=r['issue_seconds'],case_id=r['id'])
        os.replace(tmp,path);files[path.name]=sha256(path);cases.append(cm)
        print(json.dumps(dict(event='development_case_complete',epoch=epoch,case=i+1,total=len(data.cohort['development']))),flush=True)
    mean=lambda v:float(np.mean(v)) if v else None
    mae=lambda v:mean(np.abs(v).tolist())
    rmse=lambda v:float(np.sqrt(np.mean(np.square(v)))) if v else None
    metrics=dict(track_mean_km=mean(route),track_scored=len(route),track_label_count=possible,unsupported_track_outputs=unsupported,
        central_pressure_MAE_hPa=mae(pressure),central_pressure_RMSE_hPa=rmse(pressure),central_pressure_count=len(pressure),
        field_MAE_hPa=field_abs/field_weight,field_RMSE_hPa=(field_sq/field_weight)**.5,field_area_weight=field_weight,
        per_lead=[dict(lead_hours=(i+1)*6,track_mean_km=mean(p['track_km']),track_count=len(p['track_km']),
            central_pressure_MAE_hPa=mae(p['central_pressure_error_hPa']),pressure_count=len(p['central_pressure_error_hPa']),
            field_MAE_hPa=p['field_abs']/p['field_weight'],field_RMSE_hPa=(p['field_sq']/p['field_weight'])**.5) for i,p in enumerate(perlead)],
        center_to_local_mslp_min_mean_km=mean([v['local_min_distance_km'] for v in alignments if v['local_min_distance_km'] is not None]),
        local_low_depth_mean_hPa=mean([v['local_low_depth_hPa'] for v in alignments if v['local_low_depth_hPa'] is not None]),
        alignment_definition='minimum within400km of predicted center; depth versus400..600km annulus; diagnostic, not cyclone detection',
        scope='24 reused development starts, one per storm; all20 leads; not full811 or untouched holdout',
        pressure_map_support='native241x321 WP, original finite MSLP masks, cosine-latitude weights; includes core in map error',
        persistence_track_km=mean(persistence),constant_motion_track_km=mean(constant_motion),baseline_track_count=len(persistence),
        baseline_comparison_completed=True,released_1_2_comparison=False,promotion=False)
    save(dest/'metrics.json',metrics);save(dest/'cases.json',cases)
    files.update({n:sha256(dest/n) for n in ('metrics.json','cases.json')})
    receipt=dict(complete=True,epoch=epoch,checkpoint_sha256=checkpoint_sha,config=config,cases=len(cases),leads=list(LEAD_HOURS),files=files)
    save(receipt_path,receipt);return receipt


def run(a):
    if not torch.cuda.is_available():raise RuntimeError('Select a GPU runtime in Colab: Runtime > Change runtime type > T4 GPU')
    device=torch.device('cuda');seed(a.seed);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data=Data(a.data);norm=data.normalizer(output/'normalizer.json')
    model=TrackPressure(norm['mean'],norm['scale'],a.width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    config=dict(experiment='trackformer-1.3.3-direct-context-20261103',code=identity(),data=data.identity,
        width=a.width,epochs=a.epochs,lr=a.lr,effective_batch=a.batch,seed=a.seed,
        normalizer_sha256=sha256(output/'normalizer.json'),precision='FP32; TF32 disabled',
        horizon_curriculum=[20]*a.epochs,forecast_mode='direct lead conditioned; fixed causal history',
        pressure_target='dense ERA5 including core; JMA central pressure diagnostic only',
        gradient_method='exact feature-leaf accumulation; one encoder backward per sample',
        initialization='fresh random weights; no old1.3 or1.2 checkpoint',torch_version=torch.__version__)
    hardware=dict(device=torch.cuda.get_device_name(),total_memory_GiB=torch.cuda.get_device_properties(0).total_memory/2**30,
        parameters=sum(p.numel() for p in model.parameters()),cuda=torch.version.cuda,config=config)
    if a.preflight:
        began=time.monotonic();torch.cuda.reset_peak_memory_stats()
        print(json.dumps(dict(event='cuda_preflight_start',device=hardware['device'],horizon=20,disposable=True)),flush=True)
        # Full native grid, longest backward segment; disposable optimizer.
        result=update(model,opt,data,[data.cohort['train'][0]],20,device)
        torch.cuda.synchronize()
        hardware.update(passed=True,seconds=time.monotonic()-began,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            result=result,disposable=True,production_updates=0,horizon=20)
        save(output/'cuda-preflight.json',hardware);print(json.dumps(hardware,indent=2));return
    # Direct production start. The optional disposable preflight is not a gate.
    # CPU direct-lead/gradient/causality/resume fixtures are checked before publication.
    save(output/'config.json',config);save(output/'hardware.json',hardware)
    print(json.dumps(dict(event='production_start',**hardware)),flush=True)
    latest=output/'latest.pt';cursor=dict(epoch=0,index=0,updates=0)
    if latest.exists():cursor=load_checkpoint(latest,model,opt,config)
    else:save_checkpoint(latest,model,opt,config,cursor)
    rows=data.cohort['train'];planned=a.epochs*math.ceil(len(rows)/a.batch);durations=[]
    start=time.monotonic()
    try:
        for epoch in range(cursor['epoch'],a.epochs):
            order=list(rows);random.Random(a.seed+epoch).shuffle(order)
            offset=cursor['index'] if epoch==cursor['epoch'] else 0
            for i in range(offset,len(order),a.batch):
                began=time.monotonic();result=update(model,opt,data,order[i:i+a.batch],20,device)
                cursor=dict(epoch=epoch,index=min(i+a.batch,len(order)),updates=cursor['updates']+1)
                checkpoint_sha=save_checkpoint(latest,model,opt,config,cursor)
                durations.append(time.monotonic()-began);rate=sum(durations[-8:])/len(durations[-8:])
                remaining_this=math.ceil((len(rows)-cursor['index'])/a.batch)
                scaled_work=remaining_this+sum(math.ceil(len(rows)/a.batch)*(h+9)/(config['horizon_curriculum'][epoch]+9)
                    for h in config['horizon_curriculum'][epoch+1:])
                status=dict(phase='training',epoch=epoch+1,cursor=cursor,planned_updates=planned,percent=100*cursor['updates']/planned,
                    seconds_per_update=rate,remaining_training_minutes=(planned-cursor['updates'])*rate/60,
                    curriculum_scaled_training_minutes=scaled_work*rate/60,ETA_is_estimate=True,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    validation_time_excluded=True,checkpoint_sha256=checkpoint_sha,**result)
                save(output/'status.json',status);print(json.dumps(status),flush=True)
                with (output/'training.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            # Immutable epoch snapshot and all20-lead development evaluation.
            epoch_path=output/f'epoch-{epoch+1:02d}.pt'
            if not epoch_path.exists():save_checkpoint(epoch_path,model,opt,config,cursor)
            rec=json.loads(epoch_path.with_suffix('.receipt.json').read_text())
            if sha256(epoch_path)!=rec['sha256'] or rec['cursor']!=cursor:raise ValueError('Epoch snapshot identity mismatch')
            save(output/'status.json',dict(phase='validation',epoch=epoch+1,cursor=cursor))
            validate(model,data,output,epoch+1,rec['sha256'],device,config)
            cursor=dict(epoch=epoch+1,index=0,updates=cursor['updates']);save_checkpoint(latest,model,opt,config,cursor)
        save(output/'receipt.json',dict(complete=True,config=config,cursor=cursor,elapsed_this_session_seconds=time.monotonic()-start,
            final_checkpoint_sha256=sha256(latest),development_receipts={f'development-E{e}/receipt.json':sha256(output/f'development-E{e}/receipt.json') for e in range(1,a.epochs+1)},
            released_1_2_comparison=False,full811=False,untouched_holdout=False))
        save(output/'status.json',dict(phase='complete',cursor=cursor,percent=100))
    except KeyboardInterrupt:
        save(output/'interruption.json',dict(reason='user interrupt',last_committed=json.loads(latest.with_suffix('.receipt.json').read_text()),
            resume='Rerun exactly the same training cell; unfinished optimizer work will be replayed'))
        raise
    except Exception as e:
        save(output/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),last_committed=json.loads(latest.with_suffix('.receipt.json').read_text())))
        raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--width',type=int,default=64);p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=1332026)
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if not 1<=a.epochs<=20 or not 1<=a.batch<=64:raise ValueError('Invalid explicit training configuration')
    run(a)
