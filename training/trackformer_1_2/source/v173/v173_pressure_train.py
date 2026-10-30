"""Full-dataset, resumable Metal training for the v168 moving-core prototype."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import socket
import time
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import CoreForecaster, objective, base, VERSION, ARCHITECTURE

ROOT=Path(__file__).resolve().parents[1]
HORIZONS=[4,8,12,20]


class Tee:
    def __init__(self, terminal, log): self.terminal,self.log=terminal,log
    def write(self, value):
        self.terminal.write(value);self.log.write(value);self.log.flush()
    def flush(self): self.terminal.flush();self.log.flush()


def batches(indices, rows, effective, physical):
    for start in range(0,len(indices),effective):
        window=indices[start:start+effective];parts=[]
        for native in (True,False):
            group=[i for i in window if (rows[i]['patch']>=0)==native]
            parts.extend(group[j:j+physical] for j in range(0,len(group),physical))
        yield parts,len(window)


def save(path,model,optimizer,m,epoch,history,best,progress=None):
    base.atomic_checkpoint(path,dict(version=VERSION,architecture=ARCHITECTURE,
        dataset_sha256=m['identity'],model=model.state_dict(),optimizer=optimizer.state_dict(),
        data_contract={k:m[k] for k in ('channels','units','global_lat','global_lon','regional_lat','regional_lon','normalization')},
        epoch=epoch,history=history,best=best,progress=progress,
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state()))


def restore(path,model,optimizer,m):
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p['architecture']!=ARCHITECTURE or p['dataset_sha256']!=m['identity']:
        raise ValueError('Architecture/dataset mismatch')
    model.load_state_dict(p['model'],strict=True);optimizer.load_state_dict(p['optimizer'])
    random.setstate(p['python_rng']);np.random.set_state(p['numpy_rng']);torch.set_rng_state(p['torch_rng'])
    return p


def warm_start(model,path,m):
    if base.digest(path)!='99ed776097e05ad6df03375465d9618741bed63a4ee57724d3ba98e295039d30':
        raise ValueError('Expected the verified v168 epoch 3 baseline checkpoint')
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p['version']!='1.2.68' or p['epoch']!=3 or p['dataset_sha256']!=m['identity']:
        raise ValueError('Warm-start identity mismatch')
    weights=p['model']
    missing,extra=model.load_state_dict(weights,strict=False)
    if extra or any(not k.startswith('context.') for k in missing):
        raise ValueError('Incomplete environmental transfer')


def main():
    p=argparse.ArgumentParser()
    for name,default in [('batch',10),('workers',2),('amp',0)]:p.add_argument(name,type=int,nargs='?',default=default)
    p.add_argument('--microbatch',type=int,default=2)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'mps',choices=['mps','cuda','cpu'])
    here=Path(__file__).resolve().parent
    colab=Path('/content/drive/MyDrive/typhoon_predict')
    p.add_argument('--manifest',type=Path,default=(here if colab.exists() else ROOT)/'data/v164_reuse/manifest.json')
    p.add_argument('--output',type=Path,default=colab/'training/v173' if colab.exists() else here/'runs/mac')
    p.add_argument('--warm-start',type=Path,default=here/'baseline_epoch_003.pt' if colab.exists() else ROOT/'v168/runs/mac/epoch_003.pt')
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    if min(a.batch,a.microbatch,a.epochs)<1 or a.workers<0 or a.amp not in (0,1):p.error('Positive batches/epochs required; amp must be 0 or 1')
    if a.amp: print('AMP request recorded; pressure dynamics use FP32 for numerical stability.',flush=True)
    if a.preflight:
        m=base.load_reuse(a.manifest)
        assert m['counts']==dict(train=13949,validation=1041,test=1195)
        assert m['patch_counts']==dict(train=800,validation=100,test=100)
        model=CoreForecaster(m);warm_start(model,a.warm_start,m)
        print('Verified full dataset and v168 E3 warm start; parameters=',sum(p.numel() for p in model.parameters()))
        return
    torch.set_num_threads(4);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    base.EvolutionNet.block=lambda self,module,x:module(x)
    device=base.device_for(a.device);a.output.mkdir(parents=True,exist_ok=True)
    handle=(a.output/'training.lock').open('a+')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Training already active for this output')
    handle.seek(0);handle.truncate();json.dump(dict(pid=os.getpid(),host=socket.gethostname()),handle);handle.flush()
    log=(a.output/'train.log').open('a',buffering=1)
    sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    m=base.load_reuse(a.manifest)
    assert m['counts']==dict(train=13949,validation=1041,test=1195)
    assert m['patch_counts']==dict(train=800,validation=100,test=100)
    model=CoreForecaster(m).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    last=a.output/'last.pt';history=[];best=float('inf');start=1;progress=None
    if last.exists():
        saved=restore(last,model,optimizer,m)
        history,best,start,progress=saved['history'],saved['best'],saved['epoch']+1,saved['progress']
        if progress and progress['batch']!=a.batch:raise ValueError('Mid-epoch resume needs identical effective batch')
    else:warm_start(model,a.warm_start,m)
    manifest=dict(version=VERSION,architecture=ARCHITECTURE,parameters=sum(x.numel() for x in model.parameters()),
        samples=m['counts'],native_samples=m['patch_counts'],dataset_sha256=m['identity'],device=str(device),
        effective_batch=a.batch,microbatch=a.microbatch,precision='float32',future_weather_as_inputs=False,
        future_best_track_as_inputs=False,official_forecasts_as_inputs=False,
        pressure_map='moving learned core plus resolved environment; observation-conditioned initialization, not raw analysis',
        internal_core_spacing_km=20,initial_core_prior_sigma_km=100,inner_absolute_reanalysis_loss_exclusion_km=150,
        core_structure_status='prototype; unresolved inner structure is not spatially validated',
        pressure_change_supervision=True,motion_change_supervision=True,
        environmental_attention='two spatial pooling scales, two transformer blocks, cross attention; zero initialized correction',
        route='pressure-field association; no independent displayed route or pressure head',
        warm_start=str(a.warm_start),warm_start_sha256=base.digest(a.warm_start),
        model_sha256=base.digest(Path(__file__).with_name('model.py')),trainer_sha256=base.digest(__file__))
    base.atomic_json(a.output/'manifest.json',manifest)
    print(json.dumps(manifest),flush=True)
    for epoch in range(start,a.epochs+1):
        metrics={}
        for split in ('train','validation'):
            training=split=='train';model.train(training);data=base.ReuseDataset(m,split)
            indices=base.training_indices(data.rows,epoch) if training else list(range(len(data)))
            windows=list(batches(indices,data.rows,a.batch,a.microbatch))
            offset=progress['window'] if training and progress else 0
            totals=progress['totals'] if training and progress else [[0.,0],[0.,0]]
            parts=[part for group,_ in windows[offset:] for part in group]
            loader=iter(DataLoader(data,batch_sampler=parts,num_workers=a.workers,**({'prefetch_factor':1} if a.workers else {})))
            began=time.time();steps=HORIZONS[min(epoch-1,3)] if training else 20
            for wi,(group,count) in enumerate(windows[offset:],offset):
                optimizer.zero_grad(set_to_none=True)
                for _ in group:
                    x,y=next(loader);x={k:v.to(device) for k,v in x.items()};size=x['center'].shape[0]
                    with torch.set_grad_enabled(training):
                        state=model.initial(x);block=0;total=0.
                        for lead in range(steps):
                            state,out=model.step(state);target={k:v[:,lead].to(device) for k,v in y.items()}
                            target['previous_pressure']=y['pressure'][:,lead-1].to(device) if lead else x['issue_intensity'][:,1]
                            target['previous_pressure_mask']=y['pressure_mask'][:,lead-1].to(device) if lead else x['issue_mask'][:,1]
                            target['previous_center']=y['center'][:,lead-1].to(device) if lead else x['center']
                            target['previous_storm_mask']=y['storm_mask'][:,lead-1].to(device) if lead else torch.ones_like(target['storm_mask'])
                            loss,terms=objective(out,target,x,m)
                            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss; previous checkpoint retained')
                            total+=float(loss.detach())/steps
                            if training:
                                block=block+loss*size/(count*steps)
                                if (lead+1)%4==0 or lead+1==steps:
                                    block.backward();block=0;state={k:v.detach() for k,v in state.items()}
                    native=int(bool(x['detail_available'][0,0]));totals[native][0]+=total*size;totals[native][1]+=size
                if training:
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite gradient; previous checkpoint retained')
                    optimizer.step()
                    if (wi+1)%50==0 or wi+1==len(windows) or stop[0]:
                        save(last,model,optimizer,m,epoch-1,history,best,dict(window=wi+1,totals=totals,batch=a.batch))
                if wi%5==0 or wi+1==len(windows):
                    print(f'{split} epoch={epoch}/{a.epochs} window={wi+1}/{len(windows)} horizon={steps} loss={total:.5f} pressure={float(terms["pressure"].detach()):.5f} route={float(terms["route"].detach()):.5f} elapsed_s={time.time()-began:.0f}',flush=True)
                if stop[0]:print('Stopped at optimizer boundary; resume from last.pt',flush=True);return
            metrics[split]=sum(v/n for v,n in totals if n)/sum(n>0 for _,n in totals);progress=None
        promote=metrics['validation']<best
        if promote:best=metrics['validation']
        history.append(dict(epoch=epoch,horizon=HORIZONS[min(epoch-1,3)],**metrics))
        for path in [a.output/f'epoch_{epoch:03d}.pt',last]+([a.output/'best.pt'] if promote else []):
            save(path,model,optimizer,m,epoch,history,best)
        base.atomic_json(a.output/'history.json',history)
        print(f'Saved completed epoch {epoch}; best={promote}',flush=True)
    print('Training complete; best selected on fixed 120h validation.',flush=True)


if __name__=='__main__':main()
