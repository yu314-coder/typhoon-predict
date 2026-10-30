"""Real-data Metal preflight, pressure coupling, gradients and resume."""
from pathlib import Path
import tempfile
import torch
from model import CoreForecaster, objective, base
from v173_pressure_train import save, restore, warm_start, batches
from baseline_model import CoreForecaster as Baseline


def main():
    root=Path(__file__).resolve().parents[1]
    torch.set_num_threads(4);torch.manual_seed(168)
    base.EvolutionNet.block=lambda self,module,x:module(x)
    m=base.load_reuse(root/'data/v164_reuse/manifest.json')
    assert m['counts']==dict(train=13949,validation=1041,test=1195)
    assert not any('1979275N06159'==r['storm_id'] for r in m['rows'])
    model=CoreForecaster(m).to('mps');warm_start(model,root/'v168/runs/mac/epoch_003.pt',m)
    reference=Baseline(m).to('mps')
    reference.load_state_dict(torch.load(root/'v168/runs/mac/epoch_003.pt',map_location='cpu',weights_only=False)['model'])
    reference.eval();model.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    data=base.ReuseDataset(m,'train')
    choices=[next(i for i,r in enumerate(data.rows) if r['patch']<0 and r['issue_mask'][1]),
             next(i for i,r in enumerate(data.rows) if r['patch']>=0 and r['issue_mask'][1])]
    for index in choices:
        x,y=data[index];x={k:v[None].to('mps') for k,v in x.items()}
        if index==choices[0]:
            with torch.no_grad():
                expected=reference(x,1);actual=model(x,1)
                for key in expected: assert torch.allclose(actual[key],expected[key],atol=1e-5,rtol=1e-5), key
            print('PASS zero-correction warm start equals v168 E3 field and route',flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True);state=model.initial(x)
        initial_p=state['core'][:,0,32,32]*model.std[0,0,0,0]+model.mean[0,0,0,0]
        assert torch.allclose(initial_p,x['issue_intensity'][:,1],atol=1e-3)
        total=0
        for lead in range(4):
            state,out=model.step(state)
            target={k:v[lead:lead+1].to('mps') for k,v in y.items()}
            target['previous_pressure']=y['pressure'][lead-1:lead].to('mps') if lead else x['issue_intensity'][:,1]
            target['previous_pressure_mask']=y['pressure_mask'][lead-1:lead].to('mps') if lead else x['issue_mask'][:,1]
            target['previous_center']=y['center'][lead-1:lead].to('mps') if lead else x['center']
            target['previous_storm_mask']=y['storm_mask'][lead-1:lead].to('mps') if lead else torch.ones_like(target['storm_mask'])
            loss,_=objective(out,target,x,m);total=total+loss/4
            assert all(bool(torch.isfinite(v).all()) for v in out.values())
            grid=model.core_grid(out['center'][:,0,None,None],out['center'][:,1,None,None],state['origin'])
            field_pressure=base.sample_field(out['core']*model.std[:,:1]+model.mean[:,:1],grid)[:,0,0,0]
            assert torch.equal(field_pressure,out['pressure'])
        total.backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        for layer in [model.core_net.out,model.coarse.out,model.motion_net[-1]]:
            assert layer.weight.grad is not None and layer.weight.grad.abs().sum()>0
        assert model.context.output.weight.grad.abs().sum()>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        print('PASS real 4-step backward, anchored pressure and field-derived output; native=',bool(x['detail_available'].item()),flush=True)
    model.eval()
    with torch.no_grad():
        out=model(x,20)
        assert all(bool(torch.isfinite(v).all()) for v in out.values())
        first=model(x,1)
        altered={k:v.clone() for k,v in target.items()};altered['center']+=10
        objective({k:v[:,0] for k,v in first.items()},altered,x,m)
        again=model(x,1)
        assert all(torch.equal(first[k],again[k]) for k in first)
        try:model.initial(dict(x,future_pressure=target['regional']))
        except ValueError:pass
        else:raise AssertionError('Future input accepted')
        missing={k:v.clone() for k,v in x.items()};missing['issue_mask'].zero_();missing['issue_intensity'].zero_()
        absent=model(missing,1)
        assert all(bool(torch.isfinite(v).all()) for v in absent.values())
    print('PASS 120h rollout, label invariance, missing intensity and future-input rejection',flush=True)
    with tempfile.TemporaryDirectory(dir=root/'v173') as tmp:
        path=Path(tmp)/'resume.pt';save(path,model,optimizer,m,0,[],100.,dict(window=1,totals=[[1.,1],[1.,1]],batch=10))
        def update():
            model.train();optimizer.zero_grad(set_to_none=True)
            o=model(x,1);loss,_=objective({k:v[:,0] for k,v in o.items()},target,x,m)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            return model.core_net.out.weight.detach().clone()
        expected=update();payload=restore(path,model,optimizer,m);actual=update()
        assert torch.allclose(expected,actual,atol=1e-6,rtol=1e-5)
        assert payload['progress']['window']==1
    indices=base.training_indices(data.rows,1)
    grouped=list(batches(indices,data.rows,10,2))
    assert sorted(i for group,_ in grouped for part in group for i in part)==sorted(indices)
    assert set(indices)==set(range(len(data)))
    print('PASS checkpoint optimizer resume and full dataset coverage',flush=True)
    print('parameters=',sum(p.numel() for p in model.parameters()),flush=True)


if __name__=='__main__':main()
