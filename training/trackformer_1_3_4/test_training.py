"""CPU contracts and synthetic learnability; these are not weather validation."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import torch
from model import TrackPressure,loss,transport,detach_state,sample_at
from train import update,seed,save_checkpoint,load_checkpoint,validate
from contracts import sha256

MEAN=[1000,1000,0,0,290,290]+[v for _ in range(5) for v in (0,0,270,.005,50000)]
SCALE=[20,20,10,10,10,10]+[v for _ in range(5) for v in (10,10,10,.005,10000)]
TENDENCY=[s*.2 for s in SCALE]


class Fixture:
    def __init__(self):
        self.lat=torch.linspace(40,10,25);self.lon=torch.linspace(120,160,33)
        yy,xx=torch.meshgrid(self.lat,self.lon,indexing='ij')
        self.maps=[]
        for k in range(-8,21):
            a=torch.tensor(MEAN).float()[:,None,None].expand(31,25,33).clone()
            a[0]-=25*torch.exp(-((yy-(23.+.1*k))**2+(xx-(138.+.8*k))**2)/8)
            self.maps.append(a)
        self.history_a=torch.stack(self.maps[:9])[None]
    def history(self,r,device):
        return dict(history=self.history_a.clone(),valid=torch.ones_like(self.history_a,dtype=torch.bool),
                    center=torch.tensor([[23.,138.]]),motion=torch.tensor([[82.,11.12]]),issue_pressure=torch.tensor([975.]),
                    pressure_valid=torch.tensor([True]),lat=self.lat,lon=self.lon)
    def label(self,r,k,device):
        a=self.maps[9+k][None]
        return dict(target=a,valid=torch.ones_like(a,dtype=torch.bool),center=torch.tensor([[23.+.1*(k+1),138.+.8*(k+1)]]),
                    center_valid=torch.tensor([True]),pressure=torch.tensor([975.]),pressure_valid=torch.tensor([True]),
                    motion=torch.tensor([[82.,11.12]]),motion_valid=torch.tensor([True]))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def model(self):return TrackPressure(MEAN,SCALE,8,TENDENCY)

    def test_transport_sign_identity_and_center(self):
        lat=torch.linspace(40,10,61);lon=torch.linspace(120,160,81)
        yy,xx=torch.meshgrid(lat,lon,indexing='ij')
        low=(-30*torch.exp(-((yy-25)**2+(xx-140)**2)/2))[None,None]
        flow=torch.zeros(1,2,61,81)
        torch.testing.assert_close(transport(low,flow,lat,lon),low,rtol=1e-6,atol=1e-5)
        # Exact +1 degree east/north forward movement, including descending latitude.
        flow[:,0]=111.2*lat.deg2rad().cos()[:,None];flow[:,1]=111.2
        moved=transport(low,flow,lat,lon)
        flat=int(moved.argmin());iy,ix=divmod(flat,81)
        self.assertEqual(float(lat[iy]),26.);self.assertEqual(float(lon[ix]),141.)
        center=torch.tensor([[25.,140.]]);velocity=sample_at(flow,center,lat,lon)
        next_center=center+torch.stack((velocity[:,1]/111.2,velocity[:,0]/(111.2*center[:,0].deg2rad().cos())),1)
        torch.testing.assert_close(next_center,torch.tensor([[26.,141.]]),rtol=1e-6,atol=1e-7)

    def test_evolving_state_and_map_pressure_identity(self):
        seed(134);d=Fixture();m=self.model().eval()
        with torch.no_grad():
            m.flow.bias.copy_(torch.tensor([.02,.005]));m.increment.bias[0]=.1
            s=m.initial(**d.history(None,'cpu'));initial=s['x'].clone();previous=s['x']
            for k in range(20):
                s,out=m.step(s)
                torch.testing.assert_close(s['previous'],previous,rtol=0,atol=0);previous=s['x']
                expected=sample_at(out['weather'][:,:1],out['center'],d.lat,d.lon)[:,0]
                torch.testing.assert_close(out['pressure'],expected,rtol=0,atol=0)
            self.assertGreater(float((s['x']-initial).abs().mean()),.01)
            self.assertGreater(float(s['center'][0,1]),138.)
            self.assertEqual(s['step'],20)

    def test_checkpoint_gradient_matches_full_segment(self):
        seed(134);d=Fixture();m=self.model()
        torch.nn.init.normal_(m.increment.weight,std=.001);torch.nn.init.normal_(m.flow.weight,std=.0001)
        full=copy.deepcopy(m);full.checkpoint_activations=False
        packet=d.history(None,'cpu');s=full.initial(**packet);objective=0
        prev=packet['history'][:,-1];valid=packet['valid'][:,-1]
        for k in range(3):
            s,out=full.step(s);y=d.label({},k,'cpu')
            objective+=loss(full,out,**y,lat=d.lat,lon=d.lon,previous_target=prev,previous_valid=valid)[0]/3
            prev=y['target'];valid=y['valid']
        objective.backward();torch.nn.utils.clip_grad_norm_(full.parameters(),5.,error_if_nonfinite=True)
        update(m,torch.optim.SGD(m.parameters(),lr=0.),d,[{}],3,'cpu',4)
        nonzero=0
        for (name,p),(other,q) in zip(m.named_parameters(),full.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-7,msg=name)
                nonzero+=int(bool(p.grad.abs().sum()>0))
        self.assertGreater(nonzero,80)

    def test_optimizer_exact_resume_across_truncation(self):
        seed(134);d=Fixture();m=self.model();opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
        result=update(m,opt,d,[{}],3,'cpu',2);self.assertGreater(result['gradient_norm'],0)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'c.pt';config={'code':{'fixture':'134'}};cursor={'updates':1}
            save_checkpoint(path,m,opt,config,cursor)
            update(m,opt,d,[{}],3,'cpu',2);expected=copy.deepcopy(m.state_dict())
            m2=self.model();opt2=torch.optim.AdamW(m2.parameters(),lr=1e-4)
            self.assertEqual(load_checkpoint(path,m2,opt2,config),cursor)
            update(m2,opt2,d,[{}],3,'cpu',2)
            for key,value in m2.state_dict().items():torch.testing.assert_close(value,expected[key],rtol=1e-6,atol=1e-7)
            with self.assertRaises(ValueError):load_checkpoint(path,m2,opt2,{'changed':True})

    def test_masked_nan_inputs_targets_and_future_separation(self):
        seed(134);d=Fixture();m=self.model();packet=d.history(None,'cpu')
        packet['valid'][:,:,8,:2]=False;packet['history'][:,:,8,:2]=float('nan')
        state=m.initial(**packet);_,out=m.step(state);y=d.label({},0,'cpu')
        y['target']=y['target'].clone();y['valid'][:,:,0,:]=False;y['target'][:,:,0,:]=float('nan')
        value,_=loss(m,out,**y,lat=d.lat,lon=d.lon,previous_target=packet['history'][:,-1],previous_valid=packet['valid'][:,-1])
        self.assertTrue(torch.isfinite(value));value.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        m.eval()
        with torch.no_grad():
            s=m.initial(**packet);_,a=m.step(s)
            y['target'].fill_(99999.);y['pressure'].fill_(700.)
            _,b=m.step(s)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            before=loss(m,a,**y,lat=d.lat,lon=d.lon)[0];y['pressure'].fill_(900.)
            after=loss(m,a,**y,lat=d.lat,lon=d.lon)[0]
            torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_synthetic_moving_low_is_learnable(self):
        seed(134);d=Fixture();m=self.model();opt=torch.optim.AdamW(m.parameters(),lr=2e-4)
        def score():
            m.eval()
            with torch.no_grad():
                s=m.initial(**d.history(None,'cpu'));s,out=m.step(s);s,out=m.step(s)
                y=d.label({},1,'cpu')
                return float((out['weather'][:,0]-y['target'][:,0]).square().mean()),float((out['center']-y['center']).square().sum())
        before=score()
        for _ in range(24):update(m,opt,d,[{}],2,'cpu',2)
        after=score();print('SYNTHETIC_ONLY',dict(before=before,after=after),flush=True)
        self.assertLess(after[0],before[0]*.8);self.assertLess(after[1],before[1]*.5)

    def test_validation_receipts_all24_and20_leads(self):
        seed(134);d=Fixture();m=self.model()
        d.cohort={'development':[dict(id=f'fixture-{i}',storm_id=f'storm-{i}',issue_seconds=123456,
                                   initial=[23.,138.,975.],motion=[82.,11.12]) for i in range(24)]}
        with tempfile.TemporaryDirectory() as t,patch('train.LATITUDE',d.lat.numpy()),patch('train.LONGITUDE',d.lon.numpy()):
            root=Path(t);config={'fixture':True};rec=validate(m,d,root,1,'fixture-checkpoint','cpu',config)
            self.assertEqual(rec['cases'],24);self.assertEqual(rec['leads'],list(range(6,121,6)))
            for name,digest in rec['files'].items():self.assertEqual(sha256(root/'development-E1'/name),digest)
            for i in range(24):
                with np.load(root/'development-E1'/f'case-{i:03d}.npz',allow_pickle=False) as z:
                    self.assertEqual(z['pressure_hPa'].shape,(20,25,33))
                    np.testing.assert_array_equal(z['lead_hours'],np.arange(6,121,6))
                    self.assertEqual(z['truth_valid'].dtype,np.dtype(bool))
            metrics=json.loads((root/'development-E1/metrics.json').read_text())
            self.assertGreater(metrics['observed_change_mean_abs_hPa_per6h'],.01)
            self.assertLess(metrics['predicted_change_mean_abs_hPa_per6h'],.001)
            self.assertFalse(metrics['promotion'])
            self.assertEqual(validate(m,d,root,1,'fixture-checkpoint','cpu',config),rec)


if __name__=='__main__':unittest.main()
