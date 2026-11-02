"""CPU scientific contracts, gradient equivalence and resumability; no skill claim."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from model import TrackPressure,loss
from train import update,seed,save_checkpoint,load_checkpoint
from contracts import sample_pressure

MEAN=[1000,1000,0,0,290,290]+[v for _ in range(5) for v in (0,0,270,.005,50000)]
SCALE=[20,20,10,10,10,10]+[v for _ in range(5) for v in (10,10,10,.005,10000)]


class Fixture:
    def __init__(self):
        self.lat=torch.linspace(60,0,17);self.lon=torch.linspace(100,180,25)
        self.a=torch.tensor(MEAN).float()[None,None,:,None,None].expand(1,9,31,17,25).clone()
        self.a[:,:,0]+=torch.linspace(-12,12,25)[None,None,None]
    def history(self,r,device):
        return dict(history=self.a,valid=torch.ones_like(self.a,dtype=torch.bool),center=torch.tensor([[20.,135.]]),
            motion=torch.tensor([[60.,20.]]),issue_pressure=torch.tensor([980.]),pressure_valid=torch.tensor([True]),lat=self.lat,lon=self.lon)
    def label(self,r,k,device):
        target=self.a[:,-1].clone();target[:,0]-=k+1
        return dict(target=target,valid=torch.ones_like(target,dtype=torch.bool),center=torch.tensor([[20.+.2*(k+1),135.+.5*(k+1)]]),
            center_valid=torch.tensor([True]),pressure=torch.tensor([978.-k]),pressure_valid=torch.tensor([True]),
            motion=torch.tensor([[52.,22.24]]),motion_valid=torch.tensor([True]))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_map_pressure_identity_all20_and_direct_lead_order(self):
        seed(133);d=Fixture();m=TrackPressure(MEAN,SCALE,8).eval()
        with torch.no_grad():
            state=m.initial(**d.history(None,'cpu'));saved={k:m.forecast(state,k) for k in (19,7,0)}
            for k in range(20):
                state,out=m.step(state)
                expected=sample_pressure(out['weather'][0,0].numpy(),np.ones((17,25),bool),d.lat.numpy(),d.lon.numpy(),out['center'][0].numpy())
                self.assertAlmostEqual(expected,float(out['pressure'][0]),places=3)
                if k in saved:
                    for key in out:torch.testing.assert_close(out[key],saved[k][key],rtol=0,atol=0)
            self.assertEqual(state['step'],20)
            torch.testing.assert_close(state['center'],torch.tensor([[20.,135.]]))
            self.assertGreater(float(out['center'][0,1]),135.)

    def test_feature_gradient_matches_full_joint_backward(self):
        seed(133);d=Fixture();m=TrackPressure(MEAN,SCALE,8)
        # Nonzero heads ensure the test exercises encoder gradients too.
        torch.nn.init.normal_(m.increment.weight,std=.01)
        torch.nn.init.normal_(m.route[-1].weight,std=.01)
        full=copy.deepcopy(m);full.checkpoint_activations=False
        packet=d.history(None,'cpu');state=full.initial(**packet)
        objective=sum(loss(full,full.forecast(state,k),**d.label({},k,'cpu'),lat=d.lat,lon=d.lon)[0]/3 for k in range(3))
        objective.backward();torch.nn.utils.clip_grad_norm_(full.parameters(),5.,error_if_nonfinite=True)
        opt=torch.optim.SGD(m.parameters(),lr=0.)
        update(m,opt,d,[{}],3,'cpu')
        nonzero=0
        for (name,p),(other,q) in zip(m.named_parameters(),full.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-7,msg=name)
                nonzero+=int(bool(p.grad.abs().sum()>0))
        self.assertGreater(nonzero,70)

    def test_optimizer_and_exact_resume(self):
        seed(133);d=Fixture();m=TrackPressure(MEAN,SCALE,8);opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
        result=update(m,opt,d,[{}],3,'cpu');self.assertGreater(result['gradient_norm'],0)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'c.pt';config={'code':{'fixture':'133'}};cursor={'updates':1}
            save_checkpoint(p,m,opt,config,cursor)
            update(m,opt,d,[{}],3,'cpu');expected=copy.deepcopy(m.state_dict())
            m2=TrackPressure(MEAN,SCALE,8);o2=torch.optim.AdamW(m2.parameters(),lr=1e-4)
            self.assertEqual(load_checkpoint(p,m2,o2,config),cursor)
            update(m2,o2,d,[{}],3,'cpu')
            for k,v in m2.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=1e-6,atol=1e-7)
            with self.assertRaises(ValueError):load_checkpoint(p,m2,o2,{'changed':True})

    def test_masked_labels_and_inputs(self):
        d=Fixture();m=TrackPressure(MEAN,SCALE,8);packet=d.history(None,'cpu')
        packet['valid'][:,:,8,:2]=False;packet['history'][:,:,8,:2]=float('nan')
        out=m.forecast(m.initial(**packet),0);y=d.label({},0,'cpu')
        y['center_valid'].fill_(False);y['pressure_valid'].fill_(False)
        y['valid'][:,:,0,:]=False;y['target'][:,:,0,:]=float('nan')
        v,_=loss(m,out,**y,lat=d.lat,lon=d.lon);self.assertTrue(torch.isfinite(v));v.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_future_labels_do_not_enter_forward_or_jma_map_loss(self):
        d=Fixture();m=TrackPressure(MEAN,SCALE,8).eval();packet=d.history(None,'cpu')
        with torch.no_grad():
            state=m.initial(**packet);a=m.forecast(state,4)
            label=d.label({},4,'cpu');v,_=loss(m,a,**label,lat=d.lat,lon=d.lon)
            label['pressure'].fill_(800.)
            v2,_=loss(m,a,**label,lat=d.lat,lon=d.lon)
            self.assertEqual(float(v),float(v2))
            label['target'].fill_(100000);b=m.forecast(state,4)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
