"""CPU contract fixtures; not a CUDA/T4 qualification or measured skill test."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from model import TrackPressure,loss,advect,environmental_wind,annulus_weights
from train import update,seed,save_checkpoint,load_checkpoint
from contracts import sample_pressure

MEAN=[1000,1000,0,0,290,290]+[v for _ in range(5) for v in (0,0,270,.005,50000)]
SCALE=[20,20,10,10,10,10]+[v for _ in range(5) for v in (10,10,10,.005,10000)]


class Fixture:
    def __init__(self):
        self.lat=torch.linspace(60,0,24);self.lon=torch.linspace(100,180,32)
        self.a=torch.tensor(MEAN).float()[None,None,:,None,None].expand(1,9,31,24,32).clone()
        self.a[:,:,0]+=torch.linspace(-12,12,32)[None,None,None]
        self.a[:,:,6]=2.;self.a[:,:,16]=3.
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

    def test_map_pressure_identity_and_20_leads(self):
        seed(131);d=Fixture();m=TrackPressure(MEAN,SCALE,8).eval();s=m.initial(**d.history(None,'cpu'))
        with torch.no_grad():
            for k in range(20):
                s,o=m.step(s)
                expected=sample_pressure(o['weather'][0,0].numpy(),np.ones((24,32),bool),d.lat.numpy(),d.lon.numpy(),o['center'][0].numpy())
                self.assertAlmostEqual(expected,float(o['pressure'][0]),places=3)
                self.assertTrue(torch.isfinite(o['weather']).all())
            self.assertFalse(torch.equal(s['center'],torch.tensor([[20.,135.]])))
            self.assertEqual(s['step'],20)

    def test_optimizer_and_exact_cpu_resume(self):
        seed(131);d=Fixture();m=TrackPressure(MEAN,SCALE,8);opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
        r=update(m,opt,d,[{}],4,2,'cpu');self.assertGreater(r['gradient_norm'],0)
        self.assertGreater(float(m.increment.weight.detach().abs().sum()),0)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'c.pt';config={'code':{'fixture':'yes'}};cursor={'updates':1}
            save_checkpoint(p,m,opt,config,cursor)
            update(m,opt,d,[{}],4,2,'cpu');expected={k:v.detach().clone() for k,v in m.state_dict().items()}
            m2=TrackPressure(MEAN,SCALE,8);o2=torch.optim.AdamW(m2.parameters(),lr=1e-4)
            self.assertEqual(load_checkpoint(p,m2,o2,config),cursor)
            update(m2,o2,d,[{}],4,2,'cpu')
            for k,v in m2.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=1e-6,atol=1e-7)

    def test_masked_labels_have_finite_loss(self):
        d=Fixture();m=TrackPressure(MEAN,SCALE,8);p=d.history(None,'cpu');s,o=m.step(m.initial(**p));y=d.label({},0,'cpu')
        y['center_valid'].fill_(False);y['pressure_valid'].fill_(False)
        y['valid'][:,:,0,:]=False;y['target'][:,:,0,:]=float('nan')
        v,_=loss(m,o,**y,lat=p['lat'],lon=p['lon']);self.assertTrue(torch.isfinite(v));v.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_transport_zero_and_east_north_direction(self):
        lat=torch.linspace(30,20,41);lon=torch.linspace(120,130,41)
        yy,xx=torch.meshgrid(lat,lon,indexing='ij');x=torch.stack((yy,xx))[None]
        zero=torch.zeros((1,2,41,41));torch.testing.assert_close(advect(x,zero,lat,lon),x,rtol=1e-6,atol=1e-7)
        wind=zero.clone();wind[:,0]=5.;wind[:,1]=3.;wind.requires_grad_()
        moved=advect(x,wind,lat,lon)
        self.assertAlmostEqual(float(moved[0,0,20,20].detach()),25.-3*21.6/111.2,places=4)
        self.assertAlmostEqual(float(moved[0,1,20,20].detach()),125.-5*21.6/(111.2*np.cos(np.deg2rad(25.))),places=4)
        moved.sum().backward();self.assertTrue(torch.isfinite(wind.grad).all())

    def test_environment_excludes_cyclone_core_point(self):
        lat=torch.linspace(26,14,49);lon=torch.linspace(129,141,49)
        weather=torch.zeros(1,31,49,49);mask=torch.ones_like(weather,dtype=torch.bool)
        weather[:,[6,11,16]]=4.;weather[:,[7,12,17]]=2.
        weather[:,[6,7,11,12,16,17],24,24]=500.
        w=annulus_weights(lat,lon,torch.tensor([[20.,135.]]),mask)
        flow=(environmental_wind(weather,mask)*w[:,None]).sum((2,3))
        torch.testing.assert_close(flow,torch.tensor([[4.,2.]]),rtol=1e-6,atol=1e-7)

    def test_future_labels_do_not_enter_forward(self):
        d=Fixture();m=TrackPressure(MEAN,SCALE,8).eval();p=d.history(None,'cpu')
        with torch.no_grad():
            a=m.step(m.initial(**p))[1]
            future=d.label({},0,'cpu');future['target'].fill_(100000)
            b=m.step(m.initial(**p))[1]
        torch.testing.assert_close(a['weather'],b['weather'],rtol=0,atol=0)
        torch.testing.assert_close(a['center'],b['center'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
