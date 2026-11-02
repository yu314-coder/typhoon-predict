"""Direct lead-conditioned regional weather and track forecasts.

Every lead reads the same causal history encoding. Predicted weather and
pressure-minimum association are never fed into a later forecast. This is an
experimental architecture, not a reproduction of Pangu or released TrackFormer1.2.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

FEATURE_KEYS = ('a', 'c', 'd', 'e')


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),
                                 nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
    def forward(self,x): return self.net(x)


class Memory(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gates=nn.Conv2d(2*c,2*c,3,padding=1)
        self.proposal=nn.Conv2d(2*c,c,3,padding=1)
    def forward(self,x,h):
        r,z=self.gates(torch.cat((x,h),1)).sigmoid().chunk(2,1)
        return (1-z)*h+z*self.proposal(torch.cat((x,r*h),1)).tanh()


class TrackPressure(nn.Module):
    def __init__(self,mean,scale,width=64):
        super().__init__()
        if width<8 or width%4: raise ValueError('width must be a multiple of4, >=8')
        self.width=width;self.checkpoint_activations=True
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32).reshape(1,31,1,1))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32).reshape(1,31,1,1))
        self.history=nn.Sequential(nn.Conv2d(62,width,3,stride=2,padding=1),nn.GroupNorm(4,width),nn.SiLU(),Block(width,width))
        self.memory=Memory(width)
        self.enc0=Block(width+62+4,width)
        self.enc1=Block(width,2*width);self.enc2=Block(2*width,4*width);self.enc3=Block(4*width,6*width)
        layer=nn.TransformerEncoderLayer(6*width,4,12*width,dropout=0,batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.time=nn.Sequential(nn.Linear(4,6*width),nn.SiLU(),nn.Linear(6*width,12*width))
        self.route=nn.Sequential(nn.Linear(7*width+6,4*width),nn.SiLU(),nn.Linear(4*width,40))
        self.dec2=Block(10*width,4*width);self.dec1=Block(6*width,2*width);self.dec0=Block(3*width,width)
        self.refine=Block(width+31+31+5,width)
        self.increment=nn.Conv2d(width,31,1)
        nn.init.zeros_(self.route[-1].weight);nn.init.zeros_(self.route[-1].bias)
        nn.init.zeros_(self.increment.weight);nn.init.zeros_(self.increment.bias)

    @staticmethod
    def grid(lat,lon,center):
        return torch.stack((2*(center[:,1]-lon[0])/(lon[-1]-lon[0])-1,
                            2*(center[:,0]-lat[0])/(lat[-1]-lat[0])-1),-1)[:,None,None]

    @staticmethod
    def up(x,reference):
        return F.interpolate(x,reference.shape[-2:],mode='bilinear',align_corners=True)

    def initial(self,history,valid,center,motion,issue_pressure,pressure_valid,lat,lon):
        if history.ndim!=5 or history.shape[1:3]!=(9,31) or valid.shape!=history.shape:
            raise ValueError('Require9x31 causal history and original masks')
        if not bool(torch.isfinite(history[valid]).all()):raise ValueError('Nonfinite valid input')
        x=torch.where(valid,(history-self.mean[:,None])/self.scale[:,None],0.)
        h=None
        for k in range(9):
            v=torch.cat((x[:,k],valid[:,k].float()),1)
            z=checkpoint(self.history,v,use_reentrant=False) if self.training and self.checkpoint_activations else self.history(v)
            if h is None:h=torch.zeros_like(z)
            h=checkpoint(self.memory,z,h,use_reentrant=False) if self.training and self.checkpoint_activations else self.memory(z,h)
        yy,xx=torch.meshgrid(lat,lon,indexing='ij');b=x.shape[0]
        pos=torch.stack((yy.expand(b,-1,-1)/60,xx.expand(b,-1,-1)/80-1.75,
            (yy[None]-center[:,0,None,None])/20,(xx[None]-center[:,1,None,None])/20),1)
        def encode(h,x0,mask,pos):
            a=self.enc0(torch.cat((h,self.up(x0,h),self.up(mask,h),self.up(pos,h)),1))
            c=self.enc1(F.avg_pool2d(a,2,ceil_mode=True));d=self.enc2(F.avg_pool2d(c,2,ceil_mode=True))
            e=self.enc3(F.avg_pool2d(d,2,ceil_mode=True))
            tokens=F.adaptive_avg_pool2d(e,(8,10));shape=tokens.shape
            tokens=self.context(tokens.flatten(2).transpose(1,2)).transpose(1,2).reshape(shape)
            return a,c,d,e+self.up(tokens,e)
        args=(h,x[:,-1],valid[:,-1].float(),pos)
        features=checkpoint(encode,*args,use_reentrant=False) if self.training and self.checkpoint_activations else encode(*args)
        return dict(zip(FEATURE_KEYS,features),x=x[:,-1],mask=valid[:,-1],center=center,motion=motion,
                    issue_pressure=issue_pressure,pressure_valid=pressure_valid,lat=lat,lon=lon,step=0)

    def forecast(self,s,k):
        if not 0<=k<20:raise ValueError('Lead must be +6..+120h')
        def compute(a,c,d,e):
            b=a.shape[0];lat,lon=s['lat'],s['lon'];t=(k+1)/20
            local=F.grid_sample(a,self.grid(lat,lon,s['center']),align_corners=True,padding_mode='border')[:,:,0,0]
            meta=torch.cat((s['center']/torch.tensor([60.,180.],device=a.device),s['motion']/100,
                ((s['issue_pressure']-1010)/20*s['pressure_valid'])[:,None],s['pressure_valid'][:,None]),1)
            # Twenty velocities are predicted jointly from issue-time context.
            # No pressure association or prior predicted centre enters this head.
            increments=s['motion'][:,None]+100*self.route(torch.cat((local,e.mean((2,3)),meta),1)).reshape(b,20,2).tanh()
            displacement=increments[:,:k+1].sum(1)
            center=s['center']+torch.stack((displacement[:,1]/111.2,
                displacement[:,0]/(111.2*s['center'][:,0].deg2rad().cos().clamp_min(.2))),1)
            time=a.new_tensor([t,t*t,math.sin(math.pi*t),math.cos(math.pi*t)]).expand(b,-1)
            gain,bias=self.time(time).chunk(2,1)
            e=e*(1+.1*gain[:,:,None,None])+.1*bias[:,:,None,None]
            d=self.dec2(torch.cat((d,self.up(e,d)),1));c=self.dec1(torch.cat((c,self.up(d,c)),1))
            features=self.dec0(torch.cat((a,self.up(c,a)),1))
            yy,xx=torch.meshgrid(lat,lon,indexing='ij')
            pos=torch.stack(((yy[None]-center[:,0,None,None]).expand(b,-1,-1)/10,
                (xx[None]-center[:,1,None,None]).expand(b,-1,-1)/10,
                yy.expand(b,-1,-1)/60,xx.expand(b,-1,-1)/80-1.75,torch.full_like(yy,t).expand(b,-1,-1)),1)
            features=self.refine(torch.cat((self.up(features,s['x']),s['x'],s['mask'].float(),pos),1))
            normalized=s['x']+self.increment(features)
            physical=normalized*self.scale+self.mean
            pressure=F.grid_sample(physical[:,:1],self.grid(lat,lon,center),align_corners=True,padding_mode='border')[:,0,0,0]
            covered=(center[:,0]>=lat[-1])&(center[:,0]<=lat[0])&(center[:,1]>=lon[0])&(center[:,1]<=lon[-1])
            return physical,normalized,center,pressure,covered,increments[:,k]
        args=tuple(s[key] for key in FEATURE_KEYS)
        values=checkpoint(compute,*args,use_reentrant=False) if self.training and self.checkpoint_activations else compute(*args)
        return dict(zip(('weather','normalized','center','pressure','track_valid','motion'),values))

    def step(self,s):
        out=self.forecast(s,s['step'])
        # Only the integer lead cursor advances. All conditioning stays fixed.
        return dict(s,step=s['step']+1),out


def loss(model,out,target,valid,center,center_valid,pressure,pressure_valid,lat,lon,motion=None,motion_valid=None):
    y=torch.nan_to_num((target-model.mean)/model.scale)
    delta=torch.where(valid,out['normalized']-y,0.)
    area=lat.deg2rad().cos()[None,None,:,None]
    weight=valid.float()*area
    field=(delta.square()*weight).sum()/weight.sum().clamp_min(1)
    pweight=weight[:,:1]
    mslp=(delta[:,:1].square()*pweight).sum()/pweight.sum().clamp_min(1)
    yy,xx=torch.meshgrid(lat,lon,indexing='ij')
    dy=(yy[None]-center[:,0,None,None])*111.2
    dx=(xx[None]-center[:,1,None,None])*111.2*center[:,0].deg2rad().cos()[:,None,None]
    core_weight=torch.exp(-(dx.square()+dy.square())/(2*400**2))*center_valid[:,None,None]*pweight[:,0]
    # Dense observed reanalysis core remains supervised; JMA scalar pressure is
    # a separate diagnostic, never an incompatible replacement map target.
    core=(delta[:,0].square()*core_weight).sum()/core_weight.sum().clamp_min(1)
    route_error=torch.stack(((out['center'][:,1]-center[:,1])*111.2*center[:,0].deg2rad().cos(),
                             (out['center'][:,0]-center[:,0])*111.2),1)
    route=(F.smooth_l1_loss(route_error/250,torch.zeros_like(route_error),reduction='none').sum(1)*center_valid).sum()/center_valid.sum().clamp_min(1)
    velocity=field*0
    if motion is not None:
        velocity=(F.smooth_l1_loss(out['motion']/100,motion/100,reduction='none').sum(1)*motion_valid).sum()/motion_valid.sum().clamp_min(1)
    grad=field*0
    for axis in (-2,-1):
        residual=out['normalized'][:,:1].diff(dim=axis)-y[:,:1].diff(dim=axis)
        good=valid[:,:1].narrow(axis,0,residual.shape[axis])&valid[:,:1].narrow(axis,1,residual.shape[axis])
        grad=grad+(torch.where(good,residual,0.).square()).sum()/good.sum().clamp_min(1)
    return field+mslp+.5*core+.5*route+.1*velocity+.1*grad,dict(field=field,mslp=mslp,core_mslp=core,
        track=route,motion=velocity,pressure_gradient=grad)
