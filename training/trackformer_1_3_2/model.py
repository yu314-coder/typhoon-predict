"""1.3.2: regional transport, environmental steering and learned weather residuals.

Transport is an approximate numerical prior trained against actual weather, not a
post-hoc translation onto a prescribed track. All future inputs are own forecasts.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def environmental_wind(weather, mask):
    """Masked 850/700/500-hPa mean, in m/s; no vortex-centre point sample."""
    indices=[6,7,11,12,16,17]
    a=weather[:,indices].reshape(weather.shape[0],3,2,*weather.shape[-2:])
    m=mask[:,indices].reshape_as(a).float()
    return (a*m).sum(1)/m.sum(1).clamp_min(1)


def advect(x, wind, lat, lon):
    """Six-hour backward characteristic sampling, N-to-S latitude, border hold.

    A differentiable approximate prior, not a mass-conserving atmospheric solver.
    The learned weather residual also represents unresolved dynamics and boundaries.
    """
    yy,xx=torch.meshgrid(lat,lon,indexing='ij')
    departure_lat=yy[None]-wind[:,1]*21.6/111.2
    departure_lon=xx[None]-wind[:,0]*21.6/(111.2*yy.deg2rad().cos().clamp_min(.2))[None]
    gx=2*(departure_lon-lon[0])/(lon[-1]-lon[0])-1
    gy=2*(departure_lat-lat[0])/(lat[-1]-lat[0])-1
    return F.grid_sample(x,torch.stack((gx,gy),-1),align_corners=True,padding_mode='border')


def annulus_weights(lat,lon,center,mask):
    yy,xx=torch.meshgrid(lat,lon,indexing='ij')
    dy=(yy[None]-center[:,0,None,None])*111.2
    dx=(xx[None]-center[:,1,None,None])*111.2*center[:,0].deg2rad().cos()[:,None,None]
    r2=dy.square()+dx.square()
    good=mask[:,[6,7,11,12,16,17]].any(1)
    weight=((r2>=150**2)&(r2<=600**2)&good).float()*lat.deg2rad().cos()[None,:,None]
    # Tiny synthetic grids may have no annulus cells; explicit nearest support.
    fallback=(-r2/(2*300**2)).flatten(1).softmax(1).reshape_as(r2)*good
    weight=torch.where((weight.sum((1,2))>0)[:,None,None],weight,fallback)
    return weight/weight.sum((1,2),keepdim=True).clamp_min(1e-9)


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),
                                 nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
    def forward(self,x): return self.net(x)


class Memory(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gates=nn.Conv2d(2*c,2*c,3,padding=1)
        self.proposal=nn.Conv2d(2*c,c,3,padding=1)
    def forward(self,x,h):
        r,z=self.gates(torch.cat((x,h),1)).sigmoid().chunk(2,1)
        n=self.proposal(torch.cat((x,r*h),1)).tanh()
        return (1-z)*h+z*n


class TrackPressure(nn.Module):
    def __init__(self, mean, scale, width=64):
        super().__init__()
        if width < 8 or width % 4: raise ValueError('width must be a positive multiple of4, >=8')
        self.width=width
        self.checkpoint_activations=True
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32).reshape(1,31,1,1))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32).reshape(1,31,1,1))
        self.history=Block(62,width)
        self.memory=Memory(width)
        self.enc0=Block(31+31+width+9,width)
        self.enc1=Block(width,2*width)
        self.enc2=Block(2*width,4*width)
        self.enc3=Block(4*width,6*width)
        layer=nn.TransformerEncoderLayer(6*width,4,12*width,dropout=0,batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.dec2=Block(10*width,4*width)
        self.dec1=Block(6*width,2*width)
        self.dec0=Block(3*width,width)
        self.increment=nn.Conv2d(width,31,1)
        self.motion=nn.Sequential(nn.Linear(2*width+6,2*width),nn.SiLU(),nn.Linear(2*width,2))
        nn.init.zeros_(self.increment.weight);nn.init.zeros_(self.increment.bias)
        nn.init.zeros_(self.motion[-1].weight);nn.init.zeros_(self.motion[-1].bias)

    @staticmethod
    def grid(lat,lon,center):
        x=2*(center[:,1]-lon[0])/(lon[-1]-lon[0])-1
        y=2*(center[:,0]-lat[0])/(lat[-1]-lat[0])-1
        return torch.stack((x,y),-1)[:,None,None]

    def initial(self,history,valid,center,motion,issue_pressure,pressure_valid,lat,lon):
        if history.ndim!=5 or history.shape[1:3]!=(9,31) or valid.shape!=history.shape:
            raise ValueError('Require9x31 history and explicit masks')
        x=torch.where(valid,(history-self.mean[:,None])/self.scale[:,None],0.)
        h=torch.zeros((history.shape[0],self.width,*history.shape[-2:]),device=x.device)
        for k in range(9):
            v=torch.cat((x[:,k],valid[:,k].float()),1)
            if self.training and self.checkpoint_activations:
                h=checkpoint(lambda v,h:self.memory(self.history(v),h),v,h,use_reentrant=False)
            else:h=self.memory(self.history(v),h)
        return dict(x=x[:,-1],mask=valid[:,-1],h=h,center=center,motion=motion,
                    issue_pressure=issue_pressure,pressure_valid=pressure_valid,lat=lat,lon=lon,step=0)

    def step(self,s):
        # Checkpoint the large regional block, not the causal state bookkeeping.
        if self.training and self.checkpoint_activations:
            def compute(x,h,center,motion):
                state,out=self._step(dict(s,x=x,h=h,center=center,motion=motion))
                return state['x'],state['h'],state['center'],state['motion'],out['weather'],out['pressure'],out['track_valid'],out['motion']
            x,h,center,motion,weather,pressure,covered,predicted_motion=checkpoint(compute,s['x'],s['h'],s['center'],s['motion'],use_reentrant=False)
            return dict(s,x=x,h=h,center=center,motion=motion,step=s['step']+1),dict(weather=weather,normalized=x,pressure=pressure,center=center,track_valid=covered,motion=predicted_motion)
        return self._step(s)

    def _step(self,s):
        x,h,lat,lon=s['x'],s['h'],s['lat'],s['lon']
        yy,xx=torch.meshgrid(lat,lon,indexing='ij')
        b=x.shape[0]
        dy=(yy[None]-s['center'][:,0,None,None])/10
        dx=(xx[None]-s['center'][:,1,None,None])*s['center'][:,0].deg2rad().cos()[:,None,None]/10
        def plane(a):return a[:,None,None].expand(b,*yy.shape)
        pos=torch.stack((dy,dx,yy.expand(b,-1,-1)/90,xx.expand(b,-1,-1)/180-1,
            plane((s['issue_pressure']-1010)/20*s['pressure_valid']),plane(s['pressure_valid'].float()),
            plane(s['motion'][:,0]/100),plane(s['motion'][:,1]/100),
            plane(torch.full((b,),s['step']/20,device=x.device))),1)
        a=self.enc0(torch.cat((x,s['mask'].float(),h,pos),1))
        c=self.enc1(F.avg_pool2d(a,2,ceil_mode=True));d=self.enc2(F.avg_pool2d(c,2,ceil_mode=True))
        e=self.enc3(F.avg_pool2d(d,2,ceil_mode=True))
        tokens=F.adaptive_avg_pool2d(e,(8,10));shape=tokens.shape
        tokens=self.context(tokens.flatten(2).transpose(1,2)).transpose(1,2).reshape(shape)
        e=e+F.interpolate(tokens,e.shape[-2:],mode='bilinear',align_corners=True)
        up=lambda q,r:F.interpolate(q,r.shape[-2:],mode='bilinear',align_corners=True)
        d=self.dec2(torch.cat((d,up(e,d)),1));c=self.dec1(torch.cat((c,up(d,c)),1))
        features=self.dec0(torch.cat((a,up(c,a)),1))
        # Spatially varying environmental flow moves the weather before the
        # learned residual. It is NOT driven by a future best-track position.
        wind=environmental_wind(x*self.scale+self.mean,s['mask'])
        smooth=F.avg_pool2d(F.pad(wind,(10,10,10,10),mode='replicate'),21,stride=1)
        transported=advect(x,smooth,lat,lon)
        nxt=transported+.1*self.increment(features)
        physical=nxt*self.scale+self.mean
        at=self.grid(lat,lon,s['center'])
        local=F.grid_sample(features,at,align_corners=True)[:,:,0,0]
        weights=annulus_weights(lat,lon,s['center'],s['mask'])
        surrounding=(features*weights[:,None]).sum((2,3))
        flow=(wind*weights[:,None]).sum((2,3))*21.6
        correction=200*self.motion(torch.cat((local,surrounding,s['motion']/100,flow/100,
            ((s['issue_pressure']-1010)/20)[:,None]*s['pressure_valid'][:,None],s['pressure_valid'][:,None]),1)).tanh()
        motion=.5*flow+.5*s['motion']+correction
        proposal=s['center']+torch.stack((motion[:,1]/111.2,
            motion[:,0]/(111.2*s['center'][:,0].deg2rad().cos().clamp_min(.2))),1)
        # Associate a centre on the transported, predicted physical pressure map.
        km_y=(yy[None]-proposal[:,0,None,None])*111.2
        km_x=(xx[None]-proposal[:,1,None,None])*111.2*proposal[:,0].deg2rad().cos()[:,None,None]
        r2=km_y.square()+km_x.square();support=(r2<400**2)&s['mask'][:,0]
        score=-physical[:,0]/3-r2/(2*150**2)
        pdf=score.masked_fill(~support,-1e9).flatten(1).softmax(1).reshape_as(score)
        center=torch.stack(((pdf*yy).sum((1,2)),(pdf*xx).sum((1,2))),1)
        exists=support.flatten(1).any(1)
        center=torch.where(exists[:,None],center,proposal)
        pressure=F.grid_sample(physical[:,:1],self.grid(lat,lon,center),align_corners=True)[:,0,0,0]
        covered=exists&(center[:,0]>=lat[-1])&(center[:,0]<=lat[0])&(center[:,1]>=lon[0])&(center[:,1]<=lon[-1])
        actual_motion=torch.stack(((center[:,1]-s['center'][:,1])*111.2*s['center'][:,0].deg2rad().cos(),
                                   (center[:,0]-s['center'][:,0])*111.2),1)
        out=dict(weather=physical,normalized=nxt,pressure=pressure,center=center,track_valid=covered,motion=motion)
        return dict(s,x=nxt,h=self.memory(features,advect(h,smooth,lat,lon)),center=center,
                    motion=actual_motion,step=s['step']+1),out


def loss(model,out,target,valid,center,center_valid,pressure,pressure_valid,lat,lon,motion=None,motion_valid=None):
    # Dense ERA5 core MSLP is not the same observation as best-track pressure.
    y=(target-model.mean)/model.scale
    yy,xx=torch.meshgrid(lat,lon,indexing='ij')
    km2=((yy[None]-center[:,0,None,None])*111.2).square()+((xx[None]-center[:,1,None,None])*111.2*center[:,0].deg2rad().cos()[:,None,None]).square()
    mask=valid.clone();mask[:,0]&=~((km2<150**2)&center_valid[:,None,None]&pressure_valid[:,None,None])
    weight=mask.float()*lat.deg2rad().cos()[None,None,:,None]
    delta=torch.where(valid,out['normalized']-torch.nan_to_num(y),0.)
    field=(delta.square()*weight).sum()/weight.sum().clamp_min(1)
    dy=(out['center'][:,0]-center[:,0])*111.2
    dx=(out['center'][:,1]-center[:,1])*111.2*center[:,0].deg2rad().cos()
    route=(F.smooth_l1_loss(torch.stack((dx,dy),1)/100,torch.zeros_like(torch.stack((dx,dy),1)),
            reduction='none').sum(1)*center_valid).sum()/center_valid.sum().clamp_min(1)
    pm=pressure_valid&center_valid
    core=(F.smooth_l1_loss(out['pressure']/20,pressure/20,reduction='none')*pm).sum()/pm.sum().clamp_min(1)
    velocity=field*0
    if motion is not None:
        velocity=(F.smooth_l1_loss(out['motion']/100,motion/100,reduction='none').sum(1)*motion_valid).sum()/motion_valid.sum().clamp_min(1)
    grad=field*0
    for axis in (-2,-1):
        residual=out['normalized'][:,:1].diff(dim=axis)-torch.nan_to_num(y[:,:1]).diff(dim=axis)
        good=mask[:,:1].narrow(axis,0,residual.shape[axis])&mask[:,:1].narrow(axis,1,residual.shape[axis])
        grad=grad+(residual.square()*good).sum()/good.sum().clamp_min(1)
    return field+.3*route+.3*velocity+.5*core+.05*grad,dict(field=field,track=route,motion=velocity,central_pressure=core,pressure_gradient=grad)


def detach_state(s):
    return {k:v.detach() if torch.is_tensor(v) else v for k,v in s.items()}
