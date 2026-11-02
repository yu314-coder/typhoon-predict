"""Causal six-hour learned transport and tendency; one motion field for maps/track.

This regional pilot is not a GraphCast reproduction. The learned transport is
pattern displacement in km/6h, not measured atmospheric wind. Every forecast
step consumes its own last two states; future analyses are supervision only.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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


def coordinate_grid(lat,lon,batch):
    yy,xx=torch.meshgrid(lat,lon,indexing='ij')
    return torch.stack((xx,yy),-1)[None].expand(batch,-1,-1,-1)


def normalized_grid(coords,lat,lon):
    return torch.stack((2*(coords[...,0]-lon[0])/(lon[-1]-lon[0])-1,
                        2*(coords[...,1]-lat[0])/(lat[-1]-lat[0])-1),-1)


def flow_degrees(flow,lat):
    return torch.stack((flow[:,0]/(111.2*lat.deg2rad().cos().clamp_min(.2)[None,:,None]),
                        flow[:,1]/111.2),1)


def transport(field,flow_km,lat,lon):
    """Pull back a forward pattern displacement with two inverse fixed-point steps."""
    base=coordinate_grid(lat,lon,field.shape[0])
    degrees=flow_degrees(flow_km,lat)
    source=base
    for _ in range(2):
        sampled=F.grid_sample(degrees,normalized_grid(source,lat,lon),align_corners=True,padding_mode='border')
        source=base-sampled.permute(0,2,3,1)
    return F.grid_sample(field,normalized_grid(source,lat,lon),align_corners=True,padding_mode='border')


def sample_at(field,center,lat,lon):
    coords=torch.stack((center[:,1],center[:,0]),-1)[:,None,None]
    return F.grid_sample(field,normalized_grid(coords,lat,lon),align_corners=True,padding_mode='border')[:,:,0,0]


def centered_patch(field,center,lat,lon):
    # A fixed physical 600x600 km window; no moving-window observations as input.
    d=torch.linspace(-300,300,17,device=field.device,dtype=field.dtype)
    north,east=torch.meshgrid(d,d,indexing='ij')
    coords=torch.stack((center[:,1,None,None]+east[None]/(111.2*center[:,0].deg2rad().cos().clamp_min(.2)[:,None,None]),
                        center[:,0,None,None]+north[None]/111.2),-1)
    grid=normalized_grid(coords,lat,lon)
    covered=(grid.abs()<=1).all(-1)[:,None]
    return F.grid_sample(field,grid,align_corners=True,padding_mode='border'),covered


class TrackPressure(nn.Module):
    def __init__(self,mean,scale,width=64,tendency_scale=None):
        super().__init__()
        if width<8 or width%4: raise ValueError('width must be a multiple of4, >=8')
        self.width=width;self.checkpoint_activations=True
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32).reshape(1,31,1,1))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32).reshape(1,31,1,1))
        self.register_buffer('tendency_scale',torch.tensor(tendency_scale if tendency_scale is not None else scale,
                             dtype=torch.float32).reshape(1,31,1,1))
        self.history=nn.Sequential(nn.Conv2d(62,width,3,stride=2,padding=1),nn.GroupNorm(4,width),nn.SiLU(),Block(width,width))
        self.memory=Memory(width)
        self.enc0=Block(width+31*3+7,width)
        self.enc1=Block(width,2*width);self.enc2=Block(2*width,4*width);self.enc3=Block(4*width,6*width)
        layer=nn.TransformerEncoderLayer(6*width,4,12*width,dropout=0,batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,3,enable_nested_tensor=False)
        self.dec2=Block(10*width,4*width);self.dec1=Block(6*width,2*width);self.dec0=Block(3*width,width)
        self.flow=nn.Conv2d(width,2,1)
        self.refine=Block(width+31*3+2,width)
        self.increment=nn.Conv2d(width,31,1)
        nn.init.zeros_(self.flow.weight);nn.init.zeros_(self.flow.bias)
        nn.init.zeros_(self.increment.weight);nn.init.zeros_(self.increment.bias)

    @staticmethod
    def up(x,reference):return F.interpolate(x,reference.shape[-2:],mode='bilinear',align_corners=True)

    def initial(self,history,valid,center,motion,issue_pressure,pressure_valid,lat,lon):
        if history.ndim!=5 or history.shape[1:3]!=(9,31) or valid.shape!=history.shape:
            raise ValueError('Require9x31 causal history and original masks')
        if not bool(torch.isfinite(history[valid]).all()):raise ValueError('Nonfinite valid input')
        x=torch.where(valid,(history-self.mean[:,None])/self.scale[:,None],0.)
        h=None
        for k in range(9):
            args=torch.cat((x[:,k],valid[:,k].float()),1)
            z=checkpoint(self.history,args,use_reentrant=False) if self.training and self.checkpoint_activations else self.history(args)
            if h is None:h=torch.zeros_like(z)
            h=checkpoint(self.memory,z,h,use_reentrant=False) if self.training and self.checkpoint_activations else self.memory(z,h)
        return dict(x=x[:,-1],previous=x[:,-2],h=h,mask=valid[:,-1],center=center,motion=motion,
                    issue_pressure=issue_pressure,pressure_valid=pressure_valid,lat=lat,lon=lon,step=0)

    def step(self,s):
        if not 0<=s['step']<20:raise ValueError('Only20 forecast leads are supported')
        def compute(x,previous,h,center,motion):
            lat,lon=s['lat'],s['lon'];b=x.shape[0];yy,xx=torch.meshgrid(lat,lon,indexing='ij')
            time=torch.full_like(yy,(s['step']+1)/20).expand(b,-1,-1)
            positions=torch.stack((yy.expand(b,-1,-1)/60,xx.expand(b,-1,-1)/80-1.75,
                (yy[None]-center[:,0,None,None])/20,(xx[None]-center[:,1,None,None])/20,time,
                motion[:,0,None,None].expand(b,*yy.shape)/100,motion[:,1,None,None].expand(b,*yy.shape)/100),1)
            tendency=(x-previous)*self.scale/self.tendency_scale
            a=self.enc0(torch.cat((h,self.up(x,h),self.up(tendency,h),self.up(s['mask'].float(),h),self.up(positions,h)),1))
            c=self.enc1(F.avg_pool2d(a,2,ceil_mode=True));d=self.enc2(F.avg_pool2d(c,2,ceil_mode=True))
            e=self.enc3(F.avg_pool2d(d,2,ceil_mode=True))
            tokens=F.adaptive_avg_pool2d(e,(8,10));shape=tokens.shape
            tokens=self.context(tokens.flatten(2).transpose(1,2)).transpose(1,2).reshape(shape)
            e=e+self.up(tokens,e)
            d=self.dec2(torch.cat((d,self.up(e,d)),1));c=self.dec1(torch.cat((c,self.up(d,c)),1))
            features=self.dec0(torch.cat((a,self.up(c,a)),1))
            # One learned forward displacement drives both field evolution and centre motion.
            flow=400*self.up(self.flow(features),x).tanh()
            velocity=sample_at(flow,center,lat,lon)
            next_center=center+torch.stack((velocity[:,1]/111.2,
                velocity[:,0]/(111.2*center[:,0].deg2rad().cos().clamp_min(.2))),1)
            advected=transport(x,flow,lat,lon)
            refined=self.refine(torch.cat((self.up(features,x),advected,tendency,s['mask'].float(),flow/100),1))
            nxt=advected+self.increment(refined)*(self.tendency_scale/self.scale)
            weather=nxt*self.scale+self.mean
            cp=sample_at(weather[:,:1],next_center,lat,lon)[:,0]
            covered=(next_center[:,0]>=lat[-1])&(next_center[:,0]<=lat[0])&(next_center[:,1]>=lon[0])&(next_center[:,1]<=lon[-1])
            # The recurrent memory evolves with the same forecast flow.
            h_lat=torch.linspace(float(lat[0]),float(lat[-1]),h.shape[-2],device=h.device)
            h_lon=torch.linspace(float(lon[0]),float(lon[-1]),h.shape[-1],device=h.device)
            next_h=self.memory(features,transport(h,self.up(flow,h),h_lat,h_lon))
            return nxt,next_h,next_center,cp,covered,velocity,flow
        args=(s['x'],s['previous'],s['h'],s['center'],s['motion'])
        values=checkpoint(compute,*args,use_reentrant=False) if self.training and self.checkpoint_activations else compute(*args)
        nxt,h,center,cp,covered,velocity,flow=values
        out=dict(normalized=nxt,weather=nxt*self.scale+self.mean,center=center,pressure=cp,track_valid=covered,
                 motion=velocity,flow_km=flow,previous_weather=s['x']*self.scale+self.mean)
        return dict(s,previous=s['x'],x=nxt,h=h,center=center,motion=velocity,step=s['step']+1),out


def masked_mean(values,weight):return (values*weight).sum()/weight.sum().clamp_min(1)


def loss(model,out,target,valid,center,center_valid,pressure,pressure_valid,lat,lon,
         motion=None,motion_valid=None,previous_target=None,previous_valid=None):
    y=torch.nan_to_num((target-model.mean)/model.scale)
    delta=torch.where(valid,out['normalized']-y,0.)
    weights=valid.float()*lat.deg2rad().cos()[None,None,:,None]
    field=masked_mean(delta.square(),weights);mslp=masked_mean(delta[:,:1].square(),weights[:,:1])
    yy,xx=torch.meshgrid(lat,lon,indexing='ij')
    km2=((yy[None]-center[:,0,None,None])*111.2).square()+((xx[None]-center[:,1,None,None])*111.2*center[:,0].deg2rad().cos()[:,None,None]).square()
    core_weight=torch.exp(-km2/(2*250**2))*center_valid[:,None,None]*weights[:,0]
    core=masked_mean(delta[:,0].square(),core_weight)
    # Changes are supervised separately from mean fields. A constant map cannot
    # hide behind good average pressure when observed systems are moving.
    evolution=field*0
    if previous_target is not None:
        good=valid&previous_valid
        tendency_error=((out['weather']-out['previous_weather'])-torch.nan_to_num(target-previous_target))/model.tendency_scale
        evolution=masked_mean(torch.where(good,tendency_error,0.).square(),good.float()*lat.deg2rad().cos()[None,None,:,None])
        evolution=evolution+masked_mean(torch.where(good[:,:1],tendency_error[:,:1],0.).square(),weights[:,:1]*good[:,:1])
    predicted_patch,inside=centered_patch(out['weather'][:,:1],out['center'],lat,lon)
    observed_patch,truth_inside=centered_patch(torch.nan_to_num(target[:,:1]),center,lat,lon)
    observed_valid,_=centered_patch(valid[:,:1].float(),center,lat,lon)
    patch_weight=inside&truth_inside&(observed_valid>.999)&center_valid[:,None,None,None]
    patch=masked_mean(((predicted_patch-observed_patch)/model.scale[:,:1]).square(),patch_weight)
    err=torch.stack(((out['center'][:,1]-center[:,1])*111.2*center[:,0].deg2rad().cos(),(out['center'][:,0]-center[:,0])*111.2),1)
    route=masked_mean(F.smooth_l1_loss(err/250,torch.zeros_like(err),reduction='none').sum(1),center_valid)
    velocity=field*0
    if motion is not None:
        velocity=masked_mean(F.smooth_l1_loss(out['motion']/100,motion/100,reduction='none').sum(1),motion_valid)
    gradient=field*0;flow_smooth=field*0
    for axis in (-2,-1):
        d=out['normalized'][:,:1].diff(dim=axis)-y[:,:1].diff(dim=axis)
        good=valid[:,:1].narrow(axis,0,d.shape[axis])&valid[:,:1].narrow(axis,1,d.shape[axis])
        gradient+=masked_mean(torch.where(good,d,0.).square(),good)
        flow_smooth+=(out['flow_km'].diff(dim=axis)/100).square().mean()
    objective=field+mslp+core+.5*evolution+.5*patch+.5*route+.2*velocity+.1*gradient+.002*flow_smooth
    return objective,dict(field=field,mslp=mslp,core_mslp=core,evolution=evolution,storm_patch=patch,
                          track=route,motion=velocity,pressure_gradient=gradient,flow_smooth=flow_smooth)


def detach_state(s):return {k:v.detach() if torch.is_tensor(v) else v for k,v in s.items()}
