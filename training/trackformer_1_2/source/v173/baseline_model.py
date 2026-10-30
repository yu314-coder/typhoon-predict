"""Observation-initialized moving pressure state; no independent pressure head."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import v165_base as base

VERSION = '1.2.68'
ARCHITECTURE = 'moving-pressure-core-v168-prototype'


def weighted(value, mask):
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1)


class CoreForecaster(base.TransportForecaster):
    def __init__(self, manifest, tiny=False):
        super().__init__(manifest, tiny)
        del self.fine, self.fine_memory, self.wind
        self.core_net = base.EvolutionNet(1 + 8 + 32 + 4, 3, 8 if tiny else 64, tiny)
        self.core_memory = base.RecurrentFeatures(9)
        self.motion_net = nn.Sequential(nn.Linear(32 + 8 + 4, 64), nn.GELU(), nn.Linear(64, 3))
        nn.init.zeros_(self.motion_net[-1].weight)
        nn.init.zeros_(self.motion_net[-1].bias)
        # 20 km internal grid is a reconstruction, not new resolved observations.
        yy, xx = torch.meshgrid(torch.linspace(640, -640, 65), torch.linspace(-640, 640, 65), indexing='ij')
        self.register_buffer('east', xx[None])
        self.register_buffer('north', yy[None])
        radius = (xx.square() + yy.square()).sqrt()
        self.register_buffer('taper', ((640-radius)/160).clamp(0, 1)[None, None])

    def coordinates(self, center):
        lat = center[:, 0, None, None] + self.north / 111.2
        lon = center[:, 1, None, None] + self.east / (111.2 * center[:, 0, None, None].deg2rad().cos().clamp_min(.2))
        return lat, lon

    def basin_grid(self, lat, lon):
        return torch.stack([(lon-100)/40-1, 1-lat/30], -1)

    def core_grid(self, lat, lon, origin):
        east = (lon-origin[:, 1, None, None])*111.2*origin[:, 0, None, None].deg2rad().cos().clamp_min(.2)
        north = (lat-origin[:, 0, None, None])*111.2
        return torch.stack([east/640, -north/640], -1)

    def initial(self, inputs):
        required = {'global_history','regional_history','global_static','regional_static','detail_available','center','motion','issue_intensity','issue_mask'}
        if set(inputs) != required:
            raise ValueError('Only declared past and issue-time inputs are allowed')
        history = inputs['global_history']
        lat, lon = self.coordinates(inputs['center'])
        grid = self.basin_grid(lat, lon)
        regional = inputs['regional_static']
        rlat, rlon = regional[:, 0]*90, (regional[:, 1]+1)*180
        native_grid = torch.stack([2*(lon-rlon[:, :1, :1])/(rlon[:, :1, -1:]-rlon[:, :1, :1])-1,
                                   2*(lat-rlat[:, :1, :1])/(rlat[:, -1:, :1]-rlat[:, :1, :1])-1], -1)
        parent = base.sample_field(history[:, -1, :1], grid)
        available = inputs['detail_available'][:, :, None, None].bool()
        past = []
        for i in range(9):
            coarse = base.sample_field(history[:, i, :1], grid)
            native = base.sample_field(inputs['regional_history'][:, i], native_grid)
            past.append(torch.where(available, native, coarse))
        core = past[-1]
        observed = (inputs['issue_intensity'][:, 1] - self.mean[0, 0, 0, 0]) / self.std[0, 0, 0, 0]
        # Only initialization uses an explicit compact core prior. Subsequent
        # changes are learned field tendencies, not repeated scalar insertion.
        radius2 = self.east.square() + self.north.square()
        prior = torch.exp(-radius2/(2*100**2))[:, None] * self.taper
        difference = observed-core[:, 0, 32, 32]
        core = core + difference[:, None, None, None]*prior*inputs['issue_mask'][:, 1, None, None, None]
        return dict(g=history[:, -1], gh=self.coarse_memory.encode(history.flatten(1, 2)),
                    core=core, ch=self.core_memory.encode(torch.cat(past, 1)),
                    anomaly=core-parent, origin=inputs['center'], center=inputs['center'], motion=inputs['motion'],
                    vmax=inputs['issue_intensity'][:, 0], gs=inputs['global_static'], rs=regional,
                    issue=inputs['issue_intensity'], issue_mask=inputs['issue_mask'])

    def step(self, s):
        delta, features = self.coarse(torch.cat([s['g'], s['gh'], s['gs']], 1))
        velocity = self.steering(s['g']) + 15*torch.tanh(delta[:, 8:])
        departure = base.departure_grid(s['g'], velocity, self.lat, 2.5)
        g = base.sample_field(s['g'], departure) + .2*torch.tanh(delta[:, :8])
        gh = self.coarse_memory(base.sample_field(s['gh'], departure), features)
        at = self.basin_grid(s['center'][:, 0, None, None], s['center'][:, 1, None, None])
        local = base.sample_field(torch.cat([gh, g], 1), at)[:, :, 0, 0]
        conditioning = torch.cat([s['motion']/100, s['issue_mask']], 1)
        tendencies = self.motion_net(torch.cat([local, conditioning], 1))
        flow = base.sample_field(velocity, at)[:, :, 0, 0]
        motion = .7*flow*21.6 + .3*s['motion'] + 100*torch.tanh(tendencies[:, :2])
        origin = s['center'] + torch.stack([motion[:, 1]/111.2,
                    motion[:, 0]/(111.2*s['center'][:, 0].deg2rad().cos().clamp_min(.2))], 1)
        lat, lon = self.coordinates(origin)
        grid = self.basin_grid(lat, lon)
        environment = base.sample_field(g, grid)
        static = base.sample_field(s['gs'], grid)
        # Translate the moving frame without resampling away bulk storm motion.
        old_lat, old_lon = self.coordinates(s['origin'])
        old_grid = self.basin_grid(old_lat, old_lon)
        old_environment = base.sample_field(s['g'], old_grid)
        rd, rf = self.core_net(torch.cat([s['core'], old_environment, s['ch'], static], 1))
        local_flow = base.sample_field(velocity, grid) + 5*torch.tanh(rd[:, 1:])
        relative = local_flow - (motion/21.6)[:, :, None, None]
        shift = (s['center']-s['origin'])
        source_lat = old_lat + shift[:, 0, None, None] - relative[:, 1]*21.6/111.2
        source_lon = old_lon + shift[:, 1, None, None] - relative[:, 0]*21.6/(111.2*old_lat.deg2rad().cos().clamp_min(.2))
        dep = self.core_grid(source_lat, source_lon, s['origin'])
        anomaly = base.sample_field(s['anomaly'], dep, 'zeros') + .1*rd[:, :1]
        core = environment[:, :1] + anomaly*self.taper
        ch = self.core_memory(base.sample_field(s['ch'], dep, 'zeros'), rf)
        physical = core*self.std[:, :1]+self.mean[:, :1]
        # Field-based association, constrained to this storm's local neighborhood.
        score = -physical[:, 0]/2 - (self.east.square()+self.north.square())/(2*180**2)
        score = score.masked_fill((self.east.square()+self.north.square()) > 300**2, -1e9)
        pdf = score.flatten(1).softmax(1).reshape_as(score)
        center = torch.stack([(pdf*lat).sum((1,2)), (pdf*lon).sum((1,2))], 1)
        cg = self.core_grid(center[:, 0, None, None], center[:, 1, None, None], origin)
        pressure = base.sample_field(physical, cg)[:, 0, 0, 0]
        rlat, rlon = s['rs'][:, 0]*90, (s['rs'][:, 1]+1)*180
        rg = self.basin_grid(rlat, rlon)
        sampling = self.core_grid(rlat, rlon, origin)
        reconstructed = base.sample_field(g[:, :1], rg) + base.sample_field(anomaly*self.taper, sampling, 'zeros')
        # Resolved basin output remains at source resolution; reconstruction has
        # separate provenance and is never relabeled as raw coarse analysis.
        vmax = (s['vmax']+5*tendencies[:, 2]).clamp_min(0)
        valid = (grid.abs()<=1).all(-1)[:, None]
        out = dict(global_=g, regional=reconstructed, regional_valid=(rg.abs()<=1).all(-1)[:,None],
                   core=core, core_lat=lat, core_lon=lon, core_valid=valid,
                   center=center, pressure=pressure, grid_pressure=pressure, vmax=vmax,
                   track_valid=(center[:,0]>0)&(center[:,0]<60)&(center[:,1]>100)&(center[:,1]<180),
                   native_route=torch.ones_like(pressure,dtype=torch.bool),
                   pressure_tendency=pressure-base.sample_field(s['core']*self.std[:,:1]+self.mean[:,:1],
                       self.core_grid(s['center'][:,0,None,None],s['center'][:,1,None,None],s['origin']))[:,0,0,0])
        out['global']=out.pop('global_')
        return dict(s,g=g,gh=gh,core=core,ch=ch,anomaly=anomaly,origin=origin,center=center,motion=motion,vmax=vmax), out


def objective(out, target, inputs, manifest):
    terms={}
    area=inputs['global_static'][:,:1].mul(90).deg2rad().cos().clamp_min(0)
    terms['environment']=weighted(F.smooth_l1_loss(out['global'],target['global'],reduction='none'),area*target['global_mask'])
    # Reanalysis does not resolve the observed central pressure. Exclude the
    # inner 150 km from absolute reconstruction targets instead of erasing it.
    lat,lon=inputs['regional_static'][:,0]*90,(inputs['regional_static'][:,1]+1)*180
    ct=target['center']
    d2=((lat-ct[:,0,None,None])*111.2).square()+((lon-ct[:,1,None,None])*111.2*ct[:,0,None,None].deg2rad().cos()).square()
    mask=target['regional_mask']*out['regional_valid']*(d2>=150**2)[:,None]
    terms['regional']=weighted(F.smooth_l1_loss(out['regional'],target['regional'],reduction='none'),mask)
    for axis in (-2,-1):
        pred=out['regional'].diff(dim=axis);truth=target['regional'].diff(dim=axis)
        w=mask.narrow(axis,0,pred.shape[axis])*mask.narrow(axis,1,pred.shape[axis])
        terms['regional']=terms['regional']+.2*weighted((pred-truth).abs(),w)
    displacement=(out['center']-ct)*torch.stack([torch.full_like(ct[:,0],111.2),111.2*ct[:,0].deg2rad().cos()],1)
    terms['route']=weighted(F.smooth_l1_loss(displacement/100,torch.zeros_like(displacement),reduction='none'),target['storm_mask'][:,None])
    terms['pressure']=weighted(F.smooth_l1_loss(out['pressure']/20,target['pressure']/20,reduction='none'),target['pressure_mask'])
    terms['wind']=weighted(F.smooth_l1_loss(out['vmax']/30,target['vmax']/30,reduction='none'),target['vmax_mask'])
    return terms['environment']+terms['regional']+.2*terms['route']+.5*terms['pressure']+.05*terms['wind'],terms
