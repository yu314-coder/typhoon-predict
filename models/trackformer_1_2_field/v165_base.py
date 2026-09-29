#!/usr/bin/env python3
"""Standalone v1.2.65. Existing full archives; no web downloads or pilot fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
HOUR = 3_600_000_000_000

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_checkpoint(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python. Install a CUDA PyTorch wheel before training.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


"""Embedded in the standalone v164 trainer: coarse archive plus native MSLP patches."""
REUSE_ARCH = 'wp-coarse8-native-pressure1-v164'
REUSE_CHANNELS = ['mslp', 'hgt500', 'uwnd850', 'vwnd850', 'uwnd500', 'vwnd500', 'uwnd200', 'vwnd200']


def load_reuse(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Combined dataset missing. Run the updated Drive setup cell first: '+str(path))
    m = json.loads(path.read_text())
    if m.get('schema') != 'v164-reuse-pressure-v1' or m.get('pilot', True):
        raise ValueError('Combined non-pilot archive required; no pilot fallback')
    for name, entry in m['files'].items():
        p = path.parent/entry['path']
        print('Checking dataset file:', name, flush=True)
        if digest(p) != entry['sha256']:
            raise ValueError('Dataset checksum mismatch: '+name)
        entry['resolved'] = str(p)
    with np.load(m['files']['basin']['resolved'], allow_pickle=False) as z:
        if z['channels'].tolist() != REUSE_CHANNELS[1:]:
            raise ValueError('Coarse channel order mismatch')
        times, lat, lon = z['time'], z['lat'], z['lon']
        q = z['q']
        scale, offset = z['scale'], z['offset']
    if q.shape != (len(times),7,25,33) or not np.all(np.diff(times) == 6*HOUR):
        raise ValueError('Coarse grid/time mismatch')
    qpath = path.parent/'basin_q_verified.npy'
    # Refresh the compact worker mmap from the checksum-verified source every preflight.
    temporary = qpath.with_suffix('.pending.npy')
    np.save(temporary, q)
    os.replace(temporary, qpath)
    slp = np.load(m['files']['slp']['resolved'], mmap_mode='r')
    if slp.shape != (len(times),25,33) or not np.isfinite(slp).all():
        raise ValueError('Coarse pressure coverage invalid')
    with np.load(m['files']['track']['resolved'], allow_pickle=False) as z:
        track = {k:z[k] for k in ['base_time','storm_id','base_lat','base_lon','target','target_mask']}
        issue_intensity = z['track'][:,-1,4:6]*z['track_std'][4:6]+z['track_mean'][4:6]
    plan = json.loads(Path(m['files']['plan']['resolved']).read_text())
    if len(plan['rows']) != 1000 or plan['shape'] != [1000,29,121,121] or plan['lead_hours'] != list(range(-48,121,6)):
        raise ValueError('Expected the verified 1000-window native pressure package, not the six-sample pilot')
    pressure = np.load(m['files']['pressure']['resolved'], mmap_mode='r')
    if pressure.shape != (1000,29,121,121):
        raise ValueError('Native pressure shape mismatch')
    patches = {}
    for pi,row in enumerate(plan['rows']):
        ti = row['track_archive_row']
        if int(track['base_time'][ti]) != row['issue_ns'] or str(track['storm_id'][ti]) != row['storm_id']:
            raise ValueError('Native patch/track identity mismatch')
        patches[ti] = pi
    first_year = {}
    years = track['base_time'].astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970
    for sid,y in zip(track['storm_id'],years):
        first_year[str(sid)] = min(first_year.get(str(sid),9999),int(y))
    lookup = {(str(s),int(t)):i for i,(s,t) in enumerate(zip(track['storm_id'],track['base_time']))}
    rows = []
    for i,ns in enumerate(track['base_time']):
        sid = str(track['storm_id'][i])
        y = first_year[sid]
        split = 'train' if 2000 <= y <= 2021 else 'validation' if 2022 <= y <= 2023 else 'test' if 2024 <= y <= 2025 else None
        if split is None:
            continue
        low,high = {'train':(2000,2021),'validation':(2022,2023),'test':(2024,2025)}[split]
        bounds = np.array([ns-48*HOUR,ns+120*HOUR],dtype='int64').astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970
        ai = int(np.searchsorted(times,ns))
        center = [float(track['base_lat'][i]),float(track['base_lon'][i])%360]
        if bounds[0] < low or bounds[1] > high or ai < 8 or ai+20 >= len(times) or times[ai] != ns:
            continue
        if not np.isfinite(center).all() or not (0 < center[0] < 60 and 100 < center[1] < 180):
            continue
        previous = lookup.get((sid,int(ns-6*HOUR)))
        motion = [0.,0.]
        if previous is not None:
            old = [float(track['base_lat'][previous]),float(track['base_lon'][previous])%360]
            if np.isfinite(old).all():
                motion = [(center[1]-old[1])*111.2*np.cos(np.deg2rad(center[0])),(center[0]-old[0])*111.2]
        future = [lookup.get((sid,int(ns+j*6*HOUR))) for j in range(1,21)]
        target_centers = [[float(track['base_lat'][j]),float(track['base_lon'][j])%360] if j is not None else [0.,0.] for j in future]
        rm = [float(j is not None and np.isfinite(target_centers[k]).all()) for k,j in enumerate(future)]
        target_centers = np.nan_to_num(target_centers).tolist()
        wind,p = track['target'][i,:,2],track['target'][i,:,3]
        wm = track['target_mask'][i,:,2].astype(bool)&np.isfinite(wind)&(wind>=0)&(wind<250)
        pm = track['target_mask'][i,:,3].astype(bool)&np.isfinite(p)&(p>800)&(p<1100)
        pi = patches.get(i,-1)
        if pi >= 0 and plan['rows'][pi]['split'] != split:
            raise ValueError('Patch split mismatch')
        issue = issue_intensity[i]
        iv = [bool(np.isfinite(issue[0]) and 0<=issue[0]<250),bool(np.isfinite(issue[1]) and 800<issue[1]<1100)]
        rows.append({'split':split,'storm_id':sid,'atlas':ai,'patch':pi,'center':center,'motion':motion,
                     'issue_intensity':[float(issue[k]) if iv[k] else 0. for k in range(2)],'issue_mask':iv,
                     'target_centers':target_centers,'route_mask':rm,
                     'pressure':np.where(pm,p,0).tolist(),'pressure_mask':pm.tolist(),
                     'vmax':np.where(wm,wind,0).tolist(),'vmax_mask':wm.tolist()})
    counts = {s:sum(r['split']==s for r in rows) for s in ['train','validation','test']}
    patch_counts = {s:sum(r['split']==s and r['patch']>=0 for r in rows) for s in counts}
    if counts['train'] < 10000 or patch_counts != {'train':800,'validation':100,'test':100}:
        raise ValueError(f'Incomplete combined coverage: {counts}, native patches {patch_counts}')
    train_times = np.flatnonzero((times.astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970>=2000)&
                                (times.astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970<=2021))
    sums,squares = np.zeros(8),np.zeros(8)
    n = 0
    for start in range(0,len(train_times),256):
        ii = train_times[start:start+256]
        b = np.concatenate([np.asarray(slp[ii],dtype='float32')[:,None],q[ii].astype('float32')*scale[None,:,None,None]+offset[None,:,None,None]],1).astype('float64')
        if not np.isfinite(b).all():
            raise ValueError('Nonfinite coarse training fields')
        sums += b.sum((0,2,3)); squares += (b*b).sum((0,2,3)); n += len(ii)*25*33
    mean = sums/n
    std = np.maximum(np.sqrt(np.maximum(squares/n-mean*mean,0)),1e-4)
    m.update(rows=rows, plan=plan, qpath=str(qpath), scale=scale.tolist(), offset=offset.tolist(),
             channels=REUSE_CHANNELS,units=['hPa','m','m/s','m/s','m/s','m/s','m/s','m/s'],
             global_lat=lat.tolist(),global_lon=lon.tolist(),regional_lat=list(np.linspace(15,-15,121)),
             regional_lon=list(np.linspace(-15,15,121)),normalization={'mean':mean.tolist(),'std':std.tolist(),'fit_split':'train','end_year':2021},
             identity=digest(path),counts=counts,patch_counts=patch_counts)
    print(json.dumps({'all_archive_windows':counts,'native_pressure_windows':patch_counts,'architecture':REUSE_ARCH}),flush=True)
    return m


class ReuseDataset(Dataset):
    def __init__(self,m,split,training=False):
        self.m = m
        self.rows = [r for r in m['rows'] if r['split']==split]
        self.arrays = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        m,row = self.m,self.rows[i]
        if self.arrays is None:
            self.arrays = {k:np.load(m['files'][k]['resolved'],mmap_mode='r') for k in ['slp','pressure']}
            self.arrays['q'] = np.load(m['qpath'],mmap_mode='r')
            with np.load(m['files']['geography']['resolved'],allow_pickle=False) as geo:
                self.arrays['land'] = geo['land_fraction']
                self.arrays['height'] = geo['elevation_m']/8000
        ai = row['atlas']
        ii = slice(ai-8,ai+21)
        coarse = np.concatenate([np.array(self.arrays['slp'][ii],dtype='float32')[:,None],
              np.array(self.arrays['q'][ii],dtype='float32')*np.array(m['scale'],dtype='float32')[None,:,None,None]+np.array(m['offset'],dtype='float32')[None,:,None,None]],1)
        mean,std = np.array(m['normalization']['mean'],dtype='float32'),np.array(m['normalization']['std'],dtype='float32')
        coarse = (coarse-mean[None,:,None,None])/std[None,:,None,None]
        pi = row['patch']
        anchor = m['plan']['rows'][pi] if pi>=0 else {'center_lat':round(row['center'][0]*4)/4,'center_lon':round(row['center'][1]*4)/4}
        lat = anchor['center_lat']+np.linspace(15,-15,121,dtype='float32')
        lon = anchor['center_lon']+np.linspace(-15,15,121,dtype='float32')
        yy,xx = np.meshgrid(lat,lon,indexing='ij')
        y = np.rint((90-yy)*4).astype(int); x = np.rint(xx*4).astype(int)%1440
        rs = np.stack([yy/90,xx/180-1,self.arrays['land'][y,x],self.arrays['height'][y,x]]).astype('float32')
        cy,cx = np.meshgrid(m['global_lat'],m['global_lon'],indexing='ij')
        iy = np.rint((90-cy)*4).astype(int); ix = np.rint(cx*4).astype(int)%1440
        gs = np.stack([cy/90,cx/180-1,self.arrays['land'][iy,ix],self.arrays['height'][iy,ix]]).astype('float32')
        detail = np.zeros((29,1,121,121),dtype='float32')
        if pi>=0:
            detail[:,0] = (np.array(self.arrays['pressure'][pi],dtype='float32')-mean[0])/std[0]
            if not np.isfinite(detail).all():
                raise ValueError('Native pressure contains nonfinite fields')
        inputs = {'global_history':coarse[:9], 'regional_history':detail[:9],
                  'global_static':gs,'regional_static':rs,'detail_available':np.array([pi>=0],dtype='float32'),
                  'center':np.array(row['center'],dtype='float32'),'motion':np.array(row['motion'],dtype='float32'),
                  'issue_intensity':np.array(row['issue_intensity'],dtype='float32'),'issue_mask':np.array(row['issue_mask'],dtype='float32')}
        targets = {'global':coarse[9:],'global_mask':np.ones_like(coarse[9:]),
                   'regional_lat_weight':np.broadcast_to(np.cos(np.deg2rad(lat))[None,None,:,None],(20,1,121,1)).astype('float32'),
                   'regional':detail[9:],'regional_mask':np.full_like(detail[9:],float(pi>=0)),
                   'center':np.array(row['target_centers'],dtype='float32'),
                   'storm_mask':np.array(row['route_mask'],dtype='float32')}
        for k in ['pressure','pressure_mask','vmax','vmax_mask']:
            targets[k] = np.array(row[k],dtype='float32')
        return {k:torch.from_numpy(v.copy()) for k,v in inputs.items()},{k:torch.from_numpy(v.copy()) for k,v in targets.items()}


"""v165 implementation, embedded with verified archive readers by the builder."""
VERSION = '1.2.65'
ARCHITECTURE = REUSE_ARCH = 'wp-causal-transport-evolution-field-tracker-v165'
TRAIN_HORIZONS = [4, 8, 12, 20]


def sample_field(field, grid, padding='border'):
    # Geometry and interpolation stay FP32, including under CUDA autocast.
    with torch.autocast(device_type=field.device.type, enabled=False):
        return F.grid_sample(field.float(), grid.float(), align_corners=True,
                             mode='bilinear', padding_mode=padding)


def identity_grid(field):
    h, w = field.shape[-2:]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=field.device),
                            torch.linspace(-1, 1, w, device=field.device), indexing='ij')
    return torch.stack([xx, yy], -1)[None].expand(field.shape[0], -1, -1, -1)


def departure_grid(field, velocity, latitude, spacing):
    # East/north m/s -> backward departure coordinates on a north-to-south grid.
    h, w = field.shape[-2:]
    u, v = velocity.float().unbind(1)
    dx = u * 21.6 / (111.2 * latitude.float().deg2rad().cos().clamp_min(.2) * spacing)
    dy = -v * 21.6 / (111.2 * spacing)
    return identity_grid(field) - torch.stack([2*dx/(w-1), 2*dy/(h-1)], -1)


class TransportBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 5, padding=2, groups=width),
                                 nn.GroupNorm(8, width), nn.Conv2d(width, width*4, 1),
                                 nn.GELU(), nn.Conv2d(width*4, width, 1))
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), .01))

    def forward(self, x):
        return x + self.scale * self.net(x)


class EvolutionNet(nn.Module):
    def __init__(self, channels, outputs, base=72, tiny=False):
        super().__init__()
        widths = [base, base*2, base*4, base*6]
        depths = [1]*4 if tiny else [2, 2, 4, 4]
        self.stem = nn.Conv2d(channels, base, 3, padding=1)
        self.enc = nn.ModuleList(nn.Sequential(*[TransportBlock(w) for _ in range(d)])
                                 for w, d in zip(widths, depths))
        self.down = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1)
                                  for a, b in zip(widths, widths[1:]))
        self.up = nn.ModuleList(nn.Conv2d(widths[i+1], widths[i], 1) for i in (2, 1, 0))
        self.dec = nn.ModuleList(TransportBlock(widths[i]) for i in (2, 1, 0))
        self.out = nn.Conv2d(base, outputs, 1)
        self.features = nn.Conv2d(base, 32, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def block(self, module, x):
        return checkpoint(module, x, use_reentrant=False) if self.training and torch.is_grad_enabled() else module(x)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i, layer in enumerate(self.enc):
            x = self.block(layer, x)
            skips.append(x)
            if i < 3:
                x = self.down[i](x)
        for up, layer, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            x = F.interpolate(up(x), size=skip.shape[-2:], mode='bilinear', align_corners=False)+skip
            x = self.block(layer, x)
        return self.out(x).float(), self.features(x).float()


class RecurrentFeatures(nn.Module):
    def __init__(self, history_channels):
        super().__init__()
        self.encode = nn.Sequential(nn.Conv2d(history_channels, 32, 3, padding=1),
                                    nn.GELU(), TransportBlock(32))
        self.gates = nn.Conv2d(64, 64, 3, padding=1)
        self.candidate = nn.Conv2d(64, 32, 3, padding=1)

    def forward(self, old, features):
        reset, update = torch.sigmoid(self.gates(torch.cat([old, features], 1))).chunk(2, 1)
        proposal = torch.tanh(self.candidate(torch.cat([old*reset, features], 1)))
        return ((1-update)*old + update*proposal).float()


def track_field(pressure, lat, lon, prior, domain=None):
    """Associate a local pressure minimum; no independent learned route head."""
    lat0, lon0 = prior[:, 0, None, None], prior[:, 1, None, None]
    distance2 = ((lat-lat0)*111.2).square()+((lon-lon0)*111.2*lat0.deg2rad().cos()).square()
    candidate = distance2 <= 650**2
    if domain is not None:
        candidate = candidate & domain
    score = -pressure/3 - distance2/(2*650**2)
    peak = score.masked_fill(~candidate, -1e9).flatten(1).argmax(1)
    h, w = pressure.shape[-2:]
    y, x = torch.arange(h, device=pressure.device)[None, :, None], torch.arange(w, device=pressure.device)[None, None, :]
    local = ((y-(peak//w)[:, None, None]).abs() <= 1) & ((x-(peak%w)[:, None, None]).abs() <= 1) & candidate
    pdf = torch.softmax(score.masked_fill(~local, -1e9).flatten(1), 1).reshape_as(score)
    center = torch.stack([(pdf*lat).sum((1, 2)), (pdf*lon).sum((1, 2))], 1)
    grid = torch.stack([2*(center[:, 1]-lon[:, 0, 0])/(lon[:, 0, -1]-lon[:, 0, 0])-1,
                        2*(center[:, 0]-lat[:, 0, 0])/(lat[:, -1, 0]-lat[:, 0, 0])-1], -1)[:, None, None]
    central = sample_field(pressure[:, None], grid)[:, 0, 0, 0]
    annulus = (distance2 >= 300**2) & (distance2 <= 650**2)
    background = (pressure*annulus).sum((1, 2))/annulus.sum((1, 2)).clamp_min(1)
    interior = (peak//w > 0) & (peak//w < h-1) & (peak%w > 0) & (peak%w < w-1)
    valid = candidate.flatten(1).any(1) & interior & (background-central > 1.)
    return center, central, valid, background-central


class TransportForecaster(nn.Module):
    def __init__(self, manifest, tiny=False):
        super().__init__()
        base = 8 if tiny else 72
        self.coarse_memory, self.fine_memory = RecurrentFeatures(72), RecurrentFeatures(9)
        self.coarse = EvolutionNet(8+32+4, 8+2, base, tiny)
        self.fine = EvolutionNet(1+8+32+4+1, 1+2, base, tiny)
        self.steering_logits = nn.Parameter(torch.tensor([.4, .4, .2]).log())
        # Wind intensity is explicitly a scalar estimate, not a resolved 10-m wind map.
        self.wind = nn.Sequential(nn.Linear(32+2, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer('mean', torch.tensor(manifest['normalization']['mean'], dtype=torch.float32)[None, :, None, None])
        self.register_buffer('std', torch.tensor(manifest['normalization']['std'], dtype=torch.float32)[None, :, None, None])
        yy, xx = np.meshgrid(manifest['global_lat'], manifest['global_lon'], indexing='ij')
        self.register_buffer('lat', torch.tensor(yy, dtype=torch.float32)[None])
        self.register_buffer('lon', torch.tensor(xx, dtype=torch.float32)[None])

    def initial(self, inputs):
        required = {'global_history', 'regional_history', 'global_static', 'regional_static',
                    'detail_available', 'center', 'motion', 'issue_intensity', 'issue_mask'}
        if set(inputs) != required:
            raise ValueError('Only declared issue-time and historical inputs are allowed')
        gh, rh, rs = inputs['global_history'], inputs['regional_history'], inputs['regional_static']
        grid = torch.stack([((rs[:, 1]+1)*180-100)/40-1, 1-rs[:, 0]*3], -1)
        parent = sample_field(gh[:, -1, :1], grid)
        available = inputs['detail_available'][:, :, None, None]
        r = torch.where(available.bool(), rh[:, -1], parent)
        return dict(g=gh[:, -1], r=r, gh=self.coarse_memory.encode(gh.flatten(1, 2)).float(),
                    rh=self.fine_memory.encode(rh.flatten(1, 2)).float(), gs=inputs['global_static'], rs=rs,
                    grid=grid, available=available, anchor=(r-parent)*available,
                    coordinates=identity_grid(r).permute(0, 3, 1, 2), source=torch.zeros_like(r),
                    center=inputs['center'], motion=inputs['motion'], issue=inputs['issue_intensity'],
                    issue_mask=inputs['issue_mask'])

    def steering(self, g):
        physical = g.float()*self.std+self.mean
        winds = physical[:, 2:].reshape(g.shape[0], 3, 2, *g.shape[-2:])
        return (winds*torch.softmax(self.steering_logits, 0)[None, :, None, None, None]).sum(1)

    def step(self, s):
        delta, features = self.coarse(torch.cat([s['g'], s['gh'], s['gs']], 1))
        velocity = self.steering(s['g']) + 15*torch.tanh(delta[:, 8:])
        departure = departure_grid(s['g'], velocity, self.lat, 2.5)
        g = sample_field(s['g'], departure) + .2*torch.tanh(delta[:, :8])
        gh = self.coarse_memory(sample_field(s['gh'], departure), features)
        parent = sample_field(g, s['grid'])
        fine_velocity = sample_field(velocity, s['grid'])
        r, rh, coordinates, source = s['r'], s['rh'], s['coordinates'], s['source']
        parent_valid = (s['grid'].abs() <= 1).all(-1)[:, None].float()
        if bool(s['available'].any()):
            rd, rf = self.fine(torch.cat([r, parent, rh, s['rs'], parent_valid], 1))
            fine_velocity = fine_velocity + 10*torch.tanh(rd[:, 1:])
            dep = departure_grid(r, fine_velocity, s['rs'][:, 0]*90, .25)
            # Compose material coordinates and sample the ORIGINAL anomaly once per lead.
            # This avoids repeatedly interpolating away its core. Learned source terms
            # can strengthen, weaken or deform it; survival is not hard-coded.
            displacement = coordinates-identity_grid(r).permute(0, 3, 1, 2)
            coordinates = dep.permute(0, 3, 1, 2)+sample_field(displacement, dep)
            source = sample_field(source, dep, 'zeros') + .2*torch.tanh(rd[:, :1])
            anomaly = sample_field(s['anchor'], coordinates.permute(0, 2, 3, 1), 'zeros')+source
            proposed = parent[:, :1]+anomaly*parent_valid
            r = torch.where(s['available'].bool(), proposed, parent[:, :1])
            rh = self.fine_memory(sample_field(rh, dep), rf)
        else:
            r = parent[:, :1]
        # Association prior uses only predicted circulation, never a future center.
        at = torch.stack([(s['center'][:, 1]-100)/40-1, 1-s['center'][:, 0]/30], -1)[:, None, None]
        steering = sample_field(velocity, at)[:, :, 0, 0]
        displacement_km = .7*steering*21.6 + .3*s['motion']
        prior = s['center']+torch.stack([displacement_km[:, 1]/111.2,
                    displacement_km[:, 0]/(111.2*s['center'][:, 0].deg2rad().cos().clamp_min(.2))], 1)
        gp, rp = (g*self.std+self.mean)[:, 0], (r*self.std[:, :1]+self.mean[:, :1])[:, 0]
        coarse_center, coarse_p, coarse_ok, coarse_depth = track_field(gp, self.lat, self.lon, prior)
        fine_lat, fine_lon = s['rs'][:, 0]*90, (s['rs'][:, 1]+1)*180
        fine_center, fine_p, fine_ok, fine_depth = track_field(rp, fine_lat, fine_lon, prior, parent_valid[:, 0].bool())
        inside = (prior[:, 0] < fine_lat[:, 0, 0]) & (prior[:, 0] > fine_lat[:, -1, 0]) & \
                 (prior[:, 1] > fine_lon[:, 0, 0]) & (prior[:, 1] < fine_lon[:, 0, -1])
        inside = inside & (prior[:, 0] >= 0) & (prior[:, 0] <= 60) & (prior[:, 1] >= 100) & (prior[:, 1] <= 180)
        use_fine = s['available'][:, 0, 0, 0].bool() & inside
        center = torch.where(use_fine[:, None], fine_center, coarse_center)
        pressure = torch.where(use_fine, fine_p, coarse_p)
        ok = torch.where(use_fine, fine_ok, coarse_ok)
        wind_features = torch.cat([gh.mean((2, 3)), s['issue'][:, :1]/100, s['issue_mask'][:, :1]], 1)
        vmax = F.softplus(self.wind(wind_features)[:, 0])*30
        motion = torch.stack([(center[:, 1]-s['center'][:, 1])*111.2*((center[:, 0]+s['center'][:, 0])/2).deg2rad().cos(),
                              (center[:, 0]-s['center'][:, 0])*111.2], 1)
        out = dict(global_=g, regional=r, center=center, pressure=pressure, grid_pressure=pressure,
                   vmax=vmax, track_valid=ok, native_route=use_fine, regional_valid=parent_valid.bool(),
                   pressure_depth=torch.where(use_fine, fine_depth, coarse_depth),
                   global_velocity=velocity, regional_velocity=fine_velocity)
        out['global'] = out.pop('global_')
        return dict(s, g=g, r=r, gh=gh, rh=rh, coordinates=coordinates, source=source,
                    center=center, motion=motion), out

    def forward(self, inputs, steps=20):
        state, outputs = self.initial(inputs), []
        for _ in range(steps):
            state, output = self.step(state)
            outputs.append(output)
        return {k: torch.stack([o[k] for o in outputs], 1) for k in outputs[0]}


def weighted_mean(value, weight):
    return (value*weight).sum()/weight.expand_as(value).sum().clamp_min(1)


def objective165(output, target, inputs, manifest):
    terms = {}
    for key in ('global', 'regional'):
        pred, truth, mask = output[key].float(), target[key].float(), target[key+'_mask'].float()
        if key == 'regional':
            mask = mask*output['regional_valid']
        latitude = inputs['global_static' if key == 'global' else 'regional_static'][:, :1]*90
        area = latitude.deg2rad().cos().clamp_min(0)*mask
        terms[key] = weighted_mean(F.smooth_l1_loss(pred, truth, reduction='none'), area)
        for axis in (-2, -1):
            a, b = pred.diff(dim=axis), truth.diff(dim=axis)
            w = area.narrow(axis, 0, a.shape[axis])
            terms[key] = terms[key]+.2*weighted_mean((a-b).abs(), w)
    # Labels localize supervision only. They are never passed to initial()/step().
    lat, lon = inputs['regional_static'][:, 0]*90, (inputs['regional_static'][:, 1]+1)*180
    ct = target['center']
    dist2 = ((lat-ct[:, 0, None, None])*111.2).square()+ \
            ((lon-ct[:, 1, None, None])*111.2*ct[:, 0, None, None].deg2rad().cos()).square()
    region = (dist2 < 600**2)[:, None]*target['regional_mask']*output['regional_valid']*target['storm_mask'][:, None, None, None]
    p, truth = output['regional'].float(), target['regional'].float()
    terms['storm_field'] = weighted_mean((p-truth).abs(), region)
    # Compare resolved native pressure minima against the SAME resolved target field,
    # not subgrid best-track pressure (e.g. Tip's 875 hPa is absent in this archive).
    local = region.flatten(1).bool()
    pmin = p.flatten(1).masked_fill(~local, 1e4).min(1).values
    tmin = truth.flatten(1).masked_fill(~local, 1e4).min(1).values
    terms['resolved_core'] = weighted_mean((pmin-tmin).abs(), local.any(1).float())
    # Radial-band pressure contrasts retain size/asymmetry information without
    # imposing an idealized vortex or forcing a storm to survive.
    terms['radial_structure'] = 0
    for low, high in ((0, 150), (150, 300), (300, 600)):
        band = ((dist2 >= low**2) & (dist2 < high**2))[:, None]*region
        den = band.sum((1, 2, 3)).clamp_min(1)
        contrast = ((p-truth)*band).sum((1, 2, 3))/den
        terms['radial_structure'] = terms['radial_structure']+weighted_mean(contrast.abs(), (den > 1).float())/3
    displacement = (output['center']-ct)*torch.stack([torch.ones_like(ct[:, 0])*111.2,
                                  111.2*ct[:, 0].deg2rad().cos()], 1)
    terms['route'] = weighted_mean(F.smooth_l1_loss(displacement/100, torch.zeros_like(displacement), reduction='none'),
                                   target['storm_mask'][:, None])
    terms['wind'] = weighted_mean(F.smooth_l1_loss(output['vmax']/30, target['vmax']/30, reduction='none'), target['vmax_mask'])
    terms['flow_smooth'] = sum(v.diff(dim=a).abs().mean()/30 for v in
                    [output['global_velocity'], output['regional_velocity']] for a in [-2, -1])
    loss = terms['global']+terms['regional']+.5*terms['storm_field']+.2*terms['resolved_core']+ \
           .1*terms['radial_structure']+.1*terms['route']+.05*terms['wind']+.002*terms['flow_smooth']
    return loss, terms


def training_indices(rows, epoch):
    """Every coarse window once, plus storm-balanced native replays (~25% native)."""
    rng = np.random.default_rng(165000+epoch)
    indices = list(range(len(rows)))
    groups = {}
    for i, row in enumerate(rows):
        if row['patch'] >= 0:
            groups.setdefault(row['storm_id'], []).append(i)
    native = sum(len(v) for v in groups.values())
    extra = max(0, math.ceil((.25*len(rows)-native)/.75))
    keys = sorted(groups)
    for j in range(extra):
        # Round-robin storms, random window within each storm.
        indices.append(int(rng.choice(groups[keys[j % len(keys)]])))
    rng.shuffle(indices)
    return indices


def save_training_state(path, model, optimizer, scaler, manifest, epoch, history, best):
    atomic_checkpoint(path, dict(version=VERSION, architecture=ARCHITECTURE, dataset_sha256=manifest['identity'],
        data_contract={k:manifest[k] for k in ('channels', 'units', 'global_lat', 'global_lon', 'regional_lat', 'regional_lon', 'normalization')},
        epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
        history=history, best_value=best, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))


def restore_training_state(path, model, optimizer, scaler, manifest):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['architecture'] != ARCHITECTURE or saved['dataset_sha256'] != manifest['identity']:
        raise ValueError('Resume requires the identical v165 architecture and dataset; v164 is not compatible')
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler'])
    random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng']); torch.set_rng_state(saved['torch_rng'])
    if torch.cuda.is_available() and saved.get('cuda_rng') is not None:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['epoch']+1, saved['history'], saved['best_value']


def train165(model, m, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and bool(args.amp))
    last, best_path = args.output/'last.pt', args.output/'best.pt'
    start, history, best = 1, [], float('inf')
    if last.is_file():
        start, history, best = restore_training_state(last, model, optimizer, scaler, m)
        print('Resuming completed epoch', start-1, flush=True)
    atomic_json(args.output/'manifest.json', dict(version=VERSION, architecture=ARCHITECTURE,
        parameters=sum(p.numel() for p in model.parameters()), dataset_sha256=m['identity'],
        script_sha256=digest(__file__), samples=m['counts'], native_samples=m['patch_counts'],
        future_weather_as_inputs=False, official_forecasts_as_inputs=False, input_hours=list(range(-48, 1, 6)),
        forecast_hours=list(range(6, 121, 6)), pressure='resolved field at the reported field-derived center',
        native_sampling='all coarse windows plus storm-balanced native replays to 25 percent',
        validation='fixed 120h joint loss, balanced native/coarse strata; no test-set selection',
        physical_microbatch=1, effective_batch=args.batch, tbptt_steps=4,
        limitations=['regional WP, not global', 'native MSLP only on 1000 windows',
                     'no fine upper-air variables', 'not an exact conservation solver', 'Tip is a development case']))
    for epoch in range(start, args.epochs+1):
        horizon = TRAIN_HORIZONS[min(epoch-1, len(TRAIN_HORIZONS)-1)]
        metrics = {}
        for split in ('train', 'validation'):
            training = split == 'train'
            model.train(training)
            data = ReuseDataset(m, split)
            sampler = training_indices(data.rows, epoch) if training else None
            loader = DataLoader(data, batch_size=1, sampler=sampler, num_workers=min(args.workers, 2),
                                **({'prefetch_factor':1} if args.workers else {}))
            groups, failures = {False:[], True:[]}, []
            optimizer.zero_grad(set_to_none=True)
            began = time.time()
            for index, (inputs, targets) in enumerate(loader, 1):
                inputs = {k:v.to(device) for k, v in inputs.items()}
                steps = horizon if training else 20
                total, block_loss = 0., 0
                with torch.set_grad_enabled(training):
                    state = model.initial(inputs)
                    for step in range(steps):
                        with torch.autocast('cuda', dtype=torch.float16) if scaler.is_enabled() else nullcontext():
                            state, output = model.step(state)
                        frame = {k:v[:, step].to(device) for k, v in targets.items()}
                        loss, terms = objective165(output, frame, inputs, m)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f'Nonfinite loss epoch={epoch} sample={index} step={step}; no checkpoint advanced')
                        total += float(loss.detach())/steps
                        failures.append(float((~output['track_valid']).float().mean()))
                        if training:
                            block_loss = block_loss+loss/(steps*args.batch)
                            if (step+1) % 4 == 0 or step+1 == steps:
                                scaler.scale(block_loss).backward()
                                state = {k:v.detach() for k, v in state.items()}
                                block_loss = 0
                    if training and (index % args.batch == 0 or index == len(loader)):
                        scaler.unscale_(optimizer)
                        remainder = index % args.batch
                        if index == len(loader) and remainder:
                            for parameter in model.parameters():
                                if parameter.grad is not None:
                                    parameter.grad.mul_(args.batch/remainder)
                        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        if not torch.isfinite(norm):
                            raise RuntimeError('Nonfinite gradient; checkpoint unchanged. Retry with AMP=0.')
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                groups[bool(inputs['detail_available'][0, 0])].append(total)
                if index == 1 or index % 10 == 0:
                    print(f'{split} epoch={epoch}/{args.epochs} sample={index}/{len(loader)} horizon={steps} loss={total:.5f} elapsed_s={time.time()-began:.0f}', flush=True)
            # Keep rare native examples visible in selection, rather than 5.7% dilution.
            metrics[split+'_native'] = float(np.mean(groups[True]))
            metrics[split+'_coarse'] = float(np.mean(groups[False]))
            metrics[split] = (metrics[split+'_native']+metrics[split+'_coarse'])/2
            metrics[split+'_association_failure_fraction'] = float(np.mean(failures))
        promote = metrics['validation'] < best
        if promote:
            best = metrics['validation']
        history.append(dict(epoch=epoch, horizon=horizon, **metrics))
        save_training_state(last, model, optimizer, scaler, m, epoch, history, best)
        save_training_state(args.output/f'epoch_{epoch:03d}.pt', model, optimizer, scaler, m, epoch, history, best)
        if promote:
            save_training_state(best_path, model, optimizer, scaler, m, epoch, history, best)
        atomic_json(args.output/'history.json', history)
        print(f'Saved epoch {epoch} directly to {args.output}; best={promote}', flush=True)
    print('Training complete. Test set untouched; best.pt selected by full-120h validation.', flush=True)


def main165():
    parser = argparse.ArgumentParser(description='v1.2.65 causal pressure transport and evolution; Drive-only data')
    parser.add_argument('batch', type=int, nargs='?', default=10)
    parser.add_argument('workers', type=int, nargs='?', default=3)
    parser.add_argument('amp', type=int, nargs='?', default=1, choices=[0, 1])
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/v164_reuse/manifest.json')
    persistent = Path('/content/drive/MyDrive/typhoon_predict') if Path('/content').is_dir() else ROOT
    parser.add_argument('--output', type=Path, default=persistent/'training/v165_transport_pressure')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or args.epochs < 1:
        parser.error('Invalid batch, workers or epochs')
    if Path('/content').is_dir() and (not (persistent.parent).is_dir() or not args.output.resolve().is_relative_to(persistent.resolve())):
        raise ValueError('Mount Google Drive first; Colab checkpoints must persist under MyDrive/typhoon_predict')
    manifest = load_reuse(args.manifest)
    for split in ('train', 'validation', 'test'):
        data = ReuseDataset(manifest, split)
        data[0]; data[next(i for i, r in enumerate(data.rows) if r['patch'] >= 0)]
    if args.preflight:
        print('PASS full existing archive checksums, splits and native samples. No training started.', flush=True)
        return
    device = device_for(args.device)
    random.seed(165); np.random.seed(165); torch.manual_seed(165)
    model = TransportForecaster(manifest).to(device)
    print(json.dumps(dict(version=VERSION, architecture=ARCHITECTURE, parameters=sum(p.numel() for p in model.parameters()),
        device=str(device), samples=manifest['counts'], native_samples=manifest['patch_counts'], output=str(args.output))), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output/'training.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps(dict(pid=os.getpid(), host=socket.gethostname())).encode()); os.close(fd)
    try:
        train165(model, manifest, args, device)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main165()
