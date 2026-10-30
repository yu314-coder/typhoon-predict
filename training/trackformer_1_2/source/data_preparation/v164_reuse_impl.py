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


class ReuseForecaster(nn.Module):
    def __init__(self,m,tiny=False):
        super().__init__()
        self.global_time = nn.Conv2d(8*9,8,1)
        self.regional_time = nn.Conv2d(9,1,1)
        self.global_operator = FieldOperator(8,8 if tiny else 80,False,1 if tiny else 3,tiny)
        self.regional_operator = FieldOperator(1,8 if tiny else 88,False,0,tiny)
        self.parent_fusion = nn.Conv2d(1+8+1,1,1)
        self.object_score = nn.Conv2d(32+3,1,1)
        self.intensity = nn.Sequential(nn.Linear(32+3+4,64),nn.GELU(),nn.Linear(64,2))
        self.register_buffer('mean',torch.tensor(m['normalization']['mean'],dtype=torch.float32)[None,:,None,None])
        self.register_buffer('std',torch.tensor(m['normalization']['std'],dtype=torch.float32)[None,:,None,None])
        yy,xx = np.meshgrid(m['global_lat'],m['global_lon'],indexing='ij')
        self.register_buffer('yy',torch.tensor(yy,dtype=torch.float32)[None])
        self.register_buffer('xx',torch.tensor(xx,dtype=torch.float32)[None])

    def initial(self,inputs):
        required = {'global_history','regional_history','global_static','regional_static','detail_available','center','motion','issue_intensity','issue_mask'}
        if set(inputs)!=required:
            raise ValueError('Only declared causal inputs are allowed')
        gh,rh = inputs['global_history'],inputs['regional_history']
        rs = inputs['regional_static']
        yy,xx = rs[:,0]*90,(rs[:,1]+1)*180
        grid = torch.stack([(xx-100)/40-1,1-yy/30],-1)
        return {'g':gh[:,-1],'r':rh[:,-1],'gc':self.global_time(gh.flatten(1,2)),
                'rc':self.regional_time(rh.flatten(1,2)),'gs':inputs['global_static'],'rs':rs,
                'grid':grid,'available':inputs['detail_available'],'center':inputs['center'],
                'motion':inputs['motion'],'issue':inputs['issue_intensity'],'issue_mask':inputs['issue_mask']}

    def step(self,s):
        g,features = self.global_operator(s['g'],s['gc'],torch.ones_like(s['g']),s['gs'])
        valid = (s['grid'].abs()<=1).all(-1)[:,None].to(g.dtype)
        parent = F.grid_sample(g,s['grid'],align_corners=True,padding_mode='zeros')*valid
        r = s['r']
        if bool(s['available'].any()):
            context = self.parent_fusion(torch.cat([s['rc'],parent,valid],1))
            r,_ = self.regional_operator(r,context,torch.ones_like(r),s['rs'])
        physical = g.float()*self.std+self.mean
        p = physical[:,0]
        anomaly = p-F.avg_pool2d(F.pad(p[:,None],(2,2,2,2),mode='replicate'),5,1)[:,0]
        evidence = torch.cat([features.float(),anomaly[:,None]/10,physical[:,2:4]/30],1)
        lat0,lon0 = s['center'][:,0,None,None],s['center'][:,1,None,None]
        plat = lat0+s['motion'][:,1,None,None]/111.2
        plon = lon0+s['motion'][:,0,None,None]/(111.2*lat0.deg2rad().cos().clamp_min(.2))
        d2 = ((self.yy-plat)*111.2).square()+((self.xx-plon)*111.2*lat0.deg2rad().cos()).square()
        score = self.object_score(evidence)[:,0]-anomaly/10-d2/(2*600**2)
        peak = score.flatten(1).argmax(1)
        iy,ix = torch.arange(25,device=g.device)[None,:,None],torch.arange(33,device=g.device)[None,None,:]
        local = ((iy-(peak//33)[:,None,None]).abs()<=1)&((ix-(peak%33)[:,None,None]).abs()<=1)
        pdf = torch.softmax(score.masked_fill(~local,-1e9).flatten(1),1).reshape_as(score)
        center = torch.stack([(pdf*self.yy).sum((1,2)),(pdf*self.xx).sum((1,2))],1)
        pooled = (evidence*pdf[:,None]).sum((2,3))
        scalar = torch.cat([s['issue']/torch.tensor([100.,1000.],device=g.device),s['issue_mask']],1)
        intensity = self.intensity(torch.cat([pooled,scalar],1))
        grid_p = (p*pdf).sum((1,2))
        dy,dx = s['rs'][:,0]*90,(s['rs'][:,1]+1)*180
        sample = torch.stack([2*(center[:,1]-dx[:,0,0])/30-1,2*(dy[:,0,0]-center[:,0])/30-1],-1)[:,None,None]
        inside = (sample.abs()<=1).all((1,2,3)) & (s['available'][:,0]>0)
        detailed_p = F.grid_sample(r.float()*self.std[:,:1]+self.mean[:,:1],sample,align_corners=True)[:,0,0,0]
        grid_p = torch.where(inside,detailed_p,grid_p)
        pressure = grid_p-F.softplus(intensity[:,0])*5
        vmax = s['issue'][:,0]*s['issue_mask'][:,0]+intensity[:,1]*10
        meanlat = (center[:,0]+s['center'][:,0])/2
        motion = torch.stack([(center[:,1]-s['center'][:,1])*111.2*meanlat.deg2rad().cos(),(center[:,0]-s['center'][:,0])*111.2],1)
        return dict(s,g=g,r=r,center=center,motion=motion),{'global':g,'regional':r,'center':center,'pressure':pressure,'vmax':vmax,'grid_pressure':grid_p}

    def forward(self,inputs,steps=20):
        s = self.initial(inputs)
        out = []
        for _ in range(steps):
            s,o = self.step(s); out.append(o)
        return {k:torch.stack([o[k] for o in out],1) for k in out[0]}


def reuse_main():
    global ARCHITECTURE, AtmosphereDataset
    ARCHITECTURE,AtmosphereDataset = REUSE_ARCH,ReuseDataset
    p = argparse.ArgumentParser(description='v1.2.64 coarse weather + native pressure training; no pilot fallback')
    p.add_argument('batch',type=int,nargs='?',default=10)
    p.add_argument('workers',type=int,nargs='?',default=3)
    p.add_argument('amp',type=int,nargs='?',default=1,choices=[0,1])
    p.add_argument('--manifest',type=Path,default=ROOT/'data/v164_reuse/manifest.json')
    persistent = Path('/content/drive/MyDrive/typhoon_predict') if Path('/content').is_dir() else ROOT
    p.add_argument('--output',type=Path,default=persistent/'training/v164_reuse_pressure')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--preflight',action='store_true')
    args = p.parse_args()
    if args.batch<1 or args.workers<0 or args.epochs<1:
        p.error('Invalid batch/workers/epochs')
    m = load_reuse(args.manifest)
    for split in ['train','validation','test']:
        d = ReuseDataset(m,split)
        d[0]
        d[next(i for i,r in enumerate(d.rows) if r['patch']>=0)]
    if args.preflight:
        print('PASS combined non-pilot archives and native pressure data. No training started.',flush=True)
        return
    if Path('/content').is_dir() and not args.output.resolve().is_relative_to(persistent.resolve()):
        raise ValueError('Colab checkpoints must persist in typhoon_predict on Drive')
    device = device_for(args.device)
    random.seed(164); np.random.seed(164); torch.manual_seed(164)
    model = ReuseForecaster(m).to(device)
    print(json.dumps({'version':VERSION,'architecture':REUSE_ARCH,'parameters':sum(x.numel() for x in model.parameters()),'device':str(device),'windows':m['counts'],'native_windows':m['patch_counts'],'output':str(args.output)}),flush=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and bool(args.amp))
    args.output.mkdir(parents=True,exist_ok=True)
    lock = args.output/'training.lock'
    fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({'pid':os.getpid(),'host':socket.gethostname()}).encode()); os.close(fd)
    try:
        train(model,optimizer,scaler,m,args,device)
    finally:
        lock.unlink(missing_ok=True)
