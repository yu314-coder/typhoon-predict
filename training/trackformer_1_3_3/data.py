"""Pinned native-grid data; history inputs and future supervision are separate."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from contracts import LATITUDE,LONGITUDE,CHANNELS,effective_mask,history_times,check_times,sha256
from prepare_colab import catalog,save


class Data:
    def __init__(self,root):
        self.root=Path(root);self.catalog=catalog();self.verified={}
        ready=json.loads((self.root/'ready.json').read_text())
        if ready['catalog_sha256']!=sha256(Path(__file__).parent/'catalog.json.gz') or ready['cohort_sha256']!=sha256(self.root/'cohort.json'):
            raise ValueError('Catalog/cohort identity changed')
        self.cohort=json.loads((self.root/'cohort.json').read_text())
        a={r['storm_id'] for r in self.cohort['train']};b={r['storm_id'] for r in self.cohort['development']}
        if a&b:raise ValueError('Storm split leakage')
        self.identity=ready

    def frame(self,t):
        p=self.root/'frames'/f'{t}.npz';entry=self.catalog['weather'][str(t)]
        signature=(p.stat().st_size,p.stat().st_mtime_ns)
        with np.load(p,allow_pickle=False) as z:
            a=z['values'];raw=z['valid']
            if int(z['seconds'])!=t:raise ValueError('Frame time mismatch')
            np.testing.assert_array_equal(z['latitude'],LATITUDE);np.testing.assert_array_equal(z['longitude'],LONGITUDE)
        if a.dtype!=np.float32:raise ValueError('Source must remain FP32')
        if self.verified.get(t)!=signature:
            if hashlib.sha256(a.tobytes()).hexdigest()!=entry['values_sha256']:raise ValueError('Weather values hash changed')
            rec=json.loads(p.with_suffix('.json').read_text())
            if sha256(p)!=rec['sha256']:raise ValueError('Frame/mask archive hash changed')
            if not np.array_equal(raw,np.isfinite(a)):raise ValueError('Original finite-value mask differs')
            self.verified[t]=signature
        return np.nan_to_num(a,nan=0,posinf=0,neginf=0),effective_mask(a,raw)

    def history(self,r,device):
        ts=history_times(r['issue_seconds']);check_times(r['issue_seconds'],ts)
        pairs=[self.frame(t) for t in ts]
        ten=lambda a:torch.as_tensor(np.asarray(a),device=device)
        initial=r['initial'];pv=initial[2] is not None
        return dict(history=ten(np.stack([a for a,v in pairs]))[None],valid=ten(np.stack([v for a,v in pairs]))[None],
            center=ten([initial[:2]]).float(),motion=ten([r['motion']]).float(),
            issue_pressure=ten([initial[2] if pv else 1010.]).float(),pressure_valid=ten([pv]),
            lat=ten(LATITUDE),lon=ten(LONGITUDE))

    def label(self,r,k,device):
        if not 0<=k<20:raise ValueError('Lead must be +6..+120h')
        a,v=self.frame(r['issue_seconds']+(k+1)*21600);label=r['targets'][k]
        cv=label is not None and 0<=label[0]<=60 and 100<=label[1]<=180
        pv=cv and label[2] is not None
        previous=r['initial'] if k==0 else r['targets'][k-1]
        mv=cv and previous is not None
        motion=([111.2*np.cos(np.deg2rad(previous[0]))*(label[1]-previous[1]),111.2*(label[0]-previous[0])]
                if mv else [0.,0.])
        return dict(target=torch.as_tensor(a,device=device)[None],valid=torch.as_tensor(v,device=device)[None],
            center=torch.tensor([label[:2] if cv else [0.,100.]],device=device,dtype=torch.float32),
            center_valid=torch.tensor([cv],device=device),pressure=torch.tensor([label[2] if pv else 0.],device=device),
            pressure_valid=torch.tensor([pv],device=device),motion=torch.tensor([motion],device=device,dtype=torch.float32),
            motion_valid=torch.tensor([mv],device=device))

    def normalizer(self,path):
        path=Path(path)
        times=sorted({t for r in self.cohort['train'] for t in history_times(r['issue_seconds'])})
        # Deterministic TRAIN-only fit; every history frame, spatial stride4.
        if path.exists():
            result=json.loads(path.read_text())
            if result['times']!=times or result['cohort_sha256']!=self.identity['cohort_sha256']:raise ValueError('Normalizer cohort mismatch')
            return result
        sums=np.zeros(31);sq=sums.copy();count=sums.copy()
        for i,t in enumerate(times):
            a,v=self.frame(t);a=a[:,::4,::4].astype('f8');v=v[:,::4,::4]
            sums+=(a*v).sum((1,2));sq+=(a*a*v).sum((1,2));count+=v.sum((1,2))
            if (i+1)%100==0:print(json.dumps(dict(normalizer_frames=i+1,total=len(times))),flush=True)
        if np.any(count==0):raise ValueError('No TRAIN observations for a channel')
        mean=sums/count;scale=np.sqrt(np.maximum(sq/count-mean**2,0))
        floor=np.array([1,1,1,1,1,1]+[v for _ in range(5) for v in (1,1,1,1e-4,100)])
        result=dict(mean=mean.tolist(),scale=np.maximum(scale,floor).tolist(),counts=count.tolist(),times=times,
            cohort_sha256=self.identity['cohort_sha256'],fit='TRAIN history only; spatial stride4',channels=CHANNELS)
        save(path,result);return result
