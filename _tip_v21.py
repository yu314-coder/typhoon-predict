"""Tip (1979) with v21's chain-of-thought model, beside v20, v17 and v10."""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from make_rmt_tracks import observed_at, SIX_H
torch.set_num_threads(8); DEVICE=torch.device("cpu"); R=111.2; KM6H=6*3600/1000.0
def km(a1,o1,a2,o2): return math.hypot((o2-o1)*R*math.cos(math.radians((a1+a2)/2)),(a2-a1)*R)
nb=json.load(open("colab_train_v17.ipynb"))
src="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
G={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,"DEVICE":DEVICE,"STEER_DROP":0.0,"STEER_CLIP":4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p,src,re.S).group(0),G)
Base=G["TrackFormerV17"]; KC,TC,EC=G["KIN_COLS"],G["THERMO_COLS"],G["ENV_COLS"]
DSC=np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i,_j=np.meshgrid(np.arange(17)-8,np.arange(17)-8,indexing="ij")
ANN=torch.tensor(((np.hypot(_i,_j)*2.5>=3.0)&(np.hypot(_i,_j)*2.5<=8.0)).astype("float32"))
class CoT(Base):
    def __init__(self,**kw):
        super().__init__(**kw)
        self.flow_delta=nn.Linear(self.track_q.shape[-1],2)
        self.A=nn.Parameter(torch.tensor([0.76,0.91]))
    def forward(self,track,vpair,slp):
        b=track.shape[0]
        kin=self.kin_enc(self.kin_proj(track[:,:,KC])+self.kin_time)
        th=self.thermo_enc(self.thermo_proj(track[:,:,TC])+self.thermo_time)
        env=self.env_enc(self.env_proj(track[:,:,EC])+self.env_time)
        st=self.steer_cnn(slp).flatten(2).transpose(1,2)+self.steer_pos
        tq=(self.track_q+self.qpos.unsqueeze(0)).expand(b,-1,-1)
        h=self.track_dec(tq,torch.cat([kin,env,st],1))
        h=h+self.alpha.view(1,self.leads,1)*self.adapter(th.mean(1).detach()).unsqueeze(1)
        w=ANN/ANN.sum(); fd=self.flow_delta(h)
        v0,vp=vpair[:,:2],vpair[:,2:]
        s0=v0.norm(dim=1,keepdim=True).clamp(min=1e-3)
        phi0=torch.atan2(v0[:,1],v0[:,0]); dphi=phi0-torch.atan2(vp[:,1],vp[:,0])
        om=torch.atan2(torch.sin(dphi),torch.cos(dphi))
        phil=phi0.unsqueeze(1)+self.gturn.view(1,self.leads)*om.unsqueeze(1)
        sp=self.rho.view(1,self.leads)*s0
        base=torch.stack([sp*torch.cos(phil),sp*torch.sin(phil)],-1)/100.0
        motion=base+(self.A.view(1,1,2)*fd)*KM6H/100.0+self.track_res(h)
        iq=(self.int_q+self.qpos.unsqueeze(0)).expand(b,-1,-1)
        hi=self.int_dec(iq,torch.cat([th,env,kin.detach(),st.detach()],1))
        return torch.cat([motion,self.int_state(hi)],-1),None,None
def load(ck,cls):
    sd=torch.load(ck,map_location="cpu",weights_only=False)["model"]
    sd={k[6:]:v for k,v in sd.items() if k.startswith("inner.")} or sd
    m=cls().eval(); m.load_state_dict(sd); return m
z=np.load("track_build/tip_fixed.npz",allow_pickle=True)
o13=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
tr=z["track"].astype("float32"); tgt=z["target"].astype("float32"); nl=z["n_leads"].astype(int)
bt=z["base_time"].astype("int64"); bla=z["base_lat"].astype("float64"); blo=z["base_lon"].astype("float64")
tm=o13["track_mean"].astype("float32"); ts=o13["track_std"].astype("float32")
vp=np.concatenate([tr[:,-1,2:4]*ts[2:4]+tm[2:4], tr[:,-2,2:4]*ts[2:4]+tm[2:4]],1).astype("float32")
S20=np.load("track_build/tip_dlm4.npy").astype("float32")
S4=np.load("track_build/tip_steer4.npy").astype("float32")
S17=np.clip(S4/np.load("track_build/steer5_scale.npy")[:4][None,:,None,None],-4,4).astype("float32")
SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12); obs=observed_at(bt,bla,blo)
K=np.where(nl==20)[0]; K=K[np.argsort(bt[K])]
@torch.no_grad()
def full_storm(ms,S,cls):
    P=[]
    for i in range(0,len(K),64):
        j=K[i:i+64]
        a=[torch.from_numpy(tr[j]),torch.from_numpy(vp[j]),torch.from_numpy(S[j])]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0)*SC).numpy())
    A=np.concatenate(P); cE,cN=np.cumsum(A[...,0],1),np.cumsum(A[...,1],1)
    T=tgt[K]; tE,tN=np.cumsum(T[...,0],1),np.cumsum(T[...,1],1)
    return float(np.hypot(cE[:,19]-tE[:,19],cN[:,19]-tN[:,19]).mean())
M21=[load(f"downloads/x/v21_seed{i}.pt",CoT) for i in range(5)]
M20=[load(f"downloads/x/v20_seed{i}.pt",Base) for i in range(5)]
M17=[load(f"downloads/x/v17_seed{i}.pt",Base) for i in range(5)]
import numpy as _np
print("steering inputs differ:", float(_np.abs(S17-S20).max()), "max abs diff")
print("Typhoon Tip 1979 -- 105 full-horizon forecasts, mean 120 h error")
print(f"  v10   (no fields)            1250 km   [from the earlier export]")
print(f"  v17   (500 hPa steering)   {full_storm(M17,S17,Base):8.2f} km")
print(f"  v20   (deep-layer mean)    {full_storm(M20,S20,Base):8.2f} km")
print(f"  v21   (chain-of-thought)   {full_storm(M21,S20,CoT):8.2f} km")
