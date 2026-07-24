"""Export v21 tracks for the four test storms, for the map."""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8); DEVICE=torch.device("cpu"); R=111.2; KM6H=6*3600/1000.0
nb=json.load(open("colab_train_v17.ipynb"))
src="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
G={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,"DEVICE":DEVICE,"STEER_DROP":0.0,"STEER_CLIP":4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p,src,re.S).group(0),G)
Base=G["TrackFormerV17"]; KC,TC,EC=G["KIN_COLS"],G["THERMO_COLS"],G["ENV_COLS"]
_i,_j=np.meshgrid(np.arange(17)-8,np.arange(17)-8,indexing="ij")
ANN=torch.tensor(((np.hypot(_i,_j)*2.5>=3.0)&(np.hypot(_i,_j)*2.5<=8.0)).astype("float32"))
class CoT(Base):
    def __init__(self,**kw):
        super().__init__(**kw); self.flow_delta=nn.Linear(self.track_q.shape[-1],2)
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
        fd=self.flow_delta(h)
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
        return torch.cat([motion,self.int_state(hi)],-1),None
def load(ck,cls):
    sd=torch.load(ck,map_location="cpu",weights_only=False)["model"]
    sd={k[6:]:v for k,v in sd.items() if k.startswith("inner.")} or sd
    m=cls().eval(); m.load_state_dict(sd); return m
z=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
track=z["track"].astype("float32"); target=z["target"].astype("float32")
sid=z["storm_id"].astype(str); nl=z["n_leads"].astype(int); bt=z["base_time"].astype("int64")
bla=z["base_lat"].astype("float64"); blo=z["base_lon"].astype("float64")
tm=z["track_mean"].astype("float32"); ts=z["track_std"].astype("float32")
vp=np.concatenate([track[:,-1,2:4]*ts[2:4]+tm[2:4], track[:,-2,2:4]*ts[2:4]+tm[2:4]],1).astype("float32")
S20=np.clip(np.load("track_build/dlm4_int8.npz")["q"][:,:4].astype("float32")/31.75,-4,4)
SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12)
STORMS=[("2026182N09163","Bavi"),("1986228N19120","Wayne"),("2025203N20124","Co-may"),("2022239N22150","Hinnamnor")]
for tag,cks,cls in [("v21",[f"downloads/x/v21_seed{i}.pt" for i in range(5)],CoT),
                    ("v20",[f"downloads/x/v20_seed{i}.pt" for i in range(5)],Base)]:
    MS=[load(c,cls) for c in cks]; out={}
    for s,nm in STORMS:
        k=np.where((sid==s)&(nl==20))[0]; k=k[np.argsort(bt[k])]
        if not len(k): continue
        P=[]
        with torch.no_grad():
            for i in range(0,len(k),64):
                j=k[i:i+64]
                a=[torch.from_numpy(track[j]),torch.from_numpy(vp[j]),torch.from_numpy(S20[j])]
                P.append((torch.stack([m(*a)[0] for m in MS]).mean(0)*SC).numpy())
        A=np.concatenate(P); cE,cN=np.cumsum(A[...,0],1),np.cumsum(A[...,1],1)
        T=target[k]; tE,tN=np.cumsum(T[...,0],1),np.cumsum(T[...,1],1)
        lats,lons=[],[]
        for a2 in range(len(k)):
            la=bla[k[a2]]+cN[a2]/R
            lo=blo[k[a2]]+cE[a2]/(R*np.cos(np.radians((bla[k[a2]]+la)/2)))
            lats.append(np.round(la,3).tolist()); lons.append(np.round(lo,3).tolist())
        err=float(np.hypot(cE[:,19]-tE[:,19],cN[:,19]-tN[:,19]).mean())
        out[nm]={"lat":lats,"lon":lons,"base_time":bt[k].tolist(),
                 "base_lat":np.round(bla[k],3).tolist(),"base_lon":np.round(blo[k],3).tolist(),
                 "err120_mean":err,"n":int(len(k))}
        print(f"  {tag:5s} {nm:10s} {len(k):3d} fc | mean 120h {err:6.0f} km")
    os.makedirs("track_build/v21map",exist_ok=True)
    json.dump(out,open(f"track_build/v21map/{tag}_tracks.json","w"))
print("wrote track_build/v21map/")
