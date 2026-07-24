"""One launch, three models: v10 (no fields), v17 (500 hPa), v21 (chain-of-thought)."""
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
z=np.load("track_build/tip_fixed.npz",allow_pickle=True)
o13=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
tr=z["track"].astype("float32"); nl=z["n_leads"].astype(int)
bt=z["base_time"].astype("int64"); bla=z["base_lat"].astype("float64"); blo=z["base_lon"].astype("float64")
tm=o13["track_mean"].astype("float32"); ts=o13["track_std"].astype("float32")
vp=np.concatenate([tr[:,-1,2:4]*ts[2:4]+tm[2:4], tr[:,-2,2:4]*ts[2:4]+tm[2:4]],1).astype("float32")
S21=np.load("track_build/tip_dlm4.npy").astype("float32")
S4=np.load("track_build/tip_steer4.npy").astype("float32")
S17=np.clip(S4/np.load("track_build/steer5_scale.npy")[:4][None,:,None,None],-4,4).astype("float32")
SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12); obs=observed_at(bt,bla,blo)
ISO="1979-10-13T06:00"; T0=int(np.datetime64(ISO,"ns").astype("int64"))
i=int(np.abs(bt-T0).argmin()); assert abs(int(bt[i])-T0)<SIX_H
la0,lo0=bla[i],blo[i]
print(f"single launch {ISO}  from {la0:.2f}N {lo0:.2f}E\n")
os.makedirs("track_build/tip3",exist_ok=True)
def emit(tag,lat,lon):
    errs,pairs=[],[]
    for L in range(20):
        vt=int(round((T0+(L+1)*SIX_H)/SIX_H))*SIX_H
        if vt in obs:
            errs.append((L+1,km(obs[vt][0],obs[vt][1],lat[L],lon[L])))
            pairs.append([[float(lat[L]),float(lon[L])],[obs[vt][0],obs[vt][1]]])
    d=dict(errs)
    rec={"lat":[[la0]+list(map(float,lat))],"lon":[[lo0]+list(map(float,lon))],
         "base_time":[int(bt[i])],"base_lat":bla.tolist(),"base_lon":blo.tolist(),
         "launch":[la0,lo0],"pairs":pairs,"n":1,"err120_mean":d.get(20,float("nan"))}
    json.dump({"Tip":rec},open(f"track_build/tip3/{tag}_tracks.json","w"))
    print(f"  {tag:5s} +24h {d.get(4,float('nan')):6.0f}  +72h {d.get(12,float('nan')):6.0f}  "
          f"+120h {d.get(20,float('nan')):6.0f}  mean {np.mean([e for _,e in errs]):6.0f} km")
@torch.no_grad()
def run(cks,S,cls):
    MS=[load(c,cls) for c in cks]
    a=[torch.from_numpy(tr[i:i+1]),torch.from_numpy(vp[i:i+1]),torch.from_numpy(S[i:i+1])]
    P=(torch.stack([m(*a)[0] for m in MS]).mean(0)*SC).numpy()[0]
    cE,cN=np.cumsum(P[:,0]),np.cumsum(P[:,1])
    lat=la0+cN/R; lon=lo0+cE/(R*np.cos(np.radians((la0+lat)/2)))
    return lat,lon
v10=json.load(open("track_build/tipmap/v10_tracks.json"))["Tip"]
k=int(np.abs(np.array(v10["base_time"],dtype="int64")-T0).argmin())
emit("v10", v10["lat"][k], v10["lon"][k])
emit("v17", *run([f"downloads/x/v17_seed{j}.pt" for j in range(5)],S17,Base))
emit("v21", *run([f"downloads/x/v21_seed{j}.pt" for j in range(5)],S21,CoT))
print("\nwrote track_build/tip3/")
