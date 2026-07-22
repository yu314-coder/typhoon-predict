"""v10 vs v21 on five storms: the four test storms plus Tip (1979)."""
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
def load(ck):
    sd=torch.load(ck,map_location="cpu",weights_only=False)["model"]
    m=CoT().eval(); m.load_state_dict(sd); return m
MS=[load(f"downloads/x/v21_seed{i}.pt") for i in range(5)]
SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12)
@torch.no_grad()
def tracks(track,vp,S,bt,bla,blo,tgt,k):
    P=[]
    for i in range(0,len(k),64):
        j=k[i:i+64]
        a=[torch.from_numpy(track[j]),torch.from_numpy(vp[j]),torch.from_numpy(S[j])]
        P.append((torch.stack([m(*a)[0] for m in MS]).mean(0)*SC).numpy())
    A=np.concatenate(P); cE,cN=np.cumsum(A[...,0],1),np.cumsum(A[...,1],1)
    T=tgt[k]; tE,tN=np.cumsum(T[...,0],1),np.cumsum(T[...,1],1)
    lats,lons=[],[]
    for a2 in range(len(k)):
        la=bla[k[a2]]+cN[a2]/R
        lo=blo[k[a2]]+cE[a2]/(R*np.cos(np.radians((bla[k[a2]]+la)/2)))
        lats.append(np.round(la,3).tolist()); lons.append(np.round(lo,3).tolist())
    return {"lat":lats,"lon":lons,"base_time":bt[k].tolist(),
            "base_lat":np.round(bla[k],3).tolist(),"base_lon":np.round(blo[k],3).tolist(),
            "err120_mean":float(np.hypot(cE[:,19]-tE[:,19],cN[:,19]-tN[:,19]).mean()),"n":int(len(k))}
# --- Tip ---
z=np.load("track_build/tip_fixed.npz",allow_pickle=True); o13=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
trT=z["track"].astype("float32"); tgtT=z["target"].astype("float32"); nlT=z["n_leads"].astype(int)
btT=z["base_time"].astype("int64"); blaT=z["base_lat"].astype("float64"); bloT=z["base_lon"].astype("float64")
tm=o13["track_mean"].astype("float32"); ts=o13["track_std"].astype("float32")
vpT=np.concatenate([trT[:,-1,2:4]*ts[2:4]+tm[2:4], trT[:,-2,2:4]*ts[2:4]+tm[2:4]],1).astype("float32")
ST=np.load("track_build/tip_dlm4.npy").astype("float32")
kT=np.where(nlT==20)[0]; kT=kT[np.argsort(btT[kT])]
v21={"Tip":tracks(trT,vpT,ST,btT,blaT,bloT,tgtT,kT)}
# --- the four ---
z2=o13; tr2=z2["track"].astype("float32"); tgt2=z2["target"].astype("float32")
sid=z2["storm_id"].astype(str); nl2=z2["n_leads"].astype(int); bt2=z2["base_time"].astype("int64")
bla2=z2["base_lat"].astype("float64"); blo2=z2["base_lon"].astype("float64")
vp2=np.concatenate([tr2[:,-1,2:4]*ts[2:4]+tm[2:4], tr2[:,-2,2:4]*ts[2:4]+tm[2:4]],1).astype("float32")
S2=np.clip(np.load("track_build/dlm4_int8.npz")["q"][:,:4].astype("float32")/31.75,-4,4)
for s,nm in [("2026182N09163","Bavi"),("1986228N19120","Wayne"),("2025203N20124","Co-may"),("2022239N22150","Hinnamnor")]:
    k=np.where((sid==s)&(nl2==20))[0]; k=k[np.argsort(bt2[k])]
    if len(k): v21[nm]=tracks(tr2,vp2,S2,bt2,bla2,blo2,tgt2,k)
# --- v10: merge its two existing exports ---
v10=dict(json.load(open("track_build/v10_tracks.json")))
v10["Tip"]=json.load(open("track_build/tipmap/v10_tracks.json"))["Tip"]
os.makedirs("track_build/five",exist_ok=True)
json.dump(v21,open("track_build/five/v21_tracks.json","w"))
json.dump(v10,open("track_build/five/v10_tracks.json","w"))
print(f"{'storm':11s} {'n':>4s} {'v10':>8s} {'v21':>8s} {'delta':>8s}")
for nm in ["Tip","Bavi","Wayne","Co-may","Hinnamnor"]:
    if nm in v21 and nm in v10:
        a,b=v10[nm]["err120_mean"],v21[nm]["err120_mean"]
        print(f"{nm:11s} {v21[nm]['n']:4d} {a:8.0f} {b:8.0f} {b-a:+8.0f}")
