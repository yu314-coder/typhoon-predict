import json,re,math,glob,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
torch.set_num_threads(8)
nb=json.load(open("colab_train_v17.ipynb"));cells=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
body="\n\n".join(cells[2:7]).replace('"/content/d/steer5_int8.npz"','"track_build/dlm4_int8.npz"').replace('"/content/d/track_windows_v13.npz"','"track_build/track_windows_v13.npz"').replace('DEVICE = torch.device("cuda")','DEVICE = torch.device("cpu")')
G={"__name__":"x","torch":torch,"nn":nn,"F":F,"np":np,"os":__import__("os"),"json":json,"time":__import__("time"),"math":math}
exec(compile(body,"<nb>","exec"),G)
Base=G["TrackFormerV17"];SLP=G["SLP"];track=G["track"];target=G["target"];vpair=G["vpair"];basins=G["basins"];z=G["z"];SC=G["TARGET_SCALE"];va_idx=G["va_idx"];te_idx=G["te_idx"];nl=z["n_leads"].astype(int)
DSC=np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i,_j=np.meshgrid(np.arange(17)-8,np.arange(17)-8,indexing="ij");ANN=torch.tensor(((np.hypot(_i,_j)*2.5>=3.0)&(np.hypot(_i,_j)*2.5<=8.0)).astype("float32"));KM6H=6*3600/1000.0
g21={"Base":Base,"torch":torch,"nn":nn,"F":F,"math":math,"G":G,"ANN":ANN,"DSC":DSC,"KM6H":KM6H,"R_ROUNDS":0,"USE_FLOW":1}
exec(re.search(r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)",open("colab_v26_train.py").read(),re.S).group(0),g21);V21=g21["TrackFormerCoT"]
v28=open("colab_v28_train.py").read()
g23={"V21":V21,"torch":torch,"nn":nn,"F":F,"math":math,"G":G,"ANN":ANN,"DSC":DSC,"KM6H":KM6H,"USE_HIST":1}
exec(re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n",v28,re.S).group(0),g23)
exec(re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n",v28,re.S).group(0),g23);V23=g23["TrackFormerHist"]
sid=z["storm_id"].astype(str);bt=z["base_time"].astype("int64")
SIX=int(6*3600*1e9);key={(sid[i],int(bt[i])):i for i in range(len(sid))}
HIST=np.full((len(sid),2),-1,dtype=np.int64)
for i in range(len(sid)):
  for c,b in enumerate((2,4)): HIST[i,c]=key.get((sid[i],int(bt[i])-b*SIX),-1)
HAVE=(HIST>=0).astype("float32");HIST_S=np.where(HIST>=0,HIST,np.arange(len(sid))[:,None])
ms=[]
for p in sorted(glob.glob("downloads/x/v23_seed*.pt")):
  m=V23().eval();m.load_state_dict(torch.load(p,map_location="cpu",weights_only=False)["model"]);ms.append(m)
@torch.no_grad()
def pos(idx):
  P=[]
  for i in range(0,len(idx),128):
    j=idx[i:i+128];h=torch.from_numpy(np.concatenate([SLP[HIST_S[j,0]],SLP[HIST_S[j,1]]],1))
    a=[torch.from_numpy(track[j]),torch.from_numpy(vpair[j]),torch.from_numpy(SLP[j]),h,torch.from_numpy(HAVE[j])]
    P.append((torch.stack([m(*a)[0] for m in ms]).mean(0)[...,:2]*SC[:2]).float().numpy())
  return np.cumsum(np.concatenate(P),1)   # predicted cumulative positions [n,20,2]
full=nl==20;wp=np.isin(basins,["WP","EP"])
VA=np.array([i for i in va_idx if full[i] and wp[i]]);TE=np.array([i for i in te_idx if full[i] and wp[i]])
Pv,Pt=pos(VA),pos(TE);Ov,Ot=np.cumsum(target[VA][...,:2],1),np.cumsum(target[TE][...,:2],1)
def track_err(P,O): return float(np.sqrt(((P-O)**2).sum(-1)).mean())
def heading(P):  # predicted per-step direction
  prev=np.concatenate([np.zeros((len(P),1,2)),P[:,:-1]],1);step=P-prev
  hd=np.arctan2(step[...,1],step[...,0]);return np.cos(hd),np.sin(hd)
base_te=track_err(Pt,Ot)
# CORRECTION 1: additive per-lead along-track shift, fit on validation
uEv,uNv=heading(Pv);uEt,uNt=heading(Pt)
alv=(Ov-Pv)[...,0]*uEv+(Ov-Pv)[...,1]*uNv   # observed-minus-pred along pred heading, on VAL
shift=alv.mean(0)                            # per-lead mean along-track deficit (val)
Pt_add=Pt.copy()
Pt_add[...,0]+=shift[None,:]*uEt; Pt_add[...,1]+=shift[None,:]*uNt
add_te=track_err(Pt_add,Ot)
# CORRECTION 2: multiplicative speed scale per lead, fit on validation (scale predicted displacement)
Pprev_v=np.concatenate([np.zeros((len(Pv),1,2)),Pv[:,:-1]],1);stepv=Pv-Pprev_v
Oprev_v=np.concatenate([np.zeros((len(Ov),1,2)),Ov[:,:-1]],1);ostepv=Ov-Oprev_v
# per-lead scale = ratio of mean observed step length to mean predicted step length (val)
sc=(np.hypot(ostepv[...,0],ostepv[...,1]).mean(0))/(np.hypot(stepv[...,0],stepv[...,1]).mean(0)+1e-6)
Pprev_t=np.concatenate([np.zeros((len(Pt),1,2)),Pt[:,:-1]],1);stept=Pt-Pprev_t
stept_s=stept*sc[None,:,None];Pt_mul=np.cumsum(stept_s,1)
mul_te=track_err(Pt_mul,Ot)
print(f"v23 test track error, all-lead:        {base_te:7.2f} km")
print(f"  + per-lead along-track shift (val-fit): {add_te:7.2f} km  ({add_te-base_te:+.2f})")
print(f"  + per-lead speed scale (val-fit):       {mul_te:7.2f} km  ({mul_te-base_te:+.2f})")
print(f"per-lead speed scale (val): {np.round(sc[[0,3,7,11,19]],3).tolist()} at 6,24,48,72,120h")
print(f"seed-noise floor ~5.5 km. A gain beyond that is real.")
