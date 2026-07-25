import json,re,math,glob,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
torch.set_num_threads(8)
nb=json.load(open("colab_train_v17.ipynb"));cells=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
body="\n\n".join(cells[2:7]).replace('"/content/d/steer5_int8.npz"','"track_build/dlm4_int8.npz"').replace('"/content/d/track_windows_v13.npz"','"track_build/track_windows_v13.npz"').replace('DEVICE = torch.device("cuda")','DEVICE = torch.device("cpu")')
G={"__name__":"x","torch":torch,"nn":nn,"F":F,"np":np,"os":__import__("os"),"json":json,"time":__import__("time"),"math":math}
exec(compile(body,"<nb>","exec"),G)
Base=G["TrackFormerV17"];SLP=G["SLP"];track=G["track"];target=G["target"];vpair=G["vpair"];basins=G["basins"];z=G["z"];SC=G["TARGET_SCALE"];te_idx=G["te_idx"];nl=z["n_leads"].astype(int)
tmean=G["tmean"];tstd=G["tstd"];TM=torch.tensor(tmean);TS=torch.tensor(tstd)
DSC=np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i,_j=np.meshgrid(np.arange(17)-8,np.arange(17)-8,indexing="ij");ANN=torch.tensor(((np.hypot(_i,_j)*2.5>=3.0)&(np.hypot(_i,_j)*2.5<=8.0)).astype("float32"));KM6H=6*3600/1000.0
g21={"Base":Base,"torch":torch,"nn":nn,"F":F,"math":math,"G":G,"ANN":ANN,"DSC":DSC,"KM6H":KM6H,"R_ROUNDS":0,"USE_FLOW":1}
exec(re.search(r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)",open("colab_v26_train.py").read(),re.S).group(0),g21);V21=g21["TrackFormerCoT"]
v28=open("colab_v28_train.py").read()
g23={"V21":V21,"torch":torch,"nn":nn,"F":F,"math":math,"G":G,"ANN":ANN,"DSC":DSC,"KM6H":KM6H,"USE_HIST":1}
exec(re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n",v28,re.S).group(0),g23)
exec(re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n",v28,re.S).group(0),g23);V23=g23["TrackFormerHist"]
src=open("colab_v34_train.py").read()
gd={"V23":V23,"torch":torch,"nn":nn,"F":F,"math":math,"TM":TM,"TS":TS,"A_MAX":0.65,"USE_DRIFT":0}
exec(re.search(r"class MeridionalDrift\(nn\.Module\):.*?return sign\[:, None\] \* self\.a_max \* mag",src,re.S).group(0),gd)
exec(re.search(r"class TrackFormerDrift\(V23\):.*?return s, ls, fp\n",src,re.S).group(0),gd);V28=gd["TrackFormerDrift"]
sid=z["storm_id"].astype(str);bt=z["base_time"].astype("int64")
SIX=int(6*3600*1e9);key={(sid[i],int(bt[i])):i for i in range(len(sid))}
HIST=np.full((len(sid),2),-1,dtype=np.int64)
for i in range(len(sid)):
  for c,b in enumerate((2,4)): HIST[i,c]=key.get((sid[i],int(bt[i])-b*SIX),-1)
HAVE=(HIST>=0).astype("float32");HIST_S=np.where(HIST>=0,HIST,np.arange(len(sid))[:,None])
def ld(paths,cls):
  ms=[];
  for p in paths:
    m=cls().eval();m.load_state_dict(torch.load(p,map_location="cpu",weights_only=False)["model"]);ms.append(m)
  return ms
MSv23=ld(sorted(glob.glob("downloads/x/v23_seed*.pt")),V23)
MSabl=ld(sorted(glob.glob("downloads/v28ablck/**/v28abl_seed*.pt",recursive=True)),V28)
full=nl==20;wp=np.isin(basins,["WP","EP"]);TE=np.array([i for i in te_idx if full[i] and wp[i]])
@torch.no_grad()
def err(ms,idx,drift):
  gd["USE_DRIFT"]=drift;P=[]
  for i in range(0,len(idx),128):
    j=idx[i:i+128];h=torch.from_numpy(np.concatenate([SLP[HIST_S[j,0]],SLP[HIST_S[j,1]]],1))
    a=[torch.from_numpy(track[j]),torch.from_numpy(vpair[j]),torch.from_numpy(SLP[j]),h,torch.from_numpy(HAVE[j])]
    P.append((torch.stack([m(*a)[0] for m in ms]).mean(0)[...,:2]*SC[:2]).float().numpy())
  return np.cumsum(np.concatenate(P),1)-np.cumsum(target[idx][...,:2],1)
def ab(idx,E):
  obs=np.cumsum(target[idx][...,:2],1);prev=np.concatenate([np.zeros((len(idx),1,2)),obs[:,:-1]],1);step=obs-prev
  hd=np.arctan2(step[...,1],step[...,0]);return E[...,0]*np.cos(hd)+E[...,1]*np.sin(hd)
alv=ab(TE,err(MSv23,TE,None));ala=ab(TE,err(MSabl,TE,0))
print(f"{'lead':>5s} {'h':>4s} | {'v23 along':>10s} {'v28abl along':>13s}")
for L in (0,1,2,3,7,11,19):
  print(f"{L+1:5d} {6*(L+1):4d} | {alv[:,L].mean():10.1f} {ala[:,L].mean():13.1f}")
