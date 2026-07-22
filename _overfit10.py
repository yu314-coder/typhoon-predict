import re, os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE=torch.device("cpu")
src=open("train_track_v10.py").read()
g={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,"DEVICE":DEVICE}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p,src,re.S).group(0), g)
m=g["TrackFormerV9"](); m.load_state_dict(torch.load("track_build/track_v10_best.pt",map_location="cpu",weights_only=False)["model"]); m.eval()
z=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
track=z["track"].astype("float32"); target=z["target"].astype("float32")
yr=z["year"].astype(int); nl=z["n_leads"].astype(int); basin=z["basin"].astype(str); sids=z["storm_id"].astype(str)
tm=z["track_mean"].astype("float32"); ts=z["track_std"].astype("float32")
v0=track[:,-1,2:4]*ts[2:4]+tm[2:4]; vp=track[:,-2,2:4]*ts[2:4]+tm[2:4]
vpair=np.concatenate([v0,vp],1).astype("float32"); SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12)
fy={s:int(yr[sids==s].min()) for s in np.unique(sids)}; first=np.array([fy[s] for s in sids]); full=nl==20
SETS={"train <=2015":np.where(full&(first<=2015)&((basin=="WP")|(basin=="EP")))[0][:2500],
      "valid 16-19":np.where(full&(first>=2016)&(first<=2019)&((basin=="WP")|(basin=="EP")))[0],
      "test 2020+":np.where(full&(first>=2020)&((basin=="WP")|(basin=="EP")))[0]}
def err(idx):
    out=[]
    for i in range(0,len(idx),256):
        j=idx[i:i+256]
        with torch.no_grad(): s,_=m(torch.from_numpy(track[j]),torch.from_numpy(vpair[j]))
        out.append((s*SC).numpy())
    P=np.concatenate(out); T=target[idx]
    return float(np.sqrt(((np.cumsum(P[...,:2],1)-np.cumsum(T[...,:2],1))**2).sum(-1)).mean())
e={k:err(v) for k,v in SETS.items()}
print(f"v10    " + " ".join(f"{e[k]:14.1f}" for k in SETS) + f"   {e['test 2020+']/e['train <=2015']:9.2f}x")
