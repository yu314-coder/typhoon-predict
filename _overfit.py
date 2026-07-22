import re, os, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE=torch.device("cpu"); STEER_DROP=0.0; STEER_CLIP=4.0
nb=json.load(open("colab_train_v17.ipynb"))
src="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
G={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,"DEVICE":DEVICE,
   "STEER_DROP":STEER_DROP,"STEER_CLIP":STEER_CLIP}
for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
            r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
            r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(pat,src,re.S).group(0), G)
Net=G["TrackFormerV17"]
z=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
track=z["track"].astype("float32"); target=z["target"].astype("float32")
yr=z["year"].astype(int); nl=z["n_leads"].astype(int); basin=z["basin"].astype(str)
sids=z["storm_id"].astype(str)
tm=z["track_mean"].astype("float32"); ts=z["track_std"].astype("float32")
_q=np.load("track_build/steer5_int8.npz")
SLP=np.clip(_q["q"][:,:4].astype("float32")/31.75,-4,4); del _q
v0=track[:,-1,2:4]*ts[2:4]+tm[2:4]; vp=track[:,-2,2:4]*ts[2:4]+tm[2:4]
vpair=np.concatenate([v0,vp],1).astype("float32")
SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12)
fy={s:int(yr[sids==s].min()) for s in np.unique(sids)}
first=np.array([fy[s] for s in sids])
full=nl==20
SETS={"train <=2015":np.where(full&(first<=2015)&((basin=="WP")|(basin=="EP")))[0][:2500],
      "valid 16-19":np.where(full&(first>=2016)&(first<=2019)&((basin=="WP")|(basin=="EP")))[0],
      "test 2020+":np.where(full&(first>=2020)&((basin=="WP")|(basin=="EP")))[0]}
def load(p):
    sd=torch.load(p,map_location="cpu",weights_only=False)["model"]
    sd={k[6:]:v for k,v in sd.items() if k.startswith("inner.")} or sd
    m=Net(); m.load_state_dict(sd); m.eval(); return m
def err(models, idx):
    out=[]
    for i in range(0,len(idx),256):
        j=idx[i:i+256]
        with torch.no_grad():
            s=torch.stack([m(torch.from_numpy(track[j]),torch.from_numpy(vpair[j]),
                             torch.from_numpy(SLP[j]))[0] for m in models]).mean(0)
        out.append((s*SC).numpy())
    P=np.concatenate(out); T=target[idx]
    return float(np.sqrt(((np.cumsum(P[...,:2],1)-np.cumsum(T[...,:2],1))**2).sum(-1)).mean())
print(f"{'model':6s} " + " ".join(f"{k:>14s}" for k in SETS) + "   test/train")
for tag,n in [("v17",5),("v18",8),("v19",5)]:
    M=[load(f"downloads/x/{tag}_seed{i}.pt") for i in range(n)]
    e={k:err(M,v) for k,v in SETS.items()}
    print(f"{tag:6s} " + " ".join(f"{e[k]:14.1f}" for k in SETS) +
          f"   {e['test 2020+']/e['train <=2015']:9.2f}x")
