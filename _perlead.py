import re, math, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
def build(fn):
    src=open(fn).read(); g={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,
                            "STEER_DROP":0.0,"STEER_CLIP":4.0}
    for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(pat,src,re.S).group(0), g)
    return g["TrackFormerV9"]
z=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
track=z["track"].astype("float32"); target=z["target"].astype("float32"); mask=z["target_mask"].astype(bool)
nl=z["n_leads"].astype(int); yr=z["year"].astype(int); basin=z["basin"].astype(str)
tm=z["track_mean"].astype("float32"); ts=z["track_std"].astype("float32")
NEW=np.load("track_build/steer5_patches.npy",mmap_mode='r'); nsc=np.load("track_build/steer5_scale.npy")
S2=np.load("track_build/slp_patches.npy",mmap_mode='r')
v0=track[:,-1,2:4]*ts[2:4]+tm[2:4]; vp=track[:,-2,2:4]*ts[2:4]+tm[2:4]
vpair=np.concatenate([v0,vp],1).astype("float32"); SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12)
def load(cls,p):
    m=cls(); m.load_state_dict(torch.load(p,map_location="cpu",weights_only=False)["model"]); m.eval(); return m
M={"v10":(load(build("train_track_v10.py"),"track_build/track_v10_best.pt"),None,None),
   "v13":(load(build("train_track_v13.py"),"track_build/track_v13_best.pt"),"slp",None),
   "v14":(load(build("train_track_v14.py"),"track_build/track_v14_best.pt"),"s4",None),
   "v14.1":(load(build("train_track_v14_1.py"),"track_build/track_v14_1_best.pt"),"s4",4.0)}
EVAL=np.where((yr>=2020)&(nl==20)&((basin=="WP")|(basin=="EP")))[0]
print("eval windows:",len(EVAL))
res={}
for tag,(m,kind,cl) in M.items():
    out=[]
    for i in range(0,len(EVAL),256):
        j=EVAL[i:i+256]; a=[torch.from_numpy(track[j]),torch.from_numpy(vpair[j])]
        if kind=="slp":
            s_=np.asarray(S2[j],dtype="float32")/np.array([5.,3.],dtype="float32")[None,:,None,None]
            a.append(torch.from_numpy(s_))
        elif kind=="s4":
            s_=np.asarray(NEW[j],dtype="float32")[:,:4]/nsc[None,:4,None,None]
            if cl: s_=np.clip(s_,-cl,cl)
            a.append(torch.from_numpy(s_))
        with torch.no_grad(): s,_=m(*a)
        out.append((s*SC).numpy())
    P=np.concatenate(out); T=target[EVAL]; K=mask[EVAL]
    pt,tt=np.cumsum(P[...,:2],1),np.cumsum(T[...,:2],1)
    r={"track":np.sqrt(((pt-tt)**2).sum(-1)).mean(0).tolist()}
    for i,nm in [(2,"vmax"),(3,"pressure"),(4,"rmw")]:
        r[nm]=[float(np.abs(P[:,L,i]-T[:,L,i])[K[:,L,i]].mean()) if K[:,L,i].any() else None for L in range(20)]
    rm=K[...,5:17]
    r["radii"]=[float(np.abs(P[:,L,5:17]-T[:,L,5:17])[rm[:,L]].mean()) if rm[:,L].any() else None for L in range(20)]
    res[tag]=r
    print(f"{tag:6s} track all-lead {np.mean(r['track']):7.1f}  120h {r['track'][19]:7.1f}  "
          f"vmax {np.mean([x for x in r['vmax'] if x]):5.2f}  pres {np.mean([x for x in r['pressure'] if x]):5.2f}  "
          f"rmw {np.mean([x for x in r['rmw'] if x]):5.2f}  radii {np.mean([x for x in r['radii'] if x]):5.2f}")
json.dump(res,open("track_build/perlead_local.json","w"))
print("saved track_build/perlead_local.json")
