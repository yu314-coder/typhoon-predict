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
def predict(tag):
    m,kind,cl=M[tag]; out=[]
    for i in range(0,len(EVAL),256):
        j=EVAL[i:i+256]; a=[torch.from_numpy(track[j]),torch.from_numpy(vpair[j])]
        if kind=="slp":
            a.append(torch.from_numpy(np.asarray(S2[j],dtype="float32")/np.array([5.,3.],dtype="float32")[None,:,None,None]))
        elif kind=="s4":
            s_=np.asarray(NEW[j],dtype="float32")[:,:4]/nsc[None,:4,None,None]
            if cl: s_=np.clip(s_,-cl,cl)
            a.append(torch.from_numpy(s_))
        with torch.no_grad(): s,_=m(*a)
        out.append((s*SC).numpy())
    return np.concatenate(out)
T=target[EVAL]; K=mask[EVAL]
BINS=[(0,34,"TD <34"),(34,64,"TS 34-63"),(64,96,"Cat1-2 64-95"),(96,113,"Cat3 96-112"),(113,300,"Cat4-5 >=113")]
print(f"eval {len(EVAL)} windows, WP+EP 2020+, all 20 leads pooled\n")
print(f"{'bin':16s} {'n':>7s} | " + " | ".join(f"{t:^17s}" for t in M))
print(f"{'':16s} {'':>7s} | " + " | ".join(f"{'MAE':>7s} {'bias':>8s}" for _ in M))
P={t:predict(t) for t in M}
for lo,hi,lab in BINS:
    sel=K[...,2]&(T[...,2]>=lo)&(T[...,2]<hi)
    row=f"{lab:16s} {int(sel.sum()):7d} | "
    for t in M:
        d=(P[t][...,2]-T[...,2])[sel]
        row+=f"{np.abs(d).mean():7.2f} {d.mean():+8.2f} | "
    print(row)
print("\nPRESSURE (hPa), binned by observed vmax")
for lo,hi,lab in BINS:
    sel=K[...,3]&(T[...,2]>=lo)&(T[...,2]<hi)
    if sel.sum()<50: continue
    row=f"{lab:16s} {int(sel.sum()):7d} | "
    for t in M:
        d=(P[t][...,3]-T[...,3])[sel]
        row+=f"{np.abs(d).mean():7.2f} {d.mean():+8.2f} | "
    print(row)
print("\nFORWARD SPEED (km/h), binned by observed speed")
osp=np.hypot(T[...,0],T[...,1])/6.0
for lo,hi,lab in [(0,15,"slow <15"),(15,25,"25"),(25,35,"35"),(35,200,"fast >=35")]:
    sel=K[...,0]&(osp>=lo)&(osp<hi)
    row=f"{lab:16s} {int(sel.sum()):7d} | "
    for t in M:
        d=(np.hypot(P[t][...,0],P[t][...,1])/6.0-osp)[sel]
        row+=f"{np.abs(d).mean():7.2f} {d.mean():+8.2f} | "
    print(row)
