import re, os, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from six_metrics import VMAX_BINS
DEVICE=torch.device("cpu"); STEER_DROP=0.0; STEER_CLIP=4.0
def build_from_py(fn):
    src=open(fn).read(); g={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,
                            "DEVICE":DEVICE,"STEER_DROP":0.0,"STEER_CLIP":4.0}
    for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
              r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
              r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(p,src,re.S).group(0), g)
    return g["TrackFormerV9"]
nb=json.load(open("colab_train_v17.ipynb"))
src="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
G={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,"DEVICE":DEVICE,"STEER_DROP":0.0,"STEER_CLIP":4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p,src,re.S).group(0), G)
NetNB=G["TrackFormerV17"]

z=np.load("track_build/tip_fixed.npz",allow_pickle=True)
o=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
track=z["track"].astype("float32"); target=z["target"].astype("float32"); mask=z["target_mask"].astype(bool)
nl=z["n_leads"].astype(int)
tm=o["track_mean"].astype("float32"); ts=o["track_std"].astype("float32")
S4=np.load("track_build/tip_steer4.npy").astype("float32")
sc4=np.load("track_build/steer5_scale.npy")[:4]
v0=track[:,-1,2:4]*ts[2:4]+tm[2:4]; vp=track[:,-2,2:4]*ts[2:4]+tm[2:4]
vpair=np.concatenate([v0,vp],1).astype("float32")
SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12)
K=np.where(nl==20)[0]
print(f"Typhoon Tip (1979) — {len(track)} windows, {len(K)} with the full 5-day horizon")
print("NOT in the training data: the dataset starts at 1980.\n")

def run(models, slp_scaled):
    out=[]
    for i in range(0,len(K),64):
        j=K[i:i+64]
        with torch.no_grad():
            s=torch.stack([m(torch.from_numpy(track[j]),torch.from_numpy(vpair[j]),
                             torch.from_numpy(slp_scaled[j]))[0] for m in models]).mean(0)
        out.append((s*SC).numpy())
    return np.concatenate(out)
S_slp=(S4[:,:2]/np.array([5.,3.],dtype="float32")[None,:,None,None])
S_st=np.clip(S4/sc4[None,:,None,None],-4,4)
M={}
M["v10"]=([build_from_py("train_track_v10.py")()],None)
M["v10"][0][0].load_state_dict(torch.load("track_build/track_v10_best.pt",map_location="cpu",weights_only=False)["model"]); M["v10"][0][0].eval()
m13=build_from_py("train_track_v13.py")(); m13.load_state_dict(torch.load("track_build/track_v13_best.pt",map_location="cpu",weights_only=False)["model"]); m13.eval()
m14=build_from_py("train_track_v14.py")(); m14.load_state_dict(torch.load("track_build/track_v14_best.pt",map_location="cpu",weights_only=False)["model"]); m14.eval()
def loadnb(p):
    sd=torch.load(p,map_location="cpu",weights_only=False)["model"]
    sd={k[6:]:v for k,v in sd.items() if k.startswith("inner.")} or sd
    m=NetNB(); m.load_state_dict(sd); m.eval(); return m
v17=[loadnb(f"downloads/x/v17_seed{i}.pt") for i in range(5)]
v18=[loadnb(f"downloads/x/v18_seed{i}.pt") for i in range(8)]

T=target[K]; Km=mask[K]
res={}
# v10 takes no field
out=[]
for i in range(0,len(K),64):
    j=K[i:i+64]
    with torch.no_grad(): s,_=M["v10"][0][0](torch.from_numpy(track[j]),torch.from_numpy(vpair[j]))
    out.append((s*SC).numpy())
res["v10"]=np.concatenate(out)
res["v13"]=run([m13],S_slp); res["v14"]=run([m14],S_st)
res["v17"]=run(v17,S_st);    res["v18"]=run(v18,S_st)

print(f"{'model':6s} {'120h track':>11s} | {'peak wind: MAE':>15s} {'BIAS':>8s} | {'pressure MAE':>13s} {'BIAS':>8s}")
for tag,P in res.items():
    d=np.sqrt(((np.cumsum(P[...,:2],1)-np.cumsum(T[...,:2],1))**2).sum(-1))[:,19].mean()
    vm=Km[...,2]; dv=(P[...,2]-T[...,2])[vm]
    pm=Km[...,3]; dp=(P[...,3]-T[...,3])[pm]
    print(f"{tag:6s} {d:11.0f} | {np.abs(dv).mean():15.2f} {dv.mean():+8.2f} | {np.abs(dp).mean():13.2f} {dp.mean():+8.2f}")
print("\nPEAK WIND bias by observed strength (Tip reached 165 kt)")
print(f"{'bin':14s} {'n':>6s} | " + " ".join(f"{t:>14s}" for t in res))
for lo,hi,lab in VMAX_BINS:
    sel=Km[...,2]&(T[...,2]>=lo)&(T[...,2]<hi)
    if sel.sum()<20: continue
    row=f"{lab:14s} {int(sel.sum()):6d} | "
    for tag,P in res.items():
        row+=f"{(P[...,2]-T[...,2])[sel].mean():+14.1f} "
    print(row)
json.dump({k:v[...,:5].tolist() for k,v in res.items()}, open("track_build/tip_preds.json","w"))

print("\nTHE EXTREME TAIL — Tip peaked at 165 kt, the strongest storm ever measured")
print(f"{'observed vmax':16s} {'n':>5s} | " + " ".join(f"{t:>10s}" for t in res))
for lo,hi,lab in [(113,130,"113-129 kt"),(130,145,"130-144 kt"),(145,200,">=145 kt")]:
    sel=Km[...,2]&(T[...,2]>=lo)&(T[...,2]<hi)
    if sel.sum()<5: continue
    row=f"{lab:16s} {int(sel.sum()):5d} | "
    for tag,P in res.items():
        row+=f"{(P[...,2])[sel].mean():10.1f} "
    print(row + "   <- mean PREDICTED kt")
obs_pk=T[...,2][Km[...,2]].max()
print(f"\nobserved peak in these windows: {obs_pk:.0f} kt")
for tag,P in res.items():
    print(f"  {tag:5s} highest vmax it ever predicts: {P[...,2].max():6.1f} kt "
          f"({100*P[...,2].max()/obs_pk:.0f}% of observed peak)")
