import re, os, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE=torch.device("cpu")
def from_py(fn, ck):
    src=open(fn).read(); g={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,
                            "DEVICE":DEVICE,"STEER_DROP":0.0,"STEER_CLIP":4.0}
    for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
              r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
              r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(p,src,re.S).group(0), g)
    m=g["TrackFormerV9"](); m.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["model"]); m.eval(); return m
nb=json.load(open("colab_train_v17.ipynb"))
src="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
G={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,"DEVICE":DEVICE,"STEER_DROP":0.0,"STEER_CLIP":4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p,src,re.S).group(0), G)
def from_nb(ck):
    sd=torch.load(ck,map_location="cpu",weights_only=False)["model"]
    sd={k[6:]:v for k,v in sd.items() if k.startswith("inner.")} or sd
    m=G["TrackFormerV17"](); m.load_state_dict(sd); m.eval(); return m

z=np.load("track_build/tip_fixed.npz",allow_pickle=True)
o=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
track=z["track"].astype("float32"); target=z["target"].astype("float32"); nl=z["n_leads"].astype(int)
bt=z["base_time"].astype("int64"); bla=z["base_lat"].astype("float64"); blo=z["base_lon"].astype("float64")
tm=o["track_mean"].astype("float32"); ts=o["track_std"].astype("float32")
S4=np.load("track_build/tip_steer4.npy").astype("float32"); sc4=np.load("track_build/steer5_scale.npy")[:4]
v0=track[:,-1,2:4]*ts[2:4]+tm[2:4]; vp=track[:,-2,2:4]*ts[2:4]+tm[2:4]
vpair=np.concatenate([v0,vp],1).astype("float32"); SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12)
S_slp=S4[:,:2]/np.array([5.,3.],dtype="float32")[None,:,None,None]
S_st=np.clip(S4/sc4[None,:,None,None],-4,4)
K=np.where(nl==20)[0]; K=K[np.argsort(bt[K])]
R=111.2
MODELS={"v10":([from_py("train_track_v10.py","track_build/track_v10_best.pt")],None),
        "v14":([from_py("train_track_v14.py","track_build/track_v14_best.pt")],"st"),
        "v17":([from_nb(f"downloads/x/v17_seed{i}.pt") for i in range(5)],"st"),
        "v18":([from_nb(f"downloads/x/v18_seed{i}.pt") for i in range(8)],"st"),
        "v19":([from_nb(f"downloads/x/v19_seed{i}.pt") for i in range(5)],"st")}
os.makedirs("track_build/tipmap", exist_ok=True)
for tag,(MS,kind) in MODELS.items():
    out=[]
    for i in range(0,len(K),64):
        j=K[i:i+64]; a=[torch.from_numpy(track[j]),torch.from_numpy(vpair[j])]
        if kind=="st": a.append(torch.from_numpy(S_st[j]))
        with torch.no_grad():
            s=torch.stack([m(*a)[0] for m in MS]).mean(0)
        out.append((s*SC).numpy())
    P=np.concatenate(out)
    cE,cN=np.cumsum(P[...,0],1),np.cumsum(P[...,1],1)
    T=target[K]; tE,tN=np.cumsum(T[...,0],1),np.cumsum(T[...,1],1)
    lats,lons=[],[]
    for a in range(len(K)):
        la=bla[K[a]]+cN[a]/R
        lo=blo[K[a]]+cE[a]/(R*np.cos(np.radians((bla[K[a]]+la)/2)))
        lats.append(np.round(la,3).tolist()); lons.append(np.round(lo,3).tolist())
    err=float(np.hypot(cE[:,19]-tE[:,19],cN[:,19]-tN[:,19]).mean())
    json.dump({"Tip":{"lat":lats,"lon":lons,"base_time":bt[K].tolist(),
                      "base_lat":np.round(bla[K],3).tolist(),"base_lon":np.round(blo[K],3).tolist(),
                      "err120_mean":err,"n":int(len(K))}},
              open(f"track_build/tipmap/{tag}_tracks.json","w"))
    print(f"  {tag:5s} {len(K):3d} forecasts | mean 120h {err:6.0f} km")
