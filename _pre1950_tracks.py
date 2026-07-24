"""Tracks for three pre-1950 storms — outside EVERY model's training data, including v10.1's."""
import re, os, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE=torch.device("cpu")
def from_py(fn, ck, envcols=None):
    src=open(fn).read(); g={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,
                            "DEVICE":DEVICE,"STEER_DROP":0.0,"STEER_CLIP":4.0}
    for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
              r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
              r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(p,src,re.S).group(0), g)
    m=g["TrackFormerV9"](); m.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["model"]); m.eval()
    return m, g["KIN_COLS"], g["THERMO_COLS"], g["ENV_COLS"]
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

P=np.load("track_build/pre1950_windows.npz",allow_pickle=True)
o13=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
o20=np.load("track_build/track_windows_v20.npz",allow_pickle=True)
raw = P["track"].astype("float32")*P["track_std"].astype("float32")+P["track_mean"].astype("float32")
target=P["target"].astype("float32"); nl=P["n_leads"].astype(int); bt=P["base_time"].astype("int64")
bla=P["base_lat"].astype("float64"); blo=P["base_lon"].astype("float64"); sid=P["storm_id"].astype(str)
# renormalise per model family: 54-feature models use the v13 stats, v10.1 uses the v20 stats
t54=(raw[:,:,:54]-o13["track_mean"].astype("float32"))/o13["track_std"].astype("float32")
t55=(raw-o20["track_mean"].astype("float32"))/o20["track_std"].astype("float32")
def vpair_of(t, stats):
    tm,ts=stats
    a=t[:,-1,2:4]*ts[2:4]+tm[2:4]; b=t[:,-2,2:4]*ts[2:4]+tm[2:4]
    return np.concatenate([a,b],1).astype("float32")
vp54=vpair_of(t54,(o13["track_mean"].astype("float32"),o13["track_std"].astype("float32")))
vp55=vpair_of(t55,(o20["track_mean"].astype("float32"),o20["track_std"].astype("float32")))
SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12); R=111.2
NAMES={"1949317N09158":"Allyn","1946222N15152":"Lilly","1948011N07147":"Karen"}
# v17/v18 need a 500 hPa steering patch and NCEP reanalysis only begins in 1948, so 1946 has no
# coverage at all. Feeding them zeros would be a handicap dressed up as a comparison, so the map
# is restricted to the two models that consume no reanalysis -- which is also the question at
# hand: does 72% more data plus ONI help on storms nobody has seen?
MODELS={"v10":([from_py("train_track_v10.py","track_build/track_v10_best.pt")[0]],t54,vp54),
        "v10.1":([from_py("train_track_v20.py",f"downloads/x/v20_seed{i}.pt")[0] for i in range(3)],t55,vp55)}
os.makedirs("track_build/pre1950map",exist_ok=True)
print(f"{'model':7s} " + " ".join(f"{n:>9s}" for n in NAMES.values()))
for tag,(MS,TR,VP) in MODELS.items():
    out={}; row=[]
    for s,nm in NAMES.items():
        K=np.where((sid==s)&(nl==20))[0]; K=K[np.argsort(bt[K])]
        if not len(K): continue
        pr=[]
        for i in range(0,len(K),64):
            j=K[i:i+64]
            with torch.no_grad():
                sv=torch.stack([m(torch.from_numpy(TR[j]),torch.from_numpy(VP[j]))[0] for m in MS]).mean(0)
            pr.append((sv*SC).numpy())
        A=np.concatenate(pr)
        cE,cN=np.cumsum(A[...,0],1),np.cumsum(A[...,1],1)
        T=target[K]; tE,tN=np.cumsum(T[...,0],1),np.cumsum(T[...,1],1)
        lats,lons=[],[]
        for a in range(len(K)):
            la=bla[K[a]]+cN[a]/R
            lo=blo[K[a]]+cE[a]/(R*np.cos(np.radians((bla[K[a]]+la)/2)))
            lats.append(np.round(la,3).tolist()); lons.append(np.round(lo,3).tolist())
        err=float(np.hypot(cE[:,19]-tE[:,19],cN[:,19]-tN[:,19]).mean())
        out[nm]={"lat":lats,"lon":lons,"base_time":bt[K].tolist(),
                 "base_lat":np.round(bla[K],3).tolist(),"base_lon":np.round(blo[K],3).tolist(),
                 "err120_mean":err,"n":int(len(K))}
        row.append(err)
    json.dump(out,open(f"track_build/pre1950map/{tag}_tracks.json","w"))
    print(f"{tag:7s} " + " ".join(f"{e:9.0f}" for e in row))
