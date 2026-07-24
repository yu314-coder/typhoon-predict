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
sid=z["storm_id"].astype(str); nl=z["n_leads"].astype(int); bt=z["base_time"].astype("int64")
bla=z["base_lat"].astype("float64"); blo=z["base_lon"].astype("float64")
tm=z["track_mean"].astype("float32"); ts=z["track_std"].astype("float32")
NEW=np.load("track_build/steer5_patches.npy",mmap_mode='r'); nsc=np.load("track_build/steer5_scale.npy")
v0=track[:,-1,2:4]*ts[2:4]+tm[2:4]; vp=track[:,-2,2:4]*ts[2:4]+tm[2:4]
vpair=np.concatenate([v0,vp],1).astype("float32"); SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12)
def load(cls,p):
    m=cls(); m.load_state_dict(torch.load(p,map_location="cpu",weights_only=False)["model"]); m.eval(); return m
M={"v10":(load(build("train_track_v10.py"),"track_build/track_v10_best.pt"),False),
   "v14":(load(build("train_track_v14.py"),"track_build/track_v14_best.pt"),True)}
R=111.2
def to_ll(lat0,lon0,cumE,cumN):
    lat=lat0+cumN/R
    lon=lon0+cumE/(R*np.cos(np.radians((lat0+lat)/2)))
    return lat,lon
def sp_bear(E,N):                    # per-step displacement km over 6 h
    return np.hypot(E,N)/6.0, (np.degrees(np.arctan2(E,N))+360)%360
STORMS=[("2026182N09163","Bavi","2026"),("1986228N19120","Wayne","1986"),
        ("2025203N20124","Co-may","2025"),("2022239N22150","Hinnamnor","2022")]
out={}
for s,nm,yr in STORMS:
    k=np.where((sid==s)&(nl==20))[0]
    if not len(k): print("skip",nm); continue
    j=int(k[np.argmin(bt[k])])                      # earliest full-horizon initialisation
    T=target[j]; K=mask[j]
    obsE,obsN=np.cumsum(T[:,0]),np.cumsum(T[:,1])
    olat,olon=to_ll(bla[j],blo[j],obsE,obsN)
    osp,obr=sp_bear(T[:,0],T[:,1])
    rec={"name":nm,"year":yr,"sid":s,"n_windows":int(len(k)),
         "base":{"lat":float(bla[j]),"lon":float(blo[j]),
                 "time":str(np.datetime64(int(bt[j]),"ns"))[:16].replace("T"," ")},
         "observed":{"lat":olat.tolist(),"lon":olon.tolist(),
                     "vmax":[float(T[L,2]) if K[L,2] else None for L in range(20)],
                     "pressure":[float(T[L,3]) if K[L,3] else None for L in range(20)],
                     "rmw":[float(T[L,4]) if K[L,4] else None for L in range(20)],
                     "speed":osp.tolist(),"bearing":obr.tolist()}}
    for tag,(m,needs) in M.items():
        a=[torch.from_numpy(track[j:j+1]),torch.from_numpy(vpair[j:j+1])]
        if needs:
            a.append(torch.from_numpy(np.asarray(NEW[j:j+1],dtype="float32")[:,:4]/nsc[None,:4,None,None]))
        with torch.no_grad(): sv,_=m(*a)
        P=(sv*SC).numpy()[0]
        cE,cN=np.cumsum(P[:,0]),np.cumsum(P[:,1])
        plat,plon=to_ll(bla[j],blo[j],cE,cN)
        psp,pbr=sp_bear(P[:,0],P[:,1])
        err=float(np.hypot(cE[19]-obsE[19],cN[19]-obsN[19]))
        rec[tag]={"lat":plat.tolist(),"lon":plon.tolist(),
                  "vmax":P[:,2].tolist(),"pressure":P[:,3].tolist(),"rmw":P[:,4].tolist(),
                  "speed":psp.tolist(),"bearing":pbr.tolist(),"err120":err}
    out[nm]=rec
    print(f"{nm:10s} init {rec['base']['time']}  base {bla[j]:5.1f}N {blo[j]:6.1f}E  "
          f"windows {len(k):3d} | 120h err  v10 {rec['v10']['err120']:6.0f}  v14 {rec['v14']['err120']:6.0f} km")
json.dump(out,open("track_build/storm_forecasts.json","w"))
print("saved track_build/storm_forecasts.json")
