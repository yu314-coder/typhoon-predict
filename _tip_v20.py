"""v20's Tip forecast, beside v17 and v10, from the same three launches."""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from make_rmt_tracks import observed_at, SIX_H
torch.set_num_threads(8); DEVICE=torch.device("cpu"); R=111.2
def km(a1,o1,a2,o2): return math.hypot((o2-o1)*R*math.cos(math.radians((a1+a2)/2)),(a2-a1)*R)

nb=json.load(open("colab_train_v17.ipynb"))
src="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
G={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,"DEVICE":DEVICE,"STEER_DROP":0.0,"STEER_CLIP":4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p,src,re.S).group(0),G)
def load17(ck):
    sd=torch.load(ck,map_location="cpu",weights_only=False)["model"]
    sd={k[6:]:v for k,v in sd.items() if k.startswith("inner.")} or sd
    m=G["TrackFormerV17"]().eval(); m.load_state_dict(sd); return m

z=np.load("track_build/tip_fixed.npz",allow_pickle=True)
o13=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
tr=z["track"].astype("float32"); tgt=z["target"].astype("float32"); nl=z["n_leads"].astype(int)
bt=z["base_time"].astype("int64"); bla=z["base_lat"].astype("float64"); blo=z["base_lon"].astype("float64")
tm=o13["track_mean"].astype("float32"); ts=o13["track_std"].astype("float32")
vp=np.concatenate([tr[:,-1,2:4]*ts[2:4]+tm[2:4], tr[:,-2,2:4]*ts[2:4]+tm[2:4]],1).astype("float32")
S20=np.load("track_build/tip_dlm4.npy").astype("float32")
S4=np.load("track_build/tip_steer4.npy").astype("float32")
S17=np.clip(S4/np.load("track_build/steer5_scale.npy")[:4][None,:,None,None],-4,4).astype("float32")
SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12)
obs=observed_at(bt,bla,blo)

M20=[load17(f"downloads/x/v20_seed{i}.pt") for i in range(5)]
M17=[load17(f"downloads/x/v17_seed{i}.pt") for i in range(5)]
@torch.no_grad()
def run(ms,S,idx):
    a=[torch.from_numpy(tr[idx]),torch.from_numpy(vp[idx]),torch.from_numpy(S[idx])]
    return (torch.stack([m(*a)[0] for m in ms]).mean(0)*SC).numpy()

# v10 (field-free) for reference, from its existing export
v10=json.load(open("track_build/tipmap/v10_tracks.json"))["Tip"]
v10bt=np.array(v10["base_time"],dtype="int64")

K=np.where(nl==20)[0]; K=K[np.argsort(bt[K])]
LAUNCH=[("11 Oct 0600","1979-10-11T06:00"),("13 Oct 0600","1979-10-13T06:00"),("16 Oct 0600","1979-10-16T06:00")]
out={}
print(f"{'launch':12s} {'model':6s} {'+24h':>7s} {'+72h':>7s} {'+120h':>7s} {'mean':>7s}")
for tag,MS,S in [("v20",M20,S20),("v17",M17,S17)]:
    rec={}
    for nm,iso in LAUNCH:
        T0=int(np.datetime64(iso,"ns").astype("int64")); i=int(np.abs(bt-T0).argmin())
        assert abs(int(bt[i])-T0)<SIX_H
        P=run(MS,S,np.array([i]))[0]
        cE,cN=np.cumsum(P[:,0]),np.cumsum(P[:,1])
        la0,lo0=bla[i],blo[i]
        lat=la0+cN/R; lon=lo0+cE/(R*np.cos(np.radians((la0+lat)/2)))
        errs,pairs=[],[]
        for L in range(20):
            vt=int(round((T0+(L+1)*SIX_H)/SIX_H))*SIX_H
            if vt in obs:
                errs.append((L+1,km(obs[vt][0],obs[vt][1],lat[L],lon[L])))
                pairs.append([[float(lat[L]),float(lon[L])],[obs[vt][0],obs[vt][1]]])
        d=dict(errs)
        rec[nm]={"lat":[[la0]+lat.tolist()],"lon":[[lo0]+lon.tolist()],"base_time":[int(bt[i])],
                 "base_lat":bla.tolist(),"base_lon":blo.tolist(),"launch":[la0,lo0],
                 "pairs":pairs,"n":1,"err120_mean":d.get(20,float("nan"))}
        print(f"{nm:12s} {tag:6s} {d.get(4,float('nan')):7.0f} {d.get(12,float('nan')):7.0f} "
              f"{d.get(20,float('nan')):7.0f} {np.mean([e for _,e in errs]):7.0f}")
    os.makedirs("track_build/tipv20",exist_ok=True)
    json.dump(rec,open(f"track_build/tipv20/{tag}_tracks.json","w"))

# v10 from its export, same launches
rec={}
for nm,iso in LAUNCH:
    T0=int(np.datetime64(iso,"ns").astype("int64")); i=int(np.abs(v10bt-T0).argmin())
    lat=np.array(v10["lat"][i]); lon=np.array(v10["lon"][i])
    la0,lo0=v10["base_lat"][i],v10["base_lon"][i]
    errs,pairs=[],[]
    for L in range(20):
        vt=int(round((T0+(L+1)*SIX_H)/SIX_H))*SIX_H
        if vt in obs:
            errs.append((L+1,km(obs[vt][0],obs[vt][1],lat[L],lon[L])))
            pairs.append([[float(lat[L]),float(lon[L])],[obs[vt][0],obs[vt][1]]])
    d=dict(errs)
    rec[nm]={"lat":[[la0]+lat.tolist()],"lon":[[lo0]+lon.tolist()],"base_time":[int(v10bt[i])],
             "base_lat":v10["base_lat"],"base_lon":v10["base_lon"],"launch":[la0,lo0],
             "pairs":pairs,"n":1,"err120_mean":d.get(20,float("nan"))}
    print(f"{nm:12s} {'v10':6s} {d.get(4,float('nan')):7.0f} {d.get(12,float('nan')):7.0f} "
          f"{d.get(20,float('nan')):7.0f} {np.mean([e for _,e in errs]):7.0f}")
json.dump(rec,open("track_build/tipv20/v10_tracks.json","w"))
print("\nwrote track_build/tipv20/")
