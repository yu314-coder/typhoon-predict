"""v20's error as a function of lead time, so it can be set beside published numbers.

No paper reports a lead-averaged scalar; they all report per-lead curves. 452 km is a mean over
leads 6-120 h, which mixes a ~50 km short-lead error with a ~1000 km five-day error and is not
comparable to anything.
"""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8); DEVICE=torch.device("cpu")
nb=json.load(open("colab_train_v17.ipynb"))
src="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
G={"torch":torch,"nn":nn,"F":F,"math":math,"np":np,"os":os,"DEVICE":DEVICE,"STEER_DROP":0.0,"STEER_CLIP":4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p,src,re.S).group(0),G)
def load(ck):
    sd=torch.load(ck,map_location="cpu",weights_only=False)["model"]
    sd={k[6:]:v for k,v in sd.items() if k.startswith("inner.")} or sd
    m=G["TrackFormerV17"]().eval(); m.load_state_dict(sd); return m
z=np.load("track_build/track_windows_v13.npz",allow_pickle=True)
track=z["track"].astype("float32"); target=z["target"].astype("float32")
sids=z["storm_id"].astype(str); years=z["year"].astype(int)
basins=z["basin"].astype(str); nl=z["n_leads"].astype(int)
tm=z["track_mean"].astype("float32"); ts=z["track_std"].astype("float32")
vp=np.concatenate([track[:,-1,2:4]*ts[2:4]+tm[2:4], track[:,-2,2:4]*ts[2:4]+tm[2:4]],1).astype("float32")
def steer(p): return np.clip(np.load(p)["q"][:,:4].astype("float32")/31.75,-4,4)
fy={s:int(years[sids==s].min()) for s in np.unique(sids)}
EV=np.array([i for i in range(len(sids)) if fy[sids[i]]>=2020 and nl[i]==20 and basins[i] in ("WP","EP")])
T=np.cumsum(target[EV][...,:2],1); SC=torch.tensor([100.,100.,35.,20.,50.]+[50.]*12)
@torch.no_grad()
def curve(ck,S):
    ms=[load(c) for c in ck]; P=[]
    for i in range(0,len(EV),128):
        j=EV[i:i+128]
        a=[torch.from_numpy(track[j]),torch.from_numpy(vp[j]),torch.from_numpy(S[j])]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0)*SC).numpy())
    C=np.cumsum(np.concatenate(P)[...,:2],1)
    return np.sqrt(((C-T)**2).sum(-1)).mean(0)      # [20], mean over windows per lead
c20=curve([f"downloads/x/v20_seed{i}.pt" for i in range(5)], steer("track_build/dlm4_int8.npz"))
c17=curve([f"downloads/x/v17_seed{i}.pt" for i in range(5)], steer("track_build/steer5_int8.npz"))
# operational references (NHC 2024 verification, Atlantic; DeMaria 2024 AI models)
OFCL={24:51.9,48:None,72:123.9,96:None,120:213.5}
print(f"{'lead':>5s} {'v17':>8s} {'v20':>8s}   {'NHC 2024 official':>18s}")
for L in range(20):
    h=6*(L+1); ref=OFCL.get(h)
    r=f"{ref:.0f} km" if ref else ""
    print(f"{h:4d}h {c17[L]:8.0f} {c20[L]:8.0f}   {r:>18s}")
print(f"\nmean over all 20 leads: v17 {c17.mean():.1f}  v20 {c20.mean():.1f} km")
json.dump({"v17":c17.tolist(),"v20":c20.tolist()},open("track_build/perlead_v20.json","w"))
