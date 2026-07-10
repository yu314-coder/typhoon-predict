import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Run the Bavi typhoon checkpoint locally.")
parser.add_argument("--checkpoint", type=Path, default=ROOT / "best.pt")
parser.add_argument("--output", type=Path, default=ROOT / "forecast.json")
parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
args = parser.parse_args()
CKPT = args.checkpoint.expanduser().resolve()
OUT = args.output.expanduser().resolve()

class FieldEncoder(nn.Module):
    def __init__(self, channels, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels,64,3,padding=1), nn.GELU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,96,3,stride=2,padding=1), nn.GELU(), nn.BatchNorm2d(96),
            nn.Conv2d(96,hidden,3,stride=2,padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
    def forward(self, x): return self.net(x).flatten(1)

class ERA5CycloneEnsemble(nn.Module):
    def __init__(self, track_dim, channels, steps, output_dim, hidden=192, latent=32):
        super().__init__()
        self.steps, self.output_dim, self.latent = steps, output_dim, latent
        self.field = FieldEncoder(channels, 128)
        self.track_gru = nn.GRU(track_dim,96,batch_first=True,bidirectional=True)
        self.fuse = nn.Sequential(nn.Linear(128*steps+192,hidden),nn.GELU(),nn.Dropout(.15),nn.Linear(hidden,hidden),nn.GELU())
        self.mean = nn.Linear(hidden,output_dim)
        self.log_scale = nn.Linear(hidden,output_dim)
        self.latent_proj = nn.Sequential(nn.Linear(latent,hidden),nn.GELU(),nn.Linear(hidden,output_dim))
    def encode(self, track_x, field_x):
        b,t,c,h,w=field_x.shape
        field_z=self.field(field_x.reshape(b*t,c,h,w)).reshape(b,t,-1).flatten(1)
        _,state=self.track_gru(track_x)
        return self.fuse(torch.cat([field_z,state.transpose(0,1).flatten(1)],dim=1))
    def sample(self, track_x, field_x, n, temperature=1.0):
        z=self.encode(track_x,field_x)
        mean=self.mean(z); log_scale=self.log_scale(z).clamp(-5,2)
        eps=torch.randn(n,z.shape[0],self.latent,device=z.device)
        low=self.latent_proj(eps.reshape(-1,self.latent)).reshape(n,z.shape[0],-1)
        independent=torch.randn_like(low)*torch.exp(log_scale).unsqueeze(0)
        return mean.unsqueeze(0)+temperature*(low+independent)

raw=torch.load(CKPT,map_location="cpu",weights_only=False)
if args.device == "mps":
    device = torch.device("mps")
elif args.device == "cpu":
    device = torch.device("cpu")
else:
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
config=raw["config"]
track_scaler=raw["track_scaler"]; y_scaler=raw["y_scaler"]
field_mean=np.asarray(raw["field_mean"],dtype="float32")
field_std=np.asarray(raw["field_std"],dtype="float32")
model=ERA5CycloneEnsemble(9,10,1,28).to(device)
model.load_state_dict(raw["model_state"]); model.eval()

fixes=[
    ("2026-07-09T12:00:00",19.2,128.8,100,952),
    ("2026-07-09T18:00:00",20.1,128.2,90,953),
    ("2026-07-10T00:00:00",20.8,127.3,75,964),
    ("2026-07-10T06:00:00",21.9,126.9,75,962),
]
track=[]
for i,(stamp,lat,lon,wind,pres) in enumerate(fixes):
    dlat=0 if i==0 else lat-fixes[i-1][1]
    dlon=0 if i==0 else ((lon-fixes[i-1][2]+180)%360)-180
    track.append([lat,lon,wind,pres,dlat,dlon,float(np.hypot(dlat,dlon)),np.sin(2*np.pi*191/366),np.cos(2*np.pi*191/366)])
track=np.asarray(track,dtype="float32")
track[:,:1]-=track[-1:,0:1]
track[:,1:2]=((track[:,1:2]-track[-1:,1:2]+180)%360)-180
track[:,2]/=100.0
track[:,3]=(track[:,3]-950.0)/50.0
xtrack=track_scaler.transform(track).astype("float32")[None]
field=np.zeros((1,1,10,33,33),dtype="float32")
with torch.no_grad():
    ens=model.sample(torch.from_numpy(xtrack).to(device),torch.from_numpy(field).to(device),50,float(config.get("sample_temperature",1.0))).cpu().numpy()
ens=ens.mean(0,keepdims=True)+(ens-ens.mean(0,keepdims=True))*1.1297996044158936
pred=y_scaler.inverse_transform(ens[:,0,:])
base_lat,base_lon=fixes[-1][1],fixes[-1][2]
points=[]
for k,lead in enumerate(config["lead_hours"]):
    j=4*k
    lat=base_lat+pred[:,j]; lon=(base_lon+pred[:,j+1])%360
    points.append({"lead_hours":int(lead),"lat":float(lat.mean()),"lon":float(lon.mean()),"p10_lat":float(np.quantile(lat,.1)),"p90_lat":float(np.quantile(lat,.9)),"p10_lon":float(np.quantile(lon,.1)),"p90_lon":float(np.quantile(lon,.9))})
result={"storm":"Bavi","source":"JTWC/TCGP fixes through 2026-07-10 0600 UTC","initial_time":"2026-07-10T06:00:00Z","initial_lat":base_lat,"initial_lon":base_lon,"points":points,"device":str(device),"note":"Mac checkpoint inference using current track fixes and mean-normalized atmospheric input because 2026 ERA5 fields are unavailable locally. Not an operational forecast."}
OUT.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
