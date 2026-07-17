"""Run TrackFormer v8 on Typhoon Haiyan (2013) from its first Cat-4 fix; reconstruct real lat/lon
for history, deterministic forecast, ensemble cone, and the observed track. Dumps geo JSON for a map.

NOTE: Haiyan (2013) is in v8's TRAINING window (<=2015) -- this is an in-sample demonstration.
"""
import math, json, re
import numpy as np
import pandas as pd
import torch, torch.nn as nn

R = 111.2
RADIUS_NAMES = [f"r{t}_{q}" for t in (34, 50, 64) for q in ("ne", "se", "sw", "nw")]
HIST = 9
LEAD_HOURS = list(range(6, 121, 6))


def motion_km(lat0, lon0, lat1, lon1):
    dlat = lat1 - lat0
    dlon = ((lon1 - lon0 + 180) % 360) - 180
    return dlon * R * math.cos(math.radians((lat0 + lat1) / 2)), dlat * R


def inv_motion(e, n, lat0, lon0):
    dlat = n / R
    lat1 = lat0 + dlat
    dlon = e / (R * math.cos(math.radians((lat0 + lat1) / 2)))
    return lat1, lon0 + dlon


def valid(v):
    return v is not None and np.isfinite(v)


# ---- Haiyan fixes ----
raw = pd.read_csv("typhoon_build/data/ibtracs_ALL.csv", skiprows=[1], low_memory=False)
h = raw[(raw["NAME"] == "HAIYAN") & (raw["SEASON"].astype(str) == "2013")].copy()
num = lambda c: pd.to_numeric(h[c], errors="coerce")
h["time"] = pd.to_datetime(h["ISO_TIME"], errors="coerce")
h = h.sort_values("time").reset_index(drop=True)
fix = {"time": h["time"].values, "lat": num("LAT").values, "lon": num("LON").values,
       "vmax": num("USA_WIND").values, "pressure": num("USA_PRES").values,
       "gust": num("USA_GUST").values, "rmw": num("USA_RMW").values}
for t in (34, 50, 64):
    for q in ("NE", "SE", "SW", "NW"):
        c = f"USA_R{t}_{q}"
        fix[f"r{t}_{q.lower()}"] = num(c).values if c in h.columns else np.full(len(h), np.nan)
tns = h["time"].values.astype("datetime64[ns]").astype("int64")  # ns
base = int(np.where(num("USA_WIND").values >= 113)[0][0])  # first Cat-4
t0 = int(tns[base])
print(f"C4 onset: {h['time'].iloc[base]}  {fix['vmax'][base]:.0f}kt  ({fix['lat'][base]:.1f},{fix['lon'][base]:.1f})")

TOL = int(1.5 * 3600 * 1e9)
def nidx(target):
    p = np.searchsorted(tns, target); best, bd = -1, TOL + 1
    for c in (p - 1, p):
        if 0 <= c < len(tns) and abs(int(tns[c]) - target) < bd:
            bd, best = abs(int(tns[c]) - target), c
    return best if bd <= TOL else -1

hidx = [nidx(t0 - 6 * i * 3600 * 10**9) for i in range(HIST - 1, -1, -1)]
fidx = [nidx(t0 + hh * 3600 * 10**9) for hh in LEAD_HOURS]
assert -1 not in hidx, "history incomplete"
doy = pd.Timestamp(h["time"].iloc[base]).dayofyear
phase = 2 * math.pi * doy / 365.25

# ---- build 48-dim history exactly like build_track_v3data ----
seq = np.zeros((HIST, 48), dtype="float32")
prev, prev_dir = -1, None
for i, idx in enumerate(hidx):
    e, n = motion_km(fix["lat"][base], fix["lon"][base], fix["lat"][idx], fix["lon"][idx])
    se, sn = (0.0, 0.0) if prev < 0 else motion_km(fix["lat"][prev], fix["lon"][prev], fix["lat"][idx], fix["lon"][idx])
    f = seq[i]; f[0:4] = [e, n, se, sn]
    vals = [fix["vmax"][idx], fix["pressure"][idx], fix["gust"][idx], fix["rmw"][idx]]
    for j in range(4):
        f[4 + j] = vals[j] if valid(vals[j]) else 0.0
    for j, nm in enumerate(RADIUS_NAMES):
        f[8 + j] = fix[nm][idx] if valid(fix[nm][idx]) else 0.0
    f[21:23] = [math.sin(phase), math.cos(phase)]; f[23] = (t0 - int(tns[idx])) / 3.6e12
    fields = vals + [fix[nm][idx] for nm in RADIUS_NAMES]
    f[24:28] = [float(valid(x)) for x in fields[:4]]; f[28:40] = [float(valid(x)) for x in fields[4:]]
    speed = math.hypot(se, sn)
    hsin, hcos = (se / speed, sn / speed) if (speed > 1e-3 and prev >= 0) else (0.0, 0.0)
    f[40], f[41], f[42] = hsin, hcos, speed
    f[43] = (prev_dir[0] * hcos - prev_dir[1] * hsin) if (prev_dir and (hsin or hcos) and (prev_dir[0] or prev_dir[1])) else 0.0
    if prev >= 0:
        v0, v1, p0, p1 = fix["vmax"][prev], fix["vmax"][idx], fix["pressure"][prev], fix["pressure"][idx]
        dv, dp = valid(v0) and valid(v1), valid(p0) and valid(p1)
        f[44], f[45] = (v1 - v0 if dv else 0.0), (p1 - p0 if dp else 0.0); f[46], f[47] = float(dv), float(dp)
    if hsin or hcos:
        prev_dir = (hsin, hcos)
    prev = idx

# ---- v8 model ----
g = {"torch": torch, "nn": nn, "F": __import__("torch.nn.functional", fromlist=["x"]), "math": math, "np": np}
src = open("train_track_v3.py").read()
for blk in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM = len\(KIN_COLS\), len\(THERMO_COLS\)",
            r"def sinusoidal.*?return e", r"def encoder.*?return nn\.TransformerEncoder\(layer, depth\)",
            r"def decoder.*?return nn\.TransformerDecoder\(layer, depth\)",
            r"class TrackFormerV3.*?return state, logscale"]:
    exec(re.search(blk, src, re.S).group(0), g)
ck = torch.load("models/trackformer_v8_15M_fp16.pt", map_location="cpu", weights_only=False)
model = g["TrackFormerV3"]().eval()
model.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["model"].items()})
tmean, tstd = ck["track_mean"].astype("float32"), ck["track_std"].astype("float32")

seq_n = (seq - tmean) / tstd
v0_raw = (seq[-1, 2:4]).astype("float32")  # current 6h motion (raw km) = last step's se,sn
with torch.no_grad():
    st, _ = model(torch.from_numpy(seq_n[None]), torch.from_numpy(v0_raw[None]))
motion = (st[0, :, :2].numpy() * 100.0)  # per-step motion km, [20,2]

# ---- reconstruct lat/lon ----
def to_latlon(mot):
    lat, lon = fix["lat"][base], fix["lon"][base]; out = [[float(lat), float(lon)]]
    for e, n in mot:
        lat, lon = inv_motion(e, n, lat, lon); out.append([float(lat), float(lon)])
    return out

pred_path = to_latlon(motion)
hist_path = [[float(fix["lat"][i]), float(fix["lon"][i])] for i in hidx]
obs_path = [[float(fix["lat"][base]), float(fix["lon"][base])]]
for idx in fidx:
    obs_path.append([float(fix["lat"][idx]), float(fix["lon"][idx])] if idx != -1 else None)

# ---- ensemble from v8 residual covariance (RMT-cleaned), reconstruct each member ----
z = np.load("track_build/track_windows_v8.npz", allow_pickle=True)
tm2, ts2 = z["track_mean"].astype("float32"), z["track_std"].astype("float32")
vv0 = (z["track"][:, -1, 2:4] * ts2[2:4] + tm2[2:4]).astype("float32")
full = (z["n_leads"].astype(int) == 20) & (z["year"].astype(int) <= 2019)
fi = np.where(full)[0][:60000]
pmf = np.zeros((len(fi), 20, 2))
with torch.no_grad():
    for s in range(0, len(fi), 512):
        b = fi[s:s + 512]
        pmf[s:s + len(b)] = (model(torch.from_numpy(z["track"][b].astype("float32")), torch.from_numpy(vv0[b]))[0][..., :2] * 100).numpy()
res = (z["target"][fi][..., :2] - pmf).reshape(len(fi), 40)
S = np.cov(res.T)
vals, vecs = np.linalg.eigh((S + S.T) / 2); vals = np.clip(vals, 1e-6, None)
scale = np.mean(np.diag(S))
edge = scale * (1 + math.sqrt(40 / len(fi))) ** 2 * 2.0   # generalized-MP-style edge (a~2)
sig = vals > edge; floor = np.median(vals[~sig]) if (~sig).any() else edge / 2
Sc = (vecs * np.where(sig, vals, floor)) @ vecs.T
L = np.linalg.cholesky((Sc + Sc.T) / 2 + 1e-6 * np.eye(40))
rng = np.random.RandomState(7); N = 50
ens = []
for _ in range(N):
    m = (motion.reshape(40) + L @ rng.standard_normal(40)).reshape(20, 2)
    ens.append(to_latlon(m))

out = {"storm": "Typhoon Haiyan (2013) — SID 2013306N07162",
       "c4_time": str(h["time"].iloc[base]), "c4": [float(fix["lat"][base]), float(fix["lon"][base])],
       "c4_wind": float(fix["vmax"][base]),
       "history": hist_path, "forecast": pred_path, "observed": obs_path, "ensemble": ens,
       "note": "In-sample (2013 is in v8's training set). Forecast issued at first Cat-4 fix, 120 h horizon."}
json.dump(out, open("track_build/haiyan_geo.json", "w"))
# quick skill readout: mean position error at each valid lead
errs = []
for i, idx in enumerate(fidx):
    if idx == -1: continue
    pe, pn = motion_km(pred_path[i + 1][0], pred_path[i + 1][1], fix["lat"][idx], fix["lon"][idx])
    errs.append((LEAD_HOURS[i], math.hypot(pe, pn)))
print("forecast landfall region reached; sample lead errors (km):",
      {h_: round(e) for h_, e in errs if h_ in (24, 48, 72, 96, 120)})
print("saved track_build/haiyan_geo.json")
