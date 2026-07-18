"""Run TrackFormer v9 (triple-stream + env) on Bavi 2026, Wayne 1986, Co-may 2025; compare to v8.
Builds the 54-feature window (adds env: lat, |lat|, lon sin/cos, dist2land, sst_proxy)."""
import math, json, re
import numpy as np, pandas as pd, torch, torch.nn as nn

R = 111.2; RN = [f"r{t}_{q}" for t in (34, 50, 64) for q in ("ne", "se", "sw", "nw")]
HIST = 9; LEADS = list(range(6, 121, 6))
STORMS = [
    {"sid": "2026182N09163", "name": "Bavi (2026)",  "thr": 113, "label": "first Cat-4", "out": "bavi"},
    {"sid": "1986228N19120", "name": "Wayne (1986)", "thr": 64,  "label": "first Cat-1", "out": "wayne"},
    {"sid": "2025203N20124", "name": "Co-may (2025)","thr": 64,  "label": "first Cat-1", "out": "comay"},
    {"sid": "2022239N22150", "name": "Hinnamnor (2022)", "thr": 113, "label": "first Cat-4", "out": "hinnamnor"},
]
V8 = {"Bavi (2026)": {24:147,48:233,72:187,96:74,120:289},
      "Wayne (1986)": {24:133,48:688,72:1248,96:1547,120:1216},
      "Co-may (2025)": {24:146,48:794,72:838,96:684,120:251},
      "Hinnamnor (2022)": {24:38,48:380,72:814,96:940,120:948}}

def mkm(a,b,c,d):
    dlat=c-a; dlon=((d-b+180)%360)-180
    return dlon*R*math.cos(math.radians((a+c)/2)), dlat*R
def ok(v): return v is not None and np.isfinite(v)

g = {"torch": torch, "nn": nn, "F": __import__("torch.nn.functional", fromlist=["x"]), "math": math, "np": np}
src = open("train_track_v14.py").read()
for blk in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
            r"def sinusoidal.*?return e", r"def enc\(.*?depth\)",
            r"def dec\(d.*?depth\)", r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(blk, src, re.S).group(0), g)
ck = torch.load("track_build/track_v14_best.pt", map_location="cpu", weights_only=False)
model = g["TrackFormerV9"]().eval()
model.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["model"].items()})
tmean, tstd = ck["track_mean"].astype("float32"), ck["track_std"].astype("float32")

# v9 residual covariance (RMT-cleaned) for the ensemble
z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
tm2, ts2 = z["track_mean"].astype("float32"), z["track_std"].astype("float32")
vv0 = (z["track"][:, -1, 2:4] * ts2[2:4] + tm2[2:4]).astype("float32")
vvp = (z["track"][:, -2, 2:4] * ts2[2:4] + tm2[2:4]).astype("float32")
vpair_full = np.concatenate([vv0, vvp], axis=1).astype("float32")
SLP_ALL = np.load("track_build/steer4_patches.npy").astype("float32")/np.load("track_build/steer4_scale.npy")[None,:,None,None]
fi = np.where((z["n_leads"].astype(int) == 20) & (z["year"].astype(int) <= 2019))[0][:60000]
pmf = np.zeros((len(fi), 20, 2))
with torch.no_grad():
    for s in range(0, len(fi), 512):
        b = fi[s:s+512]; pmf[s:s+len(b)] = (model(torch.from_numpy(z["track"][b].astype("float32")), torch.from_numpy(vpair_full[b]), torch.from_numpy(SLP_ALL[b]))[0][..., :2]*100).numpy()
res = (z["target"][fi][..., :2]-pmf).reshape(len(fi), 40); S = np.cov(res.T)
vals, vecs = np.linalg.eigh((S+S.T)/2); vals = np.clip(vals, 1e-6, None)
edge = np.mean(np.diag(S))*(1+math.sqrt(40/len(fi)))**2*2.0; sig = vals > edge
Sc = (vecs*np.where(sig, vals, np.median(vals[~sig]))) @ vecs.T
L = np.linalg.cholesky((Sc+Sc.T)/2 + 1e-6*np.eye(40))

SIDS_ALL = z["storm_id"].astype(str); BT_ALL = z["base_time"].astype("int64")
def steer_patch(sid, t0):
    """4-channel steering patch (SLP anom, SLP 24h tendency, u500, v500) for this storm's base fix."""
    m = np.where(SIDS_ALL == sid)[0]
    if len(m) == 0: raise SystemExit(f"no stored window for {sid}")
    k = int(m[np.abs(BT_ALL[m] - t0).argmin()])
    dt = abs(int(BT_ALL[k]) - t0) / 3.6e12
    if dt > 3.0: print(f"  warn: nearest stored window {dt:.0f}h off")
    return SLP_ALL[k]

raw = pd.read_csv("typhoon_build/data/ibtracs_ALL.csv", skiprows=[1], low_memory=False)

def build(storm):
    h = raw[raw["SID"] == storm["sid"]].copy()
    h["t"] = pd.to_datetime(h["ISO_TIME"], errors="coerce"); h = h.sort_values("t").reset_index(drop=True)
    num = lambda c: pd.to_numeric(h[c], errors="coerce").values if c in h.columns else np.full(len(h), np.nan)
    fx = {"lat": num("LAT"), "lon": num("LON"), "vmax": num("USA_WIND"), "pressure": num("USA_PRES"),
          "gust": num("USA_GUST"), "rmw": num("USA_RMW"), "dist2land": num("DIST2LAND")}
    for t in (34, 50, 64):
        for q in ("NE","SE","SW","NW"): fx[f"r{t}_{q.lower()}"] = num(f"USA_R{t}_{q}")
    mon = h["t"].dt.month.values
    tns = h["t"].values.astype("datetime64[ns]").astype("int64"); TOL = int(1.5*3600*1e9)
    def nidx(tg):
        p = np.searchsorted(tns, tg); best, bd = -1, TOL+1
        for c in (p-1, p):
            if 0 <= c < len(tns) and abs(int(tns[c])-tg) < bd: bd, best = abs(int(tns[c])-tg), c
        return best if bd <= TOL else -1
    def fullh(c): return -1 not in [nidx(int(tns[c])-6*i*3600*10**9) for i in range(HIST-1,-1,-1)]
    cand = [c for c in np.where(fx["vmax"] >= storm["thr"])[0] if fullh(c)]
    base = int(cand[0]) if cand else next(c for c in range(len(tns)) if fullh(c))
    t0 = int(tns[base]); hidx = [nidx(t0-6*i*3600*10**9) for i in range(HIST-1,-1,-1)]; fidx = [nidx(t0+hh*3600*10**9) for hh in LEADS]
    doy = pd.Timestamp(h["t"].iloc[base]).dayofyear; phase = 2*math.pi*doy/365.25
    seq = np.zeros((HIST, 54), dtype="float32"); prev, pdir = -1, None
    for i, idx in enumerate(hidx):
        e, n = mkm(fx["lat"][base], fx["lon"][base], fx["lat"][idx], fx["lon"][idx])
        se, sn = (0.,0.) if prev < 0 else mkm(fx["lat"][prev], fx["lon"][prev], fx["lat"][idx], fx["lon"][idx])
        f = seq[i]; f[0:4] = [e,n,se,sn]
        vv = [fx["vmax"][idx], fx["pressure"][idx], fx["gust"][idx], fx["rmw"][idx]]
        for j in range(4): f[4+j] = vv[j] if ok(vv[j]) else 0.
        for j, nm in enumerate(RN): f[8+j] = fx[nm][idx] if ok(fx[nm][idx]) else 0.
        f[21:23] = [math.sin(phase), math.cos(phase)]; f[23] = (t0-int(tns[idx]))/3.6e12
        fl = vv + [fx[nm][idx] for nm in RN]
        f[24:28] = [float(ok(x)) for x in fl[:4]]; f[28:40] = [float(ok(x)) for x in fl[4:]]
        sp = math.hypot(se, sn); hs, hc = (se/sp, sn/sp) if (sp > 1e-3 and prev >= 0) else (0.,0.)
        f[40], f[41], f[42] = hs, hc, sp
        f[43] = (pdir[0]*hc - pdir[1]*hs) if (pdir and (hs or hc) and (pdir[0] or pdir[1])) else 0.
        if prev >= 0:
            dv = ok(fx["vmax"][prev]) and ok(fx["vmax"][idx]); dp = ok(fx["pressure"][prev]) and ok(fx["pressure"][idx])
            f[44] = fx["vmax"][idx]-fx["vmax"][prev] if dv else 0.; f[45] = fx["pressure"][idx]-fx["pressure"][prev] if dp else 0.
            f[46], f[47] = float(dv), float(dp)
        lat_i, lon_i, d2l, m = fx["lat"][idx], fx["lon"][idx], fx["dist2land"][idx], int(mon[idx])
        thermal = 0.5*23.44*math.sin(2*math.pi*(m-3)/12.0)
        f[48] = lat_i; f[49] = abs(lat_i); f[50] = math.sin(math.radians(lon_i)); f[51] = math.cos(math.radians(lon_i))
        f[52] = d2l if ok(d2l) else float(tmean[52]); f[53] = max(0., min(31., 30.-0.30*abs(lat_i-thermal)**1.4))
        if hs or hc: pdir = (hs, hc)
        prev = idx
    seq_n = (seq - tmean)/tstd
    vpair = np.concatenate([seq[-1, 2:4], seq[-2, 2:4]]).astype("float32")   # v0 + previous velocity
    yr_base = int(pd.Timestamp(h["t"].iloc[base]).year)
    sp = steer_patch(storm["sid"], t0)
    with torch.no_grad():
        motion = (model(torch.from_numpy(seq_n[None]), torch.from_numpy(vpair[None]),
                        torch.from_numpy(sp[None]))[0][0, :, :2].numpy())*100.0
    def latlon(mot):
        la, lo = fx["lat"][base], fx["lon"][base]; out = [[float(la), float(lo)]]
        for e, n in mot:
            la = la+n/R; lo = lo+e/(R*math.cos(math.radians(la))); out.append([float(la), float(lo)])
        return out
    fc = latlon(motion)
    errs = {}
    for i, idx in enumerate(fidx):
        if idx == -1: continue
        e, n = mkm(fc[i+1][0], fc[i+1][1], fx["lat"][idx], fx["lon"][idx]); errs[LEADS[i]] = round(math.hypot(e, n))
    v8 = V8.get(storm["name"], {})
    row = " ".join(f"{h_}h v14 {errs.get(h_,'-'):>5} v10 {v8.get(h_,'-'):>5}" for h_ in (24,48,72,96,120))
    print(f"{storm['name']:14s} @{fx['vmax'][base]:.0f}kt | {row}")
    # geo dump for maps
    rng = np.random.RandomState(7)
    ens = [latlon((motion.reshape(40) + L @ rng.standard_normal(40)).reshape(20, 2)) for _ in range(50)]
    history = [[round(float(fx["lat"][i]), 2), round(float(fx["lon"][i]), 2)] for i in hidx]
    observed = [[round(float(fx["lat"][base]), 2), round(float(fx["lon"][base]), 2)]] + \
               [[round(float(fx["lat"][i]), 2), round(float(fx["lon"][i]), 2)] if i != -1 else None for i in fidx]
    pts = fc + [q for q in observed if q] + history
    la = [p[0] for p in pts]; lo = [p[1] for p in pts]
    ext = [math.floor(min(lo) - 3), math.ceil(max(lo) + 3), math.floor(min(la) - 2), math.ceil(max(la) + 3)]
    d = {"storm": storm["name"], "issue_label": storm["label"], "issue_time": str(h["t"].iloc[base]),
         "issue": [round(float(fx["lat"][base]), 2), round(float(fx["lon"][base]), 2)], "issue_wind": float(fx["vmax"][base]),
         "extent": ext, "history": history, "forecast": [[round(a, 2), round(b, 2)] for a, b in fc],
         "observed": observed, "ensemble": [[[round(a, 2), round(b, 2)] for a, b in e] for e in ens],
         "errors": errs, "errors_v8": v8}
    json.dump(d, open(f"track_build/{storm["out"]}_v14_geo.json", "w"))
    return errs

print("=== v14 (+500 hPa steering wind) vs v10 forecast error (km) by lead ===")
for s in STORMS: build(s)
