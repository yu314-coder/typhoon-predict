"""Run TrackFormer v8 on out-of-sample storms from a chosen issue fix; reconstruct real lat/lon for
history, deterministic forecast, RMT ensemble, and observed track. Dumps per-storm geo JSON + extent.

Storms: Co-may 2025 (erratic, issue at first Cat-1) and Bavi 2026 (Cat-5, issue at first Cat-4).
Both are in v8's test period (>=2020) -- genuine out-of-sample.
"""
import math, json, re
import numpy as np
import pandas as pd
import torch, torch.nn as nn

R = 111.2
RN = [f"r{t}_{q}" for t in (34, 50, 64) for q in ("ne", "se", "sw", "nw")]
HIST = 9; LEADS = list(range(6, 121, 6))
STORMS = [
    {"sid": "2025203N20124", "name": "Co-may (2025)", "thr": 64, "label": "first Cat-1", "out": "comay"},
    {"sid": "2026182N09163", "name": "Bavi (2026)",  "thr": 113, "label": "first Cat-4", "out": "bavi"},
    {"sid": "1986228N19120", "name": "Wayne (1986)", "thr": 64, "label": "first Cat-1", "out": "wayne"},
]


def mkm(la0, lo0, la1, lo1):
    dlat = la1 - la0; dlon = ((lo1 - lo0 + 180) % 360) - 180
    return dlon * R * math.cos(math.radians((la0 + la1) / 2)), dlat * R
def inv(e, n, la0, lo0):
    la1 = la0 + n / R
    return la1, lo0 + e / (R * math.cos(math.radians((la0 + la1) / 2)))
def ok(v): return v is not None and np.isfinite(v)

# ---- v8 ----
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

# ---- ensemble covariance (v8 residuals, RMT-cleaned) computed once ----
z = np.load("track_build/track_windows_v8.npz", allow_pickle=True)
tm2, ts2 = z["track_mean"].astype("float32"), z["track_std"].astype("float32")
vv0 = (z["track"][:, -1, 2:4] * ts2[2:4] + tm2[2:4]).astype("float32")
fi = np.where((z["n_leads"].astype(int) == 20) & (z["year"].astype(int) <= 2019))[0][:60000]
pmf = np.zeros((len(fi), 20, 2))
with torch.no_grad():
    for s in range(0, len(fi), 512):
        b = fi[s:s + 512]
        pmf[s:s + len(b)] = (model(torch.from_numpy(z["track"][b].astype("float32")), torch.from_numpy(vv0[b]))[0][..., :2] * 100).numpy()
res = (z["target"][fi][..., :2] - pmf).reshape(len(fi), 40)
S = np.cov(res.T); vals, vecs = np.linalg.eigh((S + S.T) / 2); vals = np.clip(vals, 1e-6, None)
edge = np.mean(np.diag(S)) * (1 + math.sqrt(40 / len(fi))) ** 2 * 2.0
sig = vals > edge; floor = np.median(vals[~sig]) if (~sig).any() else edge / 2
Sc = (vecs * np.where(sig, vals, floor)) @ vecs.T
L = np.linalg.cholesky((Sc + Sc.T) / 2 + 1e-6 * np.eye(40))

raw = pd.read_csv("typhoon_build/data/ibtracs_ALL.csv", skiprows=[1], low_memory=False)


def build(storm):
    h = raw[raw["SID"] == storm["sid"]].copy()
    num = lambda c: pd.to_numeric(h[c], errors="coerce").values
    h["t"] = pd.to_datetime(h["ISO_TIME"], errors="coerce"); h = h.sort_values("t").reset_index(drop=True)
    num = lambda c: pd.to_numeric(h[c], errors="coerce").values
    fx = {"lat": num("LAT"), "lon": num("LON"), "vmax": num("USA_WIND"), "pressure": num("USA_PRES"),
          "gust": num("USA_GUST"), "rmw": num("USA_RMW")}
    for t in (34, 50, 64):
        for q in ("NE", "SE", "SW", "NW"):
            c = f"USA_R{t}_{q}"; fx[f"r{t}_{q.lower()}"] = num(c) if c in h.columns else np.full(len(h), np.nan)
    tns = h["t"].values.astype("datetime64[ns]").astype("int64")
    TOL = int(1.5 * 3600 * 1e9)
    def nidx(tg):
        p = np.searchsorted(tns, tg); best, bd = -1, TOL + 1
        for c in (p - 1, p):
            if 0 <= c < len(tns) and abs(int(tns[c]) - tg) < bd: bd, best = abs(int(tns[c]) - tg), c
        return best if bd <= TOL else -1
    def full_hist(c):
        t0c = int(tns[c])
        return -1 not in [nidx(t0c - 6 * i * 3600 * 10**9) for i in range(HIST - 1, -1, -1)]
    # issue at the first fix that meets the intensity threshold AND has full 48h history;
    # fall back to the earliest full-history fix if the storm is too weak/short for the threshold.
    cand = [c for c in np.where(fx["vmax"] >= storm["thr"])[0] if full_hist(c)]
    base = int(cand[0]) if cand else next(c for c in range(len(tns)) if full_hist(c))
    t0 = int(tns[base])
    hidx = [nidx(t0 - 6 * i * 3600 * 10**9) for i in range(HIST - 1, -1, -1)]
    fidx = [nidx(t0 + hh * 3600 * 10**9) for hh in LEADS]
    doy = pd.Timestamp(h["t"].iloc[base]).dayofyear; phase = 2 * math.pi * doy / 365.25
    seq = np.zeros((HIST, 48), dtype="float32"); prev, pdir = -1, None
    for i, idx in enumerate(hidx):
        e, n = mkm(fx["lat"][base], fx["lon"][base], fx["lat"][idx], fx["lon"][idx])
        se, sn = (0., 0.) if prev < 0 else mkm(fx["lat"][prev], fx["lon"][prev], fx["lat"][idx], fx["lon"][idx])
        f = seq[i]; f[0:4] = [e, n, se, sn]
        vv = [fx["vmax"][idx], fx["pressure"][idx], fx["gust"][idx], fx["rmw"][idx]]
        for j in range(4): f[4 + j] = vv[j] if ok(vv[j]) else 0.
        for j, nm in enumerate(RN): f[8 + j] = fx[nm][idx] if ok(fx[nm][idx]) else 0.
        f[21:23] = [math.sin(phase), math.cos(phase)]; f[23] = (t0 - int(tns[idx])) / 3.6e12
        fl = vv + [fx[nm][idx] for nm in RN]
        f[24:28] = [float(ok(x)) for x in fl[:4]]; f[28:40] = [float(ok(x)) for x in fl[4:]]
        sp = math.hypot(se, sn); hs, hc = (se / sp, sn / sp) if (sp > 1e-3 and prev >= 0) else (0., 0.)
        f[40], f[41], f[42] = hs, hc, sp
        f[43] = (pdir[0] * hc - pdir[1] * hs) if (pdir and (hs or hc) and (pdir[0] or pdir[1])) else 0.
        if prev >= 0:
            dv = ok(fx["vmax"][prev]) and ok(fx["vmax"][idx]); dp = ok(fx["pressure"][prev]) and ok(fx["pressure"][idx])
            f[44] = fx["vmax"][idx] - fx["vmax"][prev] if dv else 0.; f[45] = fx["pressure"][idx] - fx["pressure"][prev] if dp else 0.
            f[46], f[47] = float(dv), float(dp)
        if hs or hc: pdir = (hs, hc)
        prev = idx
    seq_n = (seq - tmean) / tstd; v0 = seq[-1, 2:4].astype("float32")
    with torch.no_grad():
        st, _ = model(torch.from_numpy(seq_n[None]), torch.from_numpy(v0[None]))
    motion = st[0, :, :2].numpy() * 100.0
    def latlon(mot):
        la, lo = fx["lat"][base], fx["lon"][base]; out = [[float(la), float(lo)]]
        for e, n in mot: la, lo = inv(e, n, la, lo); out.append([float(la), float(lo)])
        return out
    forecast = latlon(motion)
    history = [[float(fx["lat"][i]), float(fx["lon"][i])] for i in hidx]
    observed = [[float(fx["lat"][base]), float(fx["lon"][base])]] + \
               [[float(fx["lat"][i]), float(fx["lon"][i])] if i != -1 else None for i in fidx]
    rng = np.random.RandomState(7)
    ens = [latlon((motion.reshape(40) + L @ rng.standard_normal(40)).reshape(20, 2)) for _ in range(50)]
    # frame on the deterministic forecast + observed + history (ensemble spill just clips at edges)
    pts = forecast + [q for q in observed if q] + history
    la = [p[0] for p in pts]; lo = [p[1] for p in pts]
    ext = [math.floor(min(lo) - 3), math.ceil(max(lo) + 3), math.floor(min(la) - 2), math.ceil(max(la) + 3)]
    errs = {}
    for i, idx in enumerate(fidx):
        if idx == -1: continue
        e, n = mkm(forecast[i + 1][0], forecast[i + 1][1], fx["lat"][idx], fx["lon"][idx])
        errs[LEADS[i]] = round(math.hypot(e, n))
    out = {"storm": storm["name"], "issue_label": storm["label"],
           "issue_time": str(h["t"].iloc[base]), "issue": [round(float(fx["lat"][base]), 2), round(float(fx["lon"][base]), 2)],
           "issue_wind": float(fx["vmax"][base]), "extent": ext,
           "history": [[round(a, 2), round(b, 2)] for a, b in history],
           "forecast": [[round(a, 2), round(b, 2)] for a, b in forecast],
           "observed": [[round(p[0], 2), round(p[1], 2)] if p else None for p in observed],
           "ensemble": [[[round(a, 2), round(b, 2)] for a, b in e] for e in ens], "errors": errs}
    json.dump(out, open(f"track_build/{storm['out']}_geo.json", "w"))
    print(f"{storm['name']}: issue {out['issue_time']} @{out['issue']} {out['issue_wind']:.0f}kt | "
          f"extent {ext} | errs(km) {({k: errs[k] for k in (24,48,72,96,120) if k in errs})}")


for s in STORMS:
    build(s)
