"""Extract per-lead prediction series (wind, pressure, RMW, wind-radii, moving speed) for v9 and v10
vs the observed values, for several storms. Dumps JSON for the chart."""
import math, json, re
import numpy as np, pandas as pd, torch, torch.nn as nn

R = 111.2; RN = [f"r{t}_{q}" for t in (34, 50, 64) for q in ("ne", "se", "sw", "nw")]
HIST = 9; LEADS = list(range(6, 121, 6))
STORMS = [("2026182N09163", "Bavi (2026)", 113), ("2025203N20124", "Co-may (2025)", 64),
          ("2022239N22150", "Hinnamnor (2022)", 113)]

def mkm(a, b, c, d):
    dlat = c - a; dlon = ((d - b + 180) % 360) - 180
    return dlon * R * math.cos(math.radians((a + c) / 2)), dlat * R
def ok(v): return v is not None and np.isfinite(v)

def load(script, ckpt):
    g = {"torch": torch, "nn": nn, "F": __import__("torch.nn.functional", fromlist=["x"]), "math": math, "np": np}
    src = open(script).read()
    for blk in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?return e", r"def enc\(.*?depth\)", r"def dec\(d.*?depth\)",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(blk, src, re.S).group(0), g)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = g["TrackFormerV9"]().eval()
    m.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["model"].items()})
    return m, ck["track_mean"].astype("float32"), ck["track_std"].astype("float32")

m9, tm9, ts9 = load("train_track_v9.py", "track_build/track_v9_best.pt")
m10, tm10, ts10 = load("train_track_v10.py", "track_build/track_v10_best.pt")
raw = pd.read_csv("typhoon_build/data/ibtracs_ALL.csv", skiprows=[1], low_memory=False)
TS = np.array([100., 100., 35., 20., 50.] + [50.] * 12, dtype="float32")

def build_seq(fx, hidx, base, tns, t0, mon):
    doy = pd.Timestamp((tns[base]).astype("datetime64[ns]")).dayofyear; phase = 2 * math.pi * doy / 365.25
    seq = np.zeros((HIST, 54), dtype="float32"); prev, pdir = -1, None
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
        lat_i, lon_i, d2l, mo = fx["lat"][idx], fx["lon"][idx], fx["dist2land"][idx], int(mon[idx])
        thermal = 0.5 * 23.44 * math.sin(2 * math.pi * (mo - 3) / 12.0)
        f[48] = lat_i; f[49] = abs(lat_i); f[50] = math.sin(math.radians(lon_i)); f[51] = math.cos(math.radians(lon_i))
        f[52] = d2l if ok(d2l) else float(tm9[52]); f[53] = max(0., min(31., 30. - 0.30 * abs(lat_i - thermal) ** 1.4))
        if hs or hc: pdir = (hs, hc)
        prev = idx
    return seq

out = []
for sid, name, thr in STORMS:
    h = raw[raw["SID"] == sid].copy(); h["t"] = pd.to_datetime(h["ISO_TIME"], errors="coerce")
    h = h.sort_values("t").reset_index(drop=True)
    num = lambda c: pd.to_numeric(h[c], errors="coerce").values if c in h.columns else np.full(len(h), np.nan)
    fx = {"lat": num("LAT"), "lon": num("LON"), "vmax": num("USA_WIND"), "pressure": num("USA_PRES"),
          "gust": num("USA_GUST"), "rmw": num("USA_RMW"), "dist2land": num("DIST2LAND")}
    for t in (34, 50, 64):
        for q in ("NE", "SE", "SW", "NW"): fx[f"r{t}_{q.lower()}"] = num(f"USA_R{t}_{q}")
    mon = h["t"].dt.month.values; tns = h["t"].values.astype("datetime64[ns]").astype("int64"); TOL = int(1.5 * 3600 * 1e9)
    def nidx(tg):
        p = np.searchsorted(tns, tg); best, bd = -1, TOL + 1
        for c in (p - 1, p):
            if 0 <= c < len(tns) and abs(int(tns[c]) - tg) < bd: bd, best = abs(int(tns[c]) - tg), c
        return best if bd <= TOL else -1
    def fullh(c): return -1 not in [nidx(int(tns[c]) - 6 * i * 3600 * 10**9) for i in range(HIST - 1, -1, -1)]
    cand = [c for c in np.where(fx["vmax"] >= thr)[0] if fullh(c)]
    base = int(cand[0]) if cand else next(c for c in range(len(tns)) if fullh(c))
    t0 = int(tns[base]); hidx = [nidx(t0 - 6 * i * 3600 * 10**9) for i in range(HIST - 1, -1, -1)]; fidx = [nidx(t0 + hh * 3600 * 10**9) for hh in LEADS]
    seq = build_seq(fx, hidx, base, tns, t0, mon)
    with torch.no_grad():
        p9 = (m9(torch.from_numpy(((seq - tm9) / ts9)[None]), torch.from_numpy(seq[-1, 2:4][None].astype("float32")))[0][0].numpy()) * TS
        vpair = np.concatenate([seq[-1, 2:4], seq[-2, 2:4]]).astype("float32")
        p10 = (m10(torch.from_numpy(((seq - tm10) / ts10)[None]), torch.from_numpy(vpair[None]))[0][0].numpy()) * TS
    # observed + predicted series
    def series(P):  # P: [20,17] physical
        spd = np.hypot(P[:, 0], P[:, 1]) / 6.0  # km/h (per-step km over 6h)
        radii = P[:, 5:17].mean(1)
        return dict(vmax=P[:, 2].round(1).tolist(), pres=P[:, 3].round(1).tolist(),
                    rmw=P[:, 4].round(1).tolist(), radii=radii.round(1).tolist(), speed=spd.round(1).tolist())
    obs = {"vmax": [], "pres": [], "rmw": [], "radii": [], "speed": []}
    prev = base
    for idx in fidx:
        if idx == -1:
            for k in obs: obs[k].append(None)
            continue
        e, n = mkm(fx["lat"][prev], fx["lon"][prev], fx["lat"][idx], fx["lon"][idx])
        obs["speed"].append(round(math.hypot(e, n) / 6.0, 1))
        obs["vmax"].append(round(float(fx["vmax"][idx]), 1) if ok(fx["vmax"][idx]) else None)
        obs["pres"].append(round(float(fx["pressure"][idx]), 1) if ok(fx["pressure"][idx]) else None)
        obs["rmw"].append(round(float(fx["rmw"][idx]), 1) if ok(fx["rmw"][idx]) else None)
        rv = [fx[nm][idx] for nm in RN]; valid = [x for x in rv if ok(x)]
        obs["radii"].append(round(float(np.mean(valid)), 1) if valid else None)
        prev = idx
    out.append({"name": name, "issue_wind": float(fx["vmax"][base]), "leads": LEADS,
                "obs": obs, "v9": series(p9), "v10": series(p10)})
json.dump(out, open("track_build/pred_series.json", "w"))
print("saved pred_series.json for", [s["name"] for s in out])
