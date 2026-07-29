"""v23 and v35 forecasts on Typhoon Dolphin (2026), using REAL GFS deep-layer-mean steering
(track_build/dolphin_dlm4_real.npz, from _fetch_dolphin_steering.py) -- the same real-data
discipline used for Noul (_noul_v33.py): unavailable fields (RMW, gust, wind radii, pre-genesis
history) are zero-filled with an explicit availability flag, never fabricated.

Writes track_build/dolphin_v23_v35.json: for each model, the LATEST issue's (2026-07-29 06:00
UTC) full 20-lead forecast track + vmax -- i.e. "what does the model predict right now" -- plus
each earlier issue's forecast for context (all issues, not just the latest).
"""
import json, re, math, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R = 111.2; HIST = 9; KT_PER_MPH = 0.868976

DOLPHIN_TIMES = ["2026-07-27T00:00", "2026-07-27T06:00", "2026-07-27T12:00", "2026-07-27T18:00",
                 "2026-07-28T00:00", "2026-07-28T06:00", "2026-07-28T12:00", "2026-07-28T18:00",
                 "2026-07-29T00:00", "2026-07-29T06:00"]
RAW = [  # (lat, lon, wind_mph, pressure_hPa_or_None) -- Wunderground position/wind, Zoom Earth pressure
    (12.8, 178.3, 35, 1002), (13.4, 176.7, 40, 1000), (13.6, 175.2, 50, 998),
    (13.2, 173.7, 50, 992), (13.0, 172.8, 70, 990), (13.3, 171.7, 85, 980),
    (13.4, 170.7, 115, 975), (13.7, 169.9, 145, 937), (14.1, 169.1, 140, 941),
    (14.5, 168.4, 140, 941),
]
N = len(RAW)
tns = np.array([np.datetime64(t).astype("datetime64[ns]").astype("int64") for t in DOLPHIN_TIMES])
lat_a = np.array([r[0] for r in RAW]); lon_a = np.array([r[1] for r in RAW])
vmax_a = np.array([r[2] * KT_PER_MPH for r in RAW])
pres_a = np.array([r[3] for r in RAW], dtype="float64")

nb = json.load(open("colab_train_v17.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
body = body.replace('"/content/d/steer5_int8.npz"', '"track_build/dlm4_int8.npz"')
body = body.replace('"/content/d/track_windows_v13.npz"', '"track_build/track_windows_v13.npz"')
body = body.replace('DEVICE = torch.device("cuda")', 'DEVICE = torch.device("cpu")')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os,
     "json": json, "time": __import__("time"), "math": math}
exec(compile(body, "<v17-notebook>", "exec"), G)
tmean, tstd = G["tmean"].astype("float32"), G["tstd"].astype("float32")
SC = G["TARGET_SCALE"]

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))
KM6H = 6 * 3600 / 1000.0

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": G["TrackFormerV17"], "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

v28src = open("colab_v28_train.py").read()
hs_src = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28src, re.S).group(0)
tf_src = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28src, re.S).group(0)
g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(hs_src, g23); exec(tf_src, g23)
V23 = g23["TrackFormerHist"]


def load(ck):
    m = V23().eval(); m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"]); return m


def build_v10():
    s = open("train_track_v10.py").read()
    g = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
         "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
    for pat in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
                r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
                r"class TrackFormerV9.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
        exec(re.search(pat, s, re.S).group(0), g)
    return g["TrackFormerV9"]


m10 = build_v10()()
m10.load_state_dict(torch.load("track_build/track_v10_best.pt", map_location="cpu", weights_only=False)["model"])
m10.eval()

M23 = [load(f) for f in sorted(glob.glob("downloads/x/v23_seed*.pt"))]
M35 = [load(f) for f in sorted(glob.glob("downloads/v35ck/v35_seed*.pt"))]
print(f"v10: 1 | v23: {len(M23)} | v35: {len(M35)} seeds")

_T = np.load("track_build/terrain_wp.npz")
T_LAT, T_LON, LSM = _T["lat"], _T["lon"], _T["lsm"]
_LAND = LSM > 0.5
LAND_LAT = T_LAT[np.where(_LAND)[0]]; LAND_LON = T_LON[np.where(_LAND)[1]]


def dist2land(lat_i, lon_i):
    if len(LAND_LAT) == 0:
        return 3000.0
    dlat = LAND_LAT - lat_i; dlon = (LAND_LON - lon_i) * math.cos(math.radians(lat_i))
    return float(np.hypot(dlon * R, dlat * R).min())


def build_window(base):
    hidx = [max(0, base - HIST + 1 + k) for k in range(HIST)]
    n_padded = max(0, HIST - 1 - base)
    t0 = int(tns[base])
    doy = (np.datetime64(DOLPHIN_TIMES[base]) - np.datetime64(DOLPHIN_TIMES[base][:4] + "-01-01")).astype(int) + 1
    phase = 2 * math.pi * doy / 365.25
    seq = np.zeros((HIST, 54), dtype="float32"); prev, pdir = -1, None

    def mkm(a, b, c, d):
        dlat = c - a; dlon = ((d - b + 180) % 360) - 180
        return dlon * R * math.cos(math.radians((a + c) / 2)), dlat * R

    for i, idx in enumerate(hidx):
        e, n = mkm(lat_a[base], lon_a[base], lat_a[idx], lon_a[idx])
        se, sn = (0., 0.) if prev < 0 else mkm(lat_a[prev], lon_a[prev], lat_a[idx], lon_a[idx])
        f = seq[i]; f[0:4] = [e, n, se, sn]
        vv = [vmax_a[idx], pres_a[idx], np.nan, np.nan]
        for j in range(4):
            f[4 + j] = vv[j] if (vv[j] is not None and np.isfinite(vv[j])) else 0.
        f[24:28] = [float(np.isfinite(x)) if x is not None else 0. for x in vv]
        f[21:23] = [math.sin(phase), math.cos(phase)]; f[23] = (t0 - int(tns[idx])) / 3.6e12
        sp = math.hypot(se, sn); hs, hc = (se / sp, sn / sp) if (sp > 1e-3 and prev >= 0) else (0., 0.)
        f[40], f[41], f[42] = hs, hc, sp
        f[43] = (pdir[0] * hc - pdir[1] * hs) if (pdir and (hs or hc) and (pdir[0] or pdir[1])) else 0.
        if prev >= 0:
            dv = np.isfinite(vmax_a[prev]) and np.isfinite(vmax_a[idx])
            dp = np.isfinite(pres_a[prev]) and np.isfinite(pres_a[idx])
            f[44] = vmax_a[idx] - vmax_a[prev] if dv else 0.
            f[45] = pres_a[idx] - pres_a[prev] if dp else 0.
            f[46], f[47] = float(dv), float(dp)
        lat_i, lon_i = lat_a[idx], lon_a[idx]
        m = np.datetime64(DOLPHIN_TIMES[idx]).astype("datetime64[M]").astype(int) % 12 + 1
        d2l = dist2land(lat_i, lon_i % 360)
        thermal = 0.5 * 23.44 * math.sin(2 * math.pi * (m - 3) / 12.0)
        f[48] = lat_i; f[49] = abs(lat_i); f[50] = math.sin(math.radians(lon_i)); f[51] = math.cos(math.radians(lon_i))
        f[52] = d2l; f[53] = max(0., min(31., 30. - 0.30 * abs(lat_i - thermal) ** 1.4))
        if hs or hc:
            pdir = (hs, hc)
        prev = idx
    seq_n = (seq - tmean) / tstd
    vpair = np.concatenate([seq[-1, 2:4], seq[-2, 2:4]]).astype("float32")
    return seq_n, vpair, n_padded


dlm = np.load("track_build/dolphin_dlm4_real.npz")
ISSUE_KEYS = [t.replace("-", "").replace("T", "_")[:11] for t in DOLPHIN_TIMES]
# DOLPHIN_TIMES like "2026-07-27T00:00" -> "20260727_00"
ISSUE_KEYS = ["".join(t.split("T")[0].split("-")) + "_" + t.split("T")[1][:2] for t in DOLPHIN_TIMES]
_dlm_scale = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")


def real_slp(base):
    key = ISSUE_KEYS[base]
    slp = np.zeros((1, 4, 17, 17), "float32")
    if key in dlm:
        uv = dlm[key]
        uv_n = np.clip(uv / _dlm_scale[:, None, None], -4.0, 4.0)
        slp[0, 2:4] = uv_n
    return torch.from_numpy(slp)


def real_hist(base):
    cur = real_slp(base).numpy()[0]
    hist = np.zeros((1, 8, 17, 17), "float32")
    have = np.zeros((1, 2), "float32")
    for c, back in enumerate((2, 4)):
        j = base - back
        if j >= 0 and ISSUE_KEYS[j] in dlm:
            uv = dlm[ISSUE_KEYS[j]]
            uv_n = np.clip(uv / _dlm_scale[:, None, None], -4.0, 4.0)
            hist[0, c * 4 + 2:c * 4 + 4] = uv_n
            have[0, c] = 1.0
        else:
            hist[0, c * 4:(c + 1) * 4] = cur
    return torch.from_numpy(hist), torch.from_numpy(have)


@torch.no_grad()
def forecast(tag, ms, base):
    seq_n, vpair, n_padded = build_window(base)
    tr = torch.from_numpy(seq_n[None]); vp = torch.from_numpy(vpair[None])
    if tag == "v10":
        sv, _ = m10(tr, vp)
        motion = (sv[0] * SC).float().numpy()
    else:
        h, hv = real_hist(base)
        args = [tr, vp, real_slp(base), h, hv]
        motion = (torch.stack([m(*args)[0] for m in ms]).mean(0)[0] * SC).float().numpy()  # [20,17]
    la, lo = lat_a[base], lon_a[base]
    lats, lons, vmaxs, press = [], [], [], []
    for L in range(20):
        e, n = motion[L, 0], motion[L, 1]
        la = la + n / R; lo = lo + e / (R * math.cos(math.radians(la))); lats.append(la); lons.append(lo)
        vmaxs.append(float(motion[L, 2])); press.append(float(motion[L, 3]))
    return lats, lons, vmaxs, press, n_padded


out = {}
for tag, ms in (("v10", None), ("v23", M23), ("v35", M35)):
    out[tag] = {"issues": []}
    for base in range(N):
        lats, lons, vmaxs, press, n_padded = forecast(tag, ms, base)
        out[tag]["issues"].append({
            "issue_time": DOLPHIN_TIMES[base], "base_lat": float(lat_a[base]), "base_lon": float(lon_a[base]),
            "lats": [round(x, 3) for x in lats], "lons": [round(x, 3) for x in lons],
            "vmax": [round(x, 1) for x in vmaxs], "pressure": [round(x, 1) for x in press],
            "n_padded": n_padded,
        })
    latest = out[tag]["issues"][-1]
    print(f"{tag} latest forecast (issued {latest['issue_time']}): "
          f"120h -> {latest['lats'][-1]:.1f}N {latest['lons'][-1]:.1f}E, "
          f"vmax path {[round(v) for v in latest['vmax'][::4]]}")

json.dump(out, open("track_build/dolphin_v23_v35.json", "w"))
print("\nwrote track_build/dolphin_v23_v35.json")
