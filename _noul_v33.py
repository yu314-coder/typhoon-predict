"""Run v23 and v31 on Typhoon Noul (2026) -- genuinely unseen data. Noul is NOT in
track_build/track_windows_v13.npz (max base_time there is 2026-07-13); it formed 2026-07-21 and
made landfall in Guangdong 2026-07-26, entirely after the training/test data cutoff. Confirmed
absent from the local IBTrACS snapshot too (its "NOUL" entries stop at 2015).

TRACK DATA. Hand-compiled from live sources (see citations below), cross-checked where they
overlap:
  - Wunderground's 6-hourly table (lat/lon/wind, Jul 23 06h - Jul 26 00h) -- the primary source,
    since it is the only one with a complete, regular grid.
  - Pressure attached from Zoom Earth's independent intensity table at matching timestamps
    (verified consistent: e.g. both give ~40kt/996hPa at Jul 24 00h, ~85kt/967hPa at Jul 25 18h).
  - PredictWind's position table cross-checked the Jul 23-25 positions independently (small
    inter-agency differences of ~0.1-0.3 deg, as expected between real advisory sources).

WHAT IS HONESTLY MISSING, NOT FABRICATED (unavailable == exact zero + flagged, project convention):
  - RMW, gust, and R34/R50/R64 quadrant wind radii: not published in these public trackers.
  - The steering-wind patch v23's steer_cnn normally reads: ERA5 stops ~March 2026 (confirmed
    earlier this session) and NOMADS OpenDAP (which would have real-time GFS analysis) was
    retired. v23 runs on kinematic history alone here -- an honest, harder test than its usual
    evaluation, not a like-for-like comparison to the reported 434.96 km bar.
  - dist2land IS real, not a fallback: computed from track_build/terrain_wp.npz, the same
    coastline data v31's LandDrag uses, not IBTrACS's own DIST2LAND field (unavailable for a
    storm not in IBTrACS).

Sources: Wunderground (wunderground.com/hurricane/western-pacific/2026/typhoon-noul),
Zoom Earth (zoom.earth/storms/noul-2026), PredictWind (predictwind.com/weather/severe-storms/
typhoon-noul), Weathernews (wxtech.weathernews.com/en/news/20260724-01). Post-landfall extension
(Jul 26 06h/12h, verification-only) from HKO (hko.gov.hk/textonly/v2/tc/tcp.htm), cross-checked
against Zoom Earth for wind/pressure -- see inline comment at the RAW list for the sourcing detail
and why HKO was preferred over a second candidate (taifengshuo.com) that diverged badly pre-landfall.
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R = 111.2; HIST = 9; LEADS_H = list(range(6, 121, 6)); A_MAX = 0.65
KT_PER_MPH = 0.868976

# ---- hand-compiled track (see docstring for sources) ----
# (date_str, lat, lon, wind_mph, pressure_hPa_or_None)
RAW = [
    ("2026-07-23T06:00", 17.4, 128.4, 30, None),
    ("2026-07-23T12:00", 18.0, 126.9, 35, None),
    ("2026-07-23T18:00", 18.3, 125.1, 40, 998),
    ("2026-07-24T00:00", 18.7, 123.5, 45, 996),
    ("2026-07-24T06:00", 18.8, 121.8, 50, 997),
    ("2026-07-24T12:00", 19.7, 120.4, 70, 991),
    ("2026-07-24T18:00", 20.0, 119.3, 75, 983),
    ("2026-07-25T00:00", 20.8, 118.3, 80, 980),
    ("2026-07-25T06:00", 21.3, 116.6, 85, 975),
    ("2026-07-25T12:00", 21.8, 115.9, 90, 970),
    ("2026-07-25T18:00", 22.4, 115.1, 100, 967),
    ("2026-07-26T00:00", 22.9, 114.5, 85, None),
    # ---- post-landfall extension, verification-only (never issued as a forecast: no fetched
    # steering for these times, and by now the system is a dissipating inland remnant). Position
    # from HKO's textonly track (hko.gov.hk/textonly/v2/tc/tcp.htm, HKT converted to UTC), chosen
    # over a second candidate source (taifengshuo.com) because HKO's pre-landfall points matched
    # this list's already-cross-validated Wunderground/Zoom Earth positions to within ~0.1-0.4 deg
    # throughout, while the alternate source diverged by 2+ deg during the early depression phase.
    # Wind at 06:00 is Zoom Earth's real reading at that exact hour (110 km/h, 985 hPa, both exact
    # matches to HKO's independent classification); wind at 12:00 has no exact-hour reading in any
    # source, so it's linearly interpolated between Zoom Earth's real 06:00 and 18:00 values
    # (110 -> 75 km/h) -- bounded by two real observations, not invented from nothing. Pressure at
    # 12:00 is left unavailable since the 18:00 bracket lacks a reported pressure to interpolate.
    ("2026-07-26T06:00", 23.5, 114.1, 68, 985),
    ("2026-07-26T12:00", 24.4, 113.7, 57, None),
]
N = len(RAW)
tns = np.array([np.datetime64(r[0]).astype("datetime64[ns]").astype("int64") for r in RAW])
lat_a = np.array([r[1] for r in RAW]); lon_a = np.array([r[2] for r in RAW])
vmax_a = np.array([r[3] * KT_PER_MPH for r in RAW])
pres_a = np.array([r[4] if r[4] is not None else np.nan for r in RAW])

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

v31src = open("colab_v36_train.py").read()
land_src = re.search(r"class LandDrag\(nn\.Module\):.*?\n        return self\.a_max \* torch\.tanh\(knots @ self\.Wi\.t\(\)\)\n", v31src, re.S).group(0)
tfl_src = re.search(r"class TrackFormerLand\(V23\):.*?\n        return s, ls, fp\n", v31src, re.S).group(0)
USE_LAND = 1


def build_land_cls(a_max):
    g = {"V23": V23, "torch": torch, "nn": nn, "F": F, "math": math, "A_MAX": a_max, "USE_LAND": USE_LAND}
    exec(land_src, g); exec(tfl_src, g)
    return g["TrackFormerLand"]


TrackFormerLand = build_land_cls(A_MAX)     # v31, a_max=0.65
TrackFormerLand33 = build_land_cls(1.0)     # v33, a_max=1.0

# ---- real land geometry, same source v31 was trained on ----
_T = np.load("track_build/terrain_wp.npz")
T_LAT, T_LON, LSM, ELEV = _T["lat"], _T["lon"], _T["lsm"], _T["elev"]
_LAND = LSM > 0.5
LAND_LAT = T_LAT[np.where(_LAND)[0]]; LAND_LON = T_LON[np.where(_LAND)[1]]; LAND_ELEV = ELEV[_LAND]


def land_features_np(lat, lon, heading_rad):
    n = len(lat)
    dlat = LAND_LAT[None, :] - lat[:, None]
    dlon = (LAND_LON[None, :] - lon[:, None]) * np.cos(np.radians(lat[:, None]))
    dy = dlat * R; dx = dlon * R
    dist = np.hypot(dx, dy); bearing = np.arctan2(dx, dy)
    dphi = np.arctan2(np.sin(bearing - heading_rad[:, None]), np.cos(bearing - heading_rad[:, None]))
    cone = np.abs(dphi) <= np.radians(30.0)
    ahead = np.where(cone, dist, np.inf)
    dist_nearest = dist.min(1); dist_ahead = ahead.min(1)
    dist_ahead = np.where(np.isfinite(dist_ahead), dist_ahead, 3000.0)
    near_ahead = ahead < 500.0
    terr_ahead = np.array([LAND_ELEV[near_ahead[k]].max() if near_ahead[k].any() else 0.0
                            for k in range(n)])
    return np.stack([dist_nearest, dist_ahead, terr_ahead], 1).astype("float32")


def mkm(a, b, c, d):
    dlat = c - a; dlon = ((d - b + 180) % 360) - 180
    return dlon * R * math.cos(math.radians((a + c) / 2)), dlat * R


def ok(v):
    return v is not None and np.isfinite(v)


RN = [f"r{t}_{q}" for t in (34, 50, 64) for q in ("ne", "se", "sw", "nw")]


def build_window(base):
    """54-column history array ending at index `base`, matching track_windows_v13.npz's layout
    (verbatim from predict_storms_v13.py's build(), radii/gust/rmw/steering unavailable).

    PADDING FOR EARLY ISSUES. Windows with base < HIST-1 don't have 9 real timesteps behind them
    (our earliest confirmed position is Jul 23 06h, close to genesis). Rather than fabricate an
    earlier position from vague pre-formation reports ("1,080 km east of Luzon"), the missing
    early steps repeat the EARLIEST REAL point -- a stationary-storm padding, not an invented
    position. This is flagged per-issue in the printed output, never silently."""
    hidx = [max(0, base - HIST + 1 + k) for k in range(HIST)]
    n_padded = max(0, HIST - 1 - base)   # steps clamped to index 0 (repeats of the earliest real point)
    t0 = int(tns[base])
    doy = (np.datetime64(RAW[base][0]) - np.datetime64(f"{RAW[base][0][:4]}-01-01")).astype(int) + 1
    phase = 2 * math.pi * doy / 365.25
    seq = np.zeros((HIST, 54), dtype="float32"); prev, pdir = -1, None
    for i, idx in enumerate(hidx):
        e, n = mkm(lat_a[base], lon_a[base], lat_a[idx], lon_a[idx])
        se, sn = (0., 0.) if prev < 0 else mkm(lat_a[prev], lon_a[prev], lat_a[idx], lon_a[idx])
        f = seq[i]; f[0:4] = [e, n, se, sn]
        vv = [vmax_a[idx], pres_a[idx], np.nan, np.nan]   # gust, rmw unavailable
        for j in range(4):
            f[4 + j] = vv[j] if ok(vv[j]) else 0.
        # R34/R50/R64 quadrant radii: unavailable (not in these public trackers)
        fl = vv + [np.nan] * 12
        f[24:28] = [float(ok(x)) for x in fl[:4]]; f[28:40] = [float(ok(x)) for x in fl[4:]]
        f[21:23] = [math.sin(phase), math.cos(phase)]; f[23] = (t0 - int(tns[idx])) / 3.6e12
        sp = math.hypot(se, sn); hs, hc = (se / sp, sn / sp) if (sp > 1e-3 and prev >= 0) else (0., 0.)
        f[40], f[41], f[42] = hs, hc, sp
        f[43] = (pdir[0] * hc - pdir[1] * hs) if (pdir and (hs or hc) and (pdir[0] or pdir[1])) else 0.
        if prev >= 0:
            dv = ok(vmax_a[prev]) and ok(vmax_a[idx]); dp = ok(pres_a[prev]) and ok(pres_a[idx])
            f[44] = vmax_a[idx] - vmax_a[prev] if dv else 0.
            f[45] = pres_a[idx] - pres_a[prev] if dp else 0.
            f[46], f[47] = float(dv), float(dp)
        lat_i, lon_i, m = lat_a[idx], lon_a[idx], np.datetime64(RAW[idx][0]).astype("datetime64[M]").astype(int) % 12 + 1
        d2l = land_features_np(np.array([lat_i]), np.array([lon_i % 360]), np.array([0.0]))[0, 0]  # nearest land, any dir
        thermal = 0.5 * 23.44 * math.sin(2 * math.pi * (m - 3) / 12.0)
        f[48] = lat_i; f[49] = abs(lat_i); f[50] = math.sin(math.radians(lon_i)); f[51] = math.cos(math.radians(lon_i))
        f[52] = float(d2l); f[53] = max(0., min(31., 30. - 0.30 * abs(lat_i - thermal) ** 1.4))
        if hs or hc:
            pdir = (hs, hc)
        prev = idx
    seq_n = (seq - tmean) / tstd
    vpair = np.concatenate([seq[-1, 2:4], seq[-2, 2:4]]).astype("float32")
    return seq_n, vpair, n_padded


def load(cls, paths):
    ms = []
    for p in paths:
        m = cls().eval()
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        m.load_state_dict(sd)
        ms.append(m)
    return ms


MS = {"v23": load(V23, sorted(glob.glob("downloads/x/v23_seed*.pt"))),
      "v31": load(TrackFormerLand, sorted(glob.glob("downloads/v31ck/**/v31_seed*.pt", recursive=True))),
      "v33": load(TrackFormerLand33, sorted(glob.glob("downloads/v33ck/**/v33_seed*.pt", recursive=True)))}
print(f"v23: {len(MS['v23'])} | v31: {len(MS['v31'])} | v33: {len(MS['v33'])} seeds")

ZERO_HIST = torch.zeros(1, 8, 17, 17)
ZERO_HAVE = torch.zeros(1, 2)

# ---- REAL deep-layer-mean steering (850/500/200 hPa u/v, weighted 0.269/0.500/0.231 -- the exact
# recipe extract_tip_dlm.py uses), fetched from NOAA's GFS analysis via _fetch_noul_steering.py.
# Channels 0-1 (SLPanom, SLPtend) stay zero -- no real SLP source for this storm. Normalized with
# the TRAINING scale, not Noul's own statistics (extract_tip_dlm.py's rule: "Rescaling to Tip's own
# statistics would hand the model an input distribution it never saw").
_dlm_scale = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_dlm_raw = np.load("track_build/noul_dlm4_real.npz")
_ISSUE_KEYS = ["20260723_06", "20260723_12", "20260723_18", "20260724_00", "20260724_06",
               "20260724_12", "20260724_18", "20260725_00", "20260725_06", "20260725_12",
               "20260725_18", "20260726_00"]


def real_slp(base):
    uv = _dlm_raw[_ISSUE_KEYS[base]]   # [2,17,17] raw m/s
    uv_n = np.clip(uv / _dlm_scale[:, None, None], -4.0, 4.0)
    slp = np.zeros((1, 4, 17, 17), "float32")
    slp[0, 2:4] = uv_n
    return torch.from_numpy(slp)


# ---- REAL t-12h/t-24h historical steering, for free: our issues are spaced exactly 6h apart, so
# "2 steps back" (12h) and "4 steps back" (24h) land exactly on earlier issues we already fetched
# real DLM data for. Genuinely unavailable only for the two earliest issues (before 07-23T06) --
# matching _v31tracks.py's verified (Colab-reproducing) convention for that case: fall back to the
# CURRENT window's own steering with have=0, not zero. HistStem was trained on that exact fallback
# pattern (self-repeat + an explicit have-channel telling it so), never on an all-zero history
# plane, so zero-filling would hand it an input distribution it never saw.
def real_hist(base):
    cur = real_slp(base).numpy()[0]   # [4,17,17], this window's own real steering
    hist = np.zeros((1, 8, 17, 17), "float32")
    have = np.zeros((1, 2), "float32")
    for c, back in enumerate((2, 4)):
        j = base - back
        if j >= 0:
            uv = _dlm_raw[_ISSUE_KEYS[j]]
            uv_n = np.clip(uv / _dlm_scale[:, None, None], -4.0, 4.0)
            hist[0, c * 4 + 2:c * 4 + 4] = uv_n
            have[0, c] = 1.0
        else:
            hist[0, c * 4:(c + 1) * 4] = cur
    return torch.from_numpy(hist), torch.from_numpy(have)


@torch.no_grad()
def forecast(tag, base):
    seq_n, vpair, n_padded = build_window(base)
    tr = torch.from_numpy(seq_n[None]); vp = torch.from_numpy(vpair[None])
    h, hv = real_hist(base)
    args = [tr, vp, real_slp(base), h, hv]
    if tag in ("v31", "v33"):
        phi = math.atan2(vpair[1], vpair[0])
        lf3 = land_features_np(np.array([lat_a[base]]), np.array([lon_a[base] % 360]), np.array([phi]))
        spd = math.hypot(vpair[0], vpair[1])
        latv = seq_n[-1, 48] * tstd[48] + tmean[48]
        lf5 = np.concatenate([lf3[0] / np.array([1000., 1000., 1000.], "float32"),
                              [spd / 100.0], [abs(latv) / 30.0]]).astype("float32")
        lok = np.array([1.0], "float32")
        args += [torch.from_numpy(lf5[None]), torch.from_numpy(lok)]
    motion = (torch.stack([m(*args)[0] for m in MS[tag]]).mean(0)[0, :, :2] * SC[:2]).float().numpy()
    la, lo = lat_a[base], lon_a[base]
    lats, lons = [la], [lo]
    for e, n in motion:
        la = la + n / R; lo = lo + e / (R * math.cos(math.radians(la))); lats.append(la); lons.append(lo)
    return lats[1:], lons[1:], n_padded


# ---- issue times: the 12 real named-storm points only (not the 2 post-landfall extension points
# appended above -- those have no fetched steering and are verification-only ground truth). Early
# issues (base < HIST-1) use padded pre-genesis history (see build_window docstring), flagged
# per-issue below, not hidden.
issue_idx = list(range(12))
print(f"{len(issue_idx)} issue times: {[RAW[i][0] for i in issue_idx]}")

out = {tag: {"lat": [], "lon": [], "base_time": [], "base_lat": [], "base_lon": [], "n": 0}
       for tag in ("v23", "v31", "v33")}
for i in issue_idx:
    t0ns = int(tns[i]); pad_note = ""
    for tag in ("v23", "v31", "v33"):
        lats, lons, n_padded = forecast(tag, i)
        out[tag]["lat"].append([round(x, 3) for x in lats])
        out[tag]["lon"].append([round(x, 3) for x in lons])
        out[tag]["base_time"].append(t0ns)
        out[tag]["base_lat"].append(round(float(lat_a[i]), 3))
        out[tag]["base_lon"].append(round(float(lon_a[i]), 3))
        pad_note = f"  ({n_padded} pre-genesis padded steps)" if n_padded else ""
    print(f"  issued {RAW[i][0]}  vmax {vmax_a[i]:.0f}kt  pos {lat_a[i]:.1f}N {lon_a[i]:.1f}E{pad_note}", flush=True)

# ---- verification: our track only spans ~66h from the earliest issue, so a genuine 120h
# check (what every historical-storm map uses) is impossible for any single issue here. Use the
# LONGEST lead every issue except the very last one (which has zero forward data) actually has:
# 6h. The map caption reads this honestly instead of mislabeling a short-lead error as "120h".
VERIFY_LEAD_H = 6
verifiable = [i for i in issue_idx if i + 1 < N]   # excludes the final issue (no forward truth at all)
for tag in ("v23", "v31", "v33"):
    errs = []
    for i in verifiable:
        row = issue_idx.index(i)
        la1, lo1 = out[tag]["lat"][row][0], out[tag]["lon"][row][0]   # first lead = 6h
        e, n = mkm(la1, lo1, lat_a[i + 1], lon_a[i + 1])
        errs.append(math.hypot(e, n))
    out[tag]["err120_mean"] = float(np.mean(errs))
    out[tag]["lead_h"] = VERIFY_LEAD_H
    out[tag]["n"] = len(issue_idx)
    print(f"  {tag}: mean {VERIFY_LEAD_H}h error {out[tag]['err120_mean']:.0f} km "
          f"(over {len(verifiable)}/{len(issue_idx)} issues with a forward point to check)")

# ---- landfall-window check: does v31's land-drag correction actually help here? The 2
# post-landfall extension points (idx 12-13, Jul 26 06h/12h) push several issues' longer leads
# (18-78h) into range against REAL post-landfall truth for the first time. Every (issue, lead)
# pair whose target lands on our uniform 6h grid gets compared; split into "near/at landfall"
# (target idx 9-13, i.e. truth from Jul 25 12h onward -- within ~500km of the coast through well
# inland) vs "open ocean" (target idx <= 8) to isolate where land interaction actually matters,
# rather than mixing it into one aggregate the way the full WP+EP test set did.
print("\nlandfall-window check (v23 vs v31 vs v33, split by how close the VERIFIED target is to land):")
TAGS3 = ("v23", "v31", "v33")
bucket_errs = {"open_ocean": {t: [] for t in TAGS3}, "near_landfall": {t: [] for t in TAGS3}}
for i in issue_idx:
    row = issue_idx.index(i)
    for l in range(20):
        tgt = i + l + 1
        if tgt >= N:
            break
        lead_h = 6 * (l + 1)
        bucket = "near_landfall" if tgt >= 9 else "open_ocean"
        for tag in TAGS3:
            la1, lo1 = out[tag]["lat"][row][l], out[tag]["lon"][row][l]
            e, n = mkm(la1, lo1, lat_a[tgt], lon_a[tgt])
            bucket_errs[bucket][tag].append(math.hypot(e, n))
for bucket in ("open_ocean", "near_landfall"):
    n = len(bucket_errs[bucket]["v23"])
    if n:
        means = {t: float(np.mean(bucket_errs[bucket][t])) for t in TAGS3}
        print(f"  {bucket:13s} (n={n:3d}): v23 {means['v23']:6.1f} km | v31 {means['v31']:6.1f} km | "
              f"v33 {means['v33']:6.1f} km")
for tag in TAGS3:
    out[tag]["landfall_check"] = {b: float(np.mean(bucket_errs[b][tag])) if bucket_errs[b][tag] else None for b in bucket_errs}

# ---- observed continuation, for the map's dotted "what actually happened" line ----
observed_lat = [round(float(x), 3) for x in lat_a]
observed_lon = [round(float(x), 3) for x in lon_a]
observed_time = [int(x) for x in tns]

for tag in ("v23", "v31", "v33"):
    p = f"track_build/{tag}_tracks.json"
    d = json.load(open(p)) if os.path.exists(p) else {}
    d["Noul"] = out[tag]
    json.dump(d, open(p, "w"))
    print(f"added Noul to {p}")

# v10_tracks.json supplies "observed" positions on the map -- add Noul there too, without
# touching its existing storms.
p10 = "track_build/v10_tracks.json"
d10 = json.load(open(p10))
d10["Noul"] = {"lat": [observed_lat] * len(issue_idx), "lon": [observed_lon] * len(issue_idx),
               "base_time": [int(tns[i]) for i in issue_idx],
               "base_lat": [round(float(lat_a[i]), 3) for i in issue_idx],
               "base_lon": [round(float(lon_a[i]), 3) for i in issue_idx],
               "n": len(issue_idx)}
json.dump(d10, open(p10, "w"))
print(f"added Noul to {p10} (observed reference)")
print("\ndone -- steering (current + t-12h/t-24h history) was REAL GFS data, land geometry was REAL")
