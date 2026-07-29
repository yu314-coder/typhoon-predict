"""Run TrackFormer v23 on a storm's track history, in either of two modes:

  IBTrACS-only (default, no --steering given): the model sees only the storm's own recent
  positions/wind/pressure -- exactly what IBTrACS (or any best-track record) gives you for a past
  storm, nothing else. The steering field and its 12h/24h history are zero-filled with an explicit
  availability flag, the same "unavailable == exact zeros, not fabricated" convention used
  throughout this project whenever a field genuinely isn't there.

  Full data (--steering given): the model additionally sees a real deep-layer-mean steering-wind
  patch (850/500/200 hPa u/v, weighted 0.269/0.500/0.231) around the storm -- for the current fix
  and, if present, t-12h/t-24h. This is what the project's headline 434.96 km number requires; on
  Typhoon Dolphin (2026) zeroing this field out shifted the 120h forecast position by ~600 km while
  softening the intensity forecast only modestly (99 -> 90 kt) -- see the project's README for the
  full ablation.

Usage:
    python run_v23.py --track my_storm.json --out forecast.json
    python run_v23.py --track my_storm.json --steering my_steering.npz --out forecast.json

--track JSON format: a list of fixes, OLDEST to NEWEST, spaced 6 hours apart, ending at the fix to
forecast from ("now"). Up to 9 fixes are used (fewer is fine -- the model pads with the same
pre-genesis zero-fill it saw for young storms in training); extra leading fixes beyond 9 are
ignored.
    [{"time": "2026-07-29T00:00", "lat": 14.1, "lon": 169.1, "vmax_kt": 121.6, "pres_hpa": 941},
     {"time": "2026-07-29T06:00", "lat": 14.5, "lon": 168.4, "vmax_kt": 121.7, "pres_hpa": 941}]
`pres_hpa` may be null/omitted per-fix if unknown -- it is then treated as unavailable for that fix
(zero-filled, flagged), not fabricated.

--steering NPZ format (optional): float32 arrays of shape [2,17,17] (u,v in m/s, 2.5 deg
resolution, +-20 deg box centered on the storm), keyed by ISO time strings matching entries in
--track (e.g. "2026-07-29T06:00"). Only the LAST fix's key is required; keys for the fixes 12h and
24h before it are used for v23's temporal-history stack if present, and zero-filled (flagged
unavailable) if not -- so a steering file with only the current fix still runs, just without the
temporal-history benefit. See _fetch_dolphin_steering.py in the repo root for a working example of
building this from NOAA/NOMADS GFS analysis fields for a live storm, or from ERA5 for a past one.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trackformer_v23 import build_v23, TMEAN, TSTD, DSC, TARGET_SCALE  # noqa: E402

R = 111.2       # km per degree latitude
HIST = 9        # kinematic-history window length the model was trained with

_here = os.path.dirname(os.path.abspath(__file__))
_terrain = np.load(os.path.join(_here, "v23_terrain_wp.npz"))
_T_LAT, _T_LON, _LSM = _terrain["lat"], _terrain["lon"], _terrain["lsm"]
_LAND = _LSM > 0.5
_LAND_LAT = _T_LAT[np.where(_LAND)[0]]
_LAND_LON = _T_LON[np.where(_LAND)[1]]


def dist2land(lat_i, lon_i):
    if len(_LAND_LAT) == 0:
        return 3000.0
    dlat = _LAND_LAT - lat_i
    dlon = (_LAND_LON - lon_i) * math.cos(math.radians(lat_i))
    return float(np.hypot(dlon * R, dlat * R).min())


def load_track(path):
    fixes = json.load(open(path))
    fixes = fixes[-HIST:] if len(fixes) > HIST else fixes
    times = [f["time"] for f in fixes]
    lat = np.array([f["lat"] for f in fixes], dtype="float64")
    lon = np.array([f["lon"] for f in fixes], dtype="float64")
    vmax = np.array([f["vmax_kt"] for f in fixes], dtype="float64")
    pres = np.array([f.get("pres_hpa", None) if f.get("pres_hpa", None) is not None else np.nan
                      for f in fixes], dtype="float64")
    tns = np.array([np.datetime64(t).astype("datetime64[ns]").astype("int64") for t in times])
    return times, tns, lat, lon, vmax, pres


def build_window(times, tns, lat, lon, vmax, pres):
    """Kinematic/thermodynamic feature window -- same construction as this project's other
    real-storm inference scripts (e.g. _dolphin_v23_v35.py / _noul_v33.py)."""
    n = len(times)
    base = n - 1
    hidx = [max(0, base - HIST + 1 + k) for k in range(HIST)]
    n_padded = max(0, HIST - 1 - base)
    t0 = int(tns[base])
    doy = (np.datetime64(times[base]) - np.datetime64(times[base][:4] + "-01-01")).astype(int) + 1
    phase = 2 * math.pi * doy / 365.25
    seq = np.zeros((HIST, 54), dtype="float32")
    prev, pdir = -1, None

    def mkm(a, b, c, d):
        dlat = c - a; dlon = ((d - b + 180) % 360) - 180
        return dlon * R * math.cos(math.radians((a + c) / 2)), dlat * R

    for i, idx in enumerate(hidx):
        e, n_ = mkm(lat[base], lon[base], lat[idx], lon[idx])
        se, sn = (0., 0.) if prev < 0 else mkm(lat[prev], lon[prev], lat[idx], lon[idx])
        f = seq[i]; f[0:4] = [e, n_, se, sn]
        vv = [vmax[idx], pres[idx], np.nan, np.nan]
        for j in range(4):
            f[4 + j] = vv[j] if np.isfinite(vv[j]) else 0.
        f[24:28] = [float(np.isfinite(x)) for x in vv]
        f[21:23] = [math.sin(phase), math.cos(phase)]; f[23] = (t0 - int(tns[idx])) / 3.6e12
        sp = math.hypot(se, sn); hs, hc = (se / sp, sn / sp) if (sp > 1e-3 and prev >= 0) else (0., 0.)
        f[40], f[41], f[42] = hs, hc, sp
        f[43] = (pdir[0] * hc - pdir[1] * hs) if (pdir and (hs or hc) and (pdir[0] or pdir[1])) else 0.
        if prev >= 0:
            dv = np.isfinite(vmax[prev]) and np.isfinite(vmax[idx])
            dp = np.isfinite(pres[prev]) and np.isfinite(pres[idx])
            f[44] = vmax[idx] - vmax[prev] if dv else 0.
            f[45] = pres[idx] - pres[prev] if dp else 0.
            f[46], f[47] = float(dv), float(dp)
        lat_i, lon_i = lat[idx], lon[idx]
        m = np.datetime64(times[idx]).astype("datetime64[M]").astype(int) % 12 + 1
        d2l = dist2land(lat_i, lon_i % 360)
        thermal = 0.5 * 23.44 * math.sin(2 * math.pi * (m - 3) / 12.0)
        f[48] = lat_i; f[49] = abs(lat_i); f[50] = math.sin(math.radians(lon_i)); f[51] = math.cos(math.radians(lon_i))
        f[52] = d2l; f[53] = max(0., min(31., 30. - 0.30 * abs(lat_i - thermal) ** 1.4))
        if hs or hc:
            pdir = (hs, hc)
        prev = idx

    seq_n = (seq - TMEAN) / TSTD
    vpair = np.concatenate([seq[-1, 2:4], seq[-2, 2:4]]).astype("float32")
    return seq_n, vpair, n_padded


def load_steering(path, times):
    """Returns (slp[1,4,17,17], hist[1,8,17,17], have[1,2]) for the LAST fix in `times`. `path` may
    be None -- then everything is zero-filled (the IBTrACS-only ablation)."""
    if path is None:
        return (np.zeros((1, 4, 17, 17), "float32"),
                np.zeros((1, 8, 17, 17), "float32"),
                np.zeros((1, 2), "float32"))
    dlm = np.load(path)
    now_t = np.datetime64(times[-1])

    def key_at(back_h):
        return str(now_t - np.timedelta64(back_h, "h"))

    slp = np.zeros((1, 4, 17, 17), "float32")
    now_key = times[-1]
    if now_key in dlm:
        uv = dlm[now_key]
        slp[0, 2:4] = np.clip(uv / DSC[:, None, None], -4.0, 4.0)
    elif key_at(0) in dlm:
        uv = dlm[key_at(0)]
        slp[0, 2:4] = np.clip(uv / DSC[:, None, None], -4.0, 4.0)

    hist = np.zeros((1, 8, 17, 17), "float32")
    have = np.zeros((1, 2), "float32")
    cur = slp[0]
    for c, back in enumerate((12, 24)):
        k = key_at(back)
        if k in dlm:
            uv = dlm[k]
            hist[0, c * 4 + 2:c * 4 + 4] = np.clip(uv / DSC[:, None, None], -4.0, 4.0)
            have[0, c] = 1.0
        else:
            hist[0, c * 4:(c + 1) * 4] = cur
    return slp, hist, have


@torch.no_grad()
def forecast(models, times, tns, lat, lon, vmax, pres, steering_path):
    seq_n, vpair, n_padded = build_window(times, tns, lat, lon, vmax, pres)
    tr = torch.from_numpy(seq_n[None]); vp = torch.from_numpy(vpair[None])
    slp, hist, have = load_steering(steering_path, times)
    args = [tr, vp, torch.from_numpy(slp), torch.from_numpy(hist), torch.from_numpy(have)]
    motion = torch.stack([m(*args)[0] for m in models]).mean(0)[0] * TARGET_SCALE
    motion = motion.float().numpy()   # [20, 17]
    la, lo = float(lat[-1]), float(lon[-1])
    lats, lons, vmaxs, presses = [], [], [], []
    for L in range(20):
        e, n_ = motion[L, 0], motion[L, 1]
        la = la + n_ / R; lo = lo + e / (R * math.cos(math.radians(la)))
        lats.append(la); lons.append(lo)
        vmaxs.append(float(motion[L, 2])); presses.append(float(motion[L, 3]))
    return lats, lons, vmaxs, presses, n_padded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, help="track-history JSON, see module docstring")
    ap.add_argument("--steering", default=None, help="steering NPZ (omit for IBTrACS-only mode)")
    ap.add_argument("--seeds", default=os.path.join(_here, "v23_seed*.pt"), help="checkpoint glob")
    ap.add_argument("--out", default=None, help="write forecast JSON here (default: stdout)")
    args = ap.parse_args()

    import glob
    ckpts = sorted(glob.glob(args.seeds))
    if not ckpts:
        sys.exit(f"no checkpoints matched {args.seeds!r}")
    models = []
    for c in ckpts:
        m = build_v23().eval()
        m.load_state_dict(torch.load(c, map_location="cpu", weights_only=False)["model"])
        models.append(m)
    mode = "IBTrACS-only (steering zeroed)" if args.steering is None else f"full data ({args.steering})"
    print(f"loaded {len(models)} v23 seeds, mode: {mode}", file=sys.stderr)

    times, tns, lat, lon, vmax, pres = load_track(args.track)
    lats, lons, vmaxs, presses, n_padded = forecast(models, times, tns, lat, lon, vmax, pres, args.steering)

    out = {"issue_time": times[-1], "mode": mode, "base_lat": float(lat[-1]), "base_lon": float(lon[-1]),
           "lead_hours": list(range(6, 121, 6)),
           "lats": [round(float(x), 3) for x in lats], "lons": [round(float(x), 3) for x in lons],
           "vmax_kt": [round(float(x), 1) for x in vmaxs], "pres_hpa": [round(float(x), 1) for x in presses],
           "n_padded_history": n_padded}
    text = json.dumps(out, indent=2)
    if args.out:
        open(args.out, "w").write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
