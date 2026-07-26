"""GO/NO-GO for a land-interaction correction: does terrain/coastline explain PREDICTABLE
structure in v23's error that its current (purely kinematic) state doesn't already capture?

WHY. v23 has no land feature of any kind (verified 2026-07-25) -- the 54 track columns are
kinematics/intensity only. Land drag (Gaemi stalling ~a day at Taiwan, 2024) is a real candidate
for the KNOWN along-track speed bias _shortlead.py found (-7/-17/-28/-40 km at 6/12/18/24h,
growing to -331 km at 120h), which _speedfix.py showed a GLOBAL correction cannot touch --
correcting a per-storm-conditional effect with one global scale factor is mathematically the wrong
tool, so that null result does NOT rule out land.

THE TEST (same effect-size gate as _shortlead.py, which killed v30 before any training). Fit Ridge
on VALIDATION storms twice: (A) kinematic features alone (current motion, heading, turn rate --
what v23 already sees), (B) kinematic + LAND features (distance to nearest coastline, distance to
land along the current heading, terrain height in that direction). If (B) beats (A) by more than
noise on held-out TEST storms, land is carrying real, currently-untapped signal. If not, land is
not the explanation for the along-track bias -- skip, like v28 and v30.

SCOPE. terrain_wp.npz covers WP only (100-148E, 3-47N) -- this test is WP-basin storms only,
matching the same constraint as the ERA5 steering pull. EP is out of scope for this data.
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R_KM = 111.2
KM6H = 6 * 3600 / 1000.0

nb = json.load(open("colab_train_v17.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
body = body.replace('"/content/d/steer5_int8.npz"', '"track_build/dlm4_int8.npz"')
body = body.replace('"/content/d/track_windows_v13.npz"', '"track_build/track_windows_v13.npz"')
body = body.replace('DEVICE = torch.device("cuda")', 'DEVICE = torch.device("cpu")')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os,
     "json": json, "time": __import__("time"), "math": math}
exec(compile(body, "<v17-notebook>", "exec"), G)
Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
vpair = G["vpair"]; basins = G["basins"]; z = G["z"]; SC = G["TARGET_SCALE"]
va_idx, te_idx = G["va_idx"], G["te_idx"]; nl = z["n_leads"].astype(int)
tmean = G["tmean"]; tstd = G["tstd"]

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))
CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]
v28 = open("colab_v28_train.py").read()
hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", v28, re.S).group(0)
tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", v28, re.S).group(0)
g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(hs, g23); exec(tf, g23)
V23 = g23["TrackFormerHist"]

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

MS = []
for p in sorted(glob.glob("downloads/x/v23_seed*.pt")):
    m = V23().eval(); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"]); MS.append(m)
print(f"v23: {len(MS)} seeds")

TM = tmean; TS = tstd


@torch.no_grad()
def errors(idx):
    """v23 cumulative E/N error [n,20,2] km."""
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        h = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1))
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j]), h, torch.from_numpy(HAVE[j])]
        P.append((torch.stack([m(*a)[0] for m in MS]).mean(0)[..., :2] * SC[:2]).float().numpy())
    return np.cumsum(np.concatenate(P), 1) - np.cumsum(target[idx][..., :2], 1)


def along_cross(idx, E):
    obs = np.cumsum(target[idx][..., :2], 1)
    prev = np.concatenate([np.zeros((len(idx), 1, 2)), obs[:, :-1]], 1)
    step = obs - prev
    hd = np.arctan2(step[..., 1], step[..., 0])
    uE, uN = np.cos(hd), np.sin(hd)
    along = E[..., 0] * uE + E[..., 1] * uN
    cross = -E[..., 0] * uN + E[..., 1] * uE
    return along, cross


# ---- land geometry from terrain_wp.npz ----
T = np.load("track_build/terrain_wp.npz")
T_LAT, T_LON, LSM, ELEV = T["lat"], T["lon"], T["lsm"], T["elev"]
LAND = LSM > 0.5
land_lat = T_LAT[np.where(LAND)[0]]
land_lon = T_LON[np.where(LAND)[1]]
land_elev = ELEV[LAND]
print(f"terrain: {LAND.sum():,} land cells, box lat {T_LAT[-1]:.0f}-{T_LAT[0]:.0f}N lon "
      f"{T_LON[0]:.0f}-{T_LON[-1]:.0f}E")


def land_features(lat, lon, heading_rad):
    """Per-window: [dist_to_nearest_land_km, dist_to_land_ahead_km, max_terrain_ahead_m].
    'ahead' = within a +/-30 deg cone of the current heading, out to 2000 km (spans the whole
    forecast horizon's worst-case translation). All computed from CURRENT state only -- no future
    information -- matching what a model could actually use at inference time."""
    n = len(lat)
    dlat = land_lat[None, :] - lat[:, None]
    dlon = (land_lon[None, :] - lon[:, None]) * np.cos(np.radians(lat[:, None]))
    dy = dlat * R_KM; dx = dlon * R_KM
    dist = np.hypot(dx, dy)                                    # [n, n_land] km
    bearing = np.arctan2(dx, dy)                                # 0 = north, clockwise
    dphi = np.arctan2(np.sin(bearing - heading_rad[:, None]), np.cos(bearing - heading_rad[:, None]))
    cone = np.abs(dphi) <= np.radians(30.0)
    ahead = np.where(cone, dist, np.inf)
    dist_nearest = dist.min(1)
    dist_ahead = ahead.min(1)
    dist_ahead = np.where(np.isfinite(dist_ahead), dist_ahead, 3000.0)  # no land in cone -> far
    near_ahead = ahead < 500.0
    terr_ahead = np.array([land_elev[near_ahead[k]].max() if near_ahead[k].any() else 0.0
                            for k in range(n)])
    return np.stack([dist_nearest, dist_ahead, terr_ahead], 1)


def kin_features(idx):
    v0 = vpair[idx, :2]; vp = vpair[idx, 2:]
    s0 = np.hypot(v0[:, 0], v0[:, 1]) + 1e-6
    phi0 = np.arctan2(v0[:, 1], v0[:, 0])
    dphi = phi0 - np.arctan2(vp[:, 1], vp[:, 0])
    omega = np.arctan2(np.sin(dphi), np.cos(dphi))
    lat = track[idx, -1, 48] * TS[48] + TM[48]
    vmax = track[idx, -1, 4] * TS[4] + TM[4]
    vmax_p = track[idx, -2, 4] * TS[4] + TM[4]
    spd_p = np.hypot(vp[:, 0], vp[:, 1])
    X = np.stack([v0[:, 0], v0[:, 1], s0, np.sin(phi0), np.cos(phi0), omega,
                  vp[:, 0], vp[:, 1], vmax, lat, np.abs(lat), s0 - spd_p, vmax - vmax_p], 1)
    return X.astype("float64"), phi0


full = nl == 20; wp = basins == "WP"
VA = np.array([i for i in va_idx if full[i] and wp[i]])
TE = np.array([i for i in te_idx if full[i] and wp[i]])
print(f"WP-only, full 20-lead: val {len(VA)}, test {len(TE)}")

print("computing v23 errors ...", flush=True)
Ev, Et = errors(VA), errors(TE)
av_al, av_cr = along_cross(VA, Ev)
at_al, at_cr = along_cross(TE, Et)

Kv, phiv = kin_features(VA)
Kt, phit = kin_features(TE)
print("computing land geometry ...", flush=True)
Lv = land_features(bla[VA], blo[VA] % 360, phiv)
Lt = land_features(bla[TE], blo[TE] % 360, phit)
print(f"  dist_to_nearest_land (km): val mean {Lv[:,0].mean():.0f} median {np.median(Lv[:,0]):.0f}")
print(f"  dist_to_land_ahead   (km): val mean {Lv[:,1].mean():.0f} median {np.median(Lv[:,1]):.0f}  "
      f"(<=500km: {100*(Lv[:,1]<=500).mean():.1f}%)")


def standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


def ridge_r2(y_tr, X_tr, y_te, X_te, lam=10.0):
    A = X_tr.T @ X_tr + lam * np.eye(X_tr.shape[1])
    w = np.linalg.solve(A, X_tr.T @ y_tr)
    pred = X_te @ w
    ss_res = ((y_te - pred) ** 2).sum(); ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, pred


Kv_s, mu, sd = standardize(Kv); Kt_s, _, _ = standardize(Kt, mu, sd)
Lv_s, muL, sdL = standardize(Lv); Lt_s, _, _ = standardize(Lt, muL, sdL)
Xv_kin = np.hstack([Kv_s, np.ones((len(Kv_s), 1))])
Xt_kin = np.hstack([Kt_s, np.ones((len(Kt_s), 1))])
Xv_land = np.hstack([Kv_s, Lv_s, np.ones((len(Kv_s), 1))])
Xt_land = np.hstack([Kt_s, Lt_s, np.ones((len(Kt_s), 1))])

seed_floor = 5.5
print(f"\n{'lead':>5s} {'h':>4s} | {'RMS along':>9s} | {'R2 kin':>7s} {'R2 +land':>8s} | "
      f"{'removable kin':>13s} {'removable +land':>16s} {'land gain km':>12s}")
rows = []
for L in (0, 3, 7, 11, 15, 19):
    y_va, y_te = av_al[:, L], at_al[:, L]
    r2_kin, _ = ridge_r2(y_va, Xv_kin, y_te, Xt_kin)
    r2_land, pred_land = ridge_r2(y_va, Xv_land, y_te, Xt_land)
    std_te = y_te.std()
    rem_kin = std_te * math.sqrt(max(r2_kin, 0))
    rem_land = std_te * math.sqrt(max(r2_land, 0))
    gain = rem_land - rem_kin
    rmse = float(np.sqrt((y_te ** 2)).mean())
    print(f"{L+1:5d} {6*(L+1):4d} | {rmse:9.1f} | {r2_kin:7.3f} {r2_land:8.3f} | "
          f"{rem_kin:13.1f} {rem_land:16.1f} {gain:12.1f}")
    rows.append({"lead_h": 6*(L+1), "r2_kin": r2_kin, "r2_land": r2_land,
                 "removable_kin_km": rem_kin, "removable_land_km": rem_land, "gain_km": gain})

# decision: at the leads where land plausibly matters (48h+, enough time to reach terrain),
# does adding land features buy more than seed noise ABOVE the kinematic-only baseline?
gains = [r["gain_km"] for r in rows if r["lead_h"] >= 48]
max_gain = max(gains) if gains else 0.0
print("\n" + "=" * 90)
print(f"max incremental land-feature gain (48h+ leads): {max_gain:.1f} km  |  seed-noise floor: ~{seed_floor:.1f} km")
if max_gain > seed_floor:
    print("  -> land geometry adds PREDICTABLE structure beyond what v23's kinematic state already")
    print("     captures. Worth building a land-interaction correction/branch.")
elif max_gain > 0.4 * seed_floor:
    print("  -> modest incremental signal; land might buy a little. Marginal, borderline call.")
else:
    print("  -> land geometry adds essentially NOTHING beyond current kinematic state. The along-")
    print("     track bias is not explained by land proximity/terrain -- skip, like v28 and v30.")
print("=" * 90)

json.dump({"rows": rows, "max_gain_48h_plus_km": max_gain, "seed_floor_km": seed_floor,
           "n_val": len(VA), "n_test": len(TE)},
          open("track_build/landtest.json", "w"), indent=1, default=float)
print("wrote track_build/landtest.json")

# ---- DIRECT test: the ridge regression pools ALL windows, so a localized effect (most storms are
# in the open ocean -- median dist_to_land_ahead is capped at 3000 km) gets diluted to near-zero
# signal density. Compare along-track BIAS directly between "land ahead soon" vs "open ocean"
# storms -- a group comparison has much more power for an effect that only fires in a minority
# subset, exactly the shape a landfall-drag effect would have. ----
NEAR_KM = 500.0
near_t = Lt[:, 1] <= NEAR_KM
print(f"\n{'='*90}\nDIRECT GROUP COMPARISON: land within {NEAR_KM:.0f} km ahead (n={near_t.sum()}) "
      f"vs open ocean (n={(~near_t).sum()}), TEST set")
print(f"{'lead':>5s} {'h':>4s} | {'near-land bias':>15s} {'open-ocean bias':>16s} {'diff':>8s} | "
      f"{'near SE':>8s} {'far SE':>7s} {'|diff|/SE':>10s}")
group_rows = []
for L in (0, 1, 2, 3, 5, 7, 11, 15, 19):
    y = at_al[:, L]
    yn, yf = y[near_t], y[~near_t]
    bn, bf = yn.mean(), yf.mean()
    se_n = yn.std() / math.sqrt(max(len(yn), 1))
    se_f = yf.std() / math.sqrt(max(len(yf), 1))
    se_diff = math.hypot(se_n, se_f)
    z = abs(bn - bf) / se_diff if se_diff > 0 else 0.0
    print(f"{L+1:5d} {6*(L+1):4d} | {bn:15.1f} {bf:16.1f} {bn-bf:8.1f} | "
          f"{se_n:8.1f} {se_f:7.1f} {z:10.2f}")
    group_rows.append({"lead_h": 6*(L+1), "near_bias_km": float(bn), "far_bias_km": float(bf),
                        "diff_km": float(bn-bf), "z": float(z), "n_near": int(len(yn)), "n_far": int(len(yf))})
print("=" * 90)
print("z = |diff| / SE(diff): a rough normal-approximation check, not a bootstrap. z>2 is worth a")
print("real bootstrap before building anything; z<1 across all leads means the near-land subset")
print("is not behaving differently from the rest, i.e. land really is not the explanation.")
json.dump(group_rows, open("track_build/landtest_groups.json", "w"), indent=1, default=float)
print("wrote track_build/landtest_groups.json")
