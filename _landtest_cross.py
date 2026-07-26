"""Does land carry CROSS-TRACK signal v23 is currently blind to? _landtest.py only ever tested
along-track bias (speed along the current heading) because that's the only axis v31's LandDrag
can correct -- delta * unit_heading, a scalar magnitude, structurally unable to turn the storm
left or right. But the literature v31 was built without consulting says cross-track DEFLECTION,
not along-track speed change, is the dominant, best-documented near-land effect:

  - Chi et al. 2024 (MWR, Typhoon Chanthu near Taiwan, 60-15-1km MPAS): terrain-induced
    recirculating flow creates a wavenumber-1 asymmetry that deflects the track RIGHT then LEFT
    as the storm passes a mountainous island. The governing parameter is R/L_E (vortex size over
    effective terrain length), not distance-to-land. This is a genuinely 2D (turning) effect.
  - Nature Geoscience 2026 (land-sea thermal/roughness contrast): landfalling TCs actually
    ACCELERATE approaching the coast (~48% translation speed increase in the 60h before
    landfall) -- opposite sign from the "drag slows it down" framing LandDrag was implicitly
    built around -- driven by asymmetric flux between the land- and ocean-covered halves of the
    circulation, which requires knowing WHICH SIDE has land, not just how far away it is.

Both mechanisms are fundamentally about DIRECTIONAL land asymmetry relative to the storm, which
v31's 3 scalar features (dist_nearest, dist_ahead, terrain_ahead) cannot express and which its
along-track-only output cannot correct even if it could. This script reuses _landtest.py's exact
v23 error/land-geometry pipeline (same models, same WP-only val/test split) and asks two new
questions cheaply, with no new training:

  1. Does the near-land subgroup's CROSS-TRACK bias differ from the open-ocean subgroup's, the
     same direct group comparison _landtest.py ran for along-track (which found a real, z>2
     effect at 36-48h)?
  2. Within the near-land subgroup, does a signed LAND-ASYMMETRY feature (is nearby land
     systematically to the storm's left or right of its current heading) correlate with the
     SIGN of the cross-track error? This is the one thing a scalar along-track correction could
     never use even if v31 had converged perfectly.
"""
import json, re, math, os, sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
R_KM = 111.2

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
KM6H = 6 * 3600 / 1000.0
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


@torch.no_grad()
def errors(idx):
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


T = np.load("track_build/terrain_wp.npz")
T_LAT, T_LON, LSM, ELEV = T["lat"], T["lon"], T["lsm"], T["elev"]
LAND = LSM > 0.5
land_lat = T_LAT[np.where(LAND)[0]]; land_lon = T_LON[np.where(LAND)[1]]; land_elev = ELEV[LAND]
print(f"terrain: {LAND.sum():,} land cells")


def land_ahead_dist_terr(lat, lon, heading_rad):
    """Same 'dist to land within +/-30deg cone ahead' as _landtest.py, plus max terrain height in
    that cone within 500km -- so the near-land group can be split into MOUNTAINOUS coastline
    (Taiwan-CMR-like, the specific mechanism Chi et al. describe) vs flat/generic coastline, since
    a wavenumber-1 terrain-flow deflection plausibly needs real relief, not just any nearby shore."""
    n = len(lat)
    dlat = land_lat[None, :] - lat[:, None]
    dlon = (land_lon[None, :] - lon[:, None]) * np.cos(np.radians(lat[:, None]))
    dy = dlat * R_KM; dx = dlon * R_KM
    dist = np.hypot(dx, dy)
    bearing = np.arctan2(dx, dy)
    dphi = np.arctan2(np.sin(bearing - heading_rad[:, None]), np.cos(bearing - heading_rad[:, None]))
    cone = np.abs(dphi) <= np.radians(30.0)
    ahead = np.where(cone, dist, np.inf)
    dist_ahead = ahead.min(1)
    dist_ahead = np.where(np.isfinite(dist_ahead), dist_ahead, 3000.0)
    close = cone & (dist <= 500.0)
    terr_ahead = np.array([land_elev[close[k]].max() if close[k].any() else 0.0 for k in range(n)])
    return dist_ahead, terr_ahead


def land_asymmetry(lat, lon, heading_rad, radius_km=500.0):
    """Signed, inverse-distance-weighted 'is nearby land to my left or right of current heading'.
    +1 = all nearby land dead-left (dphi=+90deg), -1 = dead-right, 0 = symmetric or no land within
    radius_km. This is the one thing LandDrag's along-track-only output could never act on even
    if the feature existed -- the whole point of this script is checking whether it's predictive
    before spending effort building a head that CAN act on it."""
    n = len(lat)
    dlat = land_lat[None, :] - lat[:, None]
    dlon = (land_lon[None, :] - lon[:, None]) * np.cos(np.radians(lat[:, None]))
    dy = dlat * R_KM; dx = dlon * R_KM
    dist = np.hypot(dx, dy)
    bearing = np.arctan2(dx, dy)
    dphi = np.arctan2(np.sin(bearing - heading_rad[:, None]), np.cos(bearing - heading_rad[:, None]))
    near = dist <= radius_km
    w = 1.0 / np.clip(dist, 50.0, None)
    out = np.zeros(n, "float64")
    for k in range(n):
        m = near[k]
        if m.any():
            out[k] = float((np.sin(dphi[k][m]) * w[k][m]).sum() / w[k][m].sum())
    return out


full = nl == 20; wp = basins == "WP"
VA = np.array([i for i in va_idx if full[i] and wp[i]])
TE = np.array([i for i in te_idx if full[i] and wp[i]])
print(f"WP-only, full 20-lead: val {len(VA)}, test {len(TE)}")

print("computing v23 errors ...", flush=True)
Ev, Et = errors(VA), errors(TE)
av_al, av_cr = along_cross(VA, Ev)
at_al, at_cr = along_cross(TE, Et)


def kin_phi(idx):
    v0 = vpair[idx, :2]
    return np.arctan2(v0[:, 1], v0[:, 0])


phiv, phit = kin_phi(VA), kin_phi(TE)
print("computing land geometry (dist-ahead + terrain + asymmetry) ...", flush=True)
dist_ahead_t, terr_ahead_t = land_ahead_dist_terr(bla[TE], blo[TE] % 360, phit)
asym_t = land_asymmetry(bla[TE], blo[TE] % 360, phit)

NEAR_KM = 500.0
MOUNTAIN_M = 500.0
near_t = dist_ahead_t <= NEAR_KM
mountain_t = near_t & (terr_ahead_t >= MOUNTAIN_M)
print(f"near-land: {near_t.sum()} windows, of which mountainous (>={MOUNTAIN_M:.0f}m terrain "
      f"ahead within {NEAR_KM:.0f}km): {mountain_t.sum()}")
sid_t = sid[TE]

print(f"\n{'='*90}\nQ1: CROSS-TRACK direct group comparison, land within {NEAR_KM:.0f}km ahead "
      f"(n={near_t.sum()}) vs open ocean (n={(~near_t).sum()}), TEST set")
print(f"{'lead':>5s} {'h':>4s} | {'near-land bias':>15s} {'open-ocean bias':>16s} {'diff':>8s} | "
      f"{'boot p (storm-resample)':>24s}")
rng = np.random.default_rng(0)
storms_near = np.unique(sid_t[near_t])
storms_far = np.unique(sid_t[~near_t])
NBOOT = 2000


def storm_group_stats(y, mask, storm_ids, storms):
    """Per-storm (sum, count) of y within mask, indexed to `storms` order -- lets the bootstrap
    resample storms via fast fancy-indexing instead of re-concatenating raw arrays each draw."""
    idx_of = {s: k for k, s in enumerate(storms)}
    sums = np.zeros(len(storms)); cnts = np.zeros(len(storms))
    sel = np.where(mask)[0]
    for i in sel:
        k = idx_of[storm_ids[i]]
        sums[k] += y[i]; cnts[k] += 1
    return sums, cnts


def boot_diff_pvalue(sums_n, cnts_n, sums_f, cnts_f, rng, nboot=NBOOT):
    Nn, Nf = len(sums_n), len(sums_f)
    obs = sums_n.sum() / cnts_n.sum() - sums_f.sum() / cnts_f.sum()
    dn = rng.integers(0, Nn, size=(nboot, Nn)); df = rng.integers(0, Nf, size=(nboot, Nf))
    mn = sums_n[dn].sum(1) / cnts_n[dn].sum(1)
    mf = sums_f[df].sum(1) / cnts_f[df].sum(1)
    boots = mn - mf
    p = 2 * min((boots >= 0).mean(), (boots <= 0).mean())
    return float(obs), float(p)


cross_rows = []
for L in (0, 1, 2, 3, 5, 7, 11, 15, 19):
    y = at_cr[:, L]
    bn, bf = y[near_t].mean(), y[~near_t].mean()
    sums_n, cnts_n = storm_group_stats(y, near_t, sid_t, storms_near)
    sums_f, cnts_f = storm_group_stats(y, ~near_t, sid_t, storms_far)
    diff, p = boot_diff_pvalue(sums_n, cnts_n, sums_f, cnts_f, rng)
    print(f"{L+1:5d} {6*(L+1):4d} | {bn:15.1f} {bf:16.1f} {diff:8.1f} | {p:24.3f}")
    cross_rows.append({"lead_h": 6*(L+1), "near_bias_km": float(bn), "far_bias_km": float(bf),
                        "diff_km": float(diff), "boot_p": float(p)})
print("=" * 90)

print(f"\nQ2: within near-land subgroup (n={near_t.sum()}), does land-asymmetry (signed, "
      f"+=land-to-left) correlate with cross-track error sign?")
print(f"{'lead':>5s} {'h':>4s} | {'corr(asym, cross)':>18s} | {'boot p':>8s}")
asym_rows = []
sub_asym = asym_t[near_t]
sub_sid = sid_t[near_t]
storms_sub = np.unique(sub_sid)
idx_of_sub = {s: k for k, s in enumerate(storms_sub)}
storm_idx_sub = np.array([idx_of_sub[s] for s in sub_sid])


def storm_corr_sums(x, y):
    """Per-storm (n, sum_x, sum_y, sum_xy, sum_x2, sum_y2) so a storm-resampled bootstrap can
    reconstruct Pearson r from aggregated sums, no per-draw concatenation of raw arrays."""
    K = len(storms_sub)
    n = np.bincount(storm_idx_sub, minlength=K).astype("float64")
    sx = np.bincount(storm_idx_sub, weights=x, minlength=K)
    sy = np.bincount(storm_idx_sub, weights=y, minlength=K)
    sxy = np.bincount(storm_idx_sub, weights=x * y, minlength=K)
    sx2 = np.bincount(storm_idx_sub, weights=x * x, minlength=K)
    sy2 = np.bincount(storm_idx_sub, weights=y * y, minlength=K)
    return n, sx, sy, sxy, sx2, sy2


def boot_corr_pvalue(n, sx, sy, sxy, sx2, sy2, rng, nboot=NBOOT):
    K = len(n)

    def r_of(N, Sx, Sy, Sxy, Sx2, Sy2):
        num = N * Sxy - Sx * Sy
        den = np.sqrt(np.clip(N * Sx2 - Sx ** 2, 1e-12, None) * np.clip(N * Sy2 - Sy ** 2, 1e-12, None))
        return num / den

    obs = r_of(n.sum(), sx.sum(), sy.sum(), sxy.sum(), sx2.sum(), sy2.sum())
    draws = rng.integers(0, K, size=(nboot, K))
    N = n[draws].sum(1); Sx = sx[draws].sum(1); Sy = sy[draws].sum(1)
    Sxy = sxy[draws].sum(1); Sx2 = sx2[draws].sum(1); Sy2 = sy2[draws].sum(1)
    boots = r_of(N, Sx, Sy, Sxy, Sx2, Sy2)
    p = 2 * min((boots >= 0).mean(), (boots <= 0).mean())
    return float(obs), float(p)


for L in (0, 1, 2, 3, 5, 7, 11, 15, 19):
    y = at_cr[near_t, L]
    if len(y) < 5 or sub_asym.std() < 1e-9:
        continue
    sums = storm_corr_sums(sub_asym, y)
    r, p = boot_corr_pvalue(*sums, rng)
    print(f"{L+1:5d} {6*(L+1):4d} | {r:18.3f} | {p:8.3f}")
    asym_rows.append({"lead_h": 6*(L+1), "corr": r, "boot_p": float(p)})

# ---- Q3: mountainous coastline specifically (Chi et al.'s actual mechanism -- CMR-scale relief,
# not just any nearby shore) vs flat near-land, both along- and cross-track. Small-n warning up
# front: this is a further subset of an already-small near-land group.
print(f"\n{'='*90}\nQ3: MOUNTAINOUS near-land (terrain>={MOUNTAIN_M:.0f}m, n={mountain_t.sum()}) "
      f"vs flat near-land (n={(near_t & ~mountain_t).sum()})")
flat_t = near_t & ~mountain_t
if mountain_t.sum() >= 10 and flat_t.sum() >= 10:
    storms_mtn = np.unique(sid_t[mountain_t]); storms_flat = np.unique(sid_t[flat_t])
    for axis_name, arr in (("along", at_al), ("cross", at_cr)):
        print(f"  -- {axis_name}-track --")
        print(f"  {'lead':>5s} {'h':>4s} | {'mountain bias':>13s} {'flat bias':>10s} {'diff':>8s} | {'boot p':>8s}")
        for L in (0, 1, 2, 3, 5, 7, 11, 15, 19):
            y = arr[:, L]
            sums_m, cnts_m = storm_group_stats(y, mountain_t, sid_t, storms_mtn)
            sums_f2, cnts_f2 = storm_group_stats(y, flat_t, sid_t, storms_flat)
            bm, bf2 = y[mountain_t].mean(), y[flat_t].mean()
            diff, p = boot_diff_pvalue(sums_m, cnts_m, sums_f2, cnts_f2, rng)
            print(f"  {L+1:5d} {6*(L+1):4d} | {bm:13.1f} {bf2:10.1f} {diff:8.1f} | {p:8.3f}")
else:
    print(f"  too few windows to test (mountain={mountain_t.sum()}, flat={flat_t.sum()}) -- skipping")

json.dump({"cross_group": cross_rows, "asymmetry_corr": asym_rows,
           "n_near": int(near_t.sum()), "n_far": int((~near_t).sum())},
          open("track_build/landtest_cross.json", "w"), indent=1, default=float)
print("\nwrote track_build/landtest_cross.json")
print("\nInterpretation: boot_p < 0.05 on Q1 means v23 has a real, land-linked CROSS-TRACK bias")
print("(a turning error) that a purely along-track correction like v31's LandDrag can never fix")
print("even in principle. boot_p < 0.05 on Q2 means the SIGN of that turning error is predictable")
print("from which side the land is on -- i.e. there's a learnable directional signal here, and a")
print("v32 candidate should output a 2D (along+cross) correction fed by a directional (not just")
print("scalar-distance) land feature, not repeat v31's along-track-only design.")
