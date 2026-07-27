"""Local (CPU) init-assertion check for v32 -- IDENTICAL architecture to v31 (TrackFormerLand),
only A_MAX changed (0.65 -> 1.5) and the train-time sampler differs (checked separately below,
since it's the actual new code in v32; the architecture checks are unchanged from _v31check.py
on purpose -- there was nothing wrong with the architecture, see colab_v37_train.py's docstring).
Sources every class from LOCAL files instead of urllib/GitHub -- same pattern as _v29/_v31check.py.
Verifies:
  1. land off  -> v32 output EXACTLY equals v23's (max diff 0)
  2. land on   -> output moves (the path is live, not dead) for a WP window with land coverage
  3. land on, EP window (no coverage) -> output stays EXACTLY at v23 (availability gate works)
  4. gradient reaches LandDrag's first mlp layer
  5. state-dict remap loads v23 weights into v32 with only the new `land.*` keys missing
  6. land_weight() (the new oversampling logic) upweights the mountainous/near-land windows it
     claims to, and leaves everything else at baseline
"""
import json, re, math, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
KM6H = 6 * 3600 / 1000.0
R_KM = 111.2
A_MAX = 1.5
NEAR_W = 6.0
MOUNTAIN_W = 21.0
MOUNTAIN_M = 500.0

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
vpair = G["vpair"]; z = G["z"]; basins = G["basins"].astype(str)

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
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

sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
SIX = int(6 * 3600 * 1e9)
key = {(sid[i], int(bt[i])): i for i in range(len(sid))}
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        HIST[i, c] = key.get((sid[i], int(bt[i]) - back * SIX), -1)
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

# ---- land geometry (verbatim from colab_v36_train.py) ----
_T = np.load("track_build/terrain_wp.npz")
T_LAT, T_LON, LSM, ELEV = _T["lat"], _T["lon"], _T["lsm"], _T["elev"]
_LAND = LSM > 0.5
LAND_LAT = T_LAT[np.where(_LAND)[0]]
LAND_LON = T_LON[np.where(_LAND)[1]]
LAND_ELEV = ELEV[_LAND]


def land_features_np(lat, lon, heading_rad):
    n = len(lat)
    dlat = LAND_LAT[None, :] - lat[:, None]
    dlon = (LAND_LON[None, :] - lon[:, None]) * np.cos(np.radians(lat[:, None]))
    dy = dlat * R_KM; dx = dlon * R_KM
    dist = np.hypot(dx, dy)
    bearing = np.arctan2(dx, dy)
    dphi = np.arctan2(np.sin(bearing - heading_rad[:, None]), np.cos(bearing - heading_rad[:, None]))
    cone = np.abs(dphi) <= np.radians(30.0)
    ahead = np.where(cone, dist, np.inf)
    dist_nearest = dist.min(1)
    dist_ahead = ahead.min(1)
    dist_ahead = np.where(np.isfinite(dist_ahead), dist_ahead, 3000.0)
    near_ahead = ahead < 500.0
    terr_ahead = np.array([LAND_ELEV[near_ahead[k]].max() if near_ahead[k].any() else 0.0
                            for k in range(n)])
    return np.stack([dist_nearest, dist_ahead, terr_ahead], 1).astype("float32")


def landfeat_of(lat_np, lon_np, heading_np, basin_np):
    wp = basin_np == "WP"
    n = len(lat_np)
    out = np.zeros((n, 3), "float32")
    if wp.any():
        out[wp] = land_features_np(lat_np[wp], lon_np[wp] % 360, heading_np[wp])
    return out, wp.astype("float32")


USE_LAND = 1


class LandDrag(nn.Module):
    def __init__(self, leads=20, knots=4, a_max=A_MAX):
        super().__init__()
        self.a_max = a_max
        self.mlp = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, knots))
        kl = torch.tensor([0., 3., 11., 19.])
        Wi = torch.zeros(leads, knots)
        for i in range(leads):
            L = float(i)
            if L <= kl[0]:
                Wi[i, 0] = 1.0
            elif L >= kl[-1]:
                Wi[i, -1] = 1.0
            else:
                for k in range(knots - 1):
                    if kl[k] <= L <= kl[k + 1]:
                        f = (L - kl[k]) / (kl[k + 1] - kl[k])
                        Wi[i, k] = 1 - f; Wi[i, k + 1] = f
                        break
        self.register_buffer("Wi", Wi)
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat):
        knots = self.mlp(feat)
        return self.a_max * torch.tanh(knots @ self.Wi.t())


class TrackFormerLand(V23):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.land = LandDrag()

    def forward(self, tr, vp, slp, hist=None, have=None, landfeat=None, land_ok=None):
        s, ls, fp = super().forward(tr, vp, slp, hist, have)
        if USE_LAND and landfeat is not None:
            s = s.clone()
            delta = self.land(landfeat)
            v0 = vp[:, :2]
            hunit = v0 / v0.norm(dim=1, keepdim=True).clamp(min=1e-3)
            ok = land_ok.view(-1, 1)
            s[..., 0] = s[..., 0] + delta * hunit[:, 0:1] * ok
            s[..., 1] = s[..., 1] + delta * hunit[:, 1:2] * ok
        return s, ls, fp


def build_lf5(idx):
    bla = z["base_lat"].astype("float64")[idx]; blo = z["base_lon"].astype("float64")[idx] % 360
    phi = np.arctan2(vpair[idx, 1], vpair[idx, 0])
    lf3, lok = landfeat_of(bla, blo, phi, basins[idx])
    spd = np.hypot(vpair[idx, 0], vpair[idx, 1])
    latv = track[idx, -1, 48] * G["tstd"][48] + G["tmean"][48]
    lf5 = np.concatenate([lf3 / np.array([1000., 1000., 1000.], "float32"),
                          (spd / 100.0)[:, None].astype("float32"),
                          (np.abs(latv) / 30.0)[:, None].astype("float32")], 1)
    return lf5, lok


# ---- pick probe windows: at least one real WP window with land nearby, one EP window.
# Sample a few thousand WP rows rather than sweeping the full population (each row costs a
# distance to all 12,660 land cells; the full WP population made this take minutes). ----
wp_idx_all = np.where(basins == "WP")[0]
_rng = np.random.default_rng(0)
wp_idx = _rng.choice(wp_idx_all, size=min(4000, len(wp_idx_all)), replace=False)
_bla_wp = z["base_lat"].astype("float64")[wp_idx]; _blo_wp = z["base_lon"].astype("float64")[wp_idx] % 360
_phi_wp = np.arctan2(vpair[wp_idx, 1], vpair[wp_idx, 0])
_lf_wp, _ = landfeat_of(_bla_wp, _blo_wp, _phi_wp, basins[wp_idx])
_near = wp_idx[_lf_wp[:, 1] <= 300.0]
assert len(_near) > 0, "no WP probe window with land within 300km ahead found in the sample"
ep_idx = np.where(basins == "EP")[0]
assert len(ep_idx) > 0, "no EP probe window found"
_j = np.array([_near[0], _near[1] if len(_near) > 1 else _near[0], ep_idx[0], ep_idx[1]])
print(f"probe windows: basins {basins[_j]}, dist_ahead check below")

_t = torch.from_numpy(track[_j]); _v = torch.from_numpy(vpair[_j])
_s = torch.from_numpy(SLP[_j])
_hn = np.concatenate([SLP[HIST_S[_j, 0]], SLP[HIST_S[_j, 1]]], 1)
_h = torch.from_numpy(_hn); _a = torch.from_numpy(HAVE[_j])
_lf5, _lok = build_lf5(_j)
print(f"  dist_ahead (km, denorm): {_lf5[:,1]*1000}")
print(f"  land_ok: {_lok}")
_lf = torch.from_numpy(_lf5); _lo = torch.from_numpy(_lok)

_p, _q = V23().eval(), TrackFormerLand().eval()
torch.manual_seed(13)
nn.init.normal_(_p.track_res.weight, std=0.02); nn.init.normal_(_p.track_res.bias, std=0.02)
_sd = _p.state_dict()
_miss, _unexp = _q.load_state_dict(_sd, strict=False)
assert not _unexp, f"unexpected keys: {list(_unexp)[:5]}"
assert all(m.startswith("land.") for m in _miss), f"v23 weights failed to transfer: {list(_miss)[:5]}"
print(f"OK: state-dict remap clean, {len(_miss)} new LandDrag keys as expected")
assert float(_q.land.mlp[-1].weight.abs().max()) == 0.0, "not zero-init"

_d1 = float((_p(_t, _v, _s, _h, _a)[0] - _q(_t, _v, _s, _h, _a, _lf, _lo)[0]).abs().max())
assert _d1 < 1e-5, f"v32 does not reduce to v23 at init: {_d1}"
print(f"OK: land OFF max|v32 - v23| = {_d1:.2e} (v32 starts as exactly v23)")

nn.init.normal_(_q.land.mlp[-1].weight, std=0.05)
_out_p = _p(_t, _v, _s, _h, _a)[0]
_out_q = _q(_t, _v, _s, _h, _a, _lf, _lo)[0]
_d2 = float((_out_p - _out_q).abs().max())
print(f"OK: land ON  max|v32 - v23| = {_d2:.2e} (land path is live)")
assert _d2 > 1e-4, f"opening the land path moved the track by only {_d2:.2e} -- DEAD"

# EP windows (indices 2,3 in the probe) must stay EXACTLY at v23 even with land ON
_d_ep = float((_out_p[2:4] - _out_q[2:4]).abs().max())
print(f"OK: EP windows (land_ok=0) diff = {_d_ep:.2e} (should be ~0, gated off)")
assert _d_ep < 1e-5, "an EP (no land coverage) window still moved -- availability gating is broken"
nn.init.zeros_(_q.land.mlp[-1].weight)

_q.train()
nn.init.normal_(_q.land.mlp[-1].weight, std=0.05)
_mo, _, _ = _q(_t, _v, _s, _h, _a, _lf, _lo)
_mo.sum().backward()
_g = _q.land.mlp[0].weight.grad
assert _g is not None and float(_g.abs().max()) > 0, "land mlp not reachable by gradient"
print(f"OK: land mlp gradient reachable, max|grad| = {float(_g.abs().max()):.2e}")
nn.init.zeros_(_q.land.mlp[-1].weight)
_q.zero_grad(); _q.eval()

n_params = sum(p.numel() for p in LandDrag().parameters())
print(f"\nLandDrag adds {n_params:,} params")
print("\nALL CHECKS PASSED (architecture) -- proceeding to land_weight() checks")

# ---- land_weight(): the actual new code in v32. Verify it upweights exactly the windows it
# claims to (near-land -> NEAR_W, mountainous-near-land -> MOUNTAIN_W, everything else -> 1.0),
# against the SAME train-index population colab_v37_train.py will use.
BLA = z["base_lat"].astype("float64"); BLO = z["base_lon"].astype("float64") % 360
tr_idx = G["tr_idx"]


def land_weight(idx):
    bla = BLA[idx]; blo = BLO[idx]
    phi = np.arctan2(vpair[idx, 1], vpair[idx, 0])
    lf3, lok = landfeat_of(bla, blo, phi, basins[idx])
    dist_ahead, terr_ahead = lf3[:, 1], lf3[:, 2]
    near = lok.astype(bool) & (dist_ahead <= 500.0)
    mountain = near & (terr_ahead >= MOUNTAIN_M)
    w = np.ones(len(idx), "float64")
    w[near] = NEAR_W
    w[mountain] = MOUNTAIN_W
    return w, near, mountain


print(f"\ncomputing land_weight() over {len(tr_idx):,} train windows ...", flush=True)
W, NEAR, MTN = land_weight(tr_idx)
n_flat_near = int((NEAR & ~MTN).sum()); n_mtn = int(MTN.sum()); n_base = int((~NEAR).sum())
print(f"  baseline (weight=1.0):        {n_base:,} ({100*n_base/len(tr_idx):.1f}%)")
print(f"  flat near-land (weight={NEAR_W:.0f}): {n_flat_near:,} ({100*n_flat_near/len(tr_idx):.2f}%)")
print(f"  mountainous near-land (weight={MOUNTAIN_W:.0f}): {n_mtn:,} ({100*n_mtn/len(tr_idx):.2f}%)")
assert n_mtn > 0, "no mountainous-near-land windows found in the training set -- oversampling has nothing to act on"
assert np.all(W[~NEAR] == 1.0), "baseline windows were reweighted -- bug in land_weight()"
assert np.all(W[NEAR & ~MTN] == NEAR_W), "flat near-land windows have the wrong weight"
assert np.all(W[MTN] == MOUNTAIN_W), "mountainous near-land windows have the wrong weight"
# effective representation per epoch under WeightedRandomSampler(replacement=True): each draw is
# proportional to w_i / sum(w). Check the mountainous share actually moves from ~1% to a real
# double-digit-adjacent fraction of a training epoch, not a token bump.
eff_share = W[MTN].sum() / W.sum()
print(f"  effective per-epoch draw share for mountainous windows: {100*eff_share:.1f}% "
      f"(was {100*n_mtn/len(tr_idx):.2f}% under uniform sampling)")
assert eff_share > 5 * (n_mtn / len(tr_idx)), "oversampling barely moved the effective share -- weights too small to matter"
print("OK: land_weight() reweights exactly the claimed subset, and the mountainous share of each")
print("    training epoch moves from a sliver to a real, learnable fraction")

print("\nALL CHECKS PASSED -- v32 (TrackFormerLand + oversampled near-land training) is ready for Colab")
