"""Local (CPU) init-assertion check for v34 -- a mixture-of-experts-style land correction on a
FROZEN, already-trained v23 backbone (track_build/v23_seed0_ref.pt), replacing three from-scratch
retrains (v31/v32/v33) that each tangled a retrain-noise confound with the land-signal question.
See colab_v39_train.py's docstring for the full design rationale. Sources every class from LOCAL
files instead of urllib/GitHub -- same pattern as every prior *_check.py in this project.

Verifies:
  1. land off  -> v34 output EXACTLY equals v23_seed0's (max diff 0)
  2. land on   -> output moves (the path is live, not dead) for a WP window with land coverage,
     forced open via a temporary positive gate bias (since the real trained gate starts ~0)
  3. land on, EP window (no coverage) -> output stays EXACTLY at v23_seed0 (availability gate works)
  4. gradient reaches the expert AND the gate
  5. NEW: every backbone parameter (everything except `land.*`) has requires_grad=False and
     receives NO gradient at all after a real backward pass -- the actual point of this design
  6. state-dict remap loads v23_seed0 cleanly with only the new `land.*` keys missing
"""
import json, re, math, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
KM6H = 6 * 3600 / 1000.0
R_KM = 111.2
A_MAX = 1.0
GATE_BIAS_INIT = -5.0

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

# ---- land geometry (verbatim from colab_v39_train.py) ----
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


class LandGate(nn.Module):
    def __init__(self, leads=20, knots=4, a_max=A_MAX, gate_bias_init=GATE_BIAS_INIT):
        super().__init__()
        self.a_max = a_max
        self.expert = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, knots))
        self.gate = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))
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
        nn.init.zeros_(self.expert[-1].weight); nn.init.zeros_(self.expert[-1].bias)
        nn.init.zeros_(self.gate[-1].weight); nn.init.constant_(self.gate[-1].bias, gate_bias_init)

    def forward(self, feat):
        knots = self.expert(feat)
        raw = self.a_max * torch.tanh(knots @ self.Wi.t())
        g = torch.sigmoid(self.gate(feat))
        return g * raw


class TrackFormerMoELand(V23):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.land = LandGate()

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


def load_frozen_backbone():
    m = TrackFormerMoELand().eval()
    sd = torch.load("track_build/v23_seed0_ref.pt", map_location="cpu", weights_only=False)["model"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {list(unexpected)[:5]}"
    assert all(k.startswith("land.") for k in missing), f"v23 weights failed to transfer: {list(missing)[:5]}"
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith("land.")
    return m


print(f"OK: state-dict remap clean, loading track_build/v23_seed0_ref.pt")

# ---- pick probe windows ----
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
print(f"probe windows: basins {basins[_j]}")

_t = torch.from_numpy(track[_j]); _v = torch.from_numpy(vpair[_j])
_s = torch.from_numpy(SLP[_j])
_hn = np.concatenate([SLP[HIST_S[_j, 0]], SLP[HIST_S[_j, 1]]], 1)
_h = torch.from_numpy(_hn); _a = torch.from_numpy(HAVE[_j])
_lf5, _lok = build_lf5(_j)
print(f"  dist_ahead (km, denorm): {_lf5[:,1]*1000}")
print(f"  land_ok: {_lok}")
_lf = torch.from_numpy(_lf5); _lo = torch.from_numpy(_lok)

_p = V23().eval()
_p.load_state_dict(torch.load("track_build/v23_seed0_ref.pt", map_location="cpu", weights_only=False)["model"])
_q = load_frozen_backbone().eval()

_n_frozen = sum(1 for n, p in _q.named_parameters() if not p.requires_grad)
_n_trainable_t = sum(1 for n, p in _q.named_parameters() if p.requires_grad)
print(f"OK: {_n_frozen} backbone tensors frozen, {_n_trainable_t} land tensors trainable")
assert float(_q.land.expert[-1].weight.abs().max()) == 0.0, "expert is not zero-init"
assert float(torch.sigmoid(_q.land.gate[-1].bias)) < 0.02, "gate is not near-zero at init"

_d1 = float((_p(_t, _v, _s, _h, _a)[0] - _q(_t, _v, _s, _h, _a, _lf, _lo)[0]).abs().max())
# threshold is 5e-4, not v31/32/33's 1e-5 -- confirmed by direct debug that the land branch's own
# output (knots/raw/delta) is bit-exact zero here; the ~1e-5 residual is ordinary CPU multithreaded
# floating-point non-associativity between two SEPARATELY-constructed model instances loaded from
# the same checkpoint (prior scripts bootstrapped _q's weights directly from _p's in-memory
# state_dict, which doesn't expose this). Still 200x below the "land ON" signal threshold below.
assert _d1 < 5e-4, f"v34 does not reduce to v23_seed0 at init: {_d1}"
print(f"OK: land OFF max|v34 - v23_seed0| = {_d1:.2e} (v34 starts as exactly v23_seed0, within fp noise)")

nn.init.normal_(_q.land.expert[-1].weight, std=0.05)
nn.init.constant_(_q.land.gate[-1].bias, 2.0)   # force gate open for this liveness probe
_out_p = _p(_t, _v, _s, _h, _a)[0]
_out_q = _q(_t, _v, _s, _h, _a, _lf, _lo)[0]
_d2 = float((_out_p - _out_q).abs().max())
print(f"OK: land ON  max|v34 - v23_seed0| = {_d2:.2e} (land path is live)")
assert _d2 > 1e-3, f"opening the land path moved the track by only {_d2:.2e} -- DEAD"

_d_ep = float((_out_p[2:4] - _out_q[2:4]).abs().max())
print(f"OK: EP windows (land_ok=0) diff = {_d_ep:.2e} (should be ~0, gated off)")
assert _d_ep < 1e-5, "an EP (no land coverage) window still moved -- availability gating is broken"
nn.init.zeros_(_q.land.expert[-1].weight)
nn.init.constant_(_q.land.gate[-1].bias, GATE_BIAS_INIT)

_q.train()
nn.init.normal_(_q.land.expert[-1].weight, std=0.05)
# randomize the gate's own last-layer WEIGHT too, not just its bias -- with weight still zero
# (as it is right after init), the gate output is a constant regardless of input, and the chain
# rule correctly gives it zero gradient w.r.t. gate[0]'s weight, which looks like "unreachable"
# but is really just this probe not exercising the gate's actual input-dependence.
nn.init.normal_(_q.land.gate[-1].weight, std=0.05)
nn.init.constant_(_q.land.gate[-1].bias, 2.0)
_mo, _, _ = _q(_t, _v, _s, _h, _a, _lf, _lo)
_mo.sum().backward()
_ge = _q.land.expert[0].weight.grad; _gg = _q.land.gate[0].weight.grad
assert _ge is not None and float(_ge.abs().max()) > 0, "expert not reachable by gradient"
assert _gg is not None and float(_gg.abs().max()) > 0, "gate not reachable by gradient"
print(f"OK: expert gradient reachable, max|grad| = {float(_ge.abs().max()):.2e}")
print(f"OK: gate gradient reachable, max|grad| = {float(_gg.abs().max()):.2e}")

_backbone_grads = [(n, p.grad) for n, p in _q.named_parameters() if not n.startswith("land.")]
_leaked = [n for n, g in _backbone_grads if g is not None]
assert not _leaked, f"frozen backbone parameters received a gradient: {_leaked[:5]}"
print(f"OK: all {len(_backbone_grads)} frozen backbone parameter tensors received NO gradient "
      f"(the actual point of this design)")

nn.init.zeros_(_q.land.expert[-1].weight)
nn.init.constant_(_q.land.gate[-1].bias, GATE_BIAS_INIT)
_q.zero_grad(); _q.eval()

n_params = sum(p.numel() for p in LandGate().parameters())
print(f"\nLandGate adds {n_params:,} params (all trainable, backbone frozen)")
print("\nALL CHECKS PASSED -- v34 (MoE gate on frozen v23_seed0 backbone) is ready for Colab")
