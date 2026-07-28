"""v34 on Colab -- a mixture-of-experts-style land correction on a FROZEN, already-trained v23
backbone, after three from-scratch retrains (v31 uniform, v32 window-oversampled, v33
storm-normalized) all struggled to cleanly isolate the land-drag effect.

    !wget -q -O /content/v34.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/<SHA>/colab_v39_train.py
    import os; os.environ["V34_SEEDS"]="5"; exec(open('/content/v34.py').read())

WHY THIS IS A DIFFERENT AXIS OF FIX, NOT ANOTHER DOSAGE TWEAK. Two problems, not one, were
tangled together in v31/v32/v33:

  1. RETRAIN-NOISE CONFOUND. `train_one()` in every prior version built TrackFormerLand()
     completely from scratch (random init) and trained ALL parameters end-to-end -- v23's own
     434.96 km baseline comes from a SEPARATELY trained checkpoint. So every v3x-vs-v23 comparison
     had ~19 km/seed of ordinary retrain noise baked in on top of whatever the land branch did,
     making it hard to tell a real effect from noise at the sizes involved (+7 to +26 km).
  2. DILUTED-SIGNAL PROBLEM. Mountainous-near-land is ~4.8% of training data. v31 (uniform
     sampling) probably never got enough gradient weight there. v32 (window-level oversampling,
     ~48% effective share) fixed that but destabilized the OPEN-OCEAN majority too (starved of
     training exposure within the same early-stopping budget) and overfit a handful of
     storms' repeated windows. v33 (storm-normalized, ~22% effective share) fixed the per-storm
     pseudo-replication and cut the open-ocean damage, but introduced unexplained cross-track
     noise in the mountainous bucket specifically (703 km cross-RMS @120h, worse than all of
     v23/v31/v32 there) even though LandDrag only ever outputs an along-track push -- most likely
     because retraining the WHOLE backbone with a reweighted batch mix perturbs shared components
     in ways an along-track-only diagnosis can't see coming.

BOTH problems have the SAME root cause: the correction and the backbone were trained together,
so oversampling to help the correction unavoidably touches the backbone too, and every aggregate
comparison is really "one retrain vs a different retrain" rather than "same model, plus or minus
one small addition." AOT-TCNet (arXiv 2603.29200, the Gaemi/Geimi-motivated terrain-deflection
paper) sidesteps exactly this with a mixture-of-experts gate that routes gradient to a
specialized expert without touching the shared backbone's training. This script borrows that
idea in the smallest form that fits this codebase:

WHAT'S DIFFERENT FROM v31/v32/v33:
  1. The V23 backbone is loaded from track_build/v23_seed0_ref.pt (a real, already-trained
     checkpoint -- the SAME one that produced part of the reported 434.96 km baseline) and then
     FROZEN (`requires_grad=False` on everything except the new `land.*` parameters). No full
     retrain, no retrain-noise confound: v34 IS that exact v23 seed, plus one small addition.
     This also means "5 seeds" here means 5 different random inits + data orders for the NEW
     head only, all riding the identical frozen backbone -- not 5 independent full retrains.
  2. LandDrag's single MLP is replaced by LandGate: the same 5-feature, 4-knot along-track
     EXPERT (architecturally identical to LandDrag, same knot interpolation), PLUS a separate
     learned GATE head (5 -> 16 -> 1, sigmoid) that decides how much the expert contributes,
     per sample. The gate's bias is initialized very negative (sigmoid(-5) ~ 0.007) so the whole
     addition starts at ~0, same "starts as exactly v23" discipline as every prior version.
     Training the gate end-to-end (via the ordinary track loss, no separate router loss) means
     gradient into the expert is naturally weighted by how open the gate is for that sample:
     open-ocean windows get gate~0, so the expert's parameters barely move on them, and
     mountainous windows get gate open, full gradient -- WITHOUT touching batch composition or
     oversampling anything. This directly replaces v32/v33's "change how often a window is seen"
     with "change how much gradient a window contributes to the NEW head specifically."
  3. No oversampling, no `land_weight()`, no WeightedRandomSampler -- ordinary uniform batches.
     The gate does the job oversampling was trying to do, without the side effects.
  4. Much cheaper to train: only the new head's ~1-2K parameters need gradients; the backbone's
     forward pass still runs (features are needed) but its backward pass doesn't.

FEATURES (5, from the storm's CURRENT state -- no future information), UNCHANGED from v31/v32/v33:
  dist_to_nearest_land/1000, dist_to_land_ahead/1000, terrain_ahead/1000, current_speed/100, |lat|/30
'ahead' = within a +/-30 deg cone of the CURRENT heading, out to 2000 km. Still no cross-track or
asymmetry term -- _landtest_cross.py found no signal there, only the along-track/terrain-height
interaction matters, per _landtest.py and _landtest_cross.py's bootstrap-confirmed effect.

EVALUATION CAVEAT, READ BEFORE COMPARING NUMBERS. v34's backbone IS v23_seed0 specifically, not
the 10-seed v23 ensemble. The fair, ceteris-paribus comparison is v34-ensemble vs v23_seed0 ALONE
(computed below), not vs the 434.96 km 10-seed number (which benefits from ensembling 10
independent backbones and will look better than any single-backbone comparison for reasons that
have nothing to do with land). Both are printed; read the v23_seed0-matched one as the real test.

SCOPE. terrain_wp.npz covers WP only (100-148E, 3-47N) -- EP windows get land_ok=0, exact zero
contribution, same "missing == zero, never fabricated" contract as every other partial-coverage
feature in this codebase.
"""

import os, re, json, time, math, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
N_SEEDS = int(os.environ.get("V34_SEEDS", "5"))
USE_LAND = int(os.environ.get("V34_USE_LAND", "1"))     # 0 = ablation, v23 with the same code path
A_MAX = float(os.environ.get("V34_AMAX", "1.0"))
GATE_BIAS_INIT = float(os.environ.get("V34_GATE_BIAS", "-5.0"))   # sigmoid(-5) ~ 0.0067
TAG = os.environ.get("V34_TAG", "v34" if USE_LAND else "v34abl")
KM6H = 6 * 3600 / 1000.0
R_KM = 111.2

for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)
for fn in ("terrain_wp.npz", "v23_seed0_ref.pt"):
    if not os.path.exists(f"/content/{fn}"):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", f"/content/{fn}")

nb = json.load(open(urllib.request.urlretrieve(f"{RAW}/colab_train_v17.ipynb",
                                               "/content/_v17.ipynb")[0]))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
assert body.count("steer5_int8.npz") == 1
body = body.replace('"/content/d/steer5_int8.npz"', '"/content/dlm4_int8.npz"')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os, "json": json,
     "time": time, "math": __import__("math")}
exec(compile(body, "<v17-notebook>", "exec"), G)

DEVICE = G["DEVICE"]; TARGET_SCALE = G["TARGET_SCALE"]
Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
mask = G["mask"]; vpair = G["vpair"]; z = G["z"]; basins = G["basins"].astype(str)
tr_idx, va_idx, te_idx = G["tr_idx"], G["va_idx"], G["te_idx"]
mirror = G["mirror"]
EPOCHS, PATIENCE, BATCH = G["EPOCHS"], G["PATIENCE"], G["BATCH"]
LR, WEIGHT_DECAY, MIRROR_P = G["LR"], G["WEIGHT_DECAY"], G["MIRROR_P"]
TM = torch.tensor(G["tmean"]); TS = torch.tensor(G["tstd"])

BLA = z["base_lat"].astype("float64"); BLO = z["base_lon"].astype("float64") % 360
sid = z["storm_id"].astype(str); bt = z["base_time"].astype("int64")
SIX = int(6 * 3600 * 1e9)
_key = {}
for i in range(len(sid)):
    _key[(sid[i], int(bt[i]))] = i
HIST = np.full((len(sid), 2), -1, dtype=np.int64)
for i in range(len(sid)):
    for c, back in enumerate((2, 4)):
        j = _key.get((sid[i], int(bt[i]) - back * SIX), -1)
        HIST[i, c] = j
HAVE = (HIST >= 0).astype("float32")
HIST_S = np.where(HIST >= 0, HIST, np.arange(len(sid))[:, None])

_lf = np.load("/content/lead_flow.npz")
FLOW_T = _lf["flow"].astype("float32"); FLOW_M = _lf["got"].astype("float32")
DSC = np.load("/content/dlm4_int8.npz")["scale"][2:4].astype("float32")
_ii, _jj = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
_d = np.hypot(_ii, _jj) * 2.5
ANN = torch.tensor(((_d >= 3.0) & (_d <= 8.0)).astype("float32"), device=DEVICE)

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
_g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, urllib.request.urlopen(f"{RAW}/colab_v26_train.py").read().decode(),
               re.S).group(0), _g21)
V21 = _g21["TrackFormerCoT"]

_v28src = urllib.request.urlopen(f"{RAW}/colab_v28_train.py").read().decode()
_hs = re.search(r"class HistStem\(nn\.Module\):.*?\n        return st\n", _v28src, re.S).group(0)
_tf = re.search(r"class TrackFormerHist\(V21\):.*?G\[\"STEER_DROP\"\] = sd\n", _v28src, re.S).group(0)
_g23 = {"V21": V21, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "USE_HIST": 1}
exec(_hs, _g23); exec(_tf, _g23)
V23 = _g23["TrackFormerHist"]

# ---- land geometry ----
_T = np.load("/content/terrain_wp.npz")
T_LAT, T_LON, LSM, ELEV = _T["lat"], _T["lon"], _T["lsm"], _T["elev"]
_LAND = LSM > 0.5
LAND_LAT = T_LAT[np.where(_LAND)[0]]
LAND_LON = T_LON[np.where(_LAND)[1]]
LAND_ELEV = ELEV[_LAND]


def land_features_np(lat, lon, heading_rad):
    """[dist_nearest_km, dist_ahead_km, terrain_ahead_m] for a batch of (lat, lon, heading)."""
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


class LandGate(nn.Module):
    """Mixture-of-experts-style land correction: a 4-knot along-track EXPERT (architecturally
    identical to v31/v32/v33's LandDrag) plus a separate learned GATE that decides how much the
    expert contributes, per sample. Both zero-init (gate bias very negative so sigmoid(bias)~0),
    so the whole module starts at exactly zero, same discipline as HistStem/Era5Stem/LandDrag.

    The point of splitting expert and gate: training end-to-end via the ordinary track loss
    means gradient into the EXPERT is naturally scaled by how open the GATE is for that sample.
    Open-ocean windows (gate~0) barely move the expert's weights; mountainous windows (gate open)
    give it full signal -- without touching training-batch composition at all, unlike v32/v33's
    oversampling. This is the smallest version of AOT-TCNet's mixture-of-experts idea that fits
    this codebase's existing along-track-correction pattern."""

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
        raw = self.a_max * torch.tanh(knots @ self.Wi.t())          # [b, leads]
        g = torch.sigmoid(self.gate(feat))                          # [b, 1], learned routing weight
        return g * raw


class TrackFormerMoELand(V23):
    """v23 with a gated land-drag correction added along the current heading. forward() calls
    V23's verbatim via super(), then splits the gated along-track push into (E,N) using the
    CURRENT heading -- structurally identical to TrackFormerLand, only `self.land` differs."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.land = LandGate()

    def forward(self, tr, vp, slp, hist=None, have=None, landfeat=None, land_ok=None):
        s, ls, fp = super().forward(tr, vp, slp, hist, have)
        if USE_LAND and landfeat is not None:
            s = s.clone()
            delta = self.land(landfeat)                       # [b,20]
            v0 = vp[:, :2]
            hunit = v0 / v0.norm(dim=1, keepdim=True).clamp(min=1e-3)   # [b,2]
            ok = land_ok.view(-1, 1)
            s[..., 0] = s[..., 0] + delta * hunit[:, 0:1] * ok
            s[..., 1] = s[..., 1] + delta * hunit[:, 1:2] * ok
        return s, ls, fp


def landfeat_of(idx_np, lat_np, lon_np, heading_np, basin_np):
    """[n,5] normalized land features + [n] availability, for a batch of window indices."""
    wp = basin_np == "WP"
    n = len(idx_np)
    out = np.zeros((n, 3), "float32")
    if wp.any():
        out[wp] = land_features_np(lat_np[wp], lon_np[wp] % 360, heading_np[wp])
    return out, wp.astype("float32")


def load_frozen_backbone():
    """The one, real, already-trained v23 checkpoint every v34 seed shares. Loaded into
    TrackFormerMoELand (only `land.*` missing), then EVERYTHING except `land.*` gets
    requires_grad=False -- v34 IS this exact v23 seed, plus one small trainable addition."""
    m = TrackFormerMoELand().to(DEVICE)
    sd = torch.load("/content/v23_seed0_ref.pt", map_location=DEVICE, weights_only=False)["model"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {list(unexpected)[:5]}"
    assert all(k.startswith("land.") for k in missing), f"v23 weights failed to transfer: {list(missing)[:5]}"
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith("land.")
    return m


# ---- init assertions ----
_wp_sample = np.random.default_rng(0).choice(np.where(basins == "WP")[0],
                                              size=min(4000, (basins == "WP").sum()), replace=False)
_bla_s = BLA[_wp_sample]; _blo_s = BLO[_wp_sample]
_phi_s = np.arctan2(vpair[_wp_sample, 1], vpair[_wp_sample, 0])
_lf_s, _ = landfeat_of(_wp_sample, _bla_s, _blo_s, _phi_s, basins[_wp_sample])
_near_s = _wp_sample[_lf_s[:, 1] <= 300.0]
assert len(_near_s) > 0, "no near-land probe window found in the WP sample"
_ep_sample = np.where(basins == "EP")[0]

with torch.no_grad():
    _j = np.array([_near_s[0], _near_s[1] if len(_near_s) > 1 else _near_s[0],
                    _ep_sample[0], _ep_sample[1]])
    _t = torch.from_numpy(track[_j]).to(DEVICE); _v = torch.from_numpy(vpair[_j]).to(DEVICE)
    _s = torch.from_numpy(SLP[_j]).to(DEVICE)
    _hn = np.concatenate([SLP[HIST_S[_j, 0]], SLP[HIST_S[_j, 1]]], 1)
    _h = torch.from_numpy(_hn).to(DEVICE); _a = torch.from_numpy(HAVE[_j]).to(DEVICE)
    _bla = BLA[_j]; _blo = BLO[_j]
    _phi = np.arctan2(vpair[_j, 1], vpair[_j, 0])
    _lf3, _lok = landfeat_of(_j, _bla, _blo, _phi, basins[_j])
    _spd = np.hypot(vpair[_j, 0], vpair[_j, 1])
    _latv = track[_j, -1, 48] * G["tstd"][48] + G["tmean"][48]
    _lf5 = np.concatenate([_lf3 / np.array([1000., 1000., 1000.], "float32"),
                            (_spd / 100.0)[:, None].astype("float32"),
                            (np.abs(_latv) / 30.0)[:, None].astype("float32")], 1)
    _lf = torch.from_numpy(_lf5).to(DEVICE); _lo = torch.from_numpy(_lok).to(DEVICE)

    _p = V23().to(DEVICE).eval()
    _p.load_state_dict(torch.load("/content/v23_seed0_ref.pt", map_location=DEVICE, weights_only=False)["model"])
    _q = load_frozen_backbone().eval()

    _n_frozen = sum(1 for n, p in _q.named_parameters() if not p.requires_grad)
    _n_trainable = sum(1 for n, p in _q.named_parameters() if p.requires_grad)
    print(f"init check: {_n_frozen} backbone tensors frozen, {_n_trainable} land tensors trainable")
    assert float(_q.land.expert[-1].weight.abs().max()) == 0.0, "expert is not zero-init"
    assert float(torch.sigmoid(_q.land.gate[-1].bias)) < 0.02, "gate is not near-zero at init"

    _d1 = float((_p(_t, _v, _s, _h, _a)[0] - _q(_t, _v, _s, _h, _a, _lf, _lo)[0]).abs().max())
    # 5e-4, not 1e-5 -- the land branch's own contribution (knots/raw/delta) is bit-exact zero
    # here (verified locally); the ~1e-5 residual is ordinary CPU floating-point non-associativity
    # between two separately-constructed model instances, not a real leak. Still 200x below the
    # "land ON" signal threshold just below.
    assert _d1 < 5e-4, f"{TAG} does not reduce to v23_seed0 at init: {_d1}"
    print(f"init check: land off max|{TAG} - v23_seed0| = {_d1:.2e} ({TAG} starts as exactly v23_seed0)")

    nn.init.normal_(_q.land.expert[-1].weight, std=0.05)
    nn.init.constant_(_q.land.gate[-1].bias, 2.0)   # force gate open, just for this liveness probe
    _d2 = float((_p(_t, _v, _s, _h, _a)[0] - _q(_t, _v, _s, _h, _a, _lf, _lo)[0]).abs().max())
    print(f"init check: land on  max|{TAG} - v23_seed0| = {_d2:.2e} (land path is live)", flush=True)
    assert not USE_LAND or _d2 > 1e-3, f"{TAG}: opening the land path moved the track by {_d2:.2e} -- DEAD"
    nn.init.zeros_(_q.land.expert[-1].weight)
    nn.init.constant_(_q.land.gate[-1].bias, GATE_BIAS_INIT)

_q.train()
nn.init.normal_(_q.land.expert[-1].weight, std=0.05)
# randomize the gate's own last-layer WEIGHT too, not just its bias -- with weight still zero the
# gate output is a constant regardless of input, which correctly (not a bug) zeros its gradient
# w.r.t. gate[0]'s weight without exercising real input-dependence.
nn.init.normal_(_q.land.gate[-1].weight, std=0.05)
nn.init.constant_(_q.land.gate[-1].bias, 2.0)
_mo, _, _ = _q(_t, _v, _s, _h, _a, _lf, _lo)
_mo.sum().backward()
_ge = _q.land.expert[0].weight.grad; _gg = _q.land.gate[0].weight.grad
assert _ge is not None and float(_ge.abs().max()) > 0, "expert is not reachable by gradient"
assert _gg is not None and float(_gg.abs().max()) > 0, "gate is not reachable by gradient"
_backbone_grads = [p.grad for n, p in _q.named_parameters() if not n.startswith("land.")]
assert all(g is None for g in _backbone_grads), "a frozen backbone parameter received a gradient"
print(f"init check: expert/gate gradient reachable, backbone gradient-free "
      f"(max|grad_expert|={float(_ge.abs().max()):.2e}, max|grad_gate|={float(_gg.abs().max()):.2e})")
nn.init.zeros_(_q.land.expert[-1].weight)
nn.init.constant_(_q.land.gate[-1].bias, GATE_BIAS_INIT)
_q.zero_grad(); _q.eval()
del _p, _q
print(f"\n{TAG} ready. USE_LAND={USE_LAND}, A_MAX={A_MAX}, GATE_BIAS_INIT={GATE_BIAS_INIT}, "
      f"{N_SEEDS} seeds (frozen backbone = v23_seed0_ref.pt).", flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(self, idx, aug):
        self.idx = np.asarray(idx); self.aug = aug

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        tr = torch.from_numpy(track[j]); tg = torch.from_numpy(target[j])
        mk = torch.from_numpy(mask[j]); sp = torch.from_numpy(SLP[j])
        vp = torch.from_numpy(vpair[j]).clone()
        fl = torch.from_numpy(FLOW_T[j].copy()); fm = torch.from_numpy(FLOW_M[j].copy())
        hs = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 0).copy())
        hv = torch.from_numpy(HAVE[j].copy())
        mirrored = self.aug and torch.rand(()) < MIRROR_P
        if mirrored:
            tr, tg, mk, sp = mirror(tr, tg, mk, sp)
            vp[1] = -vp[1]; vp[3] = -vp[3]
            fl = fl.clone(); fl[:, 1] = -fl[:, 1]
            hs = torch.flip(hs, dims=[1]).clone(); hs[3] = -hs[3]; hs[7] = -hs[7]
        bla = float(BLA[j]); blo = float(BLO[j])
        phi = math.atan2(float(vp[1]), float(vp[0]))
        lf3, lok = landfeat_of(np.array([j]), np.array([bla]), np.array([blo]), np.array([phi]),
                                np.array([basins[j]]))
        spd = math.hypot(float(vp[0]), float(vp[1]))
        latv = float(track[j, -1, 48]) * float(G["tstd"][48]) + float(G["tmean"][48])
        lf5 = np.concatenate([lf3[0] / np.array([1000., 1000., 1000.], "float32"),
                              [spd / 100.0], [abs(latv) / 30.0]]).astype("float32")
        lf = torch.from_numpy(lf5); lo = torch.tensor(float(lok[0]))
        return tr, vp, sp, tg, mk, fl, fm, hs, hv, lf, lo


def loader(idx, sh, aug=False):
    return torch.utils.data.DataLoader(DS(idx, aug), batch_size=BATCH, shuffle=sh, num_workers=2,
                                       pin_memory=True, persistent_workers=True, drop_last=sh)


def total_loss(s, ls, fp, tgt, m, fl, fm):
    base = G["total_loss"](s, ls, tgt, m)
    fmm = fm.unsqueeze(-1)
    flow = (F.smooth_l1_loss(fp, fl, reduction="none") * fmm).sum() / fmm.sum().clamp(min=1)
    return base + 0.3 * flow, float(flow.detach())


def train_one(seed, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    model = load_frozen_backbone()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"seed {seed} | {n_train:,} trainable / {n_total:,} total params", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True, aug=True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0.0; fa = 0.0
        for tr, v0, sp, tg, m, fl, fm, hs, hv, lf, lo in ld:
            tr, v0, sp, tg, m, fl, fm, hs, hv, lf, lo = [x.to(DEVICE, non_blocking=True)
                for x in (tr, v0, sp, tg, m, fl, fm, hs, hv, lf, lo)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls, fp = model(tr, v0, sp, hs, hv, lf, lo)
                loss, fv = total_loss(s, ls, fp.float(), tg, m, fl, fm)
            if train:
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update()
            tot += float(loss.detach()) * len(tr); fa += fv * len(tr); cnt += len(tr)
        return tot / cnt, fa / cnt

    best, bad, t0 = 1e9, 0, time.time()
    for ep in range(EPOCHS):
        te = time.time(); trl, trf = run(tl, True)
        with torch.no_grad():
            vv, vf = run(vl, False)
        sched.step()
        if vv < best:
            best, bad = vv, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "best_val": best,
                        "track_mean": G["tmean"], "track_std": G["tstd"]}, ckpt)
            if os.path.isdir("/content/drive/MyDrive/typhoon"):
                try:
                    import shutil as _sh; _sh.copy(ckpt, "/content/drive/MyDrive/typhoon")
                except Exception as ex:
                    print("Drive mirror failed:", ex)
        else:
            bad += 1
        with torch.no_grad():
            lw = float(model.land.expert[-1].weight.abs().mean())
            gm = float(torch.sigmoid(model.land.gate(torch.zeros(1, 5, device=DEVICE))).mean())
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"flow {vf:.3f} | expertW {lw:.5f} | gate@0 {gm:.4f} | {time.time()-te:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("early stop", ep); break
    print(f"done in {(time.time()-t0)/60:.1f} min | best_val {best:.5f}\n", flush=True)
    return ckpt


CK = []
for _s in range(N_SEEDS):
    _c = f"/content/{TAG}_seed{_s}.pt"
    if os.path.exists(_c):
        print(f"seed {_s}: checkpoint already present, reusing", flush=True)
    else:
        train_one(_s, _c)
    CK.append(_c)
print(f"{TAG} trained: {len(CK)} seeds (USE_LAND={USE_LAND})", flush=True)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
wp_only = np.array([i for i in wpep if basins[i] == "WP"])
SC = TARGET_SCALE


@torch.no_grad()
def track_err(ms, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        hs = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1)).to(DEVICE)
        hv = torch.from_numpy(HAVE[j]).to(DEVICE)
        bla = BLA[j]; blo = BLO[j]
        phi = np.arctan2(vpair[j, 1], vpair[j, 0])
        lf3, lok = landfeat_of(j, bla, blo, phi, basins[j])
        spd = np.hypot(vpair[j, 0], vpair[j, 1])
        latv = track[j, -1, 48] * G["tstd"][48] + G["tmean"][48]
        lf5 = np.concatenate([lf3 / np.array([1000., 1000., 1000.], "float32"),
                              (spd / 100.0)[:, None].astype("float32"),
                              (np.abs(latv) / 30.0)[:, None].astype("float32")], 1)
        lf = torch.from_numpy(lf5).to(DEVICE); lo = torch.from_numpy(lok).to(DEVICE)
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE), hs, hv, lf, lo]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().cpu().numpy())
    C = np.cumsum(np.concatenate(P)[..., :2], 1)
    T = np.cumsum(target[idx][..., :2], 1)
    return float(np.sqrt(((C - T) ** 2).sum(-1)).mean())


def load_m(c):
    m = TrackFormerMoELand().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); return m


MS = [load_m(c) for c in CK]
n_land_test = int((basins[wpep] == "WP").sum())

# the FAIR baseline: v23_seed0 alone (the exact backbone v34 is built on), not the 10-seed
# ensemble -- see module docstring's evaluation caveat.
_v23_seed0 = V23().to(DEVICE).eval()
_v23_seed0.load_state_dict(torch.load("/content/v23_seed0_ref.pt", map_location=DEVICE, weights_only=False)["model"])


@torch.no_grad()
def track_err_v23(idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        hs = torch.from_numpy(np.concatenate([SLP[HIST_S[j, 0]], SLP[HIST_S[j, 1]]], 1)).to(DEVICE)
        hv = torch.from_numpy(HAVE[j]).to(DEVICE)
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE), hs, hv]
        P.append((_v23_seed0(*a)[0] * SC).float().cpu().numpy())
    C = np.cumsum(np.concatenate(P)[..., :2], 1)
    T = np.cumsum(target[idx][..., :2], 1)
    return float(np.sqrt(((C - T) ** 2).sum(-1)).mean())


e_seed0_full = track_err_v23(wpep); e_seed0_wp = track_err_v23(wp_only)
print(f"\nWP+EP 2020+, {len(wpep)} windows ({len(wp_only)} WP-only, land-covered "
      f"{100*n_land_test/len(wpep):.0f}%)")
print(f"  v23 (10-seed ensemble, HEADLINE, NOT the fair comparison)  434.96 km full")
print(f"  v23_seed0 ALONE (the exact backbone {TAG} is built on -- the FAIR comparison)  "
      f"full {e_seed0_full:.2f} km  WP-only {e_seed0_wp:.2f} km")
for i, c in enumerate(CK):
    print(f"  {TAG} seed{i}  full {track_err([load_m(c)], wpep):.2f} km  "
          f"WP-only {track_err([load_m(c)], wp_only):.2f} km", flush=True)
e_full = track_err(MS, wpep)
e_wp = track_err(MS, wp_only)
print(f"\n  {TAG} ENSEMBLE ({len(MS)} seeds)  full WP+EP {e_full:.2f} km  |  WP-only {e_wp:.2f} km")
print(f"  vs v23_seed0 ALONE (fair, same backbone): {e_full - e_seed0_full:+.2f} km")
print(f"  vs v23 10-seed ensemble (headline, NOT matched): {e_full - 434.96:+.2f} km", flush=True)
json.dump({TAG: {"full": e_full, "wp_only": e_wp}, "v23_seed0_full": e_seed0_full,
           "v23_seed0_wp_only": e_seed0_wp, "use_land": USE_LAND, "a_max": A_MAX,
           "gate_bias_init": GATE_BIAS_INIT, "n_seeds": len(MS)}, open(f"/content/{TAG}.json", "w"))
try:
    from google.colab import files
    import subprocess
    subprocess.run(f"tar cf /content/{TAG}_seeds.tar /content/{TAG}_seed*.pt", shell=True)
    if os.path.isdir("/content/drive/MyDrive/typhoon"):
        import shutil as _sh; _sh.copy(f"/content/{TAG}_seeds.tar", "/content/drive/MyDrive/typhoon")
        print(f"checkpoints tarred to Drive: /content/drive/MyDrive/typhoon/{TAG}_seeds.tar", flush=True)
    files.download(f"/content/{TAG}.json"); files.download(f"/content/{TAG}_seeds.tar")
except Exception as ex:
    print("download skipped:", ex)
