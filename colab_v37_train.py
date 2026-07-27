"""v32 on Colab -- v31's EXACT architecture (LandDrag/TrackFormerLand, unchanged), retrained to
fix a diluted-training-signal problem _landtest_cross.py found, not a redesigned correction.

    !wget -q -O /content/v32.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/<SHA>/colab_v37_train.py
    import os; os.environ["V32_SEEDS"]="5"; exec(open('/content/v32.py').read())

WHY v31 (this same architecture, trained the old way) LOST to v23 EVERYWHERE it was tested (full
WP+EP Colab test +8.11 km worse, worse on 2/3 held-out storms, worse on Noul 2026 and Tip 1979's
real landfalls -- see land-interaction-lever memory) despite passing _landtest.py's pooled GO gate
(bootstrap-confirmed -22.7/-32.4 km at 36-48h, p=0.008/0.006):

_landtest_cross.py went back and tested two literature hypotheses (DeepMind's cyclone model
forecasts full physical fields rather than a bolted-on land term; Nature Geoscience 2026 found
landfalling TCs ACCELERATE approaching the coast via land-sea roughness/thermal CONTRAST, not
simple drag; Chi et al. 2024 found Typhoon Chanthu's Taiwan deflection is governed by R/L_E, a
TERRAIN-SCALE ratio) against v23's actual test errors. Cross-track deflection and a land-left/
right asymmetry feature both came back null (p>0.17, n=457 near-land windows). But splitting
_landtest.py's ORIGINAL near-land group by terrain height ahead found the real story: mountainous
coastline (>=500m terrain ahead) shows 2-2.5x the along-track bias of flat coastline at 36h
(-64.7 km diff, p=0.031) and 48h (-105.2 km diff, p=0.015) -- and only ~245 of 2698 WP test
windows are BOTH near-land AND mountainous. LandDrag already had terrain height as 1 of its 3
land features, so this was never a missing-feature problem: a tiny zero-init residual on a
FROZEN backbone, trained on the full batch mix where mountainous-near-land is a sliver of a
sliver (~1-2% of all training windows), almost certainly never got enough gradient weight to
learn the strong, narrow response the data actually supports.

WHAT CHANGES FROM v31 (nothing architectural):
  1. The train loader now uses a WeightedRandomSampler (see `land_weight()` below) that upweights
     near-land windows ~6x and mountainous-near-land windows ~21x over baseline, so LandDrag's
     gradient sees the regime where the signal actually lives, instead of being averaged toward
     zero by 98%+ open-ocean batches. Validation stays UNWEIGHTED (true population), so early
     stopping/model selection isn't biased toward overfitting the rare subgroup.
  2. A_MAX default raised 0.65 -> 1.5 (env-overridable) as a safety margin -- the found effect
     (up to -175 km bias by 48h in the mountainous group) is plausibly larger than a per-lead cap
     of 0.65 (raw units, ~65 km/step before knot interpolation) comfortably supports across the
     full cumulative track, even though a_max was probably not v31's primary bottleneck.
  3. Nothing else. Not adding a cross-track or asymmetry term -- _landtest_cross.py tested both
     and found no signal, so this stays a single along-track correction like v31, just trained to
     actually learn from the regime where it matters.

THE 120h FLIP, AND WHY IT IS NOT TARGETED HERE. At 120h the sign reverses (+164 km, near-land
storms show SMALLER error) -- refining the feature to "will the storm plausibly reach land within
THIS lead's horizon at its current speed" (rather than static distance) made the flip WORSE, not
better (+219 km), which rules out "wrong lead window" as the explanation. The more likely story:
storms with land ahead at forecast time often make landfall/dissipate for real by 120h, so their
OBSERVED track is naturally shorter -- v23's chronic under-prediction of speed happens to match
that shortened reality by coincidence at long leads, not because it understands land. Rather than
hand-pick which leads to trust, LandDrag gets a FREE (not sign-constrained, not hard-gated by lead)
per-lead learned magnitude via knot interpolation, zero-init -- exactly MeridionalDrift's
philosophy: give the model an informative feature and a flexible-but-constrained way to use it,
and let training (which optimises on the real loss and can discover whichever leads the signal
actually holds at) decide, rather than baking in a possibly-wrong manual read of 3 test points.

FEATURES (5, from the storm's CURRENT state -- no future information), UNCHANGED from v31:
  dist_to_nearest_land/1000, dist_to_land_ahead/1000, terrain_ahead/1000, current_speed/100, |lat|/30
'ahead' = within a +/-30 deg cone of the CURRENT heading, out to 2000 km.

MIRROR CONSISTENCY. Land geometry is NOT locally mirror-symmetric like atmospheric fields -- Taipei
has no mirror-image city at 24S. But 'ahead' is a DIRECTIONAL query against REAL terrain at the
storm's REAL (never-mirrored) position, and the direction is exactly `vp`'s heading -- which the
existing pipeline already mirrors consistently (vp[1] = -vp[1] on a mirrored sample). So land
features are computed from the SAME (possibly-mirrored) vp used everywhere else, AFTER mirroring:
this is a real terrain lookup in a locally-reflected search direction, precisely analogous to how
the SLP patch's mirror is not a real alternate atmosphere either, just a physically self-consistent
augmented view. dist_to_nearest_land needs no heading and is identical either way (position is
real, never altered by mirroring).

SCOPE. terrain_wp.npz covers WP only (100-148E, 3-47N) -- EP windows get land_ok=0, exact zero
contribution, same "missing == zero, never fabricated" contract as every other partial-coverage
feature in this codebase.
"""

import os, re, json, time, math, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
N_SEEDS = int(os.environ.get("V32_SEEDS", "5"))
USE_LAND = int(os.environ.get("V32_USE_LAND", "1"))     # 0 = ablation, v23 with the same code path
A_MAX = float(os.environ.get("V32_AMAX", "1.5"))          # per-lead cap, channel units (raised from
                                                            # v31's 0.65 -- see module docstring)
TAG = os.environ.get("V32_TAG", "v32" if USE_LAND else "v32abl")
NEAR_W = float(os.environ.get("V32_NEAR_W", "6.0"))        # near-land oversample weight
MOUNTAIN_W = float(os.environ.get("V32_MOUNTAIN_W", "21.0"))  # mountainous-near-land oversample weight
MOUNTAIN_M = 500.0                                          # terrain-ahead threshold, meters
KM6H = 6 * 3600 / 1000.0
R_KM = 111.2

for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)
if not os.path.exists("/content/terrain_wp.npz"):
    print("fetching terrain_wp.npz ...", flush=True)
    urllib.request.urlretrieve(f"{RAW}/track_build/terrain_wp.npz", "/content/terrain_wp.npz")

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

# materialize base_lat/base_lon as REAL arrays, not lazy z["..."] access: z is an NpzFile with a
# live zip handle, and DS.__getitem__ runs inside forked DataLoader worker processes -- concurrent
# lazy reads on the SAME (forked, shared fd) zip handle from multiple workers corrupts each other's
# reads ("zipfile.BadZipFile: Overlapped entries ... possible zip bomb"), hit on the first real run.
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
_bas_full = basins  # WP-basin windows are the only ones ever land_ok=1


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


class LandDrag(nn.Module):
    """5 features -> 4 lead-knots -> interp to 20 -> tanh -> signed along-track push. Exact zero
    at init (weight AND bias), matching HistStem/Era5Stem's convention. Unlike MeridionalDrift
    there is no forced odd/antisymmetric sign -- land drag has no reason to flip across the
    equator, so the MLP is free to learn whatever sign the data supports."""

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
    """v23 with a land-drag correction added along the current heading. forward() calls V23's
    verbatim via super(), then splits the along-track push into (E,N) using the CURRENT heading."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.land = LandDrag()

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


def land_weight(idx, chunk=4000):
    """Per-window train sample weight: NEAR_W for near-land (dist_ahead<=500km), MOUNTAIN_W for
    near-land AND mountainous (terrain_ahead>=MOUNTAIN_M) -- see module docstring. This is the
    actual fix: v31's LandDrag already had these exact features, but a uniform-random train
    sampler meant mountainous-near-land windows (~1-2% of all training data, per
    _landtest_cross.py) barely contributed gradient. computed ONCE over tr_idx, not per-seed.

    Chunked: land_features_np broadcasts every row against all 12,660 land cells at once, so
    calling it on the full ~153k-row training set in one shot allocates 153k x 12,660 float64
    intermediates (dlat/dlon/dist/bearing/...) -- tens of GB, enough to OOM even a Colab
    High-RAM runtime. 4000 rows at a time keeps peak memory in the tens-of-MB range instead."""
    n = len(idx)
    w = np.ones(n, "float64")
    for i in range(0, n, chunk):
        j = idx[i:i + chunk]
        bla = BLA[j]; blo = BLO[j]
        phi = np.arctan2(vpair[j, 1], vpair[j, 0])
        lf3, lok = landfeat_of(j, bla, blo, phi, basins[j])
        dist_ahead, terr_ahead = lf3[:, 1], lf3[:, 2]
        near = lok.astype(bool) & (dist_ahead <= 500.0)
        mountain = near & (terr_ahead >= MOUNTAIN_M)
        wi = np.ones(len(j), "float64")
        wi[near] = NEAR_W
        wi[mountain] = MOUNTAIN_W
        w[i:i + chunk] = wi
    return w


# ---- init assertions ----
# pick probe windows that GUARANTEE at least one real land-covered WP window -- an all-uncovered
# probe would trivially (and WRONGLY) trip the "opening the land path moved nothing -- DEAD"
# check, since the availability gate zeroes the contribution regardless of weight magnitude.
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

    _p, _q = V23().to(DEVICE).eval(), TrackFormerLand().to(DEVICE).eval()
    torch.manual_seed(13)
    nn.init.normal_(_p.track_res.weight, std=0.02); nn.init.normal_(_p.track_res.bias, std=0.02)
    _sd = _p.state_dict()
    _miss, _unexp = _q.load_state_dict(_sd, strict=False)
    assert not _unexp, f"unexpected keys loading v23 into v32: {list(_unexp)[:5]}"
    assert all(m.startswith("land.") for m in _miss), f"v23 weights failed to transfer: {list(_miss)[:5]}"
    assert float(_q.land.mlp[-1].weight.abs().max()) == 0.0, "land path is not zero-init"

    _d1 = float((_p(_t, _v, _s, _h, _a)[0] - _q(_t, _v, _s, _h, _a, _lf, _lo)[0]).abs().max())
    assert _d1 < 1e-5, f"{TAG} does not reduce to v23 at init: {_d1}"
    nn.init.normal_(_q.land.mlp[-1].weight, std=0.05)
    _d2 = float((_p(_t, _v, _s, _h, _a)[0] - _q(_t, _v, _s, _h, _a, _lf, _lo)[0]).abs().max())
    nn.init.zeros_(_q.land.mlp[-1].weight)
    assert not USE_LAND or _d2 > 1e-4, f"{TAG}: opening the land path moved the track by {_d2:.2e} -- DEAD"
    print(f"init check: land off max|v32 - v23| = {_d1:.2e} (v32 starts as exactly v23)")
    print(f"init check: land on  max|v32 - v23| = {_d2:.2e} (land path is live)", flush=True)

_q.train()
nn.init.normal_(_q.land.mlp[-1].weight, std=0.05)
_mo, _, _ = _q(_t, _v, _s, _h, _a, _lf, _lo)
_mo.sum().backward()
_g = _q.land.mlp[0].weight.grad
assert _g is not None and float(_g.abs().max()) > 0, "land mlp is not reachable by gradient"
nn.init.zeros_(_q.land.mlp[-1].weight)
_q.zero_grad(); _q.eval()
print(f"init check: land mlp gradient reachable, max|grad| = {float(_g.abs().max()):.2e}")
del _p, _q
print(f"\n{TAG} ready. USE_LAND={USE_LAND}, A_MAX={A_MAX}, {N_SEEDS} seeds.", flush=True)


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
        # land features from the (possibly mirrored) vp -- see module docstring on why this is
        # the physically self-consistent choice, not a bug.
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


def weighted_loader(idx, weights, aug=False):
    """Same as loader(), but samples idx with replacement proportional to `weights` instead of
    uniformly -- see land_weight(). Validation deliberately does NOT use this: early stopping and
    model selection must reflect the true population, not the oversampled training mix."""
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(idx), replacement=True)
    return torch.utils.data.DataLoader(DS(idx, aug), batch_size=BATCH, sampler=sampler, num_workers=2,
                                       pin_memory=True, persistent_workers=True, drop_last=True)


TR_WEIGHT = land_weight(tr_idx)
print(f"train sampler: {(TR_WEIGHT > 1.0).sum():,}/{len(tr_idx):,} windows upweighted "
      f"({(TR_WEIGHT >= MOUNTAIN_W).sum():,} mountainous-near-land at {MOUNTAIN_W:.0f}x, "
      f"{((TR_WEIGHT > 1.0) & (TR_WEIGHT < MOUNTAIN_W)).sum():,} flat-near-land at {NEAR_W:.0f}x)",
      flush=True)


def total_loss(s, ls, fp, tgt, m, fl, fm):
    base = G["total_loss"](s, ls, tgt, m)
    fmm = fm.unsqueeze(-1)
    flow = (F.smooth_l1_loss(fp, fl, reduction="none") * fmm).sum() / fmm.sum().clamp(min=1)
    return base + 0.3 * flow, float(flow.detach())


def train_one(seed, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrackFormerLand().to(DEVICE)
    print(f"seed {seed} | params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = weighted_loader(tr_idx, TR_WEIGHT, aug=True), loader(va_idx, False)

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
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            lw = float(model.land.mlp[-1].weight.abs().mean())
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"flow {vf:.3f} | landW {lw:.5f} | {time.time()-te:.0f}s", flush=True)
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
    m = TrackFormerLand().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); return m


MS = [load_m(c) for c in CK]
n_land_test = int((basins[wpep] == "WP").sum())
print(f"\nWP+EP 2020+, {len(wpep)} windows ({len(wp_only)} WP-only, land-covered "
      f"{100*n_land_test/len(wpep):.0f}%)")
print("  BASELINE  v23 434.96 km (the bar -- same architecture, land branch off)")
for i, c in enumerate(CK):
    print(f"  {TAG} seed{i}  full {track_err([load_m(c)], wpep):.2f} km  "
          f"WP-only {track_err([load_m(c)], wp_only):.2f} km", flush=True)
e_full = track_err(MS, wpep)
e_wp = track_err(MS, wp_only)
print(f"\n  {TAG} ENSEMBLE ({len(MS)} seeds)  full WP+EP {e_full:.2f} km  |  WP-only {e_wp:.2f} km")
print(f"  vs v23 434.96 (full): {e_full - 434.96:+.2f} km", flush=True)
print("  NOTE: seed spread is ~19 km. WP-only isolates the effect where land geometry is actually "
      "\n  available; EP windows always see the branch at zero.", flush=True)
json.dump({TAG: {"full": e_full, "wp_only": e_wp}, "use_land": USE_LAND, "a_max": A_MAX,
           "n_seeds": len(MS)}, open(f"/content/{TAG}.json", "w"))
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
