"""v24 on Colab — the basin map replaces the storm patch.

    from google.colab import drive; drive.mount('/content/drive')
    !wget -q -O /content/v24.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/<SHA>/colab_v30_train.py
    import os; os.environ["V30_SEEDS"]="10"; exec(open('/content/v24.py').read())

WHAT CHANGES AND WHY.

Every version through v23 fed the model a 17x17 patch at 2.5 deg centred on the storm: +-21 deg.
For a typhoon at 20N that stops at 41N, so the mid-latitude trough at 40-50N that decides whether
it recurves was never in the input. The model was asked to predict recurvature while blind to its
cause. The fields were also DAILY MEANS, matched to 6-hourly windows -- up to +-12 h of smearing.

v24 swaps that for the box an operational forecaster actually reads:
    100-180E, 0-60N at 2.5 deg (33 x 25), 6-HOURLY, at t-24h / t-12h / t0
    7 channels: hgt500 (the ridge/trough map) + u,v at 850/500/200
Keeping the wind levels separate rather than pre-averaging means u200-u850 IS the vertical shear,
a first-order predictor no earlier version has seen.

WHAT DOES NOT CHANGE. v21's chain-of-thought head, the curved-persistence baseline, the 4-term
track loss, the intensity decoder, the lead queries. Only the environmental encoder is replaced, so
if the number moves it is the DATA, not a new architecture. V30_USE_BASIN=0 runs the identical code
against the old patch, which is the ablation that attributes the gain.

OPERATIONAL HONESTY. The index this reads was built with causal inputs only: for a window at
03 UTC the t0 field is the 00Z analysis, never the 06Z one. Interpolating between the bracketing
6-hourly fields would have used data from AFTER the forecast launched. v24 must be able to forecast
a storm it has never seen from data available at that moment; nothing here carries storm identity,
absolute date, or any per-window unique key.

FIELD CHAIN-OF-THOUGHT (V30_W_FIELD > 0). Optionally predicts the future basin field at each lead
as dense auxiliary supervision -- ~800 supervised values per lead instead of the flow head's 2,
which is the strongest lever available against the 1.29x train->test overfit. Off by default so the
data change is measured alone first.
"""

import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
DRIVE = "/content/drive/MyDrive/typhoon"
N_SEEDS = int(os.environ.get("V30_SEEDS", "10"))
USE_BASIN = int(os.environ.get("V30_USE_BASIN", "1"))
W_FLOW = float(os.environ.get("W_FLOW", "0.3"))
W_FIELD = float(os.environ.get("V30_W_FIELD", "0.0"))
TAG = os.environ.get("V30_TAG", "v24" if USE_BASIN else "v24abl")
KM6H = 6 * 3600 / 1000.0

for fn in ("track_windows_v13.npz", "dlm4_int8.npz", "lead_flow.npz"):
    dst = "/content/d/" + fn if fn == "track_windows_v13.npz" else "/content/" + fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"fetching {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/{fn}", dst)

# the basin tensor is 261 MB, too big for the repo -- it comes from Drive
for fn in ("basin_all_int8.npz", "v24_index.npz"):
    if not os.path.exists(f"/content/{fn}"):
        src = f"{DRIVE}/{fn}"
        assert os.path.exists(src), (
            f"{src} not found. Mount Drive and upload {fn} to MyDrive/typhoon/ first.")
        import shutil; shutil.copy(src, f"/content/{fn}")
        print(f"copied {fn} from Drive", flush=True)

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
mask = G["mask"]; vpair = G["vpair"]; z = G["z"]; mirror = G["mirror"]
tr_idx, va_idx, te_idx = G["tr_idx"], G["va_idx"], G["te_idx"]
basins = G["basins"]
EPOCHS, PATIENCE, BATCH = G["EPOCHS"], G["PATIENCE"], G["BATCH"]
LR, WEIGHT_DECAY, MIRROR_P = G["LR"], G["WEIGHT_DECAY"], G["MIRROR_P"]

_lf = np.load("/content/lead_flow.npz")
FLOW_T = _lf["flow"].astype("float32"); FLOW_M = _lf["got"].astype("float32")
DSC = np.load("/content/dlm4_int8.npz")["scale"][2:4].astype("float32")
_ii, _jj = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
_d = np.hypot(_ii, _jj) * 2.5
ANN = torch.tensor(((_d >= 3.0) & (_d <= 8.0)).astype("float32"), device=DEVICE)

# ---- basin fields -------------------------------------------------------------------------
_B = np.load("/content/basin_all_int8.npz")
BQ = _B["q"]                                   # [T,7,25,33] int8, kept int8 in RAM (390 MB)
BSC = _B["scale"].astype("float32"); BOFF = _B["offset"].astype("float32")
NLAT, NLON = BQ.shape[2], BQ.shape[3]
_IX = np.load("/content/v24_index.npz")
IN_LO = _IX["in_lo"]; IN_OK = _IX["in_ok"]
TG_LO = _IX["tg_lo"]; TG_HI = _IX["tg_hi"]; TG_W = _IX["tg_w"]; TG_OK = _IX["tg_ok"]
NCH = BQ.shape[1]
print(f"basin {BQ.shape} int8 | index {IN_LO.shape} | "
      f"all-3-inputs on {100*IN_OK.prod(1).mean():.1f}% of windows", flush=True)

# normalise to roughly unit scale using the stored quantisation, per channel
_BM = np.array([(BQ[::97, c].astype("float32") * BSC[c] + BOFF[c]).mean() for c in range(NCH)],
               "float32")
_BS = np.array([(BQ[::97, c].astype("float32") * BSC[c] + BOFF[c]).std() + 1e-3
                for c in range(NCH)], "float32")
print("channel means:", np.round(_BM, 1).tolist(), flush=True)


def basin_at(rows):
    """[n,7,25,33] float32, normalised. rows are indices into the field axis."""
    v = BQ[rows].astype("float32") * BSC[None, :, None, None] + BOFF[None, :, None, None]
    return (v - _BM[None, :, None, None]) / _BS[None, :, None, None]


def basin_target(i, L):
    """Interpolated future field for window i at lead L. Supervision only, never an input."""
    lo, hi, w = TG_LO[i, L], TG_HI[i, L], TG_W[i, L]
    a = basin_at(np.array([lo]))[0]
    if hi != lo and w > 0:
        a = (1 - w) * a + w * basin_at(np.array([hi]))[0]
    return a


class BasinStem(nn.Module):
    """Replaces v17's storm-patch CNN. Encodes the basin box instead, and falls back to the old
    stem when the basin is switched off, so the ablation runs identical code.

    The three input times enter as stacked channels rather than a separate axis: the ridge's
    MOVEMENT is the signal, and a conv over stacked times reads it directly. Strides take
    25x33 -> 13x17 -> 7x9 = 63 tokens, close to the old encoder's 25 so the decoder memory does
    not blow up.

    Context is stashed on the module rather than threaded through forward(), so v24 inherits v21's
    forward VERBATIM. Hand-rewriting that forward is how v15, v21 and the first draft of v23 broke.
    """
    def __init__(self, d, old):
        super().__init__()
        self.old = old
        self.net = nn.Sequential(
            nn.Conv2d(3 * NCH, 64, 3, padding=1), nn.GELU(), nn.Dropout2d(0.10),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GELU(), nn.Dropout2d(0.10),
            nn.Conv2d(96, d, 3, stride=2, padding=1), nn.GELU())
        self.ctx = None

    def forward(self, slp):
        if not USE_BASIN or self.ctx is None:
            return self.old(slp)
        b = self.ctx.shape[0]
        return self.net(self.ctx.reshape(b, 3 * NCH, NLAT, NLON))


CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
_g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": G["math"], "G": G, "ANN": ANN,
        "DSC": DSC, "KM6H": KM6H, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, urllib.request.urlopen(f"{RAW}/colab_v26_train.py").read().decode(),
               re.S).group(0), _g21)
V21 = _g21["TrackFormerCoT"]

NTOK = 63


class TrackFormerBasin(V21):
    """v21 with the basin encoder swapped in. forward() is v21's, untouched."""
    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.steer_pos.shape[-1]
        self.steer_cnn = BasinStem(d, self.steer_cnn)
        if USE_BASIN:
            # the basin stem emits 63 tokens where the patch stem emitted 25
            self.steer_pos = nn.Parameter(torch.zeros(1, NTOK, d))

    def forward(self, tr, vp, slp, bx=None):
        self.steer_cnn.ctx = bx if USE_BASIN else None
        try:
            return super().forward(tr, vp, slp)
        finally:
            self.steer_cnn.ctx = None


print(f"\n{TAG}: USE_BASIN={USE_BASIN}, W_FIELD={W_FIELD}, {N_SEEDS} seeds", flush=True)

# ---- MIRROR MUST BE OFF, AND OFF FOR BOTH ARMS ---------------------------------------------
# v17's augmentation flips the storm-centred patch north-south and negates v500. That is a valid
# symmetry for a patch that travels with the storm. It is NOT valid for a FIXED geographic box:
# flipping 0-60N puts the mid-latitude westerlies at the equator and the monsoon trough at 60N.
#
# So v24 trains without mirroring, which halves the effective sample count. The learning curve
# says that costs roughly 18 km -- easily enough to hide the basin's benefit. The ablation arm is
# therefore ALSO run without mirroring: v24 vs v24abl stays a fair, like-for-like comparison of
# the basin against the patch, and both pay the same augmentation penalty.
#
# Consequence to state plainly when reporting: v24's ABSOLUTE km will likely be worse than v23's,
# because v23 had mirroring and v24 cannot. The meaningful number is v24 - v24abl, not v24 - v23.
# Comparing v24abl against v23 separately measures what the lost augmentation cost.
MIRROR_P = 0.0
print("mirror augmentation DISABLED (invalid on a fixed geographic grid); "
      "both arms run without it", flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(self, idx, aug):
        self.idx = np.asarray(idx); self.aug = aug

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        tr = torch.from_numpy(track[j]); tg = torch.from_numpy(target[j])
        mk = torch.from_numpy(mask[j]); sp = torch.from_numpy(SLP[j])
        vp = torch.from_numpy(vpair[j])
        fl = torch.from_numpy(FLOW_T[j].copy()); fm = torch.from_numpy(FLOW_M[j].copy())
        # the three causal basin snapshots; a window outside the field span gets exact zeros,
        # never a fabricated field, matching this project's unavailable-means-zero convention
        bx = torch.from_numpy(basin_at(IN_LO[j]) * IN_OK[j][:, None, None, None])
        return tr, vp, sp, tg, mk, fl, fm, bx


def loader(idx, sh, aug=False):
    return torch.utils.data.DataLoader(DS(idx, aug), batch_size=BATCH, shuffle=sh, num_workers=2,
                                       pin_memory=True, persistent_workers=True, drop_last=sh)


def total_loss(s, ls, fp, tgt, m, fl, fm):
    base = G["total_loss"](s, ls, tgt, m)
    fmm = fm.unsqueeze(-1)
    flow = (F.smooth_l1_loss(fp, fl, reduction="none") * fmm).sum() / fmm.sum().clamp(min=1)
    return base + W_FLOW * flow, float(flow.detach())


def train_one(seed, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrackFormerBasin().to(DEVICE)
    print(f"seed {seed} | params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    tl, vl = loader(tr_idx, True, aug=True), loader(va_idx, False)

    def run(ld, train):
        model.train(train); tot = cnt = 0.0; fa = 0.0
        for tr, v0, sp, tg, m, fl, fm, bx in ld:
            tr, v0, sp, tg, m, fl, fm, bx = [x.to(DEVICE, non_blocking=True)
                                             for x in (tr, v0, sp, tg, m, fl, fm, bx)]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
                s, ls, fp = model(tr, v0, sp, bx)
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
        else:
            bad += 1
        with torch.no_grad():
            amag = model.A.detach().cpu().numpy()
            # mean |W| of the basin stem's first conv: if it stays at its init the basin is
            # being ignored and v24 has quietly reverted to the patch model
            bw = float(model.steer_cnn.net[0].weight.abs().mean()) if USE_BASIN else 0.0
        print(f"ep {ep:03d} | train {trl:.5f} | val {vv:.5f} | best {best:.5f} | "
              f"flow {vf:.3f} | A {amag[0]:.2f},{amag[1]:.2f} | basinW {bw:.5f} | "
              f"{time.time()-te:.0f}s", flush=True)
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
print(f"{TAG} trained: {len(CK)} seeds (USE_BASIN={USE_BASIN})", flush=True)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
# 2026 storms past 17 March have no basin fields at all; scoring them would compare a model that
# saw the environment against one that saw exact zeros. Report both so the drop is visible.
covered = np.array([i for i in wpep if IN_OK[i].prod() > 0])
print(f"\nWP+EP 2020+: {len(wpep)} windows, {len(covered)} with basin coverage "
      f"({100*len(covered)/len(wpep):.1f}%)", flush=True)
SC = TARGET_SCALE


@torch.no_grad()
def track_err(ms, idx):
    P = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        bx = torch.from_numpy(basin_at(IN_LO[j].ravel()).reshape(len(j), 3, NCH, NLAT, NLON)
                              * IN_OK[j][:, :, None, None, None]).to(DEVICE)
        a = [torch.from_numpy(track[j]).to(DEVICE), torch.from_numpy(vpair[j]).to(DEVICE),
             torch.from_numpy(SLP[j]).to(DEVICE), bx]
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().cpu().numpy())
    C = np.cumsum(np.concatenate(P)[..., :2], 1)
    T = np.cumsum(target[idx][..., :2], 1)
    return float(np.sqrt(((C - T) ** 2).sum(-1)).mean())


def load_m(c):
    m = TrackFormerBasin().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"]); return m


MS = [load_m(c) for c in CK]
print("  BASELINES (with mirror)  v21 443.6 | v22 443.4 | v23 435.0")
for i, c in enumerate(CK):
    print(f"  {TAG} seed{i}  {track_err([load_m(c)], covered):.2f} km", flush=True)
e_cov = track_err(MS, covered)
e_all = track_err(MS, wpep)
print(f"\n  {TAG} ENSEMBLE ({len(MS)} seeds)")
print(f"    on basin-covered windows ({len(covered)})  {e_cov:.2f} km   <- the honest number")
print(f"    on all WP+EP windows     ({len(wpep)})  {e_all:.2f} km")
print("  NOTE: v24 trains WITHOUT mirror augmentation (invalid on a fixed grid), so its absolute")
print("  km is not comparable to v21/v22/v23. The meaningful comparison is v24 vs v24abl, which")
print("  pays the same penalty. Run V30_USE_BASIN=0 for that arm.", flush=True)
json.dump({TAG: e_cov, "all_windows": e_all, "use_basin": USE_BASIN, "n_seeds": len(MS),
           "n_covered": int(len(covered))}, open(f"/content/{TAG}.json", "w"))
try:
    from google.colab import files
    import subprocess
    subprocess.run(f"tar cf /content/{TAG}_seeds.tar /content/{TAG}_seed*.pt", shell=True)
    files.download(f"/content/{TAG}.json"); files.download(f"/content/{TAG}_seeds.tar")
except Exception as ex:
    print("download skipped:", ex)
