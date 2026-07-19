import json
nb = json.load(open("colab_train_v16.ipynb"))
C = nb["cells"]
def src(i): return "".join(C[i]["source"])
def put(i, s): C[i]["source"] = [l + "\n" for l in s.split("\n")[:-1]] + [s.split("\n")[-1]]
def sub(i, old, new, n=1):
    s = src(i); assert s.count(old) == n, f"cell {i}: expected {n}x {old[:50]!r}, got {s.count(old)}"
    put(i, s.replace(old, new))

put(0, """# TrackFormer v17 — a loss that actually constrains the track

Two findings drive this version.

**1. The v15/v16 notebook lost half the track loss.** v13/v14/v14.1 score position twice — per-step
displacement (weighted by `sqrt(lead)`) *and* the cumulative position. The notebook kept only the
per-step term:

    # v14.1                                    # v15/v16 notebook
    step = smooth_l1(pm, tm) * LEADW           d = (pm - tm) * mask
    pos  = smooth_l1(cumsum(pm), cumsum(tm))   return |d|
    return step + pos                          # <- no cumulative term at all

Nothing penalised *where the storm ended up*, only how each 6-hour hop looked in isolation. Errors
that individually look small are free to accumulate in the same direction.

**2. Speed and heading were never scored as such.** A step vector that is the right length in the
wrong direction costs the same as one the wrong length in the right direction. For a track forecast
those are not equivalent errors — direction compounds, magnitude does not.

v17 scores all four:

| term | what it constrains |
|---|---|
| `step` | per-6h displacement, weighted by sqrt(lead) — restored from v14.1 |
| `pos` | **cumulative position** — restored; this is what stops long-lead drift |
| `spd` | magnitude of each step: forward speed |
| `dir` | 1 - cos(angle) between predicted and observed step: heading |

**SST is dropped.** The v16 ablation — same architecture, same seeds, SST channel zeroed — found
removing it *improved* track from 479.13 to **466.30 km** while helping pressure by only 0.19 hPa.
v17 therefore runs on 4 channels, and 466.30 km is the baseline it must beat.

**5 seeds instead of 3**, so the RMT ensemble combination that follows has enough estimators to
work with.

## Upload these to Google Drive first (`MyDrive/typhoon/`)
- `track_windows_v13.npz` (43 MB)
- `steer5_int8.npz` (90 MB) — ch4 is simply not read

**Runtime -> Change runtime type -> L4 GPU.**""")

# ---- data cell: drop SST ----
sub(4, """_q = np.load("/content/d/steer5_int8.npz")
SLP = np.clip(_q["q"].astype("float32") / 31.75, -STEER_CLIP, STEER_CLIP)
AVAIL = _q["ok"]        # [N,3] per group: SLP pair, steering pair, SST
del _q""",
"""_q = np.load("/content/d/steer5_int8.npz")
# ch4 is SST. The v16 ablation showed it costs 12.8 km of track error and buys 0.19 hPa of
# pressure, so v17 reads only the four fields that earn their place.
SLP = np.clip(_q["q"][:, :4].astype("float32") / 31.75, -STEER_CLIP, STEER_CLIP)
AVAIL = _q["ok"]        # [N,3] per group: SLP pair, steering pair, SST (ch4 unused here)
del _q""")
sub(4, 'sp [5,17,17]', 'sp [4,17,17]')

# ---- sanity cell: no ch4 any more ----
sub(6, """print(f"SST  mean  {sp[4].mean():+7.3f} -> {sp2[4].mean():+7.3f}   (scalar: must match)")
assert torch.allclose(sp[4].mean(), sp2[4].mean(), atol=1e-4), "SST is a scalar field"
""", "")

# ---- model: 4 input channels ----
sub(8, "nn.Conv2d(5, 24, 3, padding=1)", "nn.Conv2d(4, 24, 3, padding=1)")
for i in range(len(C)):
    if C[i]["cell_type"] == "code":
        s = src(i)
        if "TrackFormerV16" in s: put(i, s.replace("TrackFormerV16", "TrackFormerV17"))
        if "v16_seed" in src(i): put(i, src(i).replace("v16_seed", "v17_seed"))

# ---- the loss ----
old = """def track_loss(s, tn, m):
    d = (s[..., :2] - tn[..., :2]) * m[..., :2]
    return (torch.sqrt((d ** 2).sum(-1) + 1e-6)).sum() / m[..., 0].sum().clamp(min=1)"""
new = '''# long leads are where the money is, and they are the minority of the loss by default
LEADW = torch.sqrt(torch.arange(1, 21, device=DEVICE).float()); LEADW = LEADW / LEADW.mean()
W_SPD = float(os.environ.get("W_SPD", "0.30"))     # forward-speed magnitude
W_DIR = float(os.environ.get("W_DIR", "0.30"))     # heading
EPS = 1e-3

def track_loss(s, tn, m):
    """Four terms: per-step, cumulative position, forward speed, heading.

    v15/v16 had only the first. Nothing constrained where the storm ENDED UP, so small same-signed
    per-step errors were free to accumulate over 20 leads. Speed and heading are scored explicitly
    because a step of the right length in the wrong direction and one of the wrong length in the
    right direction are not equally costly for a track: direction compounds, magnitude does not.
    """
    pm, tm, mm = s[..., :2], tn[..., :2], m[..., :2]
    w = mm * LEADW.view(1, 20, 1)
    step = (F.smooth_l1_loss(pm, tm, reduction="none") * w).sum() / w.sum().clamp(min=1)
    pos = (F.smooth_l1_loss(torch.cumsum(pm, 1), torch.cumsum(tm, 1), reduction="none")
           * mm).sum() / mm.sum().clamp(min=1)
    ps = torch.sqrt((pm ** 2).sum(-1) + 1e-8)          # predicted step length
    ts = torch.sqrt((tm ** 2).sum(-1) + 1e-8)          # observed step length
    mv = mm[..., 0]
    spd = (F.smooth_l1_loss(ps, ts, reduction="none") * mv).sum() / mv.sum().clamp(min=1)
    cos = (pm * tm).sum(-1) / (ps.clamp(min=EPS) * ts.clamp(min=EPS))
    # heading is undefined for a storm that has not moved -- do not train on it
    hv = mv * (ts > EPS).float()
    dirl = ((1.0 - cos) * hv).sum() / hv.sum().clamp(min=1)
    return step + pos + W_SPD * spd + W_DIR * dirl'''
done = False
for i in range(len(C)):
    if old in src(i):
        sub(i, old, new); done = True
assert done, "track_loss not found"

# ---- 5 seeds ----
for i in range(len(C)):
    if "for seed in (0, 1, 2):" in src(i):
        sub(i, "for seed in (0, 1, 2):", "for seed in (0, 1, 2, 3, 4):")

# ---- baselines in the eval cell ----
for i in range(len(C)):
    s = src(i)
    if "BASELINES (repaired data)" in s:
        put(i, s.replace(
            'print("BASELINES (repaired data)   v14 489.3 | v14.1 500.2   [WP+EP 2020+]")',
            'print("BASELINES [WP+EP 2020+, repaired]  v14 489.3 | v14.1 500.2 | v16 479.1 | '
            'v16-noSST 466.3  <- the one to beat")'))
    if "REF = {" in src(i):
        put(i, src(i).replace('{\'v16ens\':>8}', '{\'v17ens\':>8}'))

json.dump(nb, open("colab_train_v17.ipynb", "w"), indent=1)
code = "".join("".join(c["source"]) + "\n" for c in nb["cells"] if c["cell_type"] == "code")
compile(code.replace("!", "#").replace("from google.colab import drive", "pass#"), "<nb>", "exec")
print("colab_train_v17.ipynb written and compiles")
