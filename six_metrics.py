"""Six-metric evaluator: track, max wind, pressure, radius, forward speed, heading.

Every metric reports MAE *and BIAS*, stratified by the regime that matters for it. Bias is the
point: for peak wind on Cat 4-5 storms the MAE is 37.40 and the bias is -36.76, so 98% of the
error is systematic. An MAE table alone hides that completely, and a model that is merely noisy
looks identical to one that refuses to predict strong storms.

Heading error is CIRCULAR -- differences are wrapped to (-180, 180] before averaging, and the
bias is the mean wrapped difference. Averaging raw degrees would put a forecast 1 deg east of
north and one 1 deg west of north 358 deg apart.

Heading is only scored where the storm is actually moving; for a near-stationary storm the
observed heading is numerical noise and scoring against it measures nothing.

Import and call `evaluate(P, T, K)` where P/T are [n,20,17] predictions/targets in physical units
and K is the [n,20,17] validity mask.
"""
import numpy as np

VMAX_BINS = [(0, 34, "TD <34"), (34, 64, "TS 34-63"), (64, 96, "Cat1-2"),
             (96, 113, "Cat3"), (113, 400, "Cat4-5")]
SPEED_BINS = [(0, 15, "slow <15"), (15, 25, "15-25"), (25, 35, "25-35"), (35, 500, "fast >=35")]
MOVING = 8.0          # km/h below which heading is not meaningfully defined


def _mae_bias(d):
    return float(np.abs(d).mean()), float(d.mean())


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def evaluate(P, T, K):
    """Returns {metric: {bin: (n, mae, bias)}} plus per-lead track error."""
    out = {}
    pt, tt = np.cumsum(P[..., :2], 1), np.cumsum(T[..., :2], 1)
    out["track_per_lead"] = np.sqrt(((pt - tt) ** 2).sum(-1)).mean(0).tolist()
    out["track"] = {"all": (int(pt.size // 2), float(np.sqrt(((pt - tt) ** 2).sum(-1)).mean()), 0.0)}

    obs_v = T[..., 2]
    for key, idx in [("vmax", 2), ("pressure", 3), ("rmw", 4)]:
        out[key] = {}
        for lo, hi, lab in VMAX_BINS:
            sel = K[..., idx] & (obs_v >= lo) & (obs_v < hi) & K[..., 2]
            if sel.sum() < 50:
                continue
            mae, bias = _mae_bias((P[..., idx] - T[..., idx])[sel])
            out[key][lab] = (int(sel.sum()), mae, bias)

    rm = K[..., 5:17]
    out["radii"] = {}
    for lo, hi, lab in VMAX_BINS:
        sel = rm & ((obs_v >= lo) & (obs_v < hi) & K[..., 2])[..., None]
        if sel.sum() < 50:
            continue
        mae, bias = _mae_bias((P[..., 5:17] - T[..., 5:17])[sel])
        out["radii"][lab] = (int(sel.sum()), mae, bias)

    # ---- motion: speed and heading, from the per-step displacements ----
    ps = np.hypot(P[..., 0], P[..., 1]) / 6.0
    ts = np.hypot(T[..., 0], T[..., 1]) / 6.0
    pb = np.degrees(np.arctan2(P[..., 0], P[..., 1]))
    tb = np.degrees(np.arctan2(T[..., 0], T[..., 1]))
    valid = K[..., 0] & K[..., 1]
    out["speed"], out["heading"] = {}, {}
    for lo, hi, lab in SPEED_BINS:
        sel = valid & (ts >= lo) & (ts < hi)
        if sel.sum() < 50:
            continue
        mae, bias = _mae_bias((ps - ts)[sel])
        out["speed"][lab] = (int(sel.sum()), mae, bias)
        # heading only where the storm is genuinely moving
        hsel = sel & (ts >= MOVING)
        if hsel.sum() >= 50:
            d = wrap180(pb - tb)[hsel]
            out["heading"][lab] = (int(hsel.sum()), float(np.abs(d).mean()), float(d.mean()))
    return out


UNITS = {"track": "km", "vmax": "kt", "pressure": "hPa", "rmw": "km",
         "radii": "km", "speed": "km/h", "heading": "deg"}
TITLES = {"track": "TRACK (cumulative position)", "vmax": "MAX WIND by observed strength",
          "pressure": "PRESSURE by observed strength", "rmw": "RADIUS OF MAX WIND by strength",
          "radii": "WIND RADII (12) by strength", "speed": "FORWARD SPEED by observed speed",
          "heading": "HEADING by observed speed (circular, moving storms only)"}


def report(results, order=None):
    """results: {model_name: evaluate(...)}"""
    names = order or list(results)
    for key in ["track", "vmax", "pressure", "rmw", "radii", "speed", "heading"]:
        print("\n" + "=" * (26 + 20 * len(names)))
        print(f"{TITLES[key]}  [{UNITS[key]}]")
        print("=" * (26 + 20 * len(names)))
        bins = list(results[names[0]].get(key, {}))
        if not bins:
            continue
        head = f"{'bin':14s} {'n':>8s} | " + " | ".join(f"{n:^17s}" for n in names)
        sub = f"{'':14s} {'':>8s} | " + " | ".join(f"{'MAE':>7s} {'bias':>8s}" for _ in names)
        print(head); print(sub)
        for b in bins:
            n = results[names[0]][key][b][0]
            row = f"{b:14s} {n:8d} | "
            for nm in names:
                v = results[nm].get(key, {}).get(b)
                row += (f"{v[1]:7.2f} {v[2]:+8.2f} | " if v else f"{'-':>7s} {'-':>8s} | ")
            print(row)
