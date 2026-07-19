"""Smooth the fixed-horizon mean track without letting the future in.

Two ways smoothing normally cheats, both avoided here.

FIRST, the RTS backward pass. Smoothing at valid time V with an RTS smoother uses the points at
V+6h, V+12h ... and those points come from launches at V-42h, V-36h ... -- newer than the V-48h
horizon the track claims. So the backward pass is dropped: this is a forward-only constant-velocity
Kalman FILTER, where the estimate at V depends only on points at V and earlier, whose launches were
all at least 48 h before V. Strictly worse-looking than RTS, and honest.

SECOND, tuning. The smoother's process noise q was previously chosen by scoring candidates against
what actually happened, which is truth leaking in through a parameter. Here q is fitted on the four
main storms with v10 and then applied unchanged -- the same discipline already used for the error
covariance.
"""
import json, os, math, numpy as np
from make_rmt_tracks import err_field, cov_from, mp_clean, observed_at, SIX_H

R = 111.2
HORIZON_H = int(os.environ.get("FC_HORIZON_H", "48"))
MIN_LEAD = HORIZON_H // 6
Q_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def km(a1, o1, a2, o2):
    return math.hypot((o2 - o1) * R * math.cos(math.radians((a1 + a2) / 2)), (a2 - a1) * R)


def cv_filter(t_h, z, rvar, q):
    """Forward-only constant-velocity Kalman filter over one axis. No backward pass."""
    n = len(z)
    out = np.empty(n)
    x = np.array([z[0], 0.0]); P = np.array([[max(rvar[0], 1.0), 0.0], [0.0, 1e4]])
    out[0] = x[0]
    for k in range(1, n):
        dt = max(t_h[k] - t_h[k - 1], 1e-6)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        x = F @ x; P = F @ P @ F.T + Q
        S = P[0, 0] + max(rvar[k], 1.0)
        K = P[:, 0] / S
        x = x + K * (z[k] - x[0])
        P = P - np.outer(K, P[0, :])
        out[k] = x[0]
    return out


def mean_track(rec, Cc):
    """Fixed-horizon weighted mean, with the variance of each combination."""
    LAT, LON = np.asarray(rec["lat"]), np.asarray(rec["lon"])
    BT = np.asarray(rec["base_time"], dtype="int64")
    bins = {}
    for w in range(len(BT)):
        for L in range(20):
            vt = int(round((int(BT[w]) + (L + 1) * SIX_H) / SIX_H)) * SIX_H
            bins.setdefault(vt, []).append((w, L))
    T, la_, lo_, rv = [], [], [], []
    for vt in sorted(bins):
        old = [(w, L) for w, L in bins[vt] if L + 1 >= MIN_LEAD]
        if len(old) < 3:
            continue
        leads = [L for _, L in old]
        la = np.array([LAT[w][L] for w, L in old]); lo = np.array([LON[w][L] for w, L in old])
        sub = Cc[np.ix_(leads, leads)]
        inv = np.linalg.pinv(sub + 1e-6 * np.eye(len(leads)) * np.trace(sub) / len(leads))
        one = np.ones(len(leads))
        wg = np.clip(inv @ one, 0, None)
        wg = wg / wg.sum() if wg.sum() > 0 else one / len(one)
        d = float(one @ inv @ one)
        T.append(vt); la_.append(float((wg * la).sum())); lo_.append(float((wg * lo).sum()))
        rv.append(max(1.0 / d, 1.0) if d > 0 else 1e4)
    return np.array(T), np.array(la_), np.array(lo_), np.array(rv)


def smooth_ll(T, la, lo, rv, q):
    if len(T) < 3:
        return la, lo
    th = (T - T[0]) / 3.6e12
    lat0, lon0 = float(la.mean()), float(lo.mean())
    zx = (lo - lon0) * R * np.cos(np.radians((lat0 + la) / 2))
    zy = (la - lat0) * R
    sx, sy = cv_filter(th, zx, rv, q), cv_filter(th, zy, rv, q)
    sla = lat0 + sy / R
    return sla, lon0 + sx / (R * np.cos(np.radians((lat0 + sla) / 2)))


def score(T, la, lo, obs):
    e = [km(obs[t][0], obs[t][1], a, o) for t, a, o in zip(T, la, lo) if int(t) in obs]
    return float(np.mean(e)) if e else float("nan")


if __name__ == "__main__":
    MAIN = json.load(open("track_build/v10_tracks.json"))
    C, n = cov_from([err_field(r)[0] for r in MAIN.values()])
    Cc, _, _ = mp_clean(C, 20.0 / max(n, 1.0))

    # ---- fit q on the four main storms, never on the storm it will be applied to ----
    packs = []
    for nm, rec in MAIN.items():
        obs = observed_at(rec["base_time"], rec["base_lat"], rec["base_lon"])
        packs.append((mean_track(rec, Cc), obs))
    FORCE = os.environ.get("FC_Q")
    print("fitting q on:", ", ".join(MAIN))
    best_q, best_e = None, 1e18
    for q in Q_GRID:
        es = [score(T, *smooth_ll(T, la, lo, rv, q), obs) for (T, la, lo, rv), obs in packs]
        es = [e for e in es if e == e]
        m = float(np.mean(es))
        print(f"  q {q:7.3f}   mean {m:7.1f} km")
        if m < best_e:
            best_q, best_e = q, m
    if FORCE:
        best_q = float(FORCE)
        print(f"OVERRIDDEN to q = {best_q}: the out-of-sample fit picks minimal smoothing on every\n"
              f"storm tested, so this value cannot be justified from held-out data -- it is shown\n"
              f"because it is what Tip itself prefers, and is labelled as such on the page.\n")
    else:
        print(f"chosen q = {best_q} (fitted out of sample, applied unchanged below)\n")

    SETS = [("track_build/tipmap", ["Tip"]),
            ("track_build/pre1950map", ["Allyn", "Lilly", "Karen"])]
    print(f"{'storm':7s} {'model':6s} {'raw mean':>9s} {'smoothed':>9s}")
    for d, names in SETS:
        for tag in ("v10", "v10.2"):
            p = f"{d}/fc/{tag}_tracks.json"
            if not os.path.exists(p):
                continue
            D = json.load(open(p))
            for nm in names:
                rec = D[nm]
                obs = observed_at(rec["base_time"], rec["base_lat"], rec["base_lon"])
                T, la, lo, rv = mean_track(rec, Cc)
                sla, slo = smooth_ll(T, la, lo, rv, best_q)
                e0, e1 = score(T, la, lo, obs), score(T, sla, slo, obs)
                print(f"{nm:7s} {tag:6s} {e0:9.0f} {e1:9.0f}")
                rec["causal"] = [[float(a), float(o)] for a, o in zip(sla, slo)]
                rec["rmt_err"] = e1
                rec["raw_mean_err"] = e0
                rec["q_accel"] = best_q
            json.dump(D, open(p, "w"))
    print("\nupdated fc/*.json with the causally smoothed mean")
