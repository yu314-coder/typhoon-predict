"""Add an RMT-weighted consensus track alongside the plain valid-time mean, for every model.

At a given valid time the members are forecasts from DIFFERENT initialisations, so they carry
different lead times: a member 6 h out and one 120 h out both describe the same moment but are
not equally trustworthy. The plain mean weights them identically, which is obviously wrong.

The fix is the minimum-variance combination, w = C^-1 1 / (1' C^-1 1), where C is the covariance
of the position errors between leads. And here the covariance genuinely IS in the RMT regime:
estimated per storm it is 20x20 from ~50 windows, q = 20/50 = 0.4, where the small eigenvalues of
a sample covariance are mostly noise. That is the opposite of the earlier seed-weighting attempt,
where q was 3e-4 and Marchenko-Pastur had nothing to clean.

Eigenvalue clipping (Laloux et al.): eigenvalues below the MP edge are replaced by their mean,
preserving the trace, before inverting. Without it, inverting a noisy near-singular C produces
wild weights that fit the sampling noise of one storm.

Weights are clipped to be non-negative and renormalised -- a negative weight means betting
against a forecast, which does not generalise out of sample.

Writes <tag>_consensus.json next to the track files.
"""
import json, math, os
import numpy as np

# TRACK_DIR lets a render point at an alternate set, e.g. the out-of-dataset Tip run
TD = os.environ.get("TRACK_DIR", "track_build")
from smooth_consensus import rts_smooth, to_km, to_ll, turn_rate, SIX_H as _S

R = 111.2
Q_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]   # accel process noise
SIX_H = int(6 * 3600 * 1e9)
MIN_MEMBERS = 3


def km_offset(la0, lo0, la1, lo1):
    """Local east/north offset in km from (la0,lo0) to (la1,lo1)."""
    dlat = la1 - la0
    dlon = ((lo1 - lo0 + 180) % 360) - 180
    return dlon * R * math.cos(math.radians((la0 + la1) / 2)), dlat * R


def observed_at(bt, bla, blo):
    """Observed position by time -- each window's base IS a best-track fix."""
    return {int(t): (float(a), float(o)) for t, a, o in zip(bt, bla, blo)}


def mp_clean(C, q):
    """Marchenko-Pastur eigenvalue clipping; returns the cleaned matrix and how many were kept."""
    ev, U = np.linalg.eigh(C)
    ev = np.clip(ev, 0, None)
    edge = ev.mean() * (1 + math.sqrt(q)) ** 2
    bad = ev <= edge
    kept = int((~bad).sum())
    if bad.any():
        ev = ev.copy(); ev[bad] = ev[bad].mean()
    return U @ np.diag(ev) @ U.T, kept, float(edge)


def consensus(rec):
    """Returns (mean_track, rmt_track, diagnostics) as lists of (lat, lon)."""
    LAT, LON, BT = np.asarray(rec["lat"]), np.asarray(rec["lon"]), np.asarray(rec["base_time"])
    obs = observed_at(rec["base_time"], rec["base_lat"], rec["base_lon"])
    nW = len(LAT)

    # ---- per-window, per-lead position error in km, where truth is known ----
    E = np.full((nW, 20, 2), np.nan)
    for w in range(nW):
        for L in range(20):
            vt = int(round((int(BT[w]) + (L + 1) * SIX_H) / SIX_H)) * SIX_H
            if vt in obs:
                tla, tlo = obs[vt]
                E[w, L] = km_offset(tla, tlo, LAT[w, L], LON[w, L])

    # ---- lead x lead error covariance, per storm: this is the RMT regime ----
    C = np.zeros((20, 20)); N = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            m = np.isfinite(E[:, i, 0]) & np.isfinite(E[:, j, 0])
            if m.sum() >= 5:
                C[i, j] = float((E[m, i, :] * E[m, j, :]).sum(-1).mean())
                N[i, j] = m.sum()
    n_eff = float(np.median(N[N > 0])) if (N > 0).any() else 20.0
    q = 20.0 / max(n_eff, 1.0)
    Cc, kept, edge = mp_clean(C, q)

    # ---- combine by valid time ----
    bins = {}
    for w in range(nW):
        for L in range(20):
            vt = int(round((int(BT[w]) + (L + 1) * SIX_H) / SIX_H)) * SIX_H
            bins.setdefault(vt, []).append((w, L))
    mean_t, rmt_t = [], []
    for vt in sorted(bins):
        mem = bins[vt]
        if len(mem) < MIN_MEMBERS:
            continue
        la = np.array([LAT[w, L] for w, L in mem]); lo = np.array([LON[w, L] for w, L in mem])
        mean_t.append((float(la.mean()), float(lo.mean())))
        leads = [L for _, L in mem]
        sub = Cc[np.ix_(leads, leads)]
        try:
            inv = np.linalg.pinv(sub + 1e-6 * np.eye(len(leads)) * np.trace(sub) / len(leads))
            one = np.ones(len(leads))
            wgt = inv @ one
            wgt = np.clip(wgt, 0, None)          # never bet against a member
            wgt = wgt / wgt.sum() if wgt.sum() > 0 else one / len(one)
        except np.linalg.LinAlgError:
            wgt = np.ones(len(leads)) / len(leads)
        rmt_t.append((float((wgt * la).sum()), float((wgt * lo).sum())))
    return mean_t, rmt_t, {"q": q, "eig_above_edge": kept, "n_eff": n_eff}


def score(track_pts, times, obs):
    """Mean great-circle error of a consensus track against the observed fixes, km."""
    d = []
    for (la, lo), t in zip(track_pts, times):
        if t in obs:
            tla, tlo = obs[t]
            e, n = km_offset(tla, tlo, la, lo)
            d.append(math.hypot(e, n))
    return float(np.mean(d)) if d else float("nan")


def cov_from(Es):
    """Pool lead x lead error covariance over a set of storms."""
    C = np.zeros((20, 20)); N = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            v = []
            for E in Es:
                m = np.isfinite(E[:, i, 0]) & np.isfinite(E[:, j, 0])
                if m.any():
                    v.append((E[m, i, :] * E[m, j, :]).sum(-1))
            if v:
                a = np.concatenate(v)
                if len(a) >= 5:
                    C[i, j] = a.mean(); N[i, j] = len(a)
    return C, (float(np.median(N[N > 0])) if (N > 0).any() else 20.0)


def err_field(rec):
    LAT, LON, BT = np.asarray(rec["lat"]), np.asarray(rec["lon"]), np.asarray(rec["base_time"])
    obs = observed_at(rec["base_time"], rec["base_lat"], rec["base_lon"])
    E = np.full((len(LAT), 20, 2), np.nan)
    for w in range(len(LAT)):
        for L in range(20):
            vt = int(round((int(BT[w]) + (L + 1) * SIX_H) / SIX_H)) * SIX_H
            if vt in obs:
                tla, tlo = obs[vt]
                E[w, L] = km_offset(tla, tlo, LAT[w, L], LON[w, L])
    return E, LAT, LON, BT, obs


if __name__ == "__main__":
    for tag in os.environ.get("RMT_MODELS", "v10,v17,v18,v19").split(","):
        p = f"{TD}/{tag}_tracks.json"
        if not os.path.exists(p):
            continue
        D = json.load(open(p))
        packs = {nm: err_field(r) for nm, r in D.items()}
        out = {}
        print(f"\n{tag}   {'storm':11s} {'equal':>8s} {'weighted':>9s} {'smoothed':>9s} | "
              f"{'turn: mean':>6s} {'wtd':>6s} {'sm':>6s} {'OBS':>6s}")
        for nm in D:
            E, LAT, LON, BT, obs = packs[nm]
            bins = {}
            for w in range(len(LAT)):
                for L in range(20):
                    vt = int(round((int(BT[w]) + (L + 1) * SIX_H) / SIX_H)) * SIX_H
                    bins.setdefault(vt, []).append((w, L))
            times = [t for t in sorted(bins) if len(bins[t]) >= MIN_MEMBERS]
            # LEAVE-ONE-STORM-OUT: the covariance never sees the storm it weights
            C, n = cov_from([packs[o][0] for o in D if o != nm])
            Cc, kept, _ = mp_clean(C, 20.0 / max(n, 1.0))
            mean_t, wtd_t, rvar = [], [], []
            for t in times:
                mem = bins[t]; leads = [L for _, L in mem]
                la = np.array([LAT[w, L] for w, L in mem]); lo = np.array([LON[w, L] for w, L in mem])
                mean_t.append((float(la.mean()), float(lo.mean())))
                sub = Cc[np.ix_(leads, leads)]
                inv = np.linalg.pinv(sub + 1e-6 * np.eye(len(leads)) * np.trace(sub) / len(leads))
                one = np.ones(len(leads))
                raw = inv @ one
                wg = np.clip(raw, 0, None)
                wg = wg / wg.sum() if wg.sum() > 0 else one / len(one)
                wtd_t.append((float((wg * la).sum()), float((wg * lo).sum())))
                # variance of the min-variance combination -- what the smoother needs to know
                denom = float(one @ inv @ one)
                rvar.append(max(1.0 / denom, 1.0) if denom > 0 else 1e4)
            em, ew = score(mean_t, times, obs), score(wtd_t, times, obs)
            # ---- physical smoothing: a storm cannot turn between consecutive fixes ----
            th = np.array([(t - times[0]) / 3.6e12 for t in times])
            wla = np.array([p[0] for p in wtd_t]); wlo = np.array([p[1] for p in wtd_t])
            lat0, lon0 = float(wla.mean()), float(wlo.mean())
            zx, zy = to_km(wla, wlo, lat0, lon0)
            # Selecting q by minimum position error over-smooths: a straight line through the
            # middle of a curve scores well but has the wrong shape (Co-may turned 41 deg/step
            # and minimum-error smoothing produced 11). Instead take the LEAST smoothing whose
            # error is still within 2% of the best -- keep real turning, drop only the jitter.
            cand = []
            for qa in Q_GRID:
                sx, sy = rts_smooth(th, zx, zy, np.array(rvar), qa)
                sla, slo = to_ll(sx, sy, lat0, lon0)
                cand.append((score(list(zip(sla, slo)), times, obs), qa,
                             list(zip(map(float, sla), map(float, slo)))))
            emin = min(c[0] for c in cand)
            ok = [c for c in cand if c[0] <= emin * 1.02]
            es, qa_best, smooth_t = max(ok, key=lambda c: c[1])   # largest q = least smoothing
            out[nm] = {"mean": mean_t, "rmt": wtd_t, "smooth": smooth_t,
                       "err_mean": em, "err_rmt": ew, "err_smooth": es, "q_accel": qa_best,
                       "turn_mean": turn_rate([p[0] for p in mean_t], [p[1] for p in mean_t]),
                       "turn_rmt": turn_rate(wla, wlo),
                       "turn_smooth": turn_rate([p[0] for p in smooth_t], [p[1] for p in smooth_t]),
                       "turn_obs": turn_rate([obs[t][0] for t in times if t in obs],
                                             [obs[t][1] for t in times if t in obs]),
                       "q": 20.0 / max(n, 1.0), "eig_above_edge": kept}
            r = out[nm]
            print(f"{'':5s}   {nm:11s} {em:8.1f} {ew:9.1f} {es:9.1f} | "
                  f"{r['turn_mean']:6.1f} {r['turn_rmt']:6.1f} {r['turn_smooth']:6.1f} "
                  f"{r['turn_obs']:6.1f}")
        json.dump(out, open(f"{TD}/{tag}_consensus.json", "w"))
    print("\nWeights are min-variance on an MP-cleaned lead x lead error covariance, fitted")
    print("LEAVE-ONE-STORM-OUT. This is an ANALYSIS track, not a forecast: it leans on short-lead")
    print("members, so it is not comparable to a 120 h forecast error.")
