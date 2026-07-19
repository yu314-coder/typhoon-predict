"""Causal RMT consensus: what you could actually have produced AT the launch time.

The consensus on the other maps is an ANALYSIS. At each valid time it combines every forecast
describing that moment, including ones launched hours AFTER the initialisation being compared --
so it leans on information that did not exist yet at launch. That is fine for estimating where a
storm was, and useless as a forecast.

This builds the strictly causal version instead. Standing at T0, the only members available are
forecasts launched at or before T0, so for a valid time T0 + 6k hours a member launched at
T0 - 6j hours contributes at lead k + j. Membership therefore SHRINKS with lead -- 20 members at
+6 h, one at +120 h -- the opposite of the analysis, which is why it degenerates to the single
latest forecast at the far end.

No Kalman smoothing here, and no parameter tuned against truth: the smoother's q is chosen by
scoring candidates against what happened, which would smuggle the future back in. Plain
minimum-variance weights on an MP-cleaned covariance, nothing else.

The covariance is fitted on the four main storms' v10 errors and reused for every model, so the
weighting scheme is identical across models and any difference comes from the forecasts. For a
1979 storm those four are anachronistic (three are 2022-2026), but they are a climatological error
statistic, not information about Tip.
"""
import json, os, math, numpy as np
from make_rmt_tracks import err_field, cov_from, mp_clean, observed_at, SIX_H

MIN_MEMBERS = 3
R = 111.2


def causal_tracks(rec, T0, Cc, n_leads=20):
    """Direct forecast from T0, and the causal weighted consensus available at T0."""
    LAT, LON = np.asarray(rec["lat"]), np.asarray(rec["lon"])
    BT = np.asarray(rec["base_time"], dtype="int64")
    T0 = int(T0)
    i0 = int(np.abs(BT - T0).argmin())
    assert abs(int(BT[i0]) - T0) < SIX_H, "no window at that launch time"
    direct = [(float(LAT[i0][L]), float(LON[i0][L])) for L in range(n_leads)]

    out, eq, counts = [], [], []
    for k in range(1, n_leads + 1):
        V = T0 + k * SIX_H
        mem = []
        for w in range(len(BT)):
            if int(BT[w]) > T0:                       # launched in the future -- not available
                continue
            L = int(round((V - int(BT[w])) / SIX_H)) - 1
            if 0 <= L < n_leads:
                mem.append((w, L))
        if len(mem) < MIN_MEMBERS:
            break
        leads = [L for _, L in mem]
        la = np.array([LAT[w][L] for w, L in mem]); lo = np.array([LON[w][L] for w, L in mem])
        sub = Cc[np.ix_(leads, leads)]
        inv = np.linalg.pinv(sub + 1e-6 * np.eye(len(leads)) * np.trace(sub) / len(leads))
        one = np.ones(len(leads))
        wg = np.clip(inv @ one, 0, None)
        wg = wg / wg.sum() if wg.sum() > 0 else one / len(one)
        out.append((float((wg * la).sum()), float((wg * lo).sum())))
        eq.append((float(la.mean()), float(lo.mean())))     # equal weight, for comparison
        counts.append(len(mem))
    return direct, out, eq, counts, i0


def km(a1, o1, a2, o2):
    return math.hypot((o2 - o1) * R * math.cos(math.radians((a1 + a2) / 2)), (a2 - a1) * R)


def score_against(pts, T0, obs, n_from=1):
    """Mean position error of a lead-ordered list of points against the best-track fixes."""
    e = []
    for k, (la, lo) in enumerate(pts, start=n_from):
        V = int(T0) + k * SIX_H
        if V in obs:
            e.append(km(obs[V][0], obs[V][1], la, lo))
    return (float(np.mean(e)) if e else float("nan")), (e[-1] if e else float("nan")), len(e)


if __name__ == "__main__":
    MAIN = json.load(open("track_build/v10_tracks.json"))
    C, n = cov_from([err_field(r)[0] for r in MAIN.values()])
    Cc, kept, _ = mp_clean(C, 20.0 / max(n, 1.0))
    print(f"covariance: {', '.join(MAIN)} | {kept}/20 eigenvalues above the MP edge\n")

    SRC = {"v10": json.load(open("track_build/tipmap/v10_tracks.json"))["Tip"],
           "v10.2": json.load(open("track_build/v21_tracks.json"))["tip"]["Tip"]}
    LAUNCH = [("11 Oct 0600", "1979-10-11T06:00"), ("13 Oct 0600", "1979-10-13T06:00"),
              ("16 Oct 0600", "1979-10-16T06:00")]

    print(f"{'model':6s} {'launch':12s} {'':>6s} {'mean err':>9s} {'120h':>8s} {'n':>4s}")
    for tag, rec in SRC.items():
        obs = observed_at(rec["base_time"], rec["base_lat"], rec["base_lon"])
        out = {}
        for nm, iso in LAUNCH:
            T0 = int(np.datetime64(iso, "ns").astype("int64"))
            direct, cons, eq, counts, i0 = causal_tracks(rec, T0, Cc)
            dm, d120, dn = score_against(direct, T0, obs)
            cm, c120, cn = score_against(cons, T0, obs)
            em, e120, _ = score_against(eq, T0, obs)
            print(f"{tag:6s} {nm:12s} {'direct':>6s} {dm:9.0f} {d120:8.0f} {dn:4d}")
            print(f"{'':6s} {'':12s} {'rmt':>6s} {cm:9.0f} {c120:8.0f} {cn:4d}"
                  f"   members {counts[0]}->{counts[-1]} over {len(cons)} leads")
            print(f"{'':6s} {'':12s} {'equal':>6s} {em:9.0f} {e120:8.0f}")
            la0, lo0 = rec["base_lat"][i0], rec["base_lon"][i0]
            out[nm] = {"lat": [[la0] + [p[0] for p in direct]],
                       "lon": [[lo0] + [p[1] for p in direct]],
                       "causal": [[la0, lo0]] + [[p[0], p[1]] for p in cons],
                       "base_time": [int(rec["base_time"][i0])],
                       "base_lat": rec["base_lat"], "base_lon": rec["base_lon"],
                       "launch": [la0, lo0], "err120_mean": d120, "n": 1,
                       "causal_eq": [[la0, lo0]] + [[p[0], p[1]] for p in eq],
                       "causal_err": cm, "causal_eq_err": em, "direct_err": dm}
        os.makedirs("track_build/causal", exist_ok=True)
        json.dump(out, open(f"track_build/causal/{tag}_tracks.json", "w"))
    print("\nwrote track_build/causal/{v10,v10.2}_tracks.json")
