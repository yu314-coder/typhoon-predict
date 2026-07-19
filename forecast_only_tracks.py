"""Whole-storm tracks that are forecasts, not analyses.

The bold line on the earlier maps was a per-valid-time consensus. Every member was launched before
the moment it described, so it was causal in the trivial sense -- but it leaned on launches only
6 h old, which is why it scored 114 km beside a 1250 km forecast error. As a picture of "where was
the storm" that is fine. As a forecast it is meaningless, because no such product exists: at 12:00
you cannot use the 06:00 initialisation to tell someone where the storm will be at 12:00.

Two honest whole-storm products instead, both with a stated horizon:

  +120 h track   at each valid time, the position predicted by the forecast launched exactly
                 120 h earlier. One member, no averaging. This is the five-day product.

  >=+48 h mean   at each valid time, combine every forecast launched at least 48 h earlier
                 (leads 8..20). A genuine two-day-ahead product that still gets to average.
                 Available in equal-weight and RMT-weighted form -- both are reported.

Neither ever touches a launch newer than its stated horizon, so the error each one scores is
directly comparable to a forecast error at that horizon.
"""
import json, os, math, numpy as np
from make_rmt_tracks import err_field, cov_from, mp_clean, observed_at, SIX_H

R = 111.2
HORIZON_H = int(os.environ.get("FC_HORIZON_H", "48"))
MIN_LEAD = HORIZON_H // 6                      # leads >= this were launched >= HORIZON_H ago
FIXED_LEAD = 20                                # the +120 h product


def km(a1, o1, a2, o2):
    return math.hypot((o2 - o1) * R * math.cos(math.radians((a1 + a2) / 2)), (a2 - a1) * R)


def products(rec, Cc):
    LAT, LON = np.asarray(rec["lat"]), np.asarray(rec["lon"])
    BT = np.asarray(rec["base_time"], dtype="int64")
    obs = observed_at(rec["base_time"], rec["base_lat"], rec["base_lon"])

    bins = {}
    for w in range(len(BT)):
        for L in range(20):
            vt = int(round((int(BT[w]) + (L + 1) * SIX_H) / SIX_H)) * SIX_H
            bins.setdefault(vt, []).append((w, L))

    fixed, m_rmt, m_eq = [], [], []
    for vt in sorted(bins):
        mem = bins[vt]
        ex = [(w, L) for w, L in mem if L + 1 == FIXED_LEAD]
        if ex:
            w, L = ex[0]
            fixed.append((vt, float(LAT[w][L]), float(LON[w][L])))
        old = [(w, L) for w, L in mem if L + 1 >= MIN_LEAD]
        if len(old) >= 3:
            leads = [L for _, L in old]
            la = np.array([LAT[w][L] for w, L in old]); lo = np.array([LON[w][L] for w, L in old])
            m_eq.append((vt, float(la.mean()), float(lo.mean())))
            sub = Cc[np.ix_(leads, leads)]
            inv = np.linalg.pinv(sub + 1e-6 * np.eye(len(leads)) * np.trace(sub) / len(leads))
            one = np.ones(len(leads))
            wg = np.clip(inv @ one, 0, None)
            wg = wg / wg.sum() if wg.sum() > 0 else one / len(one)
            m_rmt.append((vt, float((wg * la).sum()), float((wg * lo).sum())))
    return fixed, m_rmt, m_eq, obs


def score(pts, obs):
    e = [km(obs[t][0], obs[t][1], a, o) for t, a, o in pts if t in obs]
    return (float(np.mean(e)) if e else float("nan")), len(e)


if __name__ == "__main__":
    MAIN = json.load(open("track_build/v10_tracks.json"))
    C, n = cov_from([err_field(r)[0] for r in MAIN.values()])
    Cc, _, _ = mp_clean(C, 20.0 / max(n, 1.0))

    SETS = [("track_build/tipmap", ["Tip"]),
            ("track_build/pre1950map", ["Allyn", "Lilly", "Karen"])]
    print(f"horizon {HORIZON_H} h (leads {MIN_LEAD}..20)   errors in km, all strictly causal\n")
    print(f"{'storm':7s} {'model':6s} {'+120h track':>12s} {'>=48h rmt':>10s} {'>=48h equal':>12s}")
    for d, names in SETS:
        for tag in ("v10", "v10.2"):
            p = f"{d}/{tag}_tracks.json"
            if not os.path.exists(p):
                continue
            D = json.load(open(p)); out = {}
            for nm in names:
                rec = D[nm]
                fixed, m_rmt, m_eq, obs = products(rec, Cc)
                ef, nf = score(fixed, obs)
                er, _ = score(m_rmt, obs)
                ee, _ = score(m_eq, obs)
                print(f"{nm:7s} {tag:6s} {ef:12.0f} {er:10.0f} {ee:12.0f}")
                out[nm] = dict(rec)
                out[nm]["fixed"] = [[a, o] for _, a, o in fixed]
                out[nm]["causal"] = [[a, o] for _, a, o in m_rmt]
                out[nm]["fixed_err"] = ef
                out[nm]["rmt_err"] = er
                out[nm]["eq_err"] = ee
            os.makedirs(f"{d}/fc", exist_ok=True)
            json.dump(out, open(f"{d}/fc/{tag}_tracks.json", "w"))
    print("\nwrote {tipmap,pre1950map}/fc/{v10,v10.2}_tracks.json")
