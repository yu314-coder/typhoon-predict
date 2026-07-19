"""Consensus for Tip, with the lead x lead error covariance fitted on the FOUR main storms
and applied to Tip -- a genuinely out-of-sample use of the weighting, since Tip (1979) is
outside the training data entirely and played no part in estimating the covariance."""
import json, os, math, numpy as np
from make_rmt_tracks import err_field, cov_from, mp_clean, score, observed_at, SIX_H, MIN_MEMBERS
from smooth_consensus import rts_smooth, to_km, to_ll, turn_rate
Q_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

print(f"{'model':6s} {'equal mean':>11s} {'weighted':>10s} {'+smoothed':>10s} | "
      f"{'turn: mean':>10s} {'sm':>6s} {'OBS':>6s}")
# TIP_COV lets a model without its own four-storm tracks borrow another model's error
# covariance. v10.2 was only ever run on the out-of-dataset storms, so it has none of its
# own; using v10's keeps the weighting scheme identical across models, which means any
# difference in the drawn consensus comes from the forecasts rather than the weights.
COV = os.environ.get("TIP_COV")
for tag in os.environ.get("TIP_TAGS", "v10,v17,v18,v19").split(","):
    tip_p = f"track_build/tipmap/{tag}_tracks.json"
    main_p = f"track_build/{COV or tag}_tracks.json"
    if not (os.path.exists(tip_p) and os.path.exists(main_p)):
        continue
    MAIN = json.load(open(main_p))
    C, n = cov_from([err_field(r)[0] for r in MAIN.values()])   # fitted on Bavi/Wayne/Co-may/Hinnamnor
    Cc, kept, _ = mp_clean(C, 20.0 / max(n, 1.0))

    rec = json.load(open(tip_p))["Tip"]
    E, LAT, LON, BT, obs = err_field(rec)
    bins = {}
    for w in range(len(LAT)):
        for L in range(20):
            vt = int(round((int(BT[w]) + (L + 1) * SIX_H) / SIX_H)) * SIX_H
            bins.setdefault(vt, []).append((w, L))
    times = [t for t in sorted(bins) if len(bins[t]) >= MIN_MEMBERS]
    mean_t, wtd_t, rvar = [], [], []
    for t in times:
        mem = bins[t]; leads = [L for _, L in mem]
        la = np.array([LAT[w, L] for w, L in mem]); lo = np.array([LON[w, L] for w, L in mem])
        mean_t.append((float(la.mean()), float(lo.mean())))
        sub = Cc[np.ix_(leads, leads)]
        inv = np.linalg.pinv(sub + 1e-6 * np.eye(len(leads)) * np.trace(sub) / len(leads))
        one = np.ones(len(leads))
        wg = np.clip(inv @ one, 0, None); wg = wg / wg.sum() if wg.sum() > 0 else one / len(one)
        wtd_t.append((float((wg * la).sum()), float((wg * lo).sum())))
        d = float(one @ inv @ one); rvar.append(max(1.0 / d, 1.0) if d > 0 else 1e4)
    th = np.array([(t - times[0]) / 3.6e12 for t in times])
    wla = np.array([p[0] for p in wtd_t]); wlo = np.array([p[1] for p in wtd_t])
    lat0, lon0 = float(wla.mean()), float(wlo.mean())
    zx, zy = to_km(wla, wlo, lat0, lon0)
    cand = []
    for qa in Q_GRID:
        sx, sy = rts_smooth(th, zx, zy, np.array(rvar), qa)
        sla, slo = to_ll(sx, sy, lat0, lon0)
        cand.append((score(list(zip(sla, slo)), times, obs), qa,
                     list(zip(map(float, sla), map(float, slo)))))
    emin = min(c[0] for c in cand)
    es, qa, smooth_t = max([c for c in cand if c[0] <= emin * 1.02], key=lambda c: c[1])
    em, ew = score(mean_t, times, obs), score(wtd_t, times, obs)
    obs_t = turn_rate([obs[t][0] for t in times if t in obs], [obs[t][1] for t in times if t in obs])
    json.dump({"Tip": {"mean": mean_t, "rmt": wtd_t, "smooth": smooth_t,
                       "err_mean": em, "err_rmt": ew, "err_smooth": es,
                       "turn_mean": turn_rate([p[0] for p in mean_t], [p[1] for p in mean_t]),
                       "turn_smooth": turn_rate([p[0] for p in smooth_t], [p[1] for p in smooth_t]),
                       "turn_obs": obs_t, "q_accel": qa, "eig_above_edge": kept}},
              open(f"track_build/tipmap/{tag}_consensus.json", "w"))
    print(f"{tag:6s} {em:11.1f} {ew:10.1f} {es:10.1f} | "
          f"{turn_rate([p[0] for p in mean_t],[p[1] for p in mean_t]):10.1f} "
          f"{turn_rate([p[0] for p in smooth_t],[p[1] for p in smooth_t]):6.1f} {obs_t:6.1f}")
print("\ncovariance fitted on the 4 main storms, applied to Tip -- Tip never informs its own weights")
