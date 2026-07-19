"""Make the consensus track physically coherent instead of solving each valid time in isolation.

THE PROBLEM. The lead-weighted consensus computes every valid time independently. As time advances
the member set rotates -- a new initialisation enters at lead 1 while an old one drops off at lead
20 -- and because the weights concentrate ~45% on leads 1-4, the answer jumps whenever the
short-lead membership changes. The result tracks the observed route more closely than the equal
mean but turns in ways no storm turns.

THE FIX. A storm has momentum: its heading changes over ~12-24 h, not between consecutive 6-hourly
fixes. That is prior knowledge the point-wise solve throws away. Model the track as a
constant-velocity state [x, y, vx, vy] driven by acceleration noise, treat each weighted
consensus position as an observation, and run a Kalman filter plus RTS backward smoother.

Crucially the observation variance is NOT constant: it comes from the min-variance solve itself,
R = 1 / (1' C^-1 1), so a valid time with many good short-lead members pulls hard on the track
while one held up by a handful of long-lead members is allowed to be overridden by the motion
model. That is the whole point of having estimated the covariance in the first place.

The single free parameter is the acceleration noise. It is chosen LEAVE-ONE-STORM-OUT -- fitted on
three storms, applied to the fourth -- so the smoothing is not tuned on the storm it is scored on.
"""
import json, math, os
import numpy as np

R_KM = 111.2
SIX_H = int(6 * 3600 * 1e9)


def to_km(lat, lon, lat0, lon0):
    x = ((np.asarray(lon) - lon0 + 180) % 360 - 180) * R_KM * math.cos(math.radians(lat0))
    y = (np.asarray(lat) - lat0) * R_KM
    return x, y


def to_ll(x, y, lat0, lon0):
    lat = lat0 + np.asarray(y) / R_KM
    lon = lon0 + np.asarray(x) / (R_KM * math.cos(math.radians(lat0)))
    return lat, lon


def rts_smooth(t_hours, zx, zy, rvar, q_accel):
    """Constant-velocity Kalman filter + RTS smoother on 2-D positions.

    t_hours: observation times (may have gaps); rvar: per-observation position variance (km^2);
    q_accel: acceleration process noise (km/h^2)^2 -- the one tunable.
    """
    n = len(t_hours)
    xs = np.zeros((n, 4)); Ps = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))
    H = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
    x = np.array([zx[0], zy[0], 0.0, 0.0])
    P = np.diag([rvar[0], rvar[0], 50.0 ** 2, 50.0 ** 2])
    for k in range(n):
        if k > 0:
            dt = max(t_hours[k] - t_hours[k - 1], 1e-3)
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
            # standard CV process noise from white acceleration
            q = q_accel
            Q = q * np.array([[dt**4/4, 0, dt**3/2, 0],
                              [0, dt**4/4, 0, dt**3/2],
                              [dt**3/2, 0, dt**2, 0],
                              [0, dt**3/2, 0, dt**2]], float)
            x = F @ x
            P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        Rk = np.eye(2) * max(rvar[k], 1e-3)
        S = H @ P @ H.T + Rk
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (np.array([zx[k], zy[k]]) - H @ x)
        P = (np.eye(4) - K @ H) @ P
        xs[k], Ps[k] = x, P
    # RTS backward pass
    for k in range(n - 2, -1, -1):
        dt = max(t_hours[k + 1] - t_hours[k], 1e-3)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        C = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs[:, 0], xs[:, 1]


def turn_rate(lats, lons):
    """Mean absolute heading change per step, degrees -- the 'strange turning' diagnostic."""
    if len(lats) < 3:
        return float("nan")
    x, y = to_km(lats, lons, float(np.mean(lats)), float(np.mean(lons)))
    dx, dy = np.diff(x), np.diff(y)
    h = np.degrees(np.arctan2(dx, dy))
    d = (np.diff(h) + 180) % 360 - 180
    return float(np.abs(d).mean())
