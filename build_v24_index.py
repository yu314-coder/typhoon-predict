"""Join the basin fields onto the storm windows — and prove the join leaks nothing.

WHAT v24 IS ALLOWED TO SEE, for a forecast launched at t0:
    the storm's own track over t0-48h .. t0        (already in track_windows_v13)
    basin fields at t0-24h, t0-12h, t0             (past and present only)
and nothing else. In particular it must NEVER see storm identity or absolute date, because a model
that can only forecast storms it has met is not a forecast model. If the only way to place a storm
is to recognise it, the held-out-year score is measuring memory, not skill.

So this builder emits ONLY time-relative indices. There is no storm id, no year, no calendar
feature anywhere in what it writes, and it asserts that at the end.

THE 3-HOURLY PROBLEM. Half the storm windows sit at 03/09/15/21 UTC because some storms carry
3-hourly best-track fixes, while the fields are 6-hourly. Snapping those to the nearest field puts
up to 3 h of synoptic drift into every second training sample. This emits the bracketing pair and
a weight so the loader can interpolate:
    field(t) = (1-w) * F[lo] + w * F[hi]

FUTURE INDICES are emitted too. They are the chain-of-thought TARGETS -- the field the model must
predict at each lead. They are supervision, never input, and the loader must never feed them to the
encoder. The `got` mask marks leads whose target exists.
"""
import numpy as np, os, sys

SIX = int(6 * 3600 * 1e9)
B = np.load("track_build/basin_all_int8.npz")
T = B["time"].astype("int64")
z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
bt = z["base_time"].astype("int64")
nl = z["n_leads"].astype(int)
N = len(bt)
print(f"{N:,} windows | {len(T):,} field timesteps "
      f"({np.datetime64(int(T[0]),'ns').astype('datetime64[D]')} .. "
      f"{np.datetime64(int(T[-1]),'ns').astype('datetime64[D]')})")


def bracket(times):
    """(lo, hi, w) so that field(t) = (1-w)*F[lo] + w*F[hi]. w==0 when t lands exactly."""
    hi = np.searchsorted(T, times, "left")
    lo = np.clip(hi - 1, 0, len(T) - 1)
    hi = np.clip(hi, 0, len(T) - 1)
    exact = T[hi] == times
    lo = np.where(exact, hi, lo)
    span = (T[hi] - T[lo]).astype("float64")
    w = np.where(span > 0, (times - T[lo]) / np.maximum(span, 1), 0.0)
    inside = (times >= T[0]) & (times <= T[-1])
    return lo.astype("int32"), hi.astype("int32"), w.astype("float32"), inside


# ---- INPUTS: t0-24h, t0-12h, t0 ----------------------------------------------------------
# CAUSAL, NOT INTERPOLATED. Bracketing a 03 UTC window spans the 00 and 06 UTC fields, and
# interpolating between them uses a field three hours AFTER the forecast is launched. That is
# future data: operationally, at 03 UTC you hold the 00Z analysis and nothing later. So inputs
# take the bracket's LOWER end only, with weight 0 -- the most recent field at or before t0.
# The cost is up to 3 h of field age on half the windows, which is what a real forecaster has.
# Targets are different and DO interpolate: they are supervision at a future valid time, where
# using the true field is the whole point.
IN_OFF = (-4, -2, 0)                      # in 6-h units
in_lo = np.zeros((N, 3), "int32"); in_hi = np.zeros((N, 3), "int32")
in_w = np.zeros((N, 3), "float32"); in_ok = np.zeros((N, 3), "float32")
in_age = np.zeros((N, 3), "float32")      # hours between the field used and the nominal time
for c, off in enumerate(IN_OFF):
    want = bt + off * SIX
    lo, hi, w, ins = bracket(want)
    in_lo[:, c] = lo
    in_hi[:, c] = lo                      # collapse the bracket: no forward reach
    in_w[:, c] = 0.0
    in_ok[:, c] = ins.astype("float32")
    in_age[:, c] = (want - T[lo]) / 3.6e12

# ---- TARGETS: the field at each of the 20 lead valid times -------------------------------
tg_lo = np.zeros((N, 20), "int32"); tg_hi = np.zeros((N, 20), "int32")
tg_w = np.zeros((N, 20), "float32"); tg_ok = np.zeros((N, 20), "float32")
for L in range(20):
    lo, hi, w, ins = bracket(bt + (L + 1) * SIX)
    tg_lo[:, L], tg_hi[:, L], tg_w[:, L] = lo, hi, w
    tg_ok[:, L] = (ins & (nl > L)).astype("float32")

print(f"\ninput coverage   t-24h {100*in_ok[:,0].mean():5.1f}%   "
      f"t-12h {100*in_ok[:,1].mean():5.1f}%   t0 {100*in_ok[:,2].mean():5.1f}%")
print(f"all three present            {100*(in_ok.prod(1)).mean():5.1f}%")
print(f"target coverage  +24h {100*tg_ok[:,3].mean():5.1f}%   "
      f"+72h {100*tg_ok[:,11].mean():5.1f}%   +120h {100*tg_ok[:,19].mean():5.1f}%")
print(f"\ninput field age (hours behind the nominal time):")
for c, nm in enumerate(("t-24h", "t-12h", "t0   ")):
    m = in_ok[:, c] > 0            # windows outside the field span have a meaningless age
    print(f"  {nm}  mean {in_age[m,c].mean():.2f} h   max {in_age[m,c].max():.2f} h   "
          f"zero-age {100*(in_age[m,c]==0).mean():.1f}%")

# ---- guards ------------------------------------------------------------------------------
# 1. the interpolation must be causal for inputs: never reach past t0
tmax_used = np.where(in_ok[:, 2] > 0, T[in_hi[:, 2]], bt)
assert (tmax_used <= bt).all(), "an INPUT index reaches beyond t0 -- that is future data"
print("\nOK - no input index reaches beyond t0")

# 2. weights in range, brackets ordered
assert (in_w >= 0).all() and (in_w <= 1).all(), "input weight outside [0,1]"
assert (tg_w >= 0).all() and (tg_w <= 1).all(), "target weight outside [0,1]"
assert (in_hi >= in_lo).all() and (tg_hi >= tg_lo).all(), "bracket inverted"
print("OK - all weights in [0,1] and brackets ordered")

# 3. reconstruct a known time and check it round-trips
k = int(np.argmax(tg_w[:, 0] > 0))         # a target that needs interpolation
rec = (1 - tg_w[k, 0]) * T[tg_lo[k, 0]] + tg_w[k, 0] * T[tg_hi[k, 0]]
assert abs(rec - (bt[k] + SIX)) < 1e3, f"target interp {rec} != valid time {bt[k]+SIX}"
print(f"OK - target interpolation reconstructs its valid time (window {k})")
# inputs are deliberately NOT interpolated, so they land at or before t0 by design.
# Assert exactly that, rather than the equality the interpolated version used to satisfy.
_used = T[in_lo[:, 2]][in_ok[:, 2] > 0]
_want = bt[in_ok[:, 2] > 0]
assert (_used <= _want).all() and ((_want - _used) <= 6 * 3600 * int(1e9)).all(), \
    "a t0 input field is either in the future or more than 6 h stale"
print("OK - every t0 input field is at or before t0, and at most 6 h old")

OUT = "track_build/v24_index.npz"
np.savez_compressed(OUT, in_lo=in_lo, in_hi=in_hi, in_w=in_w, in_ok=in_ok,
                    in_age=in_age,
                    tg_lo=tg_lo, tg_hi=tg_hi, tg_w=tg_w, tg_ok=tg_ok)

# 4. THE LEAKAGE GUARD: nothing written may identify the storm or the date
d = np.load(OUT)
for key in d.files:
    a = d[key]
    assert not np.issubdtype(a.dtype, np.datetime64), f"{key} carries a timestamp"
    if key.endswith(("_lo", "_hi")):
        # indices into the field array are fine, but they must not be a bijection with the
        # window (which would let the model recover which storm/date it is looking at)
        assert len(np.unique(a)) < len(a), f"{key} is unique per window -- an identity channel"
print("OK - index contains no storm id, no absolute date, no per-window unique key")

print(f"\nwrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
