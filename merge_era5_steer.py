"""Merge the per-year ERA5 0.25-degree patches (track_build/era5/era5_YYYY.npz) into ONE
COMPACT steering-wind tensor, row-indexed into the same window space as track_windows_v13.npz
(steer5_int8.npz / dlm4_int8.npz's convention), so it plugs into the existing storm-centered-patch
injection pattern without a new consumption path.

SCOPE. extract_era5.py pulled BASIN="WP" only (western Pacific) -- EP was never requested. Every
window outside WP is legitimately unavailable here, not a bug: ok=False, q=0, same "missing means
zero AND flagged" contract steer5/dlm4 already use.

THE Z FIX. Each per-year file quantized z (200/550/850 hPa geopotential) with a symmetric max-abs
scale computed on the RAW absolute value (~122,300 m^2/s^2 at 200 hPa). That fixes the int8 step
size at ~974 m^2/s^2/step -- but the actual within-patch GRADIENT structure that encodes the
ridge/trough pattern (and therefore the geostrophic steering wind) has std ~340-440 m^2/s^2,
SMALLER than one quantization step. The absolute value of z is irrelevant to steering anyway (only
its spatial derivative matters); this merge replaces z with its own per-window, per-level spatial-
mean-removed anomaly before requantizing, exactly as steer5 uses "SLPanom" instead of raw SLP. u
and v need no such fix -- they are already small, signed, physically meaningful values.

CHANNELS (6): [u200, u550, u850, v200, v550, v850] -- z DROPPED entirely (2026-07-26): the fix
above only recovers what's still in the per-year float32 pipeline BEFORE quantization, but the
per-year files on disk were already quantized with the broken scale, so z's gradient structure is
gone at the source and no merge-time transform can bring it back (refetching z was scoped in
refetch_z.py but deprioritized -- u/v already carry the primary steering signal, z would only add
secondary ridge/trough context). Quantization: global per-channel std (over WP windows with
got==1), clipped at 4 sigma, matching merge_steer5.py's convention.

COMPACT STORAGE. 86% of the full window-index space (166,395 / 193,609 rows) is EP basin or
outside 2001-2019/2020-2026 and therefore always zero -- storing it dense would be a 4.9 GB in-RAM
array on Colab, most of it wasted. Output keeps only the M populated rows plus `widx` (their
position in the full N-space) and `N` (the full window count), so a consumer reconstructs
availability with `pos = full(N, -1); pos[widx] = arange(M)` and looks up `pos[j]` per window
-- ~700 MB in RAM instead of 4.9 GB.
"""
import glob, os, time
import numpy as np

OUT = "track_build/era5_steer_int8.npz"
CLIP = 4.0
CHAN_NAMES = ["u200", "u550", "u850", "v200", "v550", "v850"]   # var index 1,2 of the source (z dropped)

t0 = time.time()
zt = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
N = len(zt["year"])
basin = zt["basin"].astype(str)

files = sorted(glob.glob("track_build/era5/era5_*.npz"))
print(f"{len(files)} year files found, target N={N}")

# ---- gather every got==1 window's u/v into one compact float32 array. Only ~28k of 193k rows
# are ever populated (WP basin, 2001-2019 + 2020-2026), so this fits comfortably in RAM. ----
comp_widx, comp_vals = [], []
n_total_rows = 0
for f in files:
    d = np.load(f)
    q, sc, got, widx = d["q"], d["scale"], d["got"], d["widx"]
    n_total_rows += len(widx)
    m = got > 0.5
    if not m.any():
        continue
    uv = q[m, 1:3].astype("float32") * sc[None, 1:3, None, None, None]   # [n,2vars,3lev,65,65]
    uv = uv.reshape(uv.shape[0], 6, 65, 65)
    comp_vals.append(uv.astype("float32"))
    comp_widx.append(widx[m])
    print(f"  {os.path.basename(f):20s} {m.sum():5d}/{len(widx):5d} got", flush=True)

comp_vals = np.concatenate(comp_vals, axis=0)     # [M, 6, 65, 65]
comp_widx = np.concatenate(comp_widx, axis=0)     # [M]
M = len(comp_widx)
assert len(set(comp_widx.tolist())) == M, "duplicate widx across year files -- year ranges overlap"
print(f"\n{M:,} usable windows / {n_total_rows:,} attempted "
      f"({100*M/n_total_rows:.1f}% fetch success) in {(time.time()-t0)/60:.1f} min")

wbasin = basin[comp_widx]
assert (wbasin == "WP").all(), f"non-WP rows leaked in: {set(wbasin[wbasin != 'WP'].tolist())}"

# ---- global per-channel scale (std over all M usable windows), matching merge_steer5.py ----
sc = comp_vals.reshape(M, 6, -1).astype("float64").std(axis=(0, 2)).astype("float32")
sc = np.maximum(sc, 1e-6)
print("\nper-channel std (== 1-sigma scale):")
for c, nm in enumerate(CHAN_NAMES):
    print(f"  ch{c} {nm:6s} std {sc[c]:9.3f}")

q_compact = np.clip(np.round(comp_vals / sc[None, :, None, None] * (127.0 / CLIP)), -127, 127)
q_compact = q_compact.astype("int8")
err = np.abs(q_compact.astype("float32") * (CLIP / 127.0 * sc[None, :, None, None])
             - np.clip(comp_vals, -CLIP * sc[None, :, None, None], CLIP * sc[None, :, None, None])).max()
print(f"max quantization error: {err:.4f} (raw units)")

# ---- save COMPACT (M rows + widx into the N-space), not dense -- see COMPACT STORAGE above ----
_tmp = OUT + ".tmp"
with open(_tmp, "wb") as fh:      # open handle: savez must not silently append .npz to the name
    np.savez_compressed(fh, q=q_compact, widx=comp_widx.astype("int64"), N=np.array(N),
                         scale=sc, chan_names=np.array(CHAN_NAMES), basin=np.array(["WP"]),
                         note=np.array(["u/v only -- z dropped, source quantization destroyed its "
                                        "gradient structure; u/v verified 0% collapsed and carry "
                                        "the primary steering signal directly"]))
os.replace(_tmp, OUT)

sz = os.path.getsize(OUT) / 1e6
print(f"\nwrote {OUT}  ({sz:.0f} MB, compact -- {M:,}/{N:,} rows, ~{q_compact.nbytes/1e6:.0f} MB in RAM)")
print(f"coverage: {M:,}/{N:,} windows ({100*M/N:.1f}%) -- WP basin, 2001-2019 + 2020-2026, "
      f"subject to the {n_total_rows-M:,} per-window fetch misses above")
print(f"total time {(time.time()-t0)/60:.1f} min")
