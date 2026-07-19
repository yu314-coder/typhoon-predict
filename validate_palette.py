"""Pairwise colour separation under normal vision and the three dichromacies.

Viénot-Brettel-Mollon LMS projection for protan/deutan/tritan, then CIEDE2000. Reports the worst
pair so a palette can be checked rather than asserted. Panels here are never overlaid, so this is
a legibility check on the legend and captions, not a hard constraint on the lines.

    python3 validate_palette.py "#eda100,#e34948,#2a78d6" [light|dark]
"""
import sys, math

def _srgb_lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

RGB2LMS = ((0.31399022, 0.63951294, 0.04649755),
           (0.15537241, 0.75789446, 0.08670142),
           (0.01775239, 0.10944209, 0.87256922))
LMS2RGB = ((5.47221206, -4.6419601, 0.16963708),
           (-1.1252419, 2.29317094, -0.1678952),
           (0.02980165, -0.19318073, 1.16364789))
# Viénot et al. 1999 dichromat projections in LMS
SIM = {"protan": ((0, 1.05118294, -0.05116099), (0, 1, 0), (0, 0, 1)),
       "deutan": ((1, 0, 0), (0.9513092, 0, 0.04866992), (0, 0, 1)),
       "tritan": ((1, 0, 0), (0, 1, 0), (-0.86744736, 1.86727089, 0))}

def _mul(M, v):
    return tuple(sum(M[i][j] * v[j] for j in range(3)) for i in range(3))

def simulate(rgb, kind):
    lin = [_srgb_lin(c) for c in rgb]
    lms = _mul(RGB2LMS, lin)
    out = _mul(LMS2RGB, _mul(SIM[kind], lms))
    def enc(c):
        c = max(0.0, min(1.0, c))
        return 255 * (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055)
    return tuple(enc(c) for c in out)

def lab(rgb):
    lin = [_srgb_lin(c) for c in rgb]
    x = 0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    z = 0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]
    def f(t):
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116
    fx, fy, fz = f(x / 0.95047), f(y), f(z / 1.08883)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)

def de2000(c1, c2):
    L1, a1, b1 = lab(c1); L2, a2, b2 = lab(c2)
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7))) if Cb > 0 else 0
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1 = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0
    h2 = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0
    dLp, dCp = L2 - L1, C2p - C1p
    dh = 0 if C1p * C2p == 0 else ((h2 - h1 + 180) % 360) - 180
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dh) / 2)
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hbp = (h1 + h2) / 2 if C1p * C2p != 0 else h1 + h2
    if C1p * C2p != 0 and abs(h1 - h2) > 180:
        hbp += 180
    T = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / math.sqrt(20 + (Lbp - 50) ** 2)
    SC, SH = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    RT = (-2 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
          * math.sin(math.radians(60 * math.exp(-(((hbp - 275) / 25) ** 2)))))
    return math.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                     + RT * (dCp / SC) * (dHp / SH))


if __name__ == "__main__":
    cols = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    rgbs = [hex_rgb(c) for c in cols]
    print(f"{len(cols)} colours, {len(cols)*(len(cols)-1)//2} pairs\n")
    worst_all = (1e9, None, None)
    for kind in ("normal", "protan", "deutan", "tritan"):
        sim = rgbs if kind == "normal" else [simulate(r, kind) for r in rgbs]
        worst, wp = 1e9, None
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                d = de2000(sim[i], sim[j])
                if d < worst:
                    worst, wp = d, (cols[i], cols[j])
        flag = "OK" if worst >= 15 else ("WEAK" if worst >= 8 else "FAIL")
        print(f"  {kind:7s} worst dE {worst:6.1f}  {wp[0]} vs {wp[1]}   {flag}")
        if worst < worst_all[0]:
            worst_all = (worst, kind, wp)
    print(f"\noverall worst: dE {worst_all[0]:.1f} under {worst_all[1]} "
          f"({worst_all[2][0]} vs {worst_all[2][1]})")
