"""v21 (chain-of-thought) vs v20, paired-storm bootstrap.

Same harness that judged v18 (+3.78 km, CI [-2.56, +9.75], not significant) and v20 (-10.35 km,
CI [-22.54, +0.82], not significant). Whole STORMS are resampled, not windows, because windows from
one storm are strongly correlated and resampling them would manufacture significance.

Both models read the deep-layer steering tensor; the only difference is that v21 derives part of its
motion from a predicted steering flow.
"""

import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8)
DEVICE = torch.device("cpu")
RNG = np.random.default_rng(0)
NBOOT = int(os.environ.get("NBOOT", "2000"))

nb = json.load(open("colab_train_v17.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
     "DEVICE": DEVICE, "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p, src, re.S).group(0), G)


KM6H = 6 * 3600 / 1000.0
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
_ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))


class CoT(G["TrackFormerV17"]):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.flow_delta = nn.Linear(self.track_q.shape[-1], 2)
        self.A = nn.Parameter(torch.tensor([0.76, 0.91]))

    def forward(self, track, vpair, slp):
        b = track.shape[0]
        KC, TC, EC = G["KIN_COLS"], G["THERMO_COLS"], G["ENV_COLS"]
        kin = self.kin_enc(self.kin_proj(track[:, :, KC]) + self.kin_time)
        th = self.thermo_enc(self.thermo_proj(track[:, :, TC]) + self.thermo_time)
        env = self.env_enc(self.env_proj(track[:, :, EC]) + self.env_time)
        st = self.steer_cnn(slp).flatten(2).transpose(1, 2) + self.steer_pos
        tq = (self.track_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h = self.track_dec(tq, torch.cat([kin, env, st], 1))
        h = h + self.alpha.view(1, self.leads, 1) * self.adapter(th.mean(1).detach()).unsqueeze(1)
        fd = self.flow_delta(h)
        v0, vp = vpair[:, :2], vpair[:, 2:]
        s0 = v0.norm(dim=1, keepdim=True).clamp(min=1e-3)
        phi0 = torch.atan2(v0[:, 1], v0[:, 0]); dphi = phi0 - torch.atan2(vp[:, 1], vp[:, 0])
        om = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        phil = phi0.unsqueeze(1) + self.gturn.view(1, self.leads) * om.unsqueeze(1)
        sp = self.rho.view(1, self.leads) * s0
        base = torch.stack([sp * torch.cos(phil), sp * torch.sin(phil)], -1) / 100.0
        motion = base + (self.A.view(1, 1, 2) * fd) * KM6H / 100.0 + self.track_res(h)
        iq = (self.int_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        hi = self.int_dec(iq, torch.cat([th, env, kin.detach(), st.detach()], 1))
        return torch.cat([motion, self.int_state(hi)], -1), None


def load(ck, cls=None):
    cls = cls or G["TrackFormerV17"]
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("inner.")} or sd
    m = cls().to(DEVICE).eval(); m.load_state_dict(sd); return m


z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
sids = z["storm_id"].astype(str); years = z["year"].astype(int)
basins = z["basin"].astype(str); nl = z["n_leads"].astype(int)
tmean = z["track_mean"].astype("float32"); tstd = z["track_std"].astype("float32")
vpair = np.concatenate([track[:, -1, 2:4] * tstd[2:4] + tmean[2:4],
                        track[:, -2, 2:4] * tstd[2:4] + tmean[2:4]], 1).astype("float32")


def steer(path):
    q = np.load(path)["q"][:, :4].astype("float32")
    return np.clip(q / 31.75, -4.0, 4.0)


S17 = steer("track_build/steer5_int8.npz")     # 500 hPa
S20 = steer("track_build/dlm4_int8.npz")       # deep-layer mean
assert S17.shape == S20.shape == (len(track), 4, 17, 17)

fy = {s: int(years[sids == s].min()) for s in np.unique(sids)}
EV = np.array([i for i in range(len(sids))
               if fy[sids[i]] >= 2020 and nl[i] == 20 and basins[i] in ("WP", "EP")])
T = np.cumsum(target[EV][..., :2], 1)
SC = torch.tensor([100., 100., 35., 20., 50.] + [50.] * 12)
ev_storm = sids[EV]
storms = np.unique(ev_storm)
print(f"test: {len(EV)} windows over {len(storms)} storms\n")


@torch.no_grad()
def per_window_err(ckpts, S, cls=None):
    P = []
    ms = [load(c, cls) for c in ckpts]
    for i in range(0, len(EV), 128):
        j = EV[i:i + 128]
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(S[j])]
        s = torch.stack([m(*a)[0] for m in ms]).mean(0)
        P.append((s * SC).numpy())
    C = np.cumsum(np.concatenate(P)[..., :2], 1)
    return np.sqrt(((C - T) ** 2).sum(-1)).mean(1)      # mean over the 20 leads, per window


e17 = per_window_err([f"downloads/x/v20_seed{i}.pt" for i in range(5)], S20)
e20 = per_window_err([f"downloads/x/v21_seed{i}.pt" for i in range(5)], S20, CoT)
print(f"v20 (5 seeds)  {e17.mean():.2f} km      [published 452.47]")
print(f"v21 (5 seeds)  {e20.mean():.2f} km      [Colab said 443.62]")
print(f"difference     {e20.mean() - e17.mean():+.2f} km\n")

# per-storm means, paired
m17 = np.array([e17[ev_storm == s].mean() for s in storms])
m20 = np.array([e20[ev_storm == s].mean() for s in storms])
n_w = np.array([(ev_storm == s).sum() for s in storms])
obs = float(np.average(m20, weights=n_w) - np.average(m17, weights=n_w))

diffs = np.empty(NBOOT)
for b in range(NBOOT):
    k = RNG.integers(0, len(storms), len(storms))     # resample STORMS, not windows
    diffs[b] = np.average(m20[k], weights=n_w[k]) - np.average(m17[k], weights=n_w[k])
lo, hi = np.percentile(diffs, [2.5, 97.5])
p_worse = float((diffs >= 0).mean())

print(f"paired storm bootstrap, {NBOOT} resamples over {len(storms)} storms")
print(f"  v21 - v20 = {obs:+.2f} km   95% CI [{lo:+.2f}, {hi:+.2f}]")
print(f"  fraction of resamples where v20 is NOT better: {p_worse:.3f}")
print()
if hi < 0:
    print("  SIGNIFICANT -- the whole CI is below zero: v21 is a real gain over v20.")
elif lo > 0:
    print("  SIGNIFICANT THE WRONG WAY -- v21 is genuinely worse.")
else:
    print("  NOT SIGNIFICANT -- the CI straddles zero. The gap is not statistically established.")
print()
n_better = int((m20 < m17).sum())
print(f"storms where v21 beats v20: {n_better}/{len(storms)}  ({100*n_better/len(storms):.0f}%)")
np.savez("track_build/per_storm_v21_v20.npz", storms=storms, m17=m17, m20=m20, n_w=n_w,
         e17=e17, e20=e20, ev_storm=ev_storm)
# where does the mean gain actually come from?
d = m20 - m17
order = np.argsort(d)
print("\nlargest v20 WINS (km better):")
for i in order[:6]:
    print(f"  {storms[i]}  {d[i]:+8.1f}   v17 {m17[i]:6.0f} -> v20 {m20[i]:6.0f}  ({n_w[i]} windows)")
print("largest v20 LOSSES (km worse):")
for i in order[::-1][:6]:
    print(f"  {storms[i]}  {d[i]:+8.1f}   v17 {m17[i]:6.0f} -> v20 {m20[i]:6.0f}  ({n_w[i]} windows)")
print(f"\nper-storm difference: median {np.median(d):+.1f} km, mean {d.mean():+.1f} km")
print(f"  a median near zero with a negative mean = a few big wins carrying it")
big = np.abs(d) > 100
print(f"  storms with |diff| > 100 km: {big.sum()}/{len(d)}; they contribute "
      f"{np.average(d[big], weights=n_w[big])*n_w[big].sum()/n_w.sum():+.2f} km of the {obs:+.2f} total")
json.dump({"v17": float(e17.mean()), "v20": float(e20.mean()), "diff": obs,
           "ci": [float(lo), float(hi)], "p_not_better": p_worse,
           "storms_better": n_better, "n_storms": int(len(storms))},
          open("track_build/bootstrap_v21_v20.json", "w"))
