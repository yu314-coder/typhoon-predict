"""Did the chain-of-thought actually work? Four tests, because the km number alone cannot say.

A track improvement is not evidence that CoT worked. v21 changes two things at once and could
improve for reasons that have nothing to do with reasoning through an intermediate:

  T1  IS THE PATH USED?      |dW| of the zero-initialised flow head, and the share of predicted
                             motion that comes from the flow term rather than persistence or the
                             residual. If the flow contributes ~0, v21 has collapsed to v20.
  T2  IS THE INTERMEDIATE RIGHT?  correlation of predicted flow against the flow the storm actually
                             experienced, ON THE TEST SET. If the model cannot predict flow, then
                             any track gain came from somewhere else and calling it CoT is wrong.
  T3  DOES IT DEGRADE WITH LEAD?  T2 broken out by lead. Steering explains motion equally well at
                             all leads (measured: corr 0.86 -> 0.87), so if predicted-flow skill
                             collapses at long lead that is a failure of forecasting, not of the
                             decomposition -- and it localises where CoT breaks.
  T4  IS IT CoT OR JUST MULTI-TASK?  needs the USE_FLOW=0 control arm; reported here if present.

Run after v21 weights are downloaded to downloads/x/.
"""
import json, re, math, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8); DEVICE = torch.device("cpu")
KM6H = 6 * 3600 / 1000.0

nb = json.load(open("colab_train_v17.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
G = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
     "DEVICE": DEVICE, "STEER_DROP": 0.0, "STEER_CLIP": 4.0}
for p in [r"KIN_COLS = .*?KIN_DIM, THERMO_DIM, ENV_DIM = len\(KIN_COLS\), len\(THERMO_COLS\), len\(ENV_COLS\)",
          r"def sinusoidal.*?\n    return e", r"def enc\(.*?depth\)\n", r"def dec\(d.*?depth\)\n",
          r"class TrackFormerV17.*?torch\.zeros_like\(motion\), ilog\], -1\)"]:
    exec(re.search(p, src, re.S).group(0), G)
Base = G["TrackFormerV17"]
KIN_COLS, THERMO_COLS, ENV_COLS = G["KIN_COLS"], G["THERMO_COLS"], G["ENV_COLS"]

z = np.load("track_build/track_windows_v13.npz", allow_pickle=True)
track = z["track"].astype("float32"); target = z["target"].astype("float32")
sids = z["storm_id"].astype(str); years = z["year"].astype(int)
basins = z["basin"].astype(str); nl = z["n_leads"].astype(int)
tmean = z["track_mean"].astype("float32"); tstd = z["track_std"].astype("float32")
vpair = np.concatenate([track[:, -1, 2:4] * tstd[2:4] + tmean[2:4],
                        track[:, -2, 2:4] * tstd[2:4] + tmean[2:4]], 1).astype("float32")
SLP = np.clip(np.load("track_build/dlm4_int8.npz")["q"][:, :4].astype("float32") / 31.75, -4, 4)
DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_lf = np.load("track_build/lead_flow.npz")
FLOW_T = _lf["flow"].astype("float32"); FLOW_M = _lf["got"]

_ii, _jj = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_ii, _jj) * 2.5 >= 3.0) &
                    (np.hypot(_ii, _jj) * 2.5 <= 8.0)).astype("float32"))

fy = {s: int(years[sids == s].min()) for s in np.unique(sids)}
EV = np.array([i for i in range(len(sids))
               if fy[sids[i]] >= 2020 and nl[i] == 20 and basins[i] in ("WP", "EP")])
print(f"test set: {len(EV):,} windows\n")


class CoT(Base):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.flow_delta = nn.Linear(self.track_q.shape[-1], 2)
        self.A = nn.Parameter(torch.tensor([0.76, 0.91]))

    def parts(self, tr, vp, slp):
        """Return the three additive pieces of motion, plus the predicted flow."""
        b = tr.shape[0]
        kin = self.kin_enc(self.kin_proj(tr[:, :, KIN_COLS]) + self.kin_time)
        thermo = self.thermo_enc(self.thermo_proj(tr[:, :, THERMO_COLS]) + self.thermo_time)
        env = self.env_enc(self.env_proj(tr[:, :, ENV_COLS]) + self.env_time)
        st = self.steer_cnn(slp).flatten(2).transpose(1, 2) + self.steer_pos
        tq = (self.track_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h = self.track_dec(tq, torch.cat([kin, env, st], 1))
        h = h + self.alpha.view(1, self.leads, 1) * self.adapter(thermo.mean(1).detach()).unsqueeze(1)
        w = ANN / ANN.sum()
        flow_now = (slp[:, 2:4] * w).sum((-2, -1)) * torch.as_tensor(DSC)
        fd = self.flow_delta(h)
        v0, vp2 = vp[:, :2], vp[:, 2:]
        s0 = v0.norm(dim=1, keepdim=True).clamp(min=1e-3)
        phi0 = torch.atan2(v0[:, 1], v0[:, 0])
        dphi = phi0 - torch.atan2(vp2[:, 1], vp2[:, 0])
        om = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        phil = phi0.unsqueeze(1) + self.gturn.view(1, self.leads) * om.unsqueeze(1)
        sp = self.rho.view(1, self.leads) * s0
        base = torch.stack([sp * torch.cos(phil), sp * torch.sin(phil)], -1) / 100.0
        flow_term = (self.A.view(1, 1, 2) * fd) * KM6H / 100.0
        res = self.track_res(h)
        return base, flow_term, res, flow_now.unsqueeze(1) + fd


def load(ck):
    m = CoT().eval()
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
    return m


cks = sorted(glob.glob("downloads/x/v21_seed*.pt"))
if not cks:
    print("no v21 checkpoints in downloads/x/ yet"); raise SystemExit
MS = [load(c) for c in cks]
print(f"loaded {len(MS)} v21 seeds\n")

print("--- T1: is the flow path used? ---")
for i, m in enumerate(MS):
    print(f"  seed{i}  |dW| {float(m.flow_delta.weight.abs().mean()):.5f}   "
          f"A [{m.A[0].item():+.3f}, {m.A[1].item():+.3f}]   (init 0.760, 0.910)")

B, Fl, R, FP = [], [], [], []
with torch.no_grad():
    for i in range(0, len(EV), 128):
        j = EV[i:i + 128]
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
        ps = [m.parts(*a) for m in MS]
        B.append(torch.stack([p[0] for p in ps]).mean(0).numpy())
        Fl.append(torch.stack([p[1] for p in ps]).mean(0).numpy())
        R.append(torch.stack([p[2] for p in ps]).mean(0).numpy())
        FP.append(torch.stack([p[3] for p in ps]).mean(0).numpy())
B, Fl, R, FP = map(np.concatenate, (B, Fl, R, FP))
nb_, nf, nr = [np.abs(x).mean() * 100 for x in (B, Fl, R)]      # back to km
print(f"\n  mean |contribution| to per-step motion, km:")
print(f"    persistence base {nb_:7.2f}")
print(f"    FLOW term        {nf:7.2f}   <- the chain-of-thought path")
print(f"    residual         {nr:7.2f}")
share = 100 * nf / (nb_ + nf + nr)
print(f"    flow share of total motion: {share:.1f}%")
print("    (near 0% => v21 collapsed to v20 and CoT bought nothing)")

print("\n--- T2/T3: is the predicted flow right, on the TEST set? ---")
ft = FLOW_T[EV]; fm = FLOW_M[EV]
print(f"{'lead':>5s} {'n':>8s} {'corr E':>7s} {'corr N':>7s} {'RMSE E':>7s} {'RMSE N':>7s}")
for L in [0, 3, 7, 11, 15, 19]:
    k = fm[:, L]
    if k.sum() < 50:
        continue
    ce = np.corrcoef(FP[k, L, 0], ft[k, L, 0])[0, 1]
    cn = np.corrcoef(FP[k, L, 1], ft[k, L, 1])[0, 1]
    re = np.sqrt(((FP[k, L, 0] - ft[k, L, 0]) ** 2).mean())
    rn = np.sqrt(((FP[k, L, 1] - ft[k, L, 1]) ** 2).mean())
    print(f"{6*(L+1):4d}h {int(k.sum()):8,d} {ce:+7.3f} {cn:+7.3f} {re:7.2f} {rn:7.2f}")
k = fm.ravel()
ce = np.corrcoef(FP[..., 0].ravel()[k], ft[..., 0].ravel()[k])[0, 1]
cn = np.corrcoef(FP[..., 1].ravel()[k], ft[..., 1].ravel()[k])[0, 1]
print(f"  pooled: corr E {ce:+.3f}  N {cn:+.3f}")
print("  a PERSISTENCE-OF-FLOW baseline (predict the present flow at every lead) is the bar:")
w = (ANN / ANN.sum()).numpy()
fnow = (SLP[EV][:, 2:4] * w).sum((-2, -1)) * DSC
pe = np.corrcoef(np.repeat(fnow[:, 0:1], 20, 1).ravel()[k], ft[..., 0].ravel()[k])[0, 1]
pn = np.corrcoef(np.repeat(fnow[:, 1:2], 20, 1).ravel()[k], ft[..., 1].ravel()[k])[0, 1]
print(f"    persistence-of-flow: corr E {pe:+.3f}  N {pn:+.3f}")
print(f"    v21 beats persistence: E {ce>pe}, N {cn>pn}")

print("\n--- T4: CoT vs plain multi-task ---")
aux = sorted(glob.glob("downloads/x/v21aux_seed*.pt"))
print(f"  control arm (USE_FLOW=0) checkpoints found: {len(aux)}")
if not aux:
    print("  run with V26_USE_FLOW=0 to separate the CoT structure from the auxiliary loss.")
json.dump({"flow_share_pct": float(share), "corr_e": float(ce), "corr_n": float(cn),
           "persist_e": float(pe), "persist_n": float(pn)},
          open("track_build/cot_diag.json", "w"))
