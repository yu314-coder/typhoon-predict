"""How much is left on the table in flow prediction? Feed v21 the TRUE future steering.

v21's chain-of-thought predicts the steering flow at each lead and derives motion from it:

    flow_pred = flow_now + fd          fd = flow_delta(h_track)
    motion    = base + A*fd*KM6H/100 + track_res(h)

lead_flow.npz holds the flow that ACTUALLY occurred at each lead. Substituting it for the model's
own prediction -- fd := flow_true - flow_now -- turns the chain-of-thought into an oracle and
answers the only question that matters before building v23: if we could predict the steering flow
perfectly, how much error would disappear?

  large drop  -> flow prediction is the bottleneck, and a better flow model is the way forward
  small drop  -> the flow path is not where the error lives, and v23 should attack something else

CAVEAT, stated up front: A, track_res and the decoder were trained against the model's own fd
distribution, not against a perfect one. This is a first-order estimate of the ceiling, not a
trained upper bound -- a model retrained with oracle flow could do better. Read it as "at least
this much is available", and only from the flow term.

Only leads with a real flow target are counted (got == 1); the rest are left at the model's own
prediction, so no window is scored against a fabricated truth.
"""
import json, re, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
nb = json.load(open("colab_train_v17.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
body = body.replace('"/content/d/steer5_int8.npz"', '"track_build/dlm4_int8.npz"')
body = body.replace('"/content/d/track_windows_v13.npz"', '"track_build/track_windows_v13.npz"')
body = body.replace('DEVICE = torch.device("cuda")', 'DEVICE = torch.device("cpu")')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os,
     "json": json, "time": __import__("time"), "math": math}
exec(compile(body, "<v17-notebook>", "exec"), G)

Base = G["TrackFormerV17"]; SLP = G["SLP"]; track = G["track"]; target = G["target"]
vpair = G["vpair"]; te_idx = G["te_idx"]; basins = G["basins"]; z = G["z"]; SC = G["TARGET_SCALE"]
KIN, THE, ENV = G["KIN_COLS"], G["THERMO_COLS"], G["ENV_COLS"]

DSC = np.load("track_build/dlm4_int8.npz")["scale"][2:4].astype("float32")
_i, _j = np.meshgrid(np.arange(17) - 8, np.arange(17) - 8, indexing="ij")
ANN = torch.tensor(((np.hypot(_i, _j) * 2.5 >= 3.0) & (np.hypot(_i, _j) * 2.5 <= 8.0)).astype("float32"))
_lf = np.load("track_build/lead_flow.npz")
FLOW_T = _lf["flow"].astype("float32")      # [N,20,2] true steering, m/s
FLOW_M = _lf["got"].astype("float32")       # [N,20] 1 where a target exists

CLS = r"class TrackFormerCoT\(Base\):.*?torch\.zeros_like\(motion\), ilog\], -1\), flow_pred\)"
g21 = {"Base": Base, "torch": torch, "nn": nn, "F": F, "math": math, "G": G, "ANN": ANN,
       "DSC": DSC, "KM6H": 6 * 3600 / 1000.0, "R_ROUNDS": 0, "USE_FLOW": 1}
exec(re.search(CLS, open("colab_v26_train.py").read(), re.S).group(0), g21)
V21 = g21["TrackFormerCoT"]

KM6H = 6 * 3600 / 1000.0
full = z["n_leads"].astype(int) == 20
TE = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
T = np.cumsum(target[TE][..., :2], 1)


class Oracle(V21):
    """v21 with fd replaced by the true flow anomaly wherever a target exists."""
    def set_truth(self, ft, fm):
        self._ft, self._fm = ft, fm

    def forward(self, tr, vp, slp):
        b = tr.shape[0]
        kin = self.kin_enc(self.kin_proj(tr[:, :, KIN]) + self.kin_time)
        th = self.thermo_enc(self.thermo_proj(tr[:, :, THE]) + self.thermo_time)
        env = self.env_enc(self.env_proj(tr[:, :, ENV]) + self.env_time)
        st = self.steer_cnn(slp).flatten(2).transpose(1, 2) + self.steer_pos
        tq = (self.track_q + self.qpos.unsqueeze(0)).expand(b, -1, -1)
        h = self.track_dec(tq, torch.cat([kin, env, st], 1))
        h = h + self.alpha.view(1, self.leads, 1) * self.adapter(th.mean(1).detach()).unsqueeze(1)
        w = ANN / ANN.sum()
        sc = torch.as_tensor(DSC, dtype=slp.dtype)
        flow_now = (slp[:, 2:4] * w).sum((-2, -1)) * sc
        fd = self.flow_delta(h)                                   # the model's own guess
        fd_true = self._ft - flow_now.unsqueeze(1)                # what it should have been
        m = self._fm.unsqueeze(-1)
        fd = m * fd_true + (1 - m) * fd                           # oracle only where truth exists
        v0, vp2 = vp[:, :2], vp[:, 2:]
        s0 = v0.norm(dim=1, keepdim=True).clamp(min=1e-3)
        phi0 = torch.atan2(v0[:, 1], v0[:, 0])
        dphi = phi0 - torch.atan2(vp2[:, 1], vp2[:, 0])
        om = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        phil = phi0.unsqueeze(1) + self.gturn.view(1, self.leads) * om.unsqueeze(1)
        sp = self.rho.view(1, self.leads) * s0
        base = torch.stack([sp * torch.cos(phil), sp * torch.sin(phil)], -1) / 100.0
        motion = base + self.track_res(h) + (self.A.view(1, 1, 2) * fd) * KM6H / 100.0
        return torch.cat([motion, torch.zeros(b, self.leads, 15)], -1), None


def load(cls, tag, s):
    m = cls().eval()
    m.load_state_dict(torch.load(f"downloads/x/{tag}_seed{s}.pt", map_location="cpu",
                                 weights_only=False)["model"])
    return m


@torch.no_grad()
def run(cls, oracle):
    ms = [load(cls, "v21", s) for s in range(5)]
    P = []
    for i in range(0, len(TE), 128):
        j = TE[i:i + 128]
        a = [torch.from_numpy(track[j]), torch.from_numpy(vpair[j]), torch.from_numpy(SLP[j])]
        if oracle:
            for m in ms:
                m.set_truth(torch.from_numpy(FLOW_T[j]), torch.from_numpy(FLOW_M[j]))
        P.append((torch.stack([m(*a)[0] for m in ms]).mean(0) * SC).float().numpy())
    return np.cumsum(np.concatenate(P)[..., :2], 1)


def err(C, L=19):
    return float(np.sqrt(((C[:, L] - T[:, L]) ** 2).sum(-1)).mean())


print(f"WP+EP 2020+, {len(TE)} windows | flow targets present on "
      f"{100*FLOW_M[TE].mean():.1f}% of lead-points\n")
Cp, Co = run(V21, False), run(Oracle, True)
print(f"{'lead':>6s} {'v21':>9s} {'oracle flow':>12s} {'gain':>8s}")
for L, nm in [(3, "24 h"), (7, "48 h"), (11, "72 h"), (15, "96 h"), (19, "120 h")]:
    a, b = err(Cp, L), err(Co, L)
    print(f"{nm:>6s} {a:9.0f} {b:12.0f} {b-a:+8.0f}")
a = float(np.sqrt(((Cp - T) ** 2).sum(-1)).mean())
b = float(np.sqrt(((Co - T) ** 2).sum(-1)).mean())
print(f"\n  all leads   v21 {a:.2f} km   oracle {b:.2f} km   {b-a:+.2f} km ({100*(b-a)/a:+.1f}%)")
print("\nNHC official for reference: 52 / 84 / 124 / 170 / 214 km")
