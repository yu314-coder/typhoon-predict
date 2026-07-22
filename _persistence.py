"""How much does the model actually add over persistence, per lead?

The oracle showed perfect steering flow is worth -1 km at 24 h and -323 km at 120 h. So at short
lead the error is NOT steering-limited -- yet we sit at 127 km against NHC's 52 km. Something
else is costing us 75 km at 24 h, and no amount of flow work will touch it.

The first suspect is that at short lead the model is barely beating persistence. Compare:

  CV persistence   constant velocity from the last observed motion vector
  v21              the trained model

Both scored on the same windows as everything else (WP+EP 2020+, full-horizon, 3763 windows),
using the cumulative displacements already saved by _errdecomp.py.

v0 is the last motion vector in km per 6 h -- the same quantity the model's curved-persistence
baseline scales by rho -- so persistence displacement at lead L is simply v0 * (L+1).
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
vpair = G["vpair"]

d = np.load("track_build/errdecomp.npz")
T, wpep = d["T"], d["wpep"]                      # truth cumulative displacement, km
v0 = vpair[wpep][:, :2].astype("float64")        # last observed motion, km per 6 h

L = np.arange(1, 21)[None, :, None]
CV = v0[:, None, :] * L                          # constant-velocity persistence


def err(C, i):
    return float(np.sqrt(((C[:, i] - T[:, i]) ** 2).sum(-1)).mean())


NHC = {3: 52, 7: 84, 11: 124, 15: 170, 19: 214}
NAME = {3: "24 h", 7: "48 h", 11: "72 h", 15: "96 h", 19: "120 h"}

print(f"WP+EP 2020+, {len(wpep)} windows\n")
print(f"{'lead':>6s} {'persistence':>12s} {'v21':>8s} {'model gain':>11s} "
      f"{'NHC':>6s} {'v21/NHC':>8s}")
for i in (3, 7, 11, 15, 19):
    p, m = err(CV, i), err(d["v21"], i)
    print(f"{NAME[i]:>6s} {p:12.0f} {m:8.0f} {m-p:+11.0f} {NHC[i]:6d} {m/NHC[i]:8.2f}x")

print("\nshare of the persistence error the model removes")
for i in (3, 7, 11, 15, 19):
    p, m = err(CV, i), err(d["v21"], i)
    print(f"  {NAME[i]:>6s}  {100*(p-m)/p:5.1f}%")

# How much of the 24 h gap to NHC could steering possibly explain? The oracle answers it.
print("\nat 24 h the oracle (perfect steering flow) scored 126 km vs v21's 127 km,")
print("so the 24 h deficit to NHC is NOT a steering-information problem.")
