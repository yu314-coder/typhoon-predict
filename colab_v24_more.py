"""v20 seeds 3-4, then the 5-seed number that is directly comparable to v17's 462.8.

    !wget -q -O v24m.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v24_more.py
    exec(open('v24m.py').read())

The first v20 run used 3 seeds and scored 453.63 against v17's FIVE-seed 462.8. A 3-seed ensemble
is normally the weaker configuration, so that comparison understated v20 if anything -- but it was
not like-for-like, and the whole point of this project is to stop accepting numbers that are not.
This trains seeds 3 and 4 and reports 1..5-seed ensembles so the k=5 row can be set beside v17's
k=5 row with nothing left to argue about.

Reuses /content/dlm4_int8.npz and any v20_seed*.pt already on the VM; re-extracts only if the
steering tensor is missing. Also re-downloads dlm4_int8.npz at the end so the paired-storm
bootstrap against v17 can be run locally, where the v17 checkpoints already live.
"""
import os, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
WANT = [0, 1, 2, 3, 4]

if not os.path.exists("/content/dlm4_int8.npz"):
    print("steering tensor missing -- re-extracting (~10 min)", flush=True)
    urllib.request.urlretrieve(f"{RAW}/colab_v24_extract.py", "/content/v24e.py")
    exec(open("/content/v24e.py").read())

os.makedirs("/content/d", exist_ok=True)
if not os.path.exists("/content/d/track_windows_v13.npz"):
    src = "/content/track_windows_v13.npz"
    if os.path.exists(src):
        import shutil; shutil.copy(src, "/content/d/track_windows_v13.npz")
    else:
        urllib.request.urlretrieve(f"{RAW}/track_build/track_windows_v13.npz",
                                   "/content/d/track_windows_v13.npz")

# ---- same v17 machinery, same one-line data swap as the first v20 run ----
nb = json.load(open(urllib.request.urlretrieve(f"{RAW}/colab_train_v17.ipynb",
                                               "/content/_v17.ipynb")[0]))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
body = "\n\n".join(cells[2:7])
assert body.count("steer5_int8.npz") == 1
body = body.replace('"/content/d/steer5_int8.npz"', '"/content/dlm4_int8.npz"')
G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os, "json": json,
     "time": time, "math": __import__("math")}
exec(compile(body, "<v17-notebook>", "exec"), G)
print(f"\nsteering source: deep-layer mean | availability {G['AVAIL'][:,1].mean():.3f}", flush=True)

train_one = G["train_one"]; metrics = G["metrics"]; load_model = G["load_model"]
te_idx = G["te_idx"]; basins = G["basins"]; z = G["z"]

CK = []
for seed in WANT:
    c = f"/content/v20_seed{seed}.pt"
    if os.path.exists(c):
        print(f"seed {seed}: checkpoint already on the VM, reusing", flush=True)
    else:
        train_one(seed, c)
    CK.append(c)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
print(f"\nWP+EP 2020+, {len(wpep)} windows\n")

per = {}
for i, c in enumerate(CK):
    per[i] = metrics([load_model(c)], wpep)["track_error_km"]
    print(f"  v20 seed{i} alone   {per[i]:.2f} km", flush=True)

# ---- ensemble size sweep: the k=5 row is the one that matches v17's published number ----
print(f"\n{'k':>2s} {'v20 ensemble':>13s}   (v17 k=5 = 462.8)")
sweep = {}
for k in range(1, len(CK) + 1):
    ms = [load_model(c) for c in CK[:k]]
    sweep[k] = metrics(ms, wpep)["track_error_km"]
    tag = "  <- comparable to v17" if k == 5 else ""
    print(f"{k:2d} {sweep[k]:13.2f}{tag}", flush=True)

e5 = sweep[len(CK)]
print(f"\nv20 ({len(CK)} seeds)  {e5:.2f} km")
print(f"  vs v17 462.8: {e5 - 462.8:+.2f} km    vs v10 549.3: {e5 - 549.3:+.2f} km", flush=True)
full_metrics = metrics([load_model(c) for c in CK], wpep)
print(f"  full: {json.dumps(full_metrics)}", flush=True)
json.dump({"v20_5seed": e5, "per_seed": per, "sweep": sweep, "full": full_metrics},
          open("/content/v20_5seed.json", "w"))

try:
    from google.colab import files
    import subprocess
    subprocess.run("tar cf /content/v20_all_seeds.tar /content/v20_seed*.pt", shell=True)
    files.download("/content/v20_5seed.json")
    files.download("/content/v20_all_seeds.tar")
    files.download("/content/dlm4_int8.npz")     # needed for the local paired bootstrap vs v17
except Exception as ex:
    print("(auto-download unavailable:", ex, ")", flush=True)
