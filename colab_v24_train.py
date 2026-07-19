"""v20 on Colab — v17 EXACTLY, but reading deep-layer-mean steering instead of 500 hPa.

    !wget -q -O v24t.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v24_train.py
    exec(open('v24t.py').read())

Run the extraction cell (colab_v24_extract.py) first; it writes /content/dlm4_int8.npz. This cell
trains on it.

Nothing about the architecture, loss, augmentation, splits, or hyperparameters changes. The model
class, the four-term track loss, the north-south mirror (which flips vDLM and keeps uDLM exactly as
it did v500/u500, because the deep-layer channels sit in the same slots), the 1980-2015 / 2016-19 /
2020+ split -- all lifted from colab_train_v17.ipynb unchanged. The single difference is that the
two steering channels now carry the 850/500/200 hPa deep-layer mean rather than 500 hPa alone.

So v20 vs v17 isolates one thing: does the deep-layer mean flow steer better than a single level?
The bar is v17's 462.8 km. A gain under ~10 km is inside the seed spread and is not a result.
"""
import os, re, json, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

RAW = "https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main"
N_SEEDS = int(os.environ.get("V24_SEEDS", "3"))

assert os.path.exists("/content/dlm4_int8.npz"), \
    "run colab_v24_extract.py first -- /content/dlm4_int8.npz is missing"
os.makedirs("/content/d", exist_ok=True)
if not os.path.exists("/content/d/track_windows_v13.npz"):
    src = "/content/track_windows_v13.npz"
    if os.path.exists(src):
        import shutil; shutil.copy(src, "/content/d/track_windows_v13.npz")
    else:
        print("fetching v13 windows ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/track_build/track_windows_v13.npz",
                                   "/content/d/track_windows_v13.npz")

# ---- lift the ENTIRE v17 training machinery from the notebook, unchanged but for the data file ----
nb = json.load(open(urllib.request.urlretrieve(f"{RAW}/colab_train_v17.ipynb",
                                               "/content/_v17.ipynb")[0]))
cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
# cells 2..6: config+data+loader (2,3), model (4), loss+train_one (5), eval helpers (6)
body = "\n\n".join(cells[2:7])
n_before = body.count("steer5_int8.npz")
body = body.replace('"/content/d/steer5_int8.npz"', '"/content/dlm4_int8.npz"')
assert n_before == 1 and body.count("steer5_int8.npz") == 0, "data-path swap did not apply cleanly"
# the loop and test-set eval (cells 7,8) are re-authored below, so strip nothing else

G = {"__name__": "v17exec", "torch": torch, "nn": nn, "F": F, "np": np, "os": os, "json": json,
     "time": time, "math": __import__("math")}
exec(compile(body, "<v17-notebook>", "exec"), G)

# sanity: the swap really changed the steering channels, not just the filename
_avail = G["AVAIL"]
print(f"\nsteering source: deep-layer mean | availability {_avail[:,1].mean():.3f}", flush=True)
Net = G["TrackFormerV17"]; train_one = G["train_one"]
metrics = G["metrics"]; load_model = G["load_model"]
te_idx = G["te_idx"]; basins = G["basins"]; z = G["z"]

CK = []
for seed in range(N_SEEDS):
    c = f"/content/v20_seed{seed}.pt"
    train_one(seed, c); CK.append(c)
print(f"\nv20 trained: {len(CK)} seeds", flush=True)

full = z["n_leads"].astype(int) == 20
wpep = np.array([i for i in te_idx if full[i] and basins[i] in ("WP", "EP")])
models = [load_model(c) for c in CK]
print(f"\nWP+EP 2020+, {len(wpep)} windows — the SAME set v17 scored 462.8 on")
print("  BASELINES   v10 549.3 | v17 462.8 (the bar) | v18 466.2 | v19 466.7")
for i, c in enumerate(CK):
    print(f"  v20 seed{i}  {json.dumps(metrics([load_model(c)], wpep))}", flush=True)
ens = metrics(models, wpep)
e = ens["track_error_km"]
print(f"\n  v20 ENSEMBLE ({len(models)} seeds)  {json.dumps(ens)}")
print(f"  track vs v17 462.8: {e - 462.8:+.2f} km   vs v10 549.3: {e - 549.3:+.2f} km", flush=True)
json.dump({"v20": e, "full": ens}, open("/content/v20.json", "w"))

# mirror weights off the VM so an idle disconnect can't take them
try:
    from google.colab import files
    import subprocess
    subprocess.run("tar cf /content/v20_seeds.tar /content/v20_seed*.pt", shell=True)
    files.download("/content/v20.json")
    files.download("/content/v20_seeds.tar")
except Exception as ex:
    print("(auto-download unavailable:", ex, ")", flush=True)
