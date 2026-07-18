"""Export v16 forecast tracks for the four test storms so the HTML map can be built locally.

Run in the finished colab_train_v16 session AFTER the ablation (which restores the SST channel):
    !wget -q -O exp.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v16_export.py
    exec(open('exp.py').read())
Produces a ~100 KB JSON and hands it to the browser.
"""
import json, numpy as np, torch

assert SLP.shape[1] == 5 and float(np.abs(SLP[:, 4]).mean()) > 1e-6, \
    "SST channel is zeroed — run the ablation to completion first (it restores it)."
R = 111.2
sid_all = z["storm_id"].astype(str); nl_all = z["n_leads"].astype(int)
bt_all = z["base_time"].astype("int64")
bla_all = z["base_lat"].astype("float64"); blo_all = z["base_lon"].astype("float64")
STORMS = [("2026182N09163", "Bavi"), ("1986228N19120", "Wayne"),
          ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]

@torch.no_grad()
def predict(k):
    P = []
    for i in range(0, len(k), 128):
        j = k[i:i + 128]
        s = torch.stack([mm(torch.from_numpy(track[j]).to(DEVICE),
                            torch.from_numpy(vpair[j]).to(DEVICE),
                            torch.from_numpy(SLP[j]).to(DEVICE))[0] for mm in models]).mean(0)
        P.append((s * TARGET_SCALE).float().cpu().numpy())
    return np.concatenate(P)

out = {}
for s, nm in STORMS:
    k = np.where((sid_all == s) & (nl_all == 20))[0]
    k = k[np.argsort(bt_all[k])]
    P = predict(k)
    cE, cN = np.cumsum(P[..., 0], 1), np.cumsum(P[..., 1], 1)
    tE, tN = np.cumsum(target[k][..., 0], 1), np.cumsum(target[k][..., 1], 1)
    lats, lons = [], []
    for a in range(len(k)):
        la = bla_all[k[a]] + cN[a] / R
        lo = blo_all[k[a]] + cE[a] / (R * np.cos(np.radians((bla_all[k[a]] + la) / 2)))
        lats.append(np.round(la, 3).tolist()); lons.append(np.round(lo, 3).tolist())
    err = float(np.hypot(cE[:, 19] - tE[:, 19], cN[:, 19] - tN[:, 19]).mean())
    out[nm] = {"widx": k.tolist(), "lat": lats, "lons_placeholder": None, "lon": lons,
               "base_time": bt_all[k].tolist(),
               "base_lat": np.round(bla_all[k], 3).tolist(),
               "base_lon": np.round(blo_all[k], 3).tolist(),
               "err120_mean": err, "n": int(len(k))}
    del out[nm]["lons_placeholder"]
    print(f"{nm:11s} {len(k):3d} windows | v16 mean 120h err {err:6.0f} km", flush=True)

json.dump(out, open("/content/v16_tracks.json", "w"))
import os; print(f"\n/content/v16_tracks.json  {os.path.getsize('/content/v16_tracks.json')/1000:.0f} KB")
try:
    from google.colab import files; files.download("/content/v16_tracks.json")
    print("handed to browser download")
except Exception as e:
    print("download failed:", e)
