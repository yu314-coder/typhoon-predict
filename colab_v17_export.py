"""Export v17 forecast tracks + per-lead quantities for the four test storms, so the HTML map
and the metric charts can be built locally.

    !wget -q -O e17.py https://raw.githubusercontent.com/yu314-coder/typhoon-predict/main/colab_v17_export.py
    exec(open('e17.py').read())
"""
import json, os, numpy as np, torch

R = 111.2
CK = [f"{DATA}/v17_seed{i}.pt" for i in range(5)]
CK = [c for c in CK if os.path.exists(c)]
assert CK, "no v17 checkpoints in Drive"
M17 = []
for c in CK:
    m = TrackFormerV17().to(DEVICE).eval()
    m.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=False)["model"])
    M17.append(m)
print(f"v17: {len(M17)} seeds", flush=True)

sid_all = z["storm_id"].astype(str); nl_all = z["n_leads"].astype(int)
bt_all = z["base_time"].astype("int64")
bla = z["base_lat"].astype("float64"); blo = z["base_lon"].astype("float64")
STORMS = [("2026182N09163", "Bavi"), ("1986228N19120", "Wayne"),
          ("2025203N20124", "Co-may"), ("2022239N22150", "Hinnamnor")]

@torch.no_grad()
def predict(k):
    P = []
    for i in range(0, len(k), 128):
        j = k[i:i + 128]
        s = torch.stack([mm(torch.from_numpy(track[j]).to(DEVICE),
                            torch.from_numpy(vpair[j]).to(DEVICE),
                            torch.from_numpy(SLP[j]).to(DEVICE))[0] for mm in M17]).mean(0)
        P.append((s * TARGET_SCALE).float().cpu().numpy())
    return np.concatenate(P)

def sp_bear(E, N):
    return np.hypot(E, N) / 6.0, (np.degrees(np.arctan2(E, N)) + 360) % 360

tracks_out, series_out = {}, {}
for s, nm in STORMS:
    k = np.where((sid_all == s) & (nl_all == 20))[0]; k = k[np.argsort(bt_all[k])]
    P = predict(k)
    cE, cN = np.cumsum(P[..., 0], 1), np.cumsum(P[..., 1], 1)
    T = target[k]; tE, tN = np.cumsum(T[..., 0], 1), np.cumsum(T[..., 1], 1)
    lats, lons = [], []
    for a in range(len(k)):
        la = bla[k[a]] + cN[a] / R
        lo = blo[k[a]] + cE[a] / (R * np.cos(np.radians((bla[k[a]] + la) / 2)))
        lats.append(np.round(la, 3).tolist()); lons.append(np.round(lo, 3).tolist())
    err = float(np.hypot(cE[:, 19] - tE[:, 19], cN[:, 19] - tN[:, 19]).mean())
    tracks_out[nm] = {"lat": lats, "lon": lons, "base_time": bt_all[k].tolist(),
                      "base_lat": np.round(bla[k], 3).tolist(),
                      "base_lon": np.round(blo[k], 3).tolist(),
                      "err120_mean": err, "n": int(len(k))}
    # earliest full-horizon window, for the per-storm quantity charts
    a = 0
    psp, pbr = sp_bear(P[a, :, 0], P[a, :, 1])
    series_out[nm] = {"lat": lats[a], "lon": lons[a],
                      "vmax": P[a, :, 2].tolist(), "pressure": P[a, :, 3].tolist(),
                      "rmw": P[a, :, 4].tolist(), "speed": psp.tolist(), "bearing": pbr.tolist(),
                      "err120": float(np.hypot(cE[a, 19] - tE[a, 19], cN[a, 19] - tN[a, 19]))}
    print(f"{nm:11s} {len(k):3d} windows | v17 mean 120h {err:6.0f} km", flush=True)

json.dump(tracks_out, open("/content/v17_tracks.json", "w"))
json.dump(series_out, open("/content/v17_series.json", "w"))
for f in ("/content/v17_tracks.json", "/content/v17_series.json"):
    print(f, f"{os.path.getsize(f)/1000:.0f} KB")
try:
    from google.colab import files
    files.download("/content/v17_tracks.json"); files.download("/content/v17_series.json")
    print("handed to browser download")
except Exception as e:
    print("download failed:", e)
