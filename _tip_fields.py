"""Extract SLP + 500 hPa steering patches for Tip 1979, with the same tolerance guard
as the repaired extractors -- no silent snapping to the nearest available timestep."""
import os, math, urllib.request, numpy as np, netCDF4
HALF=8
z=np.load("track_build/tip_fixed.npz",allow_pickle=True)
bt=z["base_time"].astype("int64"); bla=z["base_lat"].astype("float64"); blo=z["base_lon"].astype("float64")
N=len(bt); print(f"{N} Tip windows")

def times(d):
    tv=d.variables["time"]
    dts=netCDF4.num2date(tv[:],tv.units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    return np.array([np.datetime64(x).astype("datetime64[ns]").astype("int64") for x in dts])

# ---- SLP (6-hourly, tolerance 6h) ----
d=netCDF4.Dataset("track_build/geo/slp/slp.1979.nc")
tns=times(d); lat=d.variables["lat"][:].astype("float64"); lon=d.variables["lon"][:].astype("float64")
slp=d.variables["slp"][:].astype("float32")/100.0
print(f"  slp covers {tns.min().astype('datetime64[ns]')} .. {tns.max().astype('datetime64[ns]')}")
P=np.zeros((N,2,17,17),"float16"); ok_s=np.zeros(N,bool)
MAXS=int(6*3600*1e9)
for i in range(N):
    ti=int(np.abs(tns-bt[i]).argmin())
    if abs(int(tns[ti])-int(bt[i]))>MAXS or ti<4: continue
    li=int(np.abs(lat-bla[i]).argmin()); lj=int(np.abs(lon-(blo[i]%360)).argmin())
    r=np.clip(np.arange(li-HALF,li+HALF+1),0,len(lat)-1); c=np.mod(np.arange(lj-HALF,lj+HALF+1),len(lon))
    now=slp[ti][np.ix_(r,c)]; prev=slp[ti-4][np.ix_(r,c)]
    P[i,0]=(now-now.mean()).astype("float16"); P[i,1]=(now-prev).astype("float16"); ok_s[i]=True
d.close(); print(f"  SLP matched {ok_s.sum()}/{N}")

# ---- 500 hPa steering (daily, tolerance 18h) ----
BASE="https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure"
UV={}
for var in ("uwnd","vwnd"):
    f=f"track_build/geo/tmp_{var}_1979.nc"
    if not os.path.exists(f):
        print(f"  downloading {var}.1979.nc ...",flush=True); urllib.request.urlretrieve(f"{BASE}/{var}.1979.nc",f)
    dd=netCDF4.Dataset(f); lev=dd.variables["level"][:]; i5=int(np.where(lev==500)[0][0])
    UV[var]=(times(dd),dd.variables["lat"][:].astype("float64"),dd.variables["lon"][:].astype("float64"),
             np.asarray(dd.variables[var][:,i5,:,:],dtype="float32")); dd.close(); os.remove(f)
tns2,lat2,lon2,U=UV["uwnd"]; V=UV["vwnd"][3]
S=np.zeros((N,2,17,17),"float16"); ok_w=np.zeros(N,bool); MAXW=int(18*3600*1e9)
for i in range(N):
    ti=int(np.abs(tns2-bt[i]).argmin())
    if abs(int(tns2[ti])-int(bt[i]))>MAXW: continue
    li=int(np.abs(lat2-bla[i]).argmin()); lj=int(np.abs(lon2-(blo[i]%360)).argmin())
    r=np.clip(np.arange(li-HALF,li+HALF+1),0,len(lat2)-1); c=np.mod(np.arange(lj-HALF,lj+HALF+1),len(lon2))
    S[i,0]=U[ti][np.ix_(r,c)].astype("float16"); S[i,1]=V[ti][np.ix_(r,c)].astype("float16"); ok_w[i]=True
print(f"  steering matched {ok_w.sum()}/{N}")
comb=np.concatenate([P,S],axis=1)
np.save("track_build/tip_steer4.npy",comb); np.save("track_build/tip_ok.npy",np.stack([ok_s,ok_w],1))
print("saved track_build/tip_steer4.npy")
