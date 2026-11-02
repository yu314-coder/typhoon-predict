"""Create the reproducible Colab entry point and its pinned package inventory."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).parent
FILES=('contracts.py','model.py','data.py','train.py','prepare_colab.py','arco_ranges.py','source_contract.py',
       'catalog.json.gz','requirements-colab.txt','test_contracts.py','test_training.py','README.md','design.json')
manifest={'experiment':'trackformer-1.3.1-native-regional-pilot-20261102','files':{
    n:{'sha256':hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),'bytes':(ROOT/n).stat().st_size} for n in FILES}}
(ROOT/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
pin=hashlib.sha256((ROOT/'package-manifest.json').read_bytes()).hexdigest()
cells=[]
def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True)))
def code(text):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True)))
md('''# TrackFormer 1.3.1 — new track and pressure training

Fresh **2026-11-02 regional pilot** using the new31-field quarter-degree weather data.
This is separate from released1.2 and from rainfall V52. It starts from random
weights. There is no demonstrated improvement yet.

**Runtime → Change runtime type → T4 GPU → Save.** Colab GPU availability varies.
You can prepare data on a CPU runtime first to avoid spending GPU time on downloads,
then switch to T4 and reuse a Drive data folder. Switching runtimes removes `/content` data.
Run the cells in order. Do not leave a paid runtime running just for downloads.

The notebook saves model/optimizer/RNG state to Google Drive after every completed
optimizer update. Colab can still disconnect or terminate; rerun the same cells to resume.
No training starts merely by opening this notebook.

The public source is retrospective ERA5, not a live observation service. Future
weather and JMA track/pressure labels are used only for supervision.24 reused
development starts ×20 leads is a pilot, not full811 or an untouched holdout.
''')
code('''# 1. Fetch the exact small source package (not the multi-GB repository).
from pathlib import Path
import hashlib, json, urllib.request, subprocess, sys, os
CODE = Path('/content/trackformer131-code')
CODE.mkdir(parents=True, exist_ok=True)
BASE = 'https://raw.githubusercontent.com/yu314-coder/typhoon-predict/36804f8a54277c3a97f329baa2caf80f25bf796c/training/trackformer_1_3_1/'
MANIFEST_SHA256 = 'PIN'
def fetch(name):
    request = urllib.request.Request(BASE + name, headers={'User-Agent':'TrackFormer-Colab/1.3.1'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()
raw = fetch('package-manifest.json')
assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256, 'Package changed: use the matching notebook revision'
package = json.loads(raw)
for name, expected in package['files'].items():
    path = CODE / name
    payload = path.read_bytes() if path.exists() else b''
    if hashlib.sha256(payload).hexdigest() != expected['sha256']:
        payload = fetch(name)
    assert len(payload) == expected['bytes'] and hashlib.sha256(payload).hexdigest() == expected['sha256'], name
    path.write_bytes(payload)
(CODE / 'package-manifest.json').write_bytes(raw)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(CODE/'requirements-colab.txt')], check=True)
os.chdir(CODE)
print('Verified', len(package['files']), 'files. Colab CUDA PyTorch was not replaced.')
'''.replace('PIN',pin))
md('''## Data and persistent output
Authorize your own Drive mount. No credentials are written to the repository.
Keep the same configuration and output folder to resume. Use a **new output folder**
when changing width, cohort, learning rate or epochs. `DATA_DIR` can point to an
already prepared Drive folder. Default Drive caching survives runtime switches.
Preparation estimates disk use before downloading; only selected frames are fetched.
''')
code('''# 2. Mount Drive and choose the pilot size.
from google.colab import drive
drive.mount('/content/drive')
TRAIN_STARTS = 64          # 2..11978; default is a bounded pilot, not all data
WIDTH = 32                 # 2,252,513 parameters; native .25-degree grid retained
EPOCHS = 4                  # training rollout lengths: 5,10,15,20; validation always20
EFFECTIVE_BATCH = 4         # microbatch1, accumulated gradients
TBPTT = 2                  # two-step backward segments, full causal forward state
LEARNING_RATE = 2e-4
SEED = 1312026
ALLOW_LONG_DOWNLOAD = False # change to True only if you accept the measured >30min estimate
DATA_DIR = Path('/content/drive/MyDrive/TrackFormer/data/131-native-31fields')
OUTPUT_DIR = Path('/content/drive/MyDrive/TrackFormer/runs/131-native-20261102-pilot')
def command(script, *args):
    return subprocess.run([sys.executable, '-u', str(CODE/script), *map(str,args)], cwd=CODE, check=True)
command('prepare_colab.py', '--data', DATA_DIR, '--train-starts', TRAIN_STARTS)
''')
code('''# 3. Sample at most two missing frames; print a measured retrieval estimate.
# This does not launch training. Existing hash-verified frames are reused.
command('prepare_colab.py', '--data', DATA_DIR, '--train-starts', TRAIN_STARTS, '--download', '--limit', 2)
estimate = json.loads((DATA_DIR/'download-estimate.json').read_text())
print('Measured retrieval estimate:', estimate)
print('Estimate is approximate and excludes later training/validation.')
''')
code('''# 4. Prepare the complete fixed cohort. Rerun after transient network failure.
estimate = json.loads((DATA_DIR/'download-estimate.json').read_text())
if estimate['remaining_minutes'] > 30 and not ALLOW_LONG_DOWNLOAD:
    raise RuntimeError('Estimated download exceeds30minutes. Read the estimate, then set ALLOW_LONG_DOWNLOAD=True in cell2 if acceptable and rerun cells2–4. Nothing was trained.')
command('prepare_colab.py', '--data', DATA_DIR, '--train-starts', TRAIN_STARTS, '--download')
print(json.loads((DATA_DIR/'ready.json').read_text()))
''')
md('''## Qualify the actual GPU, then train
The model encodes nine history frames, retains local detail through convolutional
skip connections and exchanges regional information through coarse attention.
It predicts all31 weather fields recurrently. Track association and central
pressure use the same predicted physical pressure field, not separate painted maps.
The dense inner-core ERA5 target is treated separately from JMA central pressure.

This first regional pilot does **not** yet implement the proposed external global
branch, static terrain branch, moving fine core or calibrated ensemble. It is a
measurable starting point, not a claim to reproduce a large foundation model.

The CUDA check below performs a disposable full-native-grid20-lead update on TRAIN
only. Production starts again from the specified fresh seed. If it does not fit
your runtime, choose width24 or16 and a new output folder; do not silently change
grid resolution, precision or the saved run. Actual GPU type/memory/time is recorded.
''')
code('''# 5. CPU contracts and actual CUDA/T4 qualification.
import torch
import shutil
# Use Drive for persistence and local VM disk for repeated training reads.
# This copies only the selected cohort; checksum verification remains in Data.frame.
cohort = json.loads((DATA_DIR/'cohort.json').read_text())
times = sorted({r['issue_seconds']+h*3600 for rows in cohort.values() for r in rows for h in range(-48,121,6)})
TRAIN_DATA_DIR = Path('/content/trackformer131-data')
(TRAIN_DATA_DIR/'frames').mkdir(parents=True,exist_ok=True)
names = ['ready.json','cohort.json'] + [f'frames/{t}{suffix}' for t in times for suffix in ('.npz','.json')]
needed = sum((DATA_DIR/n).stat().st_size for n in names if not (TRAIN_DATA_DIR/n).exists())
assert shutil.disk_usage(TRAIN_DATA_DIR).free > needed+2*1024**3, 'Insufficient local VM disk for the selected cohort'
for i,name in enumerate(names):
    source,target = DATA_DIR/name,TRAIN_DATA_DIR/name
    if not target.exists() or (target.stat().st_size,target.stat().st_mtime_ns)!=(source.stat().st_size,source.stat().st_mtime_ns):
        shutil.copy2(source,target)
    if (i+1)%200==0:print('Staged',i+1,'/',len(names),'files')
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU'
print('Actual accelerator:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(CODE), '-p', 'test_*.py', '-v'], check=True)
TRAIN_ARGS = ['--data', TRAIN_DATA_DIR, '--output', OUTPUT_DIR, '--width', WIDTH, '--epochs', EPOCHS,
              '--batch', EFFECTIVE_BATCH, '--tbptt', TBPTT, '--lr', LEARNING_RATE, '--seed', SEED]
command('train.py', *TRAIN_ARGS, '--preflight')
''')
code('''# 6. START TRAINING (or resume the same saved run).
# ETA printed during fitting excludes the separately measured development validation.
command('train.py', *TRAIN_ARGS)
''')
code('''# 7. Read actual progress and completed development metrics.
for name in ('status.json','receipt.json','failure.json'):
    path = OUTPUT_DIR/name
    if path.exists():
        print(name, json.dumps(json.loads(path.read_text()), indent=2)[:5000])
for folder in sorted(OUTPUT_DIR.glob('development-E*')):
    receipt_path = folder/'receipt.json'
    if not receipt_path.exists():
        print(folder.name, 'incomplete; no final metrics claimed')
        continue
    receipt = json.loads(receipt_path.read_text())
    for name, expected in receipt['files'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == expected, name
    print(folder.name, json.dumps(json.loads((folder/'metrics.json').read_text()), indent=2))
''')
code('''# 8. Historical development example: first fixed case, +24h, common pressure scale.
import numpy as np
import matplotlib.pyplot as plt
complete = [p for p in sorted(OUTPUT_DIR.glob('development-E*')) if (p/'receipt.json').exists()]
assert complete, 'Wait for a completed development receipt'
folder = complete[-1]
receipt = json.loads((folder/'receipt.json').read_text())
path = folder/'case-000.npz'
assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['files'][path.name]
with np.load(path, allow_pickle=False) as z:
    lead = int(np.flatnonzero(z['lead_hours'] == 24)[0])
    lat,lon = z['latitude'],z['longitude']
    mask = z['truth_valid'][lead]
    fields = [np.where(mask,z['pressure_hPa'][lead],np.nan), np.where(mask,z['truth_pressure_hPa'][lead],np.nan)]
    center = z['track_lat_lon'][lead]
    title = str(z['case_id'])
lo = min(np.nanmin(a) for a in fields); hi = max(np.nanmax(a) for a in fields)
fig, axes = plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
for ax,a,label in zip(axes, fields, ['1.3.1 predicted pressure','ERA5 verification pressure']):
    im=ax.pcolormesh(lon,lat,a,cmap='RdBu_r',vmin=lo,vmax=hi,shading='nearest')
    ax.set(xlabel='Longitude E',ylabel='Latitude N',title=label)
axes[0].scatter(center[1],center[0],marker='x',color='black',s=50,label='Associated predicted centre')
axes[0].legend()
fig.colorbar(im,ax=axes,label='MSLP (hPa)')
fig.suptitle(f'{folder.name} | {title} | +24h historical development, not live')
fig.text(.5,-.02,'Sources: model output; Copernicus/ECMWF ERA5 via ARCO; NOAA IBTrACS/JMA labels',ha='center')
plt.show()
''')
md('''## Interpretation and sources
Lower training loss is not proof of better track or pressure. Check unsupported
track outputs, all20 lead errors, field MAE/RMSE and JMA central-pressure error.
These24 development cases are reused. A matched released1.2 comparison, the full811
cohort, fixed11 gate and untouched holdout remain outstanding. Released1.2 is not
replaced by this pilot; this notebook has no rainfall head.

- [Copernicus ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)
- [Public ARCO ERA5](https://github.com/google-research/arco-era5)
- [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
- [WeatherNext Cyclones](https://www.nature.com/articles/s41586-026-10953-2)
- [GraphCast](https://arxiv.org/abs/2212.12794)
- [Colab runtime limits](https://research.google.com/colaboratory/faq.html)

Source code: repository MIT license. Weather/agency data retain upstream terms
and attribution. Checkpoints are research artifacts, not operational warnings.
''')
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',colab=dict(name='TrackFormer_1_3_1_Train.ipynb',gpuType='T4'),
    kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python')),cells=cells)
for i,c in enumerate(cells):c['id']=f'tf131-{i:02d}'
(ROOT/'TrackFormer_1_3_1_Train.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(cells=len(cells),manifest_sha256=pin)))
