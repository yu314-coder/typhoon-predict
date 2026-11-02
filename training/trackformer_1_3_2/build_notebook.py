"""Build a three-cell notebook with an immutable source-package pin."""
from pathlib import Path
import argparse,hashlib,json

ROOT=Path(__file__).parent
FILES=('contracts.py','model.py','data.py','train.py','prepare_colab.py','arco_ranges.py','source_contract.py',
       'catalog.json.gz','test_contracts.py','test_training.py','test_cohort.py','README.md')


def build(commit):
    manifest=dict(experiment='trackformer-1.3.2-transport-steering-20261102',files={
        n:dict(sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),bytes=(ROOT/n).stat().st_size) for n in FILES})
    (ROOT/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pin=hashlib.sha256((ROOT/'package-manifest.json').read_bytes()).hexdigest()
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True)))
    md('''# TrackFormer 1.3.2 — larger track and pressure training

**11.29 million parameters · 186 training starts / 84 storms · eight epochs**

Uses the completed1.3.1 weather cache. No new weather download and no extra full-cache
verification pass. Select **T4 GPU**, then run these three cells in order.
The new model uses regional context, environmental steering and weather transport.
It is a new experiment, **not a proven improvement**. Released1.2 and local rainfall
V52 remain separate. This notebook starts production training in cell3.

Checkpoints go to your existing Google Drive after every optimizer update.
An interrupted run resumes by rerunning the same cells/configuration. Colab can
still disconnect; no promise of uninterrupted service is made.
''')
    code('''# 1. Fetch the exact small source package. Keep Colab's CUDA PyTorch.
from pathlib import Path
import hashlib,json,urllib.request,subprocess,sys,os
import numpy,torch
CODE=Path('/content/trackformer132-code');CODE.mkdir(parents=True,exist_ok=True)
BASE='https://raw.githubusercontent.com/yu314-coder/typhoon-predict/COMMIT/training/trackformer_1_3_2/'
MANIFEST_SHA256='PIN'
def fetch(name):
    request=urllib.request.Request(BASE+name,headers={'User-Agent':'TrackFormer-Colab/1.3.2'})
    with urllib.request.urlopen(request,timeout=90) as response:return response.read()
raw=fetch('package-manifest.json')
assert hashlib.sha256(raw).hexdigest()==MANIFEST_SHA256,'Source revision mismatch'
package=json.loads(raw)
for name,expected in package['files'].items():
    path=CODE/name;payload=path.read_bytes() if path.exists() else b''
    if hashlib.sha256(payload).hexdigest()!=expected['sha256']:payload=fetch(name)
    assert len(payload)==expected['bytes'] and hashlib.sha256(payload).hexdigest()==expected['sha256'],name
    path.write_bytes(payload)
(CODE/'package-manifest.json').write_bytes(raw)
print('1.3.2 source ready:',len(package['files']),'files; PyTorch',torch.__version__)
assert torch.cuda.is_available(),'Choose Runtime > Change runtime type > T4 GPU'
print('Actual GPU:',torch.cuda.get_device_name(0))
def command(script,*args):
    argv=[sys.executable,'-u',str(CODE/script),*map(str,args)]
    with subprocess.Popen(argv,cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as p:
        for line in p.stdout:print(line,end='',flush=True)
        result=p.wait()
    if result:raise subprocess.CalledProcessError(result,argv)
'''.replace('COMMIT',commit).replace('PIN',pin))
    code('''# 2. Reuse cached weather; stage on VM disk for fast training reads.
from google.colab import drive
drive.mount('/content/drive')
SOURCE_DATA=Path('/content/drive/MyDrive/TrackFormer/data/131-native-31fields')
TRAIN_DATA=Path('/content/trackformer132-data')
OUTPUT=Path('/content/drive/MyDrive/TrackFormer/runs/132-transport-steering-20261102-v1')
command('prepare_colab.py','--source',SOURCE_DATA,'--data',TRAIN_DATA,'--old-local','/content/trackformer131-data')
print('No new weather downloaded. Fixed24 development storms unchanged.')
''')
    code('''# 3. START / RESUME 1.3.2 PRODUCTION TRAINING.
# Explicit arguments also make this cell independent of earlier TRAIN_ARGS variables.
command('train.py','--data',TRAIN_DATA,'--output',OUTPUT,'--width',64,'--epochs',8,
        '--batch',4,'--tbptt',2,'--lr',1e-4,'--seed',1322026)
print('Final receipt:',(OUTPUT/'receipt.json').read_text())
''')
    md('''## How to judge the result

Actual production has376 optimizer updates. E1 uses+6..+30h training rollouts,
E2+60h, E3+90h, E4–E8+120h. Every epoch validates all24 fixed development starts
at all20 leads+6..+120h and saves the original pressure maps, track positions,
truth/masks, metrics and receipt hashes. The status includes a rough curriculum-aware
training ETA; validation time is additional. Training loss is not forecast accuracy.

Compare track errors against constant motion,1.3.1 and ultimately released1.2 on
the same cases and support. Check central-pressure and full-map errors too. The
reused24-case development set is not full811 validation or an untouched holdout.
There is no automatic promotion and no rainfall model in this notebook.

Sources: [ERA5/ECMWF](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels),
[ARCO ERA5](https://github.com/google-research/arco-era5),
[IBTrACS/JMA](https://www.ncei.noaa.gov/products/international-best-track-archive).
Research: [TC-Next](https://arxiv.org/html/2609.02085v1) and
[WeatherNext Cyclones](https://www.nature.com/articles/s41586-026-10953-2).
Our approximate transport prior is an engineering hypothesis, not a reproduction
or a claim to match those systems. See the source README for limitations.
''')
    for i,c in enumerate(cells):c['id']=f'tf132-{i:02d}'
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',colab=dict(name='TrackFormer_1_3_2_Train.ipynb',gpuType='T4'),
        kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python')),cells=cells)
    (ROOT/'TrackFormer_1_3_2_Train.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(manifest_sha256=pin,cells=len(cells),base_commit=commit)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-commit',default='PACKAGE_COMMIT');build(p.parse_args().base_commit)
