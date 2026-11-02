"""Build a pinned notebook; source-package commit is set after publication."""
from pathlib import Path
import argparse,hashlib,json

ROOT=Path(__file__).parent
FILES=('contracts.py','model.py','data.py','train.py','prepare_colab.py','arco_ranges.py','source_contract.py',
       'catalog.json.gz','test_contracts.py','test_training.py','test_cohort.py','README.md')


def build(commit):
    manifest=dict(experiment='trackformer-1.3.3-direct-context-20261103',files={
        n:dict(sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),bytes=(ROOT/n).stat().st_size) for n in FILES})
    (ROOT/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pin=hashlib.sha256((ROOT/'package-manifest.json').read_bytes()).hexdigest()
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True)))
    md('''# TrackFormer 1.3.3 — direct track and pressure forecasts

**New experiment · all20 leads from epoch1 · eight epochs · no new weather download**

Every lead reads the same48-hour causal history. Predicted pressure fields and
pressure-centre corrections are not fed back into later forecasts. Dense pressure
supervision includes the storm core. This is a proposed fix, not proven skill.
Select **T4 GPU**. Run the three cells below. Existing1.3.2/rainfall runs and their
checkpoints are separate; this notebook never stops another process.
''')
    code('''# 1. Fetch the small pinned1.3.3 package; retain Colab CUDA PyTorch.
from pathlib import Path
import hashlib,json,urllib.request,subprocess,sys,os
import numpy,torch
CODE133=Path('/content/trackformer133-code');CODE133.mkdir(parents=True,exist_ok=True)
BASE133='https://raw.githubusercontent.com/yu314-coder/typhoon-predict/COMMIT/training/trackformer_1_3_3/'
PIN133='PIN'
def fetch133(name):
    with urllib.request.urlopen(urllib.request.Request(BASE133+name,headers={'User-Agent':'TrackFormer-Colab/1.3.3'}),timeout=90) as r:return r.read()
raw133=fetch133('package-manifest.json')
assert hashlib.sha256(raw133).hexdigest()==PIN133,'Source manifest changed'
for name,expected in json.loads(raw133)['files'].items():
    path=CODE133/name;payload=path.read_bytes() if path.exists() else b''
    if hashlib.sha256(payload).hexdigest()!=expected['sha256']:payload=fetch133(name)
    assert len(payload)==expected['bytes'] and hashlib.sha256(payload).hexdigest()==expected['sha256'],name
    path.write_bytes(payload)
(CODE133/'package-manifest.json').write_bytes(raw133)
assert torch.cuda.is_available(),'Select T4 GPU under Runtime > Change runtime type'
print('1.3.3 source ready; actual GPU:',torch.cuda.get_device_name(0),'PyTorch:',torch.__version__)
def command133(script,*args):
    argv=[sys.executable,'-u',str(CODE133/script),*map(str,args)]
    with subprocess.Popen(argv,cwd=CODE133,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as p:
        for line in p.stdout:print(line,end='',flush=True)
        result=p.wait()
    if result:raise subprocess.CalledProcessError(result,argv)
'''.replace('COMMIT',commit).replace("'PIN'",repr(pin)))
    code('''# 2. Reuse the existing staged data and TRAIN-only normalizer.
from google.colab import drive
drive.mount('/content/drive')
SOURCE133=Path('/content/drive/MyDrive/TrackFormer/data/131-native-31fields')
DATA133=Path('/content/trackformer132-data')
OUT133=Path('/content/drive/MyDrive/TrackFormer/runs/133-direct-context-20261103-v1')
OUT133.mkdir(parents=True,exist_ok=True)
if not (DATA133/'ready.json').exists():
    command133('prepare_colab.py','--source',SOURCE133,'--data',DATA133,'--old-local','/content/trackformer131-data')
ready133=json.loads((DATA133/'ready.json').read_text())
assert ready133['catalog_sha256']=='bc14fb62a9a4e3a2bda3f84e039fa45ec4fa4305ea8654acd36236b4c6b9a975'
assert hashlib.sha256((DATA133/'cohort.json').read_bytes()).hexdigest()==ready133['cohort_sha256']=='1b16348b12899e7fce8986a7e21df4e9116ee94f42281e8b53ee3a58f423a8f3'
old_norm133=Path('/content/drive/MyDrive/TrackFormer/runs/132-transport-steering-20261102-v1/normalizer.json')
if old_norm133.exists() and not (OUT133/'normalizer.json').exists():
    payload=old_norm133.read_bytes()
    assert hashlib.sha256(payload).hexdigest()=='34ac0640f91f3b04f7cd43d5bed6bab6de0f77fc0e359f9f0b3c3c1cdfc89ffb'
    (OUT133/'normalizer.json').write_bytes(payload)
print('1.3.3 ready:186 TRAIN starts/84 storms,24 development starts; cached weather, no new download.')
''')
    code('''# 3. START / RESUME actual1.3.3 training:376 optimizer updates.
command133('train.py','--data',DATA133,'--output',OUT133,'--width',64,'--epochs',8,
           '--batch',4,'--lr',1e-4,'--seed',1332026)
print('Final1.3.3 receipt:',(OUT133/'receipt.json').read_text())
''')
    md('''## Judge by saved validation, not loss or the appearance of one map

Each epoch saves all24 fixed development cases at all20 leads+6..+120h, native
241x321 MSLP maps, original truth/masks, track, central pressure, metrics and hashes.
Checkpoints are written to Drive after every optimizer update. Re-run the same
cells/configuration to resume; no automatic restart or endless extension.

The same cached cohort permits matched1.3.2 comparisons. Released1.2 comparison,
full811 and an untouched holdout remain unmet. JMA central pressure and ERA5 map
pressure are different observations; both diagnostics are reported honestly.
No rainfall model is trained here and there is no automatic model promotion.
See README.md for architecture, losses, limitations and primary research links.
''')
    for i,c in enumerate(cells):c['id']=f'tf133-{i:02d}'
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',colab=dict(name='TrackFormer_1_3_3_Train.ipynb',gpuType='T4'),
        kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python')),cells=cells)
    (ROOT/'TrackFormer_1_3_3_Train.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(manifest_sha256=pin,base_commit=commit)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-commit',default='PACKAGE_COMMIT');build(p.parse_args().base_commit)
