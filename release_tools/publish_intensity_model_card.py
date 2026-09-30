"""Publish scoped scientific graphs/README additions while preserving HF videos.

Prepare from the freshly pinned remote model card, not a stale replacement.
No deletion, weights, paper or existing forecast/video artifact is changed.
The caller explicitly selects --publish; default only prepares and verifies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'euler314/typhoon-predict'
START = '## Matched wind, intensity and radius benchmark\n'
END = '## Fung-wong pressure forecast — MP4\n'
SOURCE_TOOLS = [
    'release_tools/STORM_STRUCTURE_DIAGNOSTICS.md',
    'release_tools/benchmark_intensity_v12_v11.py',
    'release_tools/plot_intensity_benchmark.py',
    'release_tools/export_structure_public.py',
    'release_tools/forecast_historical_video50_mac.py',
    'release_tools/plot_storm_wind_radius.py',
    'release_tools/storm_structure_diagnostics.py',
    'release_tools/test_intensity_benchmark.py',
    'release_tools/test_intensity_curves.py',
    'release_tools/test_storm_structure_diagnostics.py',
    'docs/intensity_benchmark.md',
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def merge_card(card, github):
    section = START + github.split(START, 1)[1].split(END, 1)[0]
    if card.count(END) != 1:
        raise ValueError('Unexpected model card anchor; preserve remote card')
    if START in card:
        card = card.split(START, 1)[0] + section + END + card.split(END, 1)[1]
    else:
        card = card.replace(END, section + END)
    coverage_start = '**Pressure coverage:**'
    replacement = next(line for line in github.splitlines() if line.startswith('**Pressure coverage in the original track run:**'))
    if coverage_start in card:
        line = next(line for line in card.splitlines() if line.startswith(coverage_start))
        card = card.replace(line, replacement, 1)
    if not card.startswith('---\n') or card.count('<video ') != 4:
        raise ValueError('Preserve HF front matter and all four existing inline videos')
    for asset in ('fung_wong','soudelor','mangkhut','meranti'):
        if f'docs/trackformer_1_2_{asset}.mp4' not in card:
            raise ValueError('Existing video link lost')
    return card


def run(output, publish):
    if not str(output.resolve()).startswith('/Volumes/D/'):
        raise ValueError('Publication artifacts must remain on D')
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    head = api.model_info(REPO).sha
    original = Path(hf_hub_download(REPO, 'README.md', revision=head,
        cache_dir='/Volumes/D/typhoon_predict/.cache/huggingface-intensity'))
    card = merge_card(original.read_text(), (ROOT/'README.md').read_text())
    card_path = output/'README.md'
    card_path.write_text(card)
    files = {'README.md': card_path}
    for relative in SOURCE_TOOLS:
        files[relative] = ROOT/relative
    for folder in ('evaluation/intensity','evaluation/storm_structure'):
        for path in sorted((ROOT/folder).iterdir()):
            if path.is_file() and path.suffix in ('.png','.json'):
                files[str(path.relative_to(ROOT))] = path
    benchmark = json.loads((ROOT/'evaluation/intensity/verification.json').read_text())
    if benchmark['status'] != 'complete_verified' or sha(ROOT/'evaluation/intensity/intensity_final.json') != benchmark['summary_sha256']:
        raise ValueError('Unverified final benchmark')
    for name, expected in benchmark['images'].items():
        if sha(ROOT/'evaluation/intensity'/name) != expected:
            raise ValueError('Changed benchmark chart')
    receipt = {'status':'prepared', 'repo_id': REPO, 'parent_commit': head,
        'original_card_sha256': sha(original), 'files': {p: sha(v) for p,v in files.items()},
        'weights_changed': False, 'existing_video_links_preserved': True}
    if publish:
        committed = api.create_commit(repo_id=REPO, repo_type='model', parent_commit=head,
            commit_message='Document matched daily wind/intensity benchmark and honest radius coverage',
            operations=[CommitOperationAdd(path_in_repo=p,path_or_fileobj=str(v)) for p,v in files.items()])
        commit = committed.oid
        for name, expected in receipt['files'].items():
            saved = Path(hf_hub_download(REPO,name,revision=commit,
                cache_dir='/Volumes/D/typhoon_predict/.cache/huggingface-intensity'))
            if sha(saved) != expected:
                raise ValueError('Published artifact differs: '+name)
        receipt.update(status='published_verified', commit=commit, commit_url=str(committed.commit_url),
                       all_remote_files_sha256_verified=True)
    (output/'publication.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    run(args.output,args.publish)
