"""Scoped HF video revision: preserve the remote card and all other model assets."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'euler314/typhoon-predict'
CACHE = '/Volumes/D/typhoon_predict/.cache/huggingface-intensity'
START = '## More historical pressure forecasts — MP4\n'
END = '### Illustrated technical paper\n'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def merge_card(card, github):
    """Only replace shared descriptions; keep HF front matter and inline films."""
    for text in (card, github):
        if text.count(START) != 1 or text.count(END) != 1:
            raise ValueError('Unexpected card anchors; preserve remote content')
    old = card.split(START, 1)[1].split(END, 1)[0]
    new = github.split(START, 1)[1].split(END, 1)[0]
    lead = new.split('\n\n| Soudelor', 1)[0]
    tail = new.split('**These are calendar-selected', 1)[1]
    if old.count('### Soudelor') != 1 or old.count('**These are calendar-selected') != 1:
        raise ValueError('Unexpected HF video section')
    videos = '### Soudelor'+old.split('### Soudelor', 1)[1].split('**These are calendar-selected', 1)[0]
    result = card.split(START, 1)[0]+START+'\n'+lead.strip()+'\n\n'+videos+'**These are calendar-selected'+tail+END+card.split(END, 1)[1]
    if not result.startswith('---\n') or result.count('<video ') != card.count('<video ') or result.count('<video ') != 4:
        raise ValueError('Lost HF metadata or an inline film')
    return result


def run(output, publish):
    if not str(output.resolve()).startswith('/Volumes/D/'):
        raise ValueError('All publication artifacts must stay on D')
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    parent = api.model_info(REPO).sha
    original = Path(hf_hub_download(REPO, 'README.md', revision=parent, cache_dir=CACHE))
    card = output/'README.md'
    card.write_text(merge_card(original.read_text(), (ROOT/'README.md').read_text()))
    files = {'README.md': card}
    for name in ('docs/trackformer_1_2_mangkhut.mp4', 'evaluation/trackformer_1_2_mangkhut_video_poster.png',
                 'evaluation/release_data/mangkhut_video.json', 'evaluation/release_data/mangkhut_playback_verification.json',
                 'evaluation/release_data/historical_video50_verification.json',
                 'release_tools/build_fung_wong_video.py', 'release_tools/verify_historical_video50.py',
                 'release_tools/test_historical_video_playback.py', 'release_tools/publish_mangkhut_video_revision.py'):
        files[name] = ROOT/name
    for name in ('common-pressure.npz', 'verification.json', 'core-reconstruction.json'):
        relative = 'evaluation/release_data/pressure_core/mangkhut/'+name
        files[relative] = ROOT/relative
    meta = json.loads(files['evaluation/release_data/mangkhut_video.json'].read_text())
    check = json.loads(files['evaluation/release_data/mangkhut_playback_verification.json'].read_text())
    if meta['members'] != 50 or meta['video']['duration_seconds'] != 20 or not check['verified']:
        raise ValueError('Unverified video revision')
    if sha(files['docs/trackformer_1_2_mangkhut.mp4']) != meta['video']['sha256'] or check['video_sha256'] != meta['video']['sha256']:
        raise ValueError('Video hash mismatch')
    receipt = {'status':'prepared', 'repo_id': REPO, 'parent_commit': parent,
               'files':{p:sha(v) for p,v in files.items()}, 'weights_changed':False, 'other_films_changed':False}
    if publish:
        result = api.create_commit(repo_id=REPO, repo_type='model', parent_commit=parent,
            commit_message='Repair actual moving-core Mangkhut pressure film and remove frozen ending',
            operations=[CommitOperationAdd(path_in_repo=p,path_or_fileobj=str(v)) for p,v in files.items()])
        for name, expected in receipt['files'].items():
            path = hf_hub_download(REPO, name, revision=result.oid, cache_dir=CACHE)
            if sha(path) != expected:
                raise ValueError('Remote file SHA mismatch: '+name)
        receipt.update(status='published_verified', commit=result.oid, commit_url=str(result.commit_url), all_remote_files_sha256_verified=True)
    (output/'publication.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    run(args.output, args.publish)
