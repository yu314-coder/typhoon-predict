"""Render the complete HF card from the canonical GitHub README.

Preserve remote metadata, the featured film and frozen neural weights. Default
mode only prepares artifacts on D; --publish explicitly updates the HF card,
matching existing inference wrapper/docs and this small reproduction utility.
No forecasts are rerun, no media are regenerated and no neural modules change.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = 'euler314/typhoon-predict'
MEDIA_REVISION = '87a6e366b42bb4cc0edc95d2c50c55fca21a2c93'
CACHE = '/Volumes/D/typhoon_predict/.cache/huggingface-intensity'
HF = 'https://huggingface.co/' + REPO
FILMS = ('mangkhut',)
CURRENT_FIGURES = frozenset((
    'evaluation/released_daily/model_1_2_benchmark.png',
    'docs/trackformer_1_2_architecture.svg',
))
SYNC_FILES = (
    'models/trackformer_1_2_field/README.md',
    'models/trackformer_1_2_field/WIND_ESTIMATION.md',
    'models/trackformer_1_2_field/predict.py',
    'models/trackformer_1_2_field/plot_pressure.py',
    'models/trackformer_1_2_field/wind_estimation.py',
    'release_tools/sync_public_model_cards.py',
    'release_tools/test_public_model_cards.py',
    'release_tools/test_pressure_field_export.py',
    'release_tools/verify_pressure_field_export.py',
    'RELEASE_NOTES_TRACKFORMER_1_2.md',
    'release_tools/build_release_benchmark.py',
    'release_tools/plot_release_pressure_benchmark.py',
    'release_tools/plot_daily_storm_final.py',
    'release_tools/plot_model_announcement.py',
    'release_tools/import_deepmind_release_results.py',
    'release_tools/test_deepmind_release_results.py',
    'release_tools/build_mangkhut_gif.py',
    'release_tools/deepmind_daily_benchmark.py',
    'release_tools/test_deepmind_daily_benchmark.py',
    'docs/daily_storm_benchmark.md',
    'docs/intensity_benchmark.md',
    'docs/trackformer_1_2_evaluation.md',
    'docs/deepmind_daily_benchmark.md',
    'docs/showcase_archive.md',
    'docs/trackformer_1_2_architecture.svg',
    'docs/trackformer_1_2_mangkhut.gif',
    'docs/trackformer_1_2_mangkhut.gif.json',
    'evaluation/daily_storm_final.json',
    'evaluation/released_daily/released_daily_benchmark.json',
    'evaluation/released_daily/released_daily_verification.json',
    'evaluation/intensity/intensity_final.json',
    'evaluation/intensity/verification.json',
    'evaluation/trackformer_1_2_vs_1_1_270_storms_bars.png',
    'evaluation/released_daily/pressure_comparison.png',
    'evaluation/released_daily/model_1_2_benchmark.png',
    'evaluation/released_daily/model_1_2_benchmark.json',
    'evaluation/deepmind_daily/benchmark.json',
    'evaluation/deepmind_daily/verification.json',
    'evaluation/deepmind_daily/publication_audit.json',
    'evaluation/deepmind_daily/protocol.json',
    'evaluation/deepmind_daily/case-manifest.json',
    'evaluation/deepmind_daily/canary.json',
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def media_url(stem, poster=False):
    path = ('evaluation/trackformer_1_2_' + stem + '_video_poster.png'
            if poster else 'docs/trackformer_1_2_' + stem + '.mp4')
    return HF + '/resolve/' + MEDIA_REVISION + '/' + path


def player(stem, title):
    return ('<video controls preload="metadata" width="100%" '
            'aria-label="' + html.escape(title, quote=True) + '" '
            'poster="' + media_url(stem, True) + '" '
            'src="' + media_url(stem) + '"></video>')


def render_card(original, github, figure_revision='main'):
    metadata = re.match(r'\A---\n.*?\n---\n', original, re.S)
    if not metadata:
        raise ValueError('Preserve existing HF YAML metadata; missing header')
    if github.count('## See the forecast: Mangkhut (2018)\n') != 1:
        raise ValueError('Missing or ambiguous featured-film section')
    card = github
    for stem, alt in (('mangkhut', 'Trackformer 1.2 Mangkhut pressure and route forecast'),):
        old = '[![' + alt + '](docs/trackformer_1_2_' + stem + '.gif)](' + media_url(stem) + ')'
        if card.count(old) != 1:
            raise ValueError('Missing or ambiguous primary preview: ' + stem)
        card = card.replace(old, player(stem, alt), 1)
    # Relative GitHub links need HF-specific URLs. Keep evolving documentation
    # on main; pin unchanged scientific image bytes to the verified asset commit.
    def link(match):
        image, label, url = match.groups()
        if url.startswith(('https:', 'http:', '#', 'mailto:')):
            return match.group(0)
        path, _, fragment = url.partition('#')
        resolved = (ROOT / path).resolve()
        if not resolved.is_relative_to(ROOT) or not resolved.exists():
            raise ValueError('Invalid or missing relative public link: ' + path)
        revision = figure_revision if path in CURRENT_FIGURES else MEDIA_REVISION
        kind = 'resolve/' + revision if image else ('tree/main' if resolved.is_dir() else 'blob/main')
        target = HF + '/' + kind + '/' + path + ('#' + fragment if fragment else '')
        return image + '[' + label + '](' + target + ')'
    card = re.sub(r'(!?)\[([^\[\]\n]*)\]\(([^)\s]+)\)', link, card)
    card = metadata.group(0) + '\n' + card
    if card.count('<video ') != 1:
        raise ValueError('Expected exactly one featured native player')
    for stem in FILMS:
        if card.count('src="' + media_url(stem) + '"') != 1:
            raise ValueError('Missing, repeated or stale player: ' + stem)
    if re.search(r'/resolve/main/docs/trackformer_1_2_[^"\s)]+\.mp4', card):
        raise ValueError('Unversioned movie source')
    return card


def frozen_identity(api, revision):
    manifest = json.loads((ROOT / 'models/trackformer_1_2_field/manifest.json').read_text())
    names = ['models/trackformer_1_2_field/weights.pt'] + [
        'models/trackformer_1_2_field/' + name for name in manifest['source_module_sha256']]
    entries = {i.path: i for i in api.get_paths_info(REPO, names, revision=revision)}
    weight = entries.get(names[0])
    actual = getattr(getattr(weight, 'lfs', None), 'sha256', None)
    if actual != manifest['inference_weights_sha256']:
        raise ValueError('Released weights are absent or differ from frozen identity')
    for name, expected in manifest['source_module_sha256'].items():
        path = ROOT / 'models/trackformer_1_2_field' / name
        data = path.read_bytes()
        entry = entries.get('models/trackformer_1_2_field/' + name)
        if sha(path) != expected or entry is None or entry.blob_id != hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest():
            raise ValueError('Frozen neural source differs: ' + name)
    return actual


def run(output, publish=False):
    # Lazy imports keep portable CI document tests independent of HF packages.
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    output = output.resolve()
    if not output.is_relative_to(Path('/Volumes/D')):
        raise ValueError('All generated documentation/cache artifacts must stay on D')
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    parent = api.model_info(REPO).sha
    weights_before = frozen_identity(api, parent)
    original = Path(hf_hub_download(REPO, 'README.md', revision=parent, cache_dir=CACHE))
    github = (ROOT / 'README.md').read_text()
    card_path = output / 'README.md'
    card_path.write_text(render_card(original.read_text(), github))
    files = {'README.md': card_path, **{name: ROOT / name for name in SYNC_FILES}}
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    receipt = {
        'status': 'prepared', 'repo_id': REPO, 'parent_commit': parent,
        'github_source_commit': source_commit,
        'github_readme_sha256': sha(ROOT / 'README.md'),
        'original_hf_readme_sha256': sha(original),
        'files': {name: sha(path) for name, path in files.items()},
        'media_revision': MEDIA_REVISION, 'native_video_players': 1,
        'inference_weights_sha256': weights_before,
        'weights_changed': False, 'neural_modules_changed': False,
        'forecast_arrays_changed': False, 'videos_changed': False,
    }
    if publish:
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'README.md', *SYNC_FILES], cwd=ROOT, text=True)
        if dirty:
            raise ValueError('Commit canonical README and matching source files before publishing')
        # Publish scientific assets first, then pin the new figures to that
        # immutable commit in the card. Never pin new bytes to the old movie
        # revision or let a missing future image render as a stale figure.
        assets = api.create_commit(repo_id=REPO, repo_type='model', parent_commit=parent,
            commit_message='Publish completed WeatherNext Cyclones Mini daily comparison and audited three-model chart',
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(ROOT/name)) for name in SYNC_FILES])
        card_path.write_text(render_card(original.read_text(), github, assets.oid))
        receipt['files']['README.md'] = sha(card_path)
        receipt['figure_revision'] = assets.oid
        committed = api.create_commit(repo_id=REPO, repo_type='model', parent_commit=assets.oid,
            commit_message='Update Trackformer 1.2 model card with completed DeepMind CUDA benchmark',
            operations=[CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=str(card_path))])
        for name, expected in receipt['files'].items():
            saved = hf_hub_download(REPO, name, revision=committed.oid, cache_dir=CACHE)
            if sha(saved) != expected:
                raise ValueError('Published documentation/source differs: ' + name)
        if frozen_identity(api, committed.oid) != weights_before:
            raise ValueError('Frozen identity changed during documentation publication')
        receipt.update(status='published_verified', commit=committed.oid,
            commit_url=str(committed.commit_url), all_remote_files_sha256_verified=True)
    (output / 'publication.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    run(args.output, args.publish)
