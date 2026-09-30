"""Export already audited plots and small public diagnostic receipts.

No inference, fitting, weather download, or modification of source forecasts.
Raw members remain in their immutable source folders; the public receipts give
their hashes and complete-member summaries, not fictional resolved radii.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    verification = json.loads((args.plots/'verification.json').read_text())
    cases = []
    for slug in ('fung_wong', 'soudelor', 'mangkhut', 'meranti'):
        source = args.plots/f'{slug}_structure.json'
        report = json.loads(source.read_text())
        if report['actual_members'] != 50 or not report['physical_50_member_means_verified'] or not report['observed_plus_zero_alignment_verified']:
            raise ValueError('Requires real audited 50-member source')
        # Retain the immutable member-archive digest even though the large raw
        # members are not redundantly copied into the website package.
        public = {k: v for k, v in report.items() if k not in ('member_diagnostics_by_lead', 'source_forecast_folder')}
        public['source_receipt_sha256'] = sha(source)
        public['source_reference'] = 'Original immutable /Volumes/D member archive; original files unchanged.'
        destination = args.output/f'{slug}_structure.json'
        destination.write_text(json.dumps(public, indent=2, allow_nan=False)+'\n')
        image = args.plots/f'{slug}_wind_radius.png'
        matches = [v for k, v in verification['output_sha256'].items() if Path(k).name == image.name]
        if len(matches) != 1 or sha(image) != matches[0]:
            raise ValueError('Changed audited image')
        shutil.copy2(image, args.output/image.name)
        cases.append({'slug': slug, 'name': slug.replace('_', '-').title(), 'storm_id': report['storm_id'],
            'issue_time_utc': report['issue_time_utc'], 'members': 50, 'image': image.name,
            'image_sha256': sha(image), 'receipt': destination.name, 'receipt_sha256': sha(destination),
            'coverage': report['coverage'], 'rmw_method': 'issue-time persistence, not a future RMW forecast'})
    overview = args.plots/'four_storm_wind_candidate.png'
    overview_matches = [v for k, v in verification['output_sha256'].items() if Path(k).name == overview.name]
    if len(overview_matches) != 1 or sha(overview) != overview_matches[0]:
        raise ValueError('Changed audited overview')
    shutil.copy2(overview, args.output/overview.name)
    index = {'schema': 'audited-structure-examples-v1', 'cases': cases,
        'source_verification_sha256': sha(args.plots/'verification.json'),
        'overview': overview.name, 'overview_sha256': sha(overview), 'experimental': True,
        'native_radius_skill_claim': False, 'forecast_weights_changed': False,
        'note': '50 actual member pressure diagnostics. Fung-Wong is a separate fresh replay, not the old MP4 output. '
                'RMW is persistence; USA and JMA wind periods stay separate; no radius-skill score.'}
    (args.output/'index.json').write_text(json.dumps(index, indent=2)+'\n')
    print(json.dumps({'exported_cases': len(cases), 'path': str(args.output), 'actual_members': 50}))


if __name__ == '__main__':
    main()
