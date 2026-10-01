"""Audit the appended learned wind exports, independently of pressure-core counts."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from auxiliary_wind_archive import CHECKPOINT, METHOD, sha, verify_wind


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'release_tools/history_inputs.json').read_text())
    manifest = json.loads((root/'models/trackformer_1_2_field/manifest.json').read_text())
    input_hash = sha(args.manifest)
    if input_hash != config['manifest_sha256']:
        raise ValueError('Unpinned wind recovery plan')
    planned = {r['id']: r for r in json.loads(args.manifest.read_text())['queue']}
    dest = args.output/'model-wind'
    completed = []
    resolved_points = 0
    storms_with_wind = set()
    for path in sorted(dest.glob('auto-*.json')):
        document = json.loads(path.read_text())
        if path.stem != document['forecast_id']:
            raise ValueError('Wind export filename identity mismatch')
        verify_wind(document, args.output, planned, input_hash, manifest['inference_weights_sha256'])
        completed.append(document['forecast_id'])
        valid = sum(p['valid_members'] for p in document['points'])
        resolved_points += valid
        if valid:
            storms_with_wind.add(document['storm_id'])
    status = json.loads((dest/'status.json').read_text())
    if (len(completed) != len(set(completed)) or len(completed) != status['completed']
            or status['total'] != len(planned)):
        raise ValueError('Wind published count mismatch')
    receipt = dict(method=METHOD, model='Trackformer 1.2', checkpoint_sha256=CHECKPOINT,
        members=1, verified=len(completed), planned=len(planned), resolved_points=resolved_points,
        storms_with_wind=len(storms_with_wind),
        state='complete' if set(completed) == set(planned) else 'partial',
        input_manifest_sha256=input_hash, source_archive_preserved=True,
        wind_averaging_period='unvalidated', verified_at_utc=datetime.now(timezone.utc).isoformat())
    (dest/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
