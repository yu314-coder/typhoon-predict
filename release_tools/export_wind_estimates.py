"""Append wind diagnostics from already saved physical members; no inference.

Never overwrite original forecasts/fields. The sidecar pins their identifiers,
input hash, member archive SHA and reconstruction-coordinate convention.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models/trackformer_1_2_field'))
from wind_estimation import diagnose_member, summarize_members


def export(folder, output, forecast_id=None):
    folder, output = Path(folder), Path(output)
    f = json.loads((folder/'forecast.json').read_text())
    path = folder/'ensemble-members.npz'
    with path.open('rb') as stream: member_hash = hashlib.file_digest(stream,'sha256').hexdigest()
    verification = json.loads((folder/'verification.json').read_text())
    expected = verification.get('files_sha256',{}).get('ensemble-members.npz')
    if expected != member_hash: raise ValueError('Member archive integrity mismatch')
    with np.load(path,allow_pickle=False) as z:
        n = int(f['members'])
        regional, centers, vmax = z['regional'], z['center'], z['vmax']
        if n != 50 or regional.shape != (50,20,121,121) or len(set(z['seeds'].tolist())) != 50:
            raise ValueError('Fifty actual physical members are required')
        if centers.shape != (50,20,2) or vmax.shape != (50,20):
            raise ValueError('Incomplete member route/wind outputs')
        # This is the exact released wrapper's fixed, quarter-degree issue anchor.
        anchor = np.round(np.array([f['route'][0]['lat'],f['route'][0]['lon']])*4)/4
        lat, lon = anchor[0]+np.linspace(15,-15,121), anchor[1]+np.linspace(-15,15,121)
        summaries = []
        for lead in range(20):
            if not np.allclose(centers[:,lead].mean(0),[f['route'][lead+1]['lat'],f['route'][lead+1]['lon']],atol=1e-4):
                raise ValueError('Member route does not match saved forecast')
            if not np.isclose(vmax[:,lead].mean(),f['route'][lead+1]['wind_kt_auxiliary'],atol=1e-4):
                raise ValueError('Member wind does not match saved forecast')
            members=[]
            for i in range(n):
                p=regional[i,lead].astype(float)
                # Preserve the true basin coverage, not border-filled regional pixels.
                p=np.where((lat[:,None]>=0)&(lat[:,None]<=60)&(lon[None,:]>=100)&(lon[None,:]<=180),p,np.nan)
                members.append(diagnose_member(p,lat,lon,centers[i,lead],vmax[i,lead],
                    source_spacing_km=20,sampling_radius_limit_km=300,
                    source='saved member-specific 0.25-degree moving-core composite; conservative 300-km supported diagnostic radius'))
            summaries.append(dict(lead_hours=(lead+1)*6,valid_time_utc=f['route'][lead+1]['valid_time_utc'],
                                  wind_estimation=summarize_members(members)))
    value = dict(schema_version='1.0',forecast_id=forecast_id or f.get('id'),members=n,
        issue_time_utc=f['issue_time_utc'],checkpoint_sha256=f['source_checkpoint_sha256'],
        input_tensor_sha256=f['input_tensor_sha256'],member_archive_sha256=member_hash,
        native_detail_available=False,regional_coordinate_convention='quarter-degree issue anchor +/-15 degrees',
        source_note='Read-only diagnostics of saved physical members; original forecast/pressure and weights unchanged. Legacy composites lack saved core origins: the released core spans +/-640 km and associates a centre within 300 km of its origin, so rings are conservatively limited to 300 km about the member centre. No radii beyond that supported footprint are inferred. No new training, inference or future observations.',
        points=summaries)
    if not value['forecast_id']: raise ValueError('Exact public issue id required')
    if output.exists(): raise FileExistsError('Preserve the existing diagnostic sidecar')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(value,separators=(',',':'),allow_nan=False))
    print(json.dumps({'forecast_id':value['forecast_id'],'members':n,'leads':len(summaries),
                     'first':summaries[0]['wind_estimation']['estimates']}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--forecast-id');a=ap.parse_args();export(a.folder,a.output,a.forecast_id)
