"""Verify and stage a timing-only re-encode of verified physical model frames."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def finalize(video, core, poster, output):
    if not str(output.resolve()).startswith('/Volumes/D/'):
        raise ValueError('Backups and diagnostics must stay on D')
    output.mkdir(parents=True, exist_ok=True)
    original = ROOT/'evaluation/release_data/mangkhut_video.json'
    meta = json.loads(original.read_text())
    verification = json.loads((core/'verification.json').read_text())
    for key in ('members','storm_id','issue_time_utc','checkpoint_sha256'):
        if verification[key] != meta[key]:
            raise ValueError('Wrong source core identity: '+key)
    if meta['members'] != 50 or sha(core/'common-pressure.npz') != verification['files_sha256']['common-pressure.npz']:
        raise ValueError('Wrong physical member mean')
    mean_path = ROOT/'evaluation/release_data/mangkhut_video.npz'
    if sha(mean_path) != meta['data_sha256']:
        raise ValueError('Original mean/scalar/route archive changed')
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,nb_frames:format=duration','-of','json',str(video)]))
    stream = probe['streams'][0]
    if (stream['codec_name'],stream['width'],stream['height'],int(stream['nb_frames'])) != ('h264',1600,1000,600) or float(probe['format']['duration']) != 20:
        raise ValueError('Wrong video dimensions or duration')
    hashes = subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),'-vf','fps=1','-f','framemd5','-'],text=True)
    states = [line.split(',')[-1].strip() for line in hashes.splitlines() if not line.startswith('#') and line.strip()]
    if len(states) != 20 or len(set(states)) != 20:
        raise ValueError('Repeated or missing six-hour forecast state')
    receipt = {'verified':True,'model':meta['model'],'members':50,'storm_id':meta['storm_id'],
        'issue_time_utc':meta['issue_time_utc'],'leads_hours':list(range(6,121,6)),
        'duration_seconds':20,'encoded_frames':600,'forecast_states':20,'distinct_decoded_states':20,
        'state_duration_seconds':1,'extra_final_hold_seconds':0,'interpolated_forecast_states':False,
        'decoded_state_md5':states,'last_four_states_distinct':len(set(states[-4:]))==4,
        'video_sha256':sha(video),'mean_fields_sha256':sha(mean_path),
        'common_pressure_sha256':sha(core/'common-pressure.npz'),'core_verification_sha256':sha(core/'verification.json')}
    dest = ROOT/'docs/trackformer_1_2_mangkhut.mp4'
    targets = [dest,original,ROOT/'evaluation/trackformer_1_2_mangkhut_video_poster.png',
        Path('/Users/euler/Downloads/trackformer_1_2_mangkhut.mp4'),Path('/Users/euler/Downloads/trackformer_1_2_mangkhut_corrected.mp4')]
    for path in targets:
        if path.exists():
            shutil.copy2(path,output/(path.name+'.before-'+sha(path)[:12]))
    for path in (dest,*targets[-2:]):
        shutil.copy2(video,path)
        if sha(path) != receipt['video_sha256']:
            raise ValueError('Delivery copy hash mismatch')
    shutil.copy2(poster,targets[2])
    public = ROOT/'evaluation/release_data/pressure_core/mangkhut'
    public.mkdir(parents=True,exist_ok=True)
    for name in ('common-pressure.npz','core-reconstruction.json','verification.json'):
        shutil.copy2(core/name,public/name)
    meta['video']={'file':'docs/trackformer_1_2_mangkhut.mp4','duration_seconds':20,
        'forecast_states':20,'encoded_frames':600,'codec':'H.264','encoder':'h264_videotoolbox',
        'sha256':receipt['video_sha256'],'extra_final_hold_seconds':0,'state_duration_seconds':1,
        'playback':'Each genuine six-hour state held for one second. No extra ending hold or interpolated forecast states.'}
    meta['pressure_reconstruction']=verification
    meta['pressure_reconstruction']['public_common_grid_file']='evaluation/release_data/pressure_core/mangkhut/common-pressure.npz'
    meta['field_source']='Original evolving basin plus each actual tapered moving anomaly, geographically registered before 50-member physical averaging. No scalar insertion or route/truth shift.'
    meta['grid_note']='Common WP overview sampled at 0.25 degrees; close-up at 0.1 degrees. Model core information spacing is 20 km; interpolation is not added native information.'
    meta['first_forecast_lead_outside_regional_hours']=None
    original.write_text(json.dumps(meta,indent=2)+'\n')
    (ROOT/'evaluation/release_data/mangkhut_playback_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (output/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='decoded_state_md5'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--core',type=Path,required=True)
    p.add_argument('--poster',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    finalize(args.video,args.core,args.poster,args.output)
