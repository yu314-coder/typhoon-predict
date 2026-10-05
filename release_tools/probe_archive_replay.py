"""Read-only cloud diagnosis of a fixed set of unresolved immutable replays.

No archive writes, altered weights, relaxed tolerances, or alternative inputs.
Each CPU profile is a fresh process; results are diagnostics, never forecasts.
"""
import argparse
import ast
import gzip
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

CASES = (
    'auto-hist-1978226N09162',
    'auto-tick-1996261N08184-19960929T0000',
    'auto-tick-2002057N06156-20020305T0000',
    'auto-tick-2026208N13178-20260802T1200',
)
PROFILES = {
    'original-reader': {},
    'capture-default': {},
    'single-thread': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
    'unfused-attention': {},
    'onednn-disabled': {},
    'avx2': {'ATEN_CPU_CAPABILITY': 'avx2', 'DNNL_MAX_CPU_ISA': 'AVX2', 'MKL_ENABLE_INSTRUCTIONS': 'AVX2'},
    'mkl-compatible': {'ATEN_CPU_CAPABILITY': 'avx2', 'DNNL_MAX_CPU_ISA': 'AVX2', 'MKL_CBWR': 'COMPATIBLE'},
    'sdpa-math': {},
}


def child(root, profile):
    import contextlib
    import numpy as np
    import torch
    import automatic_forecasts as a
    from pressure_core_backfill import CaptureModel, verify_replay

    torch.set_num_threads(1 if profile == 'single-thread' else 2)
    if profile == 'unfused-attention':
        torch.backends.mha.set_fastpath_enabled(False)
    if profile == 'onednn-disabled':
        torch.backends.mkldnn.enabled = False
    config = json.loads((a.MODEL/'manifest.json').read_text())
    contract = config['data_contract']
    model = (a.CoreForecaster if profile == 'original-reader' else CaptureModel)(contract).eval()
    model.load_state_dict(torch.load(root/'weights.pt', map_location='cpu', weights_only=True), strict=True)
    with np.load(root/'inputs.npz', allow_pickle=False) as z:
        inputs = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    reference = json.loads((root/'reference.json').read_text())
    field = json.loads((root/'field.json').read_text())
    context = contextlib.nullcontext()
    if profile == 'sdpa-math':
        from torch.nn.attention import sdpa_kernel, SDPBackend
        torch.backends.mha.set_fastpath_enabled(False)
        context = sdpa_kernel(SDPBackend.MATH)
    with torch.inference_mode(), context:
        state = model.initial(inputs)
        predictions = []
        for _ in range(20):
            state, prediction = model.step(state)
            predictions.append(prediction)
    if profile == 'original-reader':
        # Match the original exporter: no intervening capture operations.
        norm = contract['normalization']
        blocks = [{}]+[{'basin': p['global'][:, 0].numpy()*norm['std'][0]+norm['mean'][0]} for p in predictions]
    else:
        blocks = model.blocks[0]
    result = dict(id=reference['id'], profile=profile, torch=torch.__version__,
                  cpu_capability=torch.backends.cpu.get_cpu_capability(),
                  threads=torch.get_num_threads(), machine=platform.machine(),
                  immutable_archive_modified=False)
    try:
        result['difference'] = verify_replay(reference, inputs, predictions, blocks, field)
        result['matches_original'] = True
    except ValueError as error:
        result['matches_original'] = False
        result['error'] = str(error)
        prefix = 'Replay does not reproduce immutable outputs: '
        if str(error).startswith(prefix):
            result['difference'] = ast.literal_eval(str(error)[len(prefix):])
    print(json.dumps(result, allow_nan=False), flush=True)


def time_diagnosis(row):
    """Inspect coordinate metadata only; nearest indices are NOT model inputs."""
    import numpy as np
    import xarray as xr
    import automatic_forecasts as a
    wanted = a.ns(row['issue_time_utc'])+np.arange(-8, 1)*6*a.HOUR
    year = a.parse(row['issue_time_utc']).year
    report = []
    for variable in ('slp', 'hgt', 'uwnd', 'vwnd'):
        group = 'surface' if variable == 'slp' else 'pressure'
        url = f'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/{group}/{variable}.{year}.nc'
        try:
            with xr.open_dataset(url, engine='netcdf4') as ds:
                actual = ds.time.values.astype('datetime64[ns]').astype('int64')
                nearest = np.array([np.argmin(np.abs(actual-t)) for t in wanted])
                report.append(dict(variable=variable, time_count=len(actual),
                    exact_matches=[bool(t in actual) for t in wanted],
                    diagnostic_nearest_offsets_ns=(actual[nearest]-wanted).tolist(),
                    decoded_times=[str(x) for x in ds.time.values[nearest]],
                    time_encoding={k:str(v) for k,v in ds.time.encoding.items() if k in ('units','calendar','dtype')}))
        except Exception as error:
            report.append(dict(variable=variable, error=str(error)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--archive-sha')
    parser.add_argument('--child', choices=tuple(PROFILES))
    args = parser.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('This bounded CPU diagnosis runs on GitHub only, not beside Mac GPU training')
    if args.child:
        child(args.output, args.child)
        return
    if not args.archive_sha or len(args.archive_sha) != 40 or any(c not in '0123456789abcdef' for c in args.archive_sha):
        raise ValueError('A pinned immutable archive commit is required')
    import numpy as np
    import requests
    import automatic_forecasts as a
    from pressure_core_backfill import sha
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((a.ROOT/'release_tools/history_inputs.json').read_text())
    base = config['base_url']
    plan = json.loads(a.asset(args.cache,'input-manifest.json',base+'/manifest.json',config['manifest_sha256']).read_text())
    manifest = json.loads((a.MODEL/'manifest.json').read_text())
    if manifest['source_checkpoint_sha256'] != a.CHECKPOINT:
        raise ValueError('Wrong release checkpoint')
    for filename, expected in manifest['source_module_sha256'].items():
        if sha(a.MODEL/filename) != expected:
            raise ValueError('Frozen source changed')
    contract = manifest['data_contract']
    weights = a.asset(args.cache,'weights.pt',a.HF+'/models/trackformer_1_2_field/weights.pt',manifest['inference_weights_sha256'])
    geo = np.load(a.asset(args.cache,'geography.npz',base+'/geography.npz',plan['geography']['sha256']), allow_pickle=False)
    rows = {r['id']:r for r in plan['queue']}
    archive = f'https://raw.githubusercontent.com/yu314-coder/typhoon-predict/{args.archive_sha}/data'
    results = []
    for ident in CASES:
        row = rows[ident]
        root = args.output/ident
        root.mkdir(exist_ok=True)
        try:
            reference_bytes = a.get(f'{archive}/forecasts/{ident}.json').content
            reference = json.loads(reference_bytes)
            for suffix in ('json.gz', 'json'):
                try:
                    field_bytes = a.get(f'{archive}/fields/{ident}.{suffix}').content
                    field = json.loads(gzip.decompress(field_bytes) if suffix == 'json.gz' else field_bytes)
                    break
                except requests.HTTPError as error:
                    if error.response.status_code != 404 or suffix == 'json':
                        raise
            if row['atlas'] is not None:
                bundle = plan['bundles'][row.get('bundle_key',str(row['season']))]
                with np.load(a.asset(args.cache,bundle['file'],base+'/'+bundle['file'],bundle['sha256']), allow_pickle=False) as z:
                    times = a.ns(row['issue_time_utc'])+np.arange(-8,1)*6*a.HOUR
                    idx = np.searchsorted(z['time'], times)
                    if np.any(idx >= len(z['time'])) or not np.array_equal(z['time'][idx], times):
                        raise ValueError('Missing exact causal bundled history')
                    weather = np.concatenate([z['slp'][idx,None].astype('float32'),z['q'][idx].astype('float32')*z['scale'][None,:,None,None]+z['offset'][None,:,None,None]], axis=1)
            else:
                weather, times, _, _ = a.remote_history(row, args.cache, contract)
            inputs = a.inputs(weather, times, row, contract, geo)
            identity = a.digest(b''.join(inputs[k].numpy().tobytes() for k in sorted(inputs)))
            if identity != reference['input_tensor_sha256']:
                raise ValueError('Original causal input hash mismatch; no profile may proceed')
            (root/'weights.pt').symlink_to(weights.resolve())
            (root/'reference.json').write_bytes(reference_bytes)
            (root/'field.json').write_text(json.dumps(field, separators=(',',':')))
            np.savez(root/'inputs.npz', **{k:v.numpy() for k,v in inputs.items()})
            for profile, settings in PROFILES.items():
                environment = {**os.environ, **settings}
                process = subprocess.run([sys.executable, __file__, '--output', str(root), '--cache', str(args.cache), '--child', profile],
                    env=environment, text=True, capture_output=True, timeout=120)
                if process.returncode:
                    result = dict(id=ident, profile=profile, error=process.stderr[-2000:], matches_original=False)
                else:
                    result = json.loads(process.stdout.strip().splitlines()[-1])
                results.append(result)
                print(json.dumps(result), flush=True)
        except Exception as error:
            result = dict(id=ident, source_error=str(error), immutable_archive_modified=False)
            if row['atlas'] is None and a.parse(row['issue_time_utc']) <= a.NCEP_LAST:
                result['time_diagnosis'] = time_diagnosis(row)
            results.append(result)
            print(json.dumps(result), flush=True)
        a.write(args.output/'diagnosis.json', dict(archive_sha=args.archive_sha,
            checkpoint_sha256=a.CHECKPOINT, input_manifest_sha256=config['manifest_sha256'],
            cases=list(CASES), profiles=list(PROFILES), results=results, immutable_archive_modified=False))


if __name__ == '__main__':
    main()
