"""Portable launcher for the unchanged archived v173 trainer; no auto-training."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=here / 'data')
    p.add_argument('--warm-start', type=Path, default=here / 'checkpoints/v168_epoch_003.pt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], required=True)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch', type=int, default=10)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    source = here / 'source/v173'
    sys.path.insert(0, str(source))
    sys.argv = [str(source / 'v173_pressure_train.py'), str(a.batch), str(a.workers), '0',
        '--microbatch', str(a.microbatch), '--epochs', str(a.epochs),
        '--device', a.device, '--manifest', str(a.data_root.resolve() / 'manifest.json'),
        '--warm-start', str(a.warm_start.resolve()), '--output', str(a.output.resolve())]
    if a.preflight:
        sys.argv.append('--preflight')
    runpy.run_path(str(source / 'v173_pressure_train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
