"""CPU-only execution check of real TRAIN inputs; zero optimizer updates."""
import argparse
import json
from pathlib import Path
import sys


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=here / 'data')
    p.add_argument('--warm-start', type=Path, default=here / 'checkpoints/v168_epoch_003.pt')
    a = p.parse_args()
    sys.path.insert(0, str(here / 'source/v173'))
    import torch
    from model import CoreForecaster, objective, base
    from v173_pressure_train import warm_start, batches
    torch.set_num_threads(2)
    torch.manual_seed(168)
    base.EvolutionNet.block = lambda self, module, x: module(x)
    m = base.load_reuse(a.data_root / 'manifest.json')
    model = CoreForecaster(m).cpu()
    warm_start(model, a.warm_start, m)
    data = base.ReuseDataset(m, 'train')
    samples = [next(i for i, r in enumerate(data.rows) if (r['patch'] >= 0) == native and r['issue_mask'][1])
               for native in (False, True)]
    results = []
    for index in samples:
        x, y = data[index]
        x = {k: v[None] for k, v in x.items()}
        model.train()
        model.zero_grad(set_to_none=True)
        state = model.initial(x)
        state, prediction = model.step(state)
        target = {k: v[0:1] for k, v in y.items()}
        target.update(previous_pressure=x['issue_intensity'][:, 1],
                      previous_pressure_mask=x['issue_mask'][:, 1],
                      previous_center=x['center'],
                      previous_storm_mask=torch.ones_like(target['storm_mask']))
        loss, terms = objective(prediction, target, x, m)
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.grad is not None)
        assert model.context.output.weight.grad.abs().sum() > 0
        grid = model.core_grid(prediction['center'][:, 0, None, None],
                               prediction['center'][:, 1, None, None], state['origin'])
        pressure = base.sample_field(prediction['core'] * model.std[:, :1] + model.mean[:, :1], grid)[:, 0, 0, 0]
        assert torch.equal(pressure, prediction['pressure'])
        try:
            model.initial(dict(x, future_pressure=target['regional']))
        except ValueError:
            pass
        else:
            raise AssertionError('Future target accepted as input')
        results.append(dict(native=bool(x['detail_available'].item()), finite_backward=True,
                            attention_gradient_nonzero=True, field_pressure_exact=True,
                            future_input_rejected=True))
    indices = base.training_indices(data.rows, 1)
    assert set(indices) == set(range(len(data)))
    for parts, count in batches(indices, data.rows, 10, 4):
        assert sum(map(len, parts)) == count
        assert all(len({data.rows[i]['patch'] >= 0 for i in part}) == 1 for part in parts)
    print(json.dumps(dict(status='passed', device='cpu', optimizer_updates=0,
        full_training_rerun=False, samples=results, sampler_all_train_rows=True,
        counts=m['counts'], native_counts=m['patch_counts']), indent=2))


if __name__ == '__main__':
    main()
