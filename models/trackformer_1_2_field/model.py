"""v173: moving pressure core with multiscale environmental attention.

The public track and central pressure are extracted from the evolving core.
Wind is an auxiliary scalar estimate, not a resolved wind field.
"""
import torch
from torch import nn
from torch.nn import functional as F
from baseline_model import CoreForecaster as Baseline, objective as baseline_loss, weighted, base

VERSION = '1.2.73'
ARCHITECTURE = 'v173-moving-core-multiscale-attention'


class EnvironmentalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(44, 64, 1)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0., batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.query = nn.Linear(32, 64)
        self.cross = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.output = nn.Linear(64, 32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state):
        h, w = state['gh'].shape[-2:]
        features = self.project(torch.cat([state['g'], state['gh'], state['gs']], 1))
        tokens = torch.cat([F.avg_pool2d(features, kernel_size=size, stride=size,
                                       ceil_mode=True).flatten(2)
                            for size in (3, 5)], 2).transpose(1, 2)
        tokens = self.encoder(tokens)
        queries = self.query(state['gh'].flatten(2).transpose(1, 2))
        context, _ = self.cross(queries, tokens, tokens, need_weights=False)
        return self.output(context).transpose(1, 2).reshape(-1, 32, h, w)


class CoreForecaster(Baseline):
    def __init__(self, manifest, tiny=False):
        super().__init__(manifest, tiny)
        # Meshgrid views overlap on CPU; checkpoint loading requires owned storage.
        self.east = self.east.clone()
        self.north = self.north.clone()
        self.context = EnvironmentalAttention()

    def step(self, state):
        conditioned = dict(state, gh=state['gh'] + self.context(state))
        next_state, out = super().step(conditioned)
        out['previous_center'] = state['center']
        out['previous_pressure'] = out['pressure'] - out['pressure_tendency']
        return next_state, out


def objective(out, target, inputs, manifest):
    loss, terms = baseline_loss(out, target, inputs, manifest)
    if 'previous_pressure' in target:
        mask = target['pressure_mask'] * target['previous_pressure_mask']
        true_delta = target['pressure'] - target['previous_pressure']
        terms['pressure_change'] = weighted(F.smooth_l1_loss(
            out['pressure_tendency']/10, true_delta/10, reduction='none'), mask)
        loss = loss + .25 * terms['pressure_change']
    if 'previous_center' in target:
        center = target['center']
        scale = torch.stack([torch.full_like(center[:, 0], 111.2),
                             111.2*center[:, 0].deg2rad().cos()], 1)
        predicted = (out['center']-out['previous_center'])*scale
        truth = (center-target['previous_center'])*scale
        mask = target['storm_mask']*target['previous_storm_mask']
        terms['motion_change'] = weighted(F.smooth_l1_loss(
            predicted/100, truth/100, reduction='none'), mask[:, None])
        loss = loss + .1 * terms['motion_change']
    return loss, terms
