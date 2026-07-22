"""StormFusion-MT v3 — larger model that (a) uses the GPU properly and
(b) is a strict superset of the original architecture.

vs original:
- FrameEncoder keeps BOTH the original global-average-pool summary token (the
  "old feature") AND a 3x3 grid of local spatial tokens -> 1 global + 9 local
  per frame. Nothing from the old model is removed.
- Wider conv stack (96/192/d_model) and larger transformer (d_model 384, 8 heads,
  ffn 1024, context depth 3, decoder depth 6) so the A100 is actually utilized.
- Sinusoidal lead-time encoding on the queries.
- Dropout 0.2 for regularization; early stopping + weight decay guard overfitting.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEncoder(nn.Module):
    def __init__(self, in_channels, d_model, grid=3):
        super().__init__()
        self.grid = grid
        w0, w1 = 96, 192
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, w0, 3, padding=1),
            nn.GroupNorm(8, w0), nn.GELU(),
            nn.Conv2d(w0, w1, 3, stride=2, padding=1),
            nn.GroupNorm(8, w1), nn.GELU(),
            nn.Conv2d(w1, d_model, 3, stride=2, padding=1),
            nn.GroupNorm(8, d_model), nn.GELU(),
        )
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, grid * grid, d_model))
        self.global_pos = nn.Parameter(torch.zeros(1, 1, 1, d_model))

    def forward(self, x):
        b, s, c, h, w = x.shape
        z = self.net(x.reshape(b * s, c, h, w))                      # (b*s, d, H, W)
        local = F.adaptive_avg_pool2d(z, self.grid).flatten(2).transpose(1, 2)  # (b*s, grid^2, d)
        glob = F.adaptive_avg_pool2d(z, 1).flatten(2).transpose(1, 2)           # (b*s, 1, d)  <- original feature
        local = local.reshape(b, s, self.grid * self.grid, -1) + self.spatial_pos
        glob = glob.reshape(b, s, 1, -1) + self.global_pos
        return torch.cat([glob, local], dim=2)                       # (b, s, 1+grid^2, d)


def sinusoidal_encoding(length, d_model, device=None):
    pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
                    * (-math.log(10000.0) / d_model))
    enc = torch.zeros(length, d_model, device=device)
    enc[:, 0::2] = torch.sin(pos * div)
    enc[:, 1::2] = torch.cos(pos * div)
    return enc


class StormFusionMT(nn.Module):
    def __init__(self, tier="recommended", lead_count=20):
        super().__init__()
        d_model = 384
        heads = 8
        ffn = 1024
        context_depth = 3
        decoder_depth = 6
        dropout = 0.2

        self.d_model = d_model
        self.lead_count = lead_count
        self.inner_encoder = FrameEncoder(26, d_model)
        self.outer_encoder = FrameEncoder(14, d_model)
        self.track_proj = nn.Linear(40, d_model)
        self.env_proj = nn.Linear(10, d_model)
        self.inner_time = nn.Parameter(torch.zeros(1, 4, 1, d_model))
        self.outer_time = nn.Parameter(torch.zeros(1, 4, 1, d_model))
        self.track_time = nn.Parameter(torch.zeros(1, 9, d_model))
        self.env_time = nn.Parameter(torch.zeros(1, 4, d_model))

        context_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ffn,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.context = nn.TransformerEncoder(context_layer, num_layers=context_depth)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ffn,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)
        self.lead_queries = nn.Parameter(torch.randn(1, lead_count, d_model) * 0.02)
        self.register_buffer("lead_pos", sinusoidal_encoding(lead_count, d_model))
        self.state_head = nn.Linear(d_model, 17)
        self.log_scale_head = nn.Linear(d_model, 17)

    def forward(self, inner, outer, track, env):
        batch = inner.shape[0]
        inner_tokens = (self.inner_encoder(inner) + self.inner_time).flatten(1, 2)  # (b, 4*10, d)
        outer_tokens = (self.outer_encoder(outer) + self.outer_time).flatten(1, 2)  # (b, 4*10, d)
        track_tokens = self.track_proj(track) + self.track_time
        env_tokens = self.env_proj(env) + self.env_time
        memory = torch.cat([inner_tokens, outer_tokens, track_tokens, env_tokens], dim=1)
        memory = self.context(memory)
        queries = (self.lead_queries + self.lead_pos.unsqueeze(0)).expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        state = self.state_head(decoded)
        log_scale = self.log_scale_head(decoded).clamp(-5.0, 3.0)
        return state, log_scale
