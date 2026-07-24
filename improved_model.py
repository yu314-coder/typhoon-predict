"""StormFusion-MT v2 model cell — improved architecture (drop-in replacement).

Changes vs the original (motivation: better inductive bias on a small dataset,
NOT more raw capacity, which would overfit 883 windows):

1. FrameEncoder keeps a 3x3 grid of spatial tokens (AdaptiveAvgPool2d(3)) instead
   of collapsing each 65x65 ERA5 patch to one vector with AdaptiveAvgPool2d(1).
   The transformer can now see storm structure/asymmetry -- essential for the
   per-quadrant wind-radius targets. A learned spatial position embedding tags
   each grid cell.
2. Sinusoidal lead-time encoding added to the learned lead queries so the decoder
   knows lead=6h is adjacent to lead=12h (ordinal structure).
3. Dropout raised 0.1 -> 0.2 for regularization on the small dataset.

d_model stays 192, so parameter count barely changes -- the gain is structural.
"""
import math
import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    def __init__(self, in_channels, d_model, compact=False, grid=3):
        super().__init__()
        widths = (32, 64, d_model) if compact else (64, 128, d_model)
        self.grid = grid
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, padding=1),
            nn.GroupNorm(8, widths[0]),
            nn.GELU(),
            nn.Conv2d(widths[0], widths[1], 3, stride=2, padding=1),
            nn.GroupNorm(8, widths[1]),
            nn.GELU(),
            nn.Conv2d(widths[1], widths[2], 3, stride=2, padding=1),
            nn.GroupNorm(8, widths[2]),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(grid),  # keep a grid x grid map instead of 1x1
        )
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, grid * grid, d_model))

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        z = self.net(x.reshape(batch * steps, channels, height, width))
        z = z.flatten(2).transpose(1, 2)                      # (b*steps, grid^2, d)
        z = z.reshape(batch, steps, self.grid * self.grid, -1)
        return z + self.spatial_pos                           # (b, steps, grid^2, d)


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
        compact = tier == "compact"
        d_model = 128 if compact else 192
        heads = 4 if compact else 6
        ffn = 256 if compact else 512
        context_depth = 1 if compact else 2
        decoder_depth = 2 if compact else 4
        dropout = 0.2

        self.d_model = d_model
        self.lead_count = lead_count
        self.inner_encoder = FrameEncoder(26, d_model, compact=compact)
        self.outer_encoder = FrameEncoder(14, d_model, compact=compact)
        self.track_proj = nn.Linear(40, d_model)
        self.env_proj = nn.Linear(10, d_model)
        # time embeddings, broadcast across the spatial grid tokens
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
        inner_tokens = self.inner_encoder(inner) + self.inner_time     # (b,4,9,d)
        outer_tokens = self.outer_encoder(outer) + self.outer_time     # (b,4,9,d)
        inner_tokens = inner_tokens.flatten(1, 2)                      # (b,36,d)
        outer_tokens = outer_tokens.flatten(1, 2)                      # (b,36,d)
        track_tokens = self.track_proj(track) + self.track_time        # (b,9,d)
        env_tokens = self.env_proj(env) + self.env_time                # (b,4,d)
        memory = torch.cat([inner_tokens, outer_tokens, track_tokens, env_tokens], dim=1)
        memory = self.context(memory)
        queries = (self.lead_queries + self.lead_pos.unsqueeze(0)).expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        state = self.state_head(decoded)
        log_scale = self.log_scale_head(decoded).clamp(-5.0, 3.0)
        return state, log_scale


if __name__ == "__main__":
    torch.manual_seed(0)
    m = StormFusionMT("recommended", lead_count=20)
    B = 2
    inner = torch.randn(B, 4, 26, 65, 65)
    outer = torch.randn(B, 4, 14, 65, 65)
    track = torch.randn(B, 9, 40)
    env = torch.randn(B, 4, 10)
    state, log_scale = m(inner, outer, track, env)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("state shape     :", tuple(state.shape), "(expected (2, 20, 17))")
    print("log_scale shape :", tuple(log_scale.shape), "(expected (2, 20, 17))")
    print("memory tokens   : 36 inner + 36 outer + 9 track + 4 env = 85")
    print("finite outputs  :", torch.isfinite(state).all().item(), torch.isfinite(log_scale).all().item())
    print("trainable params:", f"{n:,}", "(original was 3,311,842)")
    # quick backward sanity
    state.sum().backward()
    print("backward OK     :", m.inner_encoder.spatial_pos.grad is not None)
