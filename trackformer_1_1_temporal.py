"""Inference-only temporal branch used by Trackformer1.1.

The training implementation is intentionally not part of the public model
package. This module contains only the architecture needed to load the
frozen temporal expert checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from trackformer_1_1_intensity import StructureSpatialExpert


class TemporalStructureSpatial(StructureSpatialExpert):
    """Spatial expert augmented with same-storm t-12/t-24 analysis fields."""

    def __init__(self, width: int, layers: int, heads: int):
        super().__init__(width, layers, heads, structure_residual=True)
        self.history_encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.history_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.history_norm = nn.LayerNorm(width)
        self.history_pos = nn.Parameter(torch.randn(1, 16, width) * 0.02)
        self.history_out = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.history_out.weight)
        nn.init.zeros_(self.history_out.bias)

    def forward(
        self,
        track: torch.Tensor,
        field: torch.Tensor,
        current: torch.Tensor,
        available: torch.Tensor,
        current_structure: torch.Tensor | None = None,
        structure_available: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
        history_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        track_tokens = self.track_encoder(
            self.track_proj(track[:, :, self._thermo_cols]) + self.track_time
        )
        field_tokens = self.field_pool(self.field_encoder(field)).flatten(2).transpose(1, 2)
        field_tokens = self.field_norm(field_tokens + self.field_pos)
        if history is not None:
            if history_available is None:
                history_available = history.new_ones((history.shape[0], 2))
            flags = history_available.view(-1, 2, 1, 1).expand(-1, 2, 17, 17)
            history_tokens = self.history_pool(
                self.history_encoder(torch.cat([history, flags], dim=1))
            )
            history_tokens = history_tokens + self.history_pos.permute(0, 2, 1).reshape(
                1, history_tokens.shape[1], 4, 4
            )
            history_tokens = self.history_norm(history_tokens.flatten(2).transpose(1, 2))
            history_tokens = history_tokens + self.history_out(
                history_tokens.transpose(1, 2).reshape(
                    history_tokens.shape[0], history_tokens.shape[2], 4, 4
                )
            ).flatten(2).transpose(1, 2)
            field_tokens = field_tokens + history_tokens
        memory = torch.cat([track_tokens, field_tokens], dim=1)
        query = (self.query + self.lead_time).expand(track.shape[0], -1, -1)
        hidden = self.decoder(query, memory)
        state = self.state(hidden).clone()
        state[:, :, :2] = state[:, :, :2] + (current * available)[:, None, :]
        if current_structure is None or structure_available is None:
            raise ValueError("Trackformer1.1 temporal expert requires current structure tensors")
        state[:, :, 2:] = (
            state[:, :, 2:]
            + current_structure[:, None, :] * structure_available[:, None, :]
        )
        return state, self.log_scale(hidden)

    @property
    def _thermo_cols(self):
        return (
            [4, 5, 6, 7]
            + list(range(8, 20))
            + list(range(24, 40))
            + [44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
        )


__all__ = ["TemporalStructureSpatial"]
