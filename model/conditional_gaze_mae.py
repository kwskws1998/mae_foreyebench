"""Text-conditioned masked autoencoder for aligned gaze features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ConditionalGazeMAEConfig:
    gaze_dim: int
    text_dim: int
    reconstruction_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    mask_ratio: float = 0.3
    valid_gaze_eps: float = 1e-8


class ConditionalGazeMAEOutput(NamedTuple):
    reconstruction_loss: torch.Tensor
    alignment_loss: torch.Tensor
    trial_representation: torch.Tensor
    masked_positions: torch.Tensor
    reconstructed_gaze: torch.Tensor


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


class ConditionalGazeMAE(nn.Module):
    """Encode observed gaze with text context and reconstruct masked gaze features."""

    def __init__(self, cfg: ConditionalGazeMAEConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads')
        if cfg.reconstruction_dim <= 0 or cfg.reconstruction_dim > cfg.gaze_dim:
            raise ValueError('reconstruction_dim must be in [1, gaze_dim]')

        self.cfg = cfg
        self.gaze_projection = nn.Linear(cfg.gaze_dim, cfg.hidden_dim)
        self.text_projection = nn.Linear(cfg.text_dim, cfg.hidden_dim)
        self.input_norm = nn.LayerNorm(cfg.hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(cfg.hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.reconstruction_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def _valid_positions(
        self,
        gaze_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        gaze_nonzero = gaze_features.abs().sum(dim=-1) > self.cfg.valid_gaze_eps
        return attention_mask.bool() & gaze_nonzero

    def _sample_mask(
        self,
        valid_positions: torch.Tensor,
        mask_ratio: float,
    ) -> torch.Tensor:
        if mask_ratio <= 0.0:
            return torch.zeros_like(valid_positions, dtype=torch.bool)

        random_mask = torch.rand(
            valid_positions.shape,
            device=valid_positions.device,
        ) < mask_ratio
        masked_positions = random_mask & valid_positions

        needs_mask = valid_positions.any(dim=1) & ~masked_positions.any(dim=1)
        if needs_mask.any():
            first_valid = valid_positions.float().argmax(dim=1)
            batch_indices = torch.arange(
                valid_positions.size(0),
                device=valid_positions.device,
            )
            masked_positions[batch_indices[needs_mask], first_valid[needs_mask]] = True

        return masked_positions

    def forward(
        self,
        gaze_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mask_ratio: float | None = None,
    ) -> ConditionalGazeMAEOutput:
        if gaze_features.ndim != 3 or text_features.ndim != 3:
            raise ValueError('gaze_features and text_features must be rank-3 tensors')
        if gaze_features.shape[:2] != text_features.shape[:2]:
            raise ValueError(
                'gaze_features and text_features must share batch and sequence dimensions'
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                gaze_features.shape[:2],
                dtype=torch.bool,
                device=gaze_features.device,
            )
        else:
            attention_mask = attention_mask.bool()

        valid_positions = self._valid_positions(gaze_features, attention_mask)
        effective_mask_ratio = self.cfg.mask_ratio if mask_ratio is None else mask_ratio
        masked_positions = self._sample_mask(valid_positions, effective_mask_ratio)

        gaze_embeds = self.gaze_projection(gaze_features)
        mask_token = self.mask_token.view(1, 1, -1).to(dtype=gaze_embeds.dtype)
        gaze_embeds = torch.where(
            masked_positions.unsqueeze(-1),
            mask_token,
            gaze_embeds,
        )

        conditioned = self.input_norm(gaze_embeds + self.text_projection(text_features))
        encoded = self.encoder(conditioned, src_key_padding_mask=~attention_mask)
        reconstructed = self.decoder(encoded)
        trial_representation = masked_mean(encoded, valid_positions)

        zero = gaze_features.new_tensor(0.0)
        if masked_positions.any():
            target = gaze_features[..., : self.cfg.reconstruction_dim]
            reconstruction_loss = F.smooth_l1_loss(
                reconstructed[masked_positions],
                target[masked_positions],
            )
        else:
            reconstruction_loss = zero

        text_trial = masked_mean(self.text_projection(text_features), valid_positions)
        if valid_positions.any():
            alignment_loss = 1.0 - F.cosine_similarity(
                trial_representation,
                text_trial,
                dim=-1,
            ).mean()
        else:
            alignment_loss = zero

        return ConditionalGazeMAEOutput(
            reconstruction_loss=reconstruction_loss,
            alignment_loss=alignment_loss,
            trial_representation=trial_representation,
            masked_positions=masked_positions,
            reconstructed_gaze=reconstructed,
        )
