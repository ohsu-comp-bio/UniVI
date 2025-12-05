# univi/models/encoders.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from .mlp import build_mlp


@dataclass
class EncoderConfig:
    input_dim: int
    hidden_dims: List[int]
    latent_dim: int
    dropout: float = 0.1
    batchnorm: bool = True


class GaussianEncoder(nn.Module):
    """x -> (mu, logvar) for a diagonal Gaussian."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.net = build_mlp(
            in_dim=cfg.input_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=2 * cfg.latent_dim,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

