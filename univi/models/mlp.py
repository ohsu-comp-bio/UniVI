# univi/models/mlp.py

from __future__ import annotations
from typing import List
from torch import nn


def build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    activation: nn.Module = nn.ReLU(),
    dropout: float = 0.0,
    batchnorm: bool = True,
) -> nn.Sequential:
    """
    Generic MLP builder: [Linear -> BN -> Act -> Dropout]* + final Linear.
    """
    layers = []
    last_dim = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        if batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last_dim = h
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)
