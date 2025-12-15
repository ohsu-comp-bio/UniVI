# univi/models/transformer.py
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TransformerConfig


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.attn_dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout_attn = nn.Dropout(cfg.dropout)
        self.activation = F.gelu if cfg.activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout_attn(attn_out))
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(y))
        return x


class TransformerEncoder(nn.Module):
    """
    Generic encoder: takes token embeddings (B, T, D_in),
    projects to d_model, runs transformer, pools to output (B, d_out).
    """
    def __init__(
        self,
        cfg: TransformerConfig,
        d_in: int,
        d_out: int,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(int(d_in), int(cfg.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(int(cfg.num_layers))])

        self.pos_emb = None
        if use_positional_encoding and cfg.max_tokens is not None:
            self.pos_emb = nn.Parameter(torch.zeros(1, int(cfg.max_tokens), int(cfg.d_model)))

        self.out_proj = nn.Linear(int(cfg.d_model), int(d_out))

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D_in)
        h = self.input_proj(x)

        if self.pos_emb is not None:
            h = h + self.pos_emb[:, : h.size(1), :]

        for blk in self.blocks:
            h = blk(h, key_padding_mask=key_padding_mask)

        if self.cfg.pooling == "cls":
            pooled = h[:, 0]
        else:
            if key_padding_mask is not None:
                mask = (~key_padding_mask).float().unsqueeze(-1)  # (B,T,1)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            else:
                pooled = h.mean(dim=1)

        return self.out_proj(pooled)


