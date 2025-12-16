# univi/models/transformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    dim_feedforward: int = 4096
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    pooling: Literal["cls", "mean"] = "mean"
    max_tokens: Optional[int] = None


def _act(name: str):
    name = str(name).lower().strip()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unknown activation: {name!r}")


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=int(cfg.num_heads),
            dropout=float(cfg.attn_dropout),
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(float(cfg.dropout))
        self.ln1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, int(cfg.dim_feedforward)),
            nn.GELU() if str(cfg.activation).lower().strip() == "gelu" else nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.dim_feedforward), d_model),
        )
        self.ff_drop = nn.Dropout(float(cfg.dropout))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B,T,D)
        attn_out, attn_w = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=bool(return_attn),
            average_attn_weights=False,  # keep per-head weights when returning
        )
        x = self.ln1(x + self.attn_drop(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.ff_drop(ff_out))

        if return_attn:
            # attn_w: (B, H, T, T)
            return x, attn_w
        return x


class TransformerEncoder(nn.Module):
    """
    Generic encoder:
      tokens (B,T,D_in) -> proj -> blocks -> pool -> out_proj -> (B,d_out)

    If return_attn=True:
      returns (pooled_out, [attn_layer0, attn_layer1, ...])
    """
    def __init__(
        self,
        *,
        cfg: TransformerConfig,
        d_in: int,
        d_out: int,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.use_positional_encoding = bool(use_positional_encoding)

        d_model = int(cfg.d_model)
        self.input_proj = nn.Identity() if int(d_in) == d_model else nn.Linear(int(d_in), d_model, bias=True)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(int(cfg.num_layers))])
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.out_proj = nn.Linear(d_model, int(d_out), bias=True)

        self.pooling = str(cfg.pooling).lower().strip()
        if self.pooling not in ("cls", "mean"):
            raise ValueError(f"Unknown pooling={cfg.pooling!r}")

        # learned positional embeddings (optional)
        self.pos_emb: Optional[nn.Parameter] = None
        if self.use_positional_encoding:
            if cfg.max_tokens is None:
                raise ValueError("use_positional_encoding=True requires cfg.max_tokens to be set.")
            max_tokens = int(cfg.max_tokens)
            self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, d_model))

    def _pool(
        self,
        x: torch.Tensor,  # (B,T,D)
        *,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return x[:, 0, :]

        # mean pooling (mask-aware)
        if key_padding_mask is None:
            return x.mean(dim=1)

        keep = (~key_padding_mask).to(dtype=x.dtype)  # (B,T)
        denom = keep.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * keep.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # tokens: (B,T,D_in)
        x = self.input_proj(tokens)

        if self.use_positional_encoding:
            assert self.pos_emb is not None
            T = x.shape[1]
            if T > self.pos_emb.shape[1]:
                raise ValueError(f"Sequence length T={T} exceeds max_tokens={self.pos_emb.shape[1]}.")
            x = x + self.pos_emb[:, :T, :]

        x = self.dropout(x)

        attn_all: List[torch.Tensor] = []
        for blk in self.blocks:
            if return_attn:
                x, aw = blk(x, key_padding_mask=key_padding_mask, return_attn=True)
                attn_all.append(aw)
            else:
                x = blk(x, key_padding_mask=key_padding_mask, return_attn=False)

        pooled = self._pool(x, key_padding_mask=key_padding_mask)
        out = self.out_proj(pooled)

        if return_attn:
            return out, attn_all
        return out

