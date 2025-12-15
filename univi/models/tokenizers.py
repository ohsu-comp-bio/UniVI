# univi/models/tokenizers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Literal

import torch
from torch import nn

from ..config import TokenizerConfig


class Tokenizer(nn.Module):
    """
    Base tokenizer interface.

    forward(x) returns:
      tokens: (B, T, D_in)
      key_padding_mask: Optional[(B, T)] where True means "PAD / ignore"
    """
    def __init__(self):
        super().__init__()

    @property
    def d_in(self) -> int:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


class TopKScalarTokenizer(Tokenizer):
    """(B,F) -> (B,K,1) using top-k by absolute value per cell."""
    def __init__(self, n_tokens: int, add_cls_token: bool = False):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.add_cls_token = bool(add_cls_token)

    @property
    def d_in(self) -> int:
        return 1

    def forward(self, x: torch.Tensor):
        # x: (B,F)
        B, F = x.shape
        K = min(self.n_tokens, F)

        # pick topK by abs value
        _, idx = torch.topk(x.abs(), k=K, dim=1, largest=True, sorted=True)
        vals = torch.gather(x, 1, idx)  # (B,K)
        tokens = vals.unsqueeze(-1)     # (B,K,1)

        key_padding_mask = None

        if self.add_cls_token:
            cls = torch.zeros((B, 1, 1), device=x.device, dtype=x.dtype)
            tokens = torch.cat([cls, tokens], dim=1)

        return tokens, key_padding_mask


class TopKChannelsTokenizer(Tokenizer):
    """
    (B,F) -> (B,K,C) multi-dim tokens, where channels can include:
      - value: raw x_i
      - rank: rank of that feature within the cell (0..1)
      - dropout: 1 if x_i == 0 else 0  (a useful "missingness" channel)
    """
    def __init__(
        self,
        n_tokens: int,
        channels: Sequence[Literal["value", "rank", "dropout"]] = ("value", "rank", "dropout"),
        add_cls_token: bool = False,
    ):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.channels = tuple(channels)
        self.add_cls_token = bool(add_cls_token)

        if len(self.channels) == 0:
            raise ValueError("TopKChannelsTokenizer requires at least one channel.")
        for c in self.channels:
            if c not in ("value", "rank", "dropout"):
                raise ValueError(f"Unknown channel {c!r}. Allowed: value, rank, dropout")

    @property
    def d_in(self) -> int:
        return len(self.channels)

    def forward(self, x: torch.Tensor):
        B, F = x.shape
        K = min(self.n_tokens, F)

        # topK by abs value
        _, idx = torch.topk(x.abs(), k=K, dim=1, largest=True, sorted=True)
        vals = torch.gather(x, 1, idx)  # (B,K)

        chans = []
        for c in self.channels:
            if c == "value":
                chans.append(vals)
            elif c == "dropout":
                chans.append((vals == 0).to(vals.dtype))
            elif c == "rank":
                # rank within the selected K features (0..1), descending by abs
                # since idx sorted by abs desc, rank is just 0..K-1
                r = torch.arange(K, device=x.device, dtype=x.dtype).view(1, K).expand(B, K)
                chans.append(r / max(K - 1, 1))
            else:
                raise RuntimeError("unreachable")

        tokens = torch.stack(chans, dim=-1)  # (B,K,C)
        key_padding_mask = None

        if self.add_cls_token:
            cls = torch.zeros((B, 1, tokens.size(-1)), device=x.device, dtype=x.dtype)
            tokens = torch.cat([cls, tokens], dim=1)

        return tokens, key_padding_mask


class PatchTokenizer(Tokenizer):
    """
    Split features into patches:

      (B,F) -> (B,T,patch_size)  where T = ceil(F/patch_size)

    Optionally learn a nonlinear projection:
      patch_vec (patch_size) -> patch_proj_dim

    That gives you the "multi-dim tokens" vibe with D_in > 1 (often much > 1).
    """
    def __init__(
        self,
        patch_size: int,
        add_cls_token: bool = False,
        patch_proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.add_cls_token = bool(add_cls_token)
        self.patch_proj_dim = int(patch_proj_dim) if patch_proj_dim is not None else None

        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")

        if self.patch_proj_dim is not None:
            self.proj = nn.Sequential(
                nn.LayerNorm(self.patch_size),
                nn.Linear(self.patch_size, self.patch_proj_dim),
                nn.GELU(),
                nn.Linear(self.patch_proj_dim, self.patch_proj_dim),
            )
        else:
            self.proj = None

    @property
    def d_in(self) -> int:
        return self.patch_proj_dim if self.patch_proj_dim is not None else self.patch_size

    def forward(self, x: torch.Tensor):
        # x: (B,F)
        B, F = x.shape
        P = self.patch_size
        T = (F + P - 1) // P
        pad = T * P - F

        if pad > 0:
            x_pad = torch.cat([x, torch.zeros((B, pad), device=x.device, dtype=x.dtype)], dim=1)
        else:
            x_pad = x

        patches = x_pad.view(B, T, P)  # (B,T,P)

        # mask padded tokens only if an entire patch is padding (this happens only if F==0)
        # Here we instead mask padded *features* implicitly by zeros; no padding tokens needed.
        key_padding_mask = None

        if self.proj is not None:
            patches = self.proj(patches)  # (B,T,patch_proj_dim)

        if self.add_cls_token:
            cls = torch.zeros((B, 1, patches.size(-1)), device=x.device, dtype=x.dtype)
            patches = torch.cat([cls, patches], dim=1)

        return patches, key_padding_mask


def build_tokenizer(cfg: TokenizerConfig) -> Tokenizer:
    mode = (cfg.mode or "").lower().strip()
    if mode == "topk_scalar":
        return TopKScalarTokenizer(n_tokens=cfg.n_tokens, add_cls_token=cfg.add_cls_token)
    if mode == "topk_channels":
        return TopKChannelsTokenizer(n_tokens=cfg.n_tokens, channels=cfg.channels, add_cls_token=cfg.add_cls_token)
    if mode == "patch":
        return PatchTokenizer(
            patch_size=cfg.patch_size,
            add_cls_token=cfg.add_cls_token,
            patch_proj_dim=cfg.patch_proj_dim,
        )
    raise ValueError(f"Unknown tokenizer mode {cfg.mode!r}")


