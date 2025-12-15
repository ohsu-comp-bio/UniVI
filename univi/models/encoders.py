# univi/models/encoders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..config import UniVIConfig, ModalityConfig, TokenizerConfig, TransformerConfig
from .mlp import build_mlp
from .transformer import TransformerEncoder


# =============================================================================
# Classic MLP Gaussian encoder
# =============================================================================

@dataclass
class EncoderConfig:
    input_dim: int
    hidden_dims: List[int]
    latent_dim: int
    dropout: float = 0.1
    batchnorm: bool = True


class GaussianEncoder(nn.Module):
    """Base: x -> (mu, logvar) for a diagonal Gaussian."""
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MLPGaussianEncoder(GaussianEncoder):
    """MLP encoder: x -> (mu, logvar) directly."""
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.net = build_mlp(
            in_dim=int(cfg.input_dim),
            hidden_dims=list(cfg.hidden_dims),
            out_dim=2 * int(cfg.latent_dim),
            dropout=float(cfg.dropout),
            batchnorm=bool(cfg.batchnorm),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar


# =============================================================================
# Vector -> tokens (TokenizerConfig)
# =============================================================================

class _VectorToTokens(nn.Module):
    """
    Turn a dense vector x (B, F) into tokens (B, T, D_in) + optional key_padding_mask.

    TokenizerConfig modes
    ---------------------
    - topk_scalar:
        Select top-k features per cell (by value), output tokens (B, K, 1)
    - topk_channels:
        Select top-k features per cell, output tokens (B, K, C) where C=len(channels)
        channels in {"value","rank","dropout"}:
          * value: raw x at selected indices
          * rank:  rank01 among selected K tokens (0..1), per cell
          * dropout: indicator (value==0)
    - patch:
        Split contiguous features into patches of size P:
          tokens (B, T, P) or (B, T, patch_proj_dim) if patch_proj_dim is set

    add_cls_token:
        If True, prepend a learned CLS token embedding:
          tokens -> (B, T+1, D_in); mask -> prepend False (not masked)

    Notes
    -----
    - Assumes dense float input (B,F). If your modality data is sparse, densify upstream.
    """
    def __init__(self, *, input_dim: int, tok: TokenizerConfig):
        super().__init__()
        self.input_dim = int(input_dim)
        self.tok = tok

        mode = str(tok.mode or "topk_scalar").lower().strip()
        if mode not in ("topk_scalar", "topk_channels", "patch"):
            raise ValueError(f"Unknown tokenizer mode {tok.mode!r}")
        self.mode = mode

        self.add_cls_token = bool(getattr(tok, "add_cls_token", False))

        # Determine token count and token embedding dimension (d_in)
        self.patch_proj: Optional[nn.Module] = None
        self.patch_size: Optional[int] = None

        if self.mode == "topk_scalar":
            self.n_tokens = int(tok.n_tokens)
            if self.n_tokens <= 0:
                raise ValueError("tokenizer.n_tokens must be > 0 for topk_scalar")
            self.channels: Sequence[str] = ("value",)
            self.d_in = 1

        elif self.mode == "topk_channels":
            self.n_tokens = int(tok.n_tokens)
            if self.n_tokens <= 0:
                raise ValueError("tokenizer.n_tokens must be > 0 for topk_channels")
            ch = list(getattr(tok, "channels", []) or [])
            if not ch:
                raise ValueError("topk_channels requires tokenizer.channels to be non-empty")
            bad = [c for c in ch if c not in ("value", "rank", "dropout")]
            if bad:
                raise ValueError(f"topk_channels invalid channels: {bad}")
            self.channels = tuple(ch)
            self.d_in = len(self.channels)

        else:  # patch
            P = int(getattr(tok, "patch_size", 0) or 0)
            if P <= 0:
                raise ValueError("patch_size must be > 0 for patch mode")
            self.patch_size = P

            # number of patches (ceil)
            T = (self.input_dim + P - 1) // P
            self.n_tokens = int(T)

            proj_dim = getattr(tok, "patch_proj_dim", None)
            if proj_dim is None:
                self.d_in = P
            else:
                proj_dim = int(proj_dim)
                if proj_dim <= 0:
                    raise ValueError("patch_proj_dim must be > 0 if set")
                self.patch_proj = nn.Linear(P, proj_dim)
                self.d_in = proj_dim

        # CLS token (learned) in token-embedding space
        self.cls_token: Optional[nn.Parameter] = None
        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_in))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() != 2:
            raise ValueError(f"_VectorToTokens expects x as (B,F); got shape {tuple(x.shape)}")
        B, Fdim = x.shape
        if Fdim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got F={Fdim}")

        key_padding_mask: Optional[torch.Tensor] = None

        if self.mode == "topk_scalar":
            K = min(int(self.n_tokens), Fdim)
            vals, _idx = torch.topk(x, k=K, dim=1, largest=True, sorted=False)  # (B,K)
            tokens = vals.unsqueeze(-1)  # (B,K,1)
            key_padding_mask = None

        elif self.mode == "topk_channels":
            K = min(int(self.n_tokens), Fdim)
            vals, _idx = torch.topk(x, k=K, dim=1, largest=True, sorted=True)  # sorted for stable rank
            feats: List[torch.Tensor] = []
            for c in self.channels:
                if c == "value":
                    feats.append(vals)
                elif c == "rank":
                    if K <= 1:
                        rank01 = torch.zeros((B, K), device=x.device, dtype=torch.float32)
                    else:
                        rank01 = torch.linspace(0.0, 1.0, steps=K, device=x.device, dtype=torch.float32)
                        rank01 = rank01.unsqueeze(0).expand(B, K)
                    feats.append(rank01)
                elif c == "dropout":
                    feats.append((vals == 0).to(torch.float32))
                else:
                    raise RuntimeError(f"Unhandled channel: {c!r}")
            tokens = torch.stack(feats, dim=-1)  # (B,K,C)
            key_padding_mask = None

        else:  # patch
            assert self.patch_size is not None
            P = int(self.patch_size)
            T = int(self.n_tokens)

            # pad to T*P then reshape
            pad = T * P - Fdim
            if pad > 0:
                x_pad = F.pad(x, (0, pad), mode="constant", value=0.0)  # (B, T*P)
            else:
                x_pad = x
            patches = x_pad.view(B, T, P)  # (B,T,P)

            if self.patch_proj is not None:
                tokens = self.patch_proj(patches)  # (B,T,D_in)
            else:
                tokens = patches  # (B,T,P)

            # With this implementation, we always have at least one real feature per patch
            # as long as input_dim>0, so we keep mask None.
            key_padding_mask = None

        # prepend CLS token if requested
        if self.add_cls_token:
            assert self.cls_token is not None
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,D_in)
            tokens = torch.cat([cls, tokens], dim=1)  # (B,1+T,D_in)
            if key_padding_mask is not None:
                cls_mask = torch.zeros((B, 1), device=x.device, dtype=torch.bool)
                key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        return tokens, key_padding_mask


# =============================================================================
# Transformer Gaussian encoder: vector -> tokens -> transformer -> (mu, logvar)
# =============================================================================

class TransformerGaussianEncoder(GaussianEncoder):
    """
    (B,F) -> tokens (B,T,D_in) -> TransformerEncoder -> (mu, logvar)
    """
    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        tokenizer: _VectorToTokens,
        tcfg: TransformerConfig,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.vec2tok = tokenizer
        self.latent_dim = int(latent_dim)

        # Learned pos-emb needs a max token length. Be careful: CLS increases token count by 1.
        if use_positional_encoding and tcfg.max_tokens is None:
            tcfg.max_tokens = int(self.vec2tok.n_tokens) + (1 if self.vec2tok.add_cls_token else 0)

        self.encoder = TransformerEncoder(
            cfg=tcfg,
            d_in=int(self.vec2tok.d_in),
            d_out=2 * int(latent_dim),
            use_positional_encoding=bool(use_positional_encoding),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, key_padding_mask = self.vec2tok(x)  # (B,T,D_in), (B,T) or None
        h = self.encoder(tokens, key_padding_mask=key_padding_mask)  # (B,2*latent_dim)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar


# =============================================================================
# Factory
# =============================================================================

def build_gaussian_encoder(*, uni_cfg: UniVIConfig, mod_cfg: ModalityConfig) -> GaussianEncoder:
    """
    Factory for per-modality Gaussian encoders.

    Config contract:
      - mod_cfg.encoder_type in {"mlp","transformer"} (default "mlp")
      - if transformer:
          mod_cfg.transformer: TransformerConfig
          mod_cfg.tokenizer:   TokenizerConfig
    """
    kind = (mod_cfg.encoder_type or "mlp").lower().strip()

    if kind == "mlp":
        return MLPGaussianEncoder(
            EncoderConfig(
                input_dim=int(mod_cfg.input_dim),
                hidden_dims=list(mod_cfg.encoder_hidden),
                latent_dim=int(uni_cfg.latent_dim),
                dropout=float(uni_cfg.encoder_dropout),
                batchnorm=bool(uni_cfg.encoder_batchnorm),
            )
        )

    if kind == "transformer":
        if mod_cfg.transformer is None:
            raise ValueError(
                f"Modality {mod_cfg.name!r}: encoder_type='transformer' requires mod_cfg.transformer."
            )
        if mod_cfg.tokenizer is None:
            raise ValueError(
                f"Modality {mod_cfg.name!r}: encoder_type='transformer' requires mod_cfg.tokenizer."
            )

        tok_cfg = mod_cfg.tokenizer
        tcfg = mod_cfg.transformer

        tokenizer = _VectorToTokens(input_dim=int(mod_cfg.input_dim), tok=tok_cfg)

        # If using learned pos-emb, align max_tokens to actual produced token length.
        if tcfg.max_tokens is None:
            tcfg.max_tokens = int(tokenizer.n_tokens) + (1 if tokenizer.add_cls_token else 0)

        return TransformerGaussianEncoder(
            input_dim=int(mod_cfg.input_dim),
            latent_dim=int(uni_cfg.latent_dim),
            tokenizer=tokenizer,
            tcfg=tcfg,
            use_positional_encoding=True,
        )

    raise ValueError(f"Unknown encoder_type={kind!r} for modality {mod_cfg.name!r}")

