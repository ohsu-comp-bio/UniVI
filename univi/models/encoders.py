# univi/models/encoders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..config import UniVIConfig, ModalityConfig, TokenizerConfig, TransformerConfig as CFGTransformerConfig
from .mlp import build_mlp
from .transformer import TransformerEncoder, TransformerConfig as ModelTransformerConfig


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


class _VectorToTokens(nn.Module):
    """
    Turn a vector x (B, F) into tokens (B, T, D_in) and optional key_padding_mask.

    TokenizerConfig modes
    ---------------------
    - topk_scalar:
        Select top-k features per cell (by value), output tokens (B, K, 1)
    - topk_channels:
        Select top-k features per cell, output tokens (B, K, C) where C=len(channels)
        channels in {"value","rank","dropout"}:
          * value: raw x at selected indices
          * rank: rank01 among selected K tokens (0..1), per cell
          * dropout: indicator (value==0)
    - patch:
        Split contiguous features into patches of size P:
          tokens (B, T, P) or (B, T, patch_proj_dim) if patch_proj_dim is set

    add_cls_token:
        If True, prepend a learned CLS token embedding to tokens:
          tokens -> (B, T+1, D_in); mask -> prepend False (not masked)

    Notes
    -----
    - This module assumes dense float input (B,F). If your modality data is sparse,
      densify before passing into the model (which you already do in your dataset).
    """
    def __init__(self, *, input_dim: int, tok: TokenizerConfig):
        super().__init__()
        self.input_dim = int(input_dim)
        self.tok = tok

        mode = str(tok.mode).lower().strip()
        if mode not in ("topk_scalar", "topk_channels", "patch"):
            raise ValueError(f"Unknown tokenizer mode {tok.mode!r}")

        self.mode = mode
        self.add_cls_token = bool(tok.add_cls_token)

        # figure out d_in per mode
        if self.mode == "topk_scalar":
            self.n_tokens = int(tok.n_tokens)
            self.d_in = 1
            self.channels: Sequence[str] = ("value",)

        elif self.mode == "topk_channels":
            self.n_tokens = int(tok.n_tokens)
            ch = list(tok.channels)
            if not ch:
                raise ValueError("topk_channels requires tokenizer.channels non-empty")
            bad = [c for c in ch if c not in ("value", "rank", "dropout")]
            if bad:
                raise ValueError(f"topk_channels invalid channels: {bad}")
            self.channels = tuple(ch)
            self.d_in = len(self.channels)

        else:  # patch
            P = int(tok.patch_size)
            if P <= 0:
                raise ValueError("patch_size must be > 0")
            self.patch_size = P

            # number of patches (ceil)
            T = (self.input_dim + P - 1) // P
            self.n_tokens = int(T)

            proj_dim = tok.patch_proj_dim
            if proj_dim is None:
                self.patch_proj = None
                self.d_in = P
            else:
                proj_dim = int(proj_dim)
                if proj_dim <= 0:
                    raise ValueError("patch_proj_dim must be > 0 if set")
                self.patch_proj = nn.Linear(P, proj_dim)
                self.d_in = proj_dim

        # CLS token (learned)
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
            # topk over features
            vals, idx = torch.topk(x, k=K, dim=1, largest=True, sorted=False)  # (B,K)
            tokens = vals.unsqueeze(-1)  # (B,K,1)
            # No padding mask needed (fixed K)
            key_padding_mask = None

        elif self.mode == "topk_channels":
            K = min(int(self.n_tokens), Fdim)
            vals, idx = torch.topk(x, k=K, dim=1, largest=True, sorted=True)  # (B,K), sorted for stable rank
            feats = []
            for c in self.channels:
                if c == "value":
                    feats.append(vals)
                elif c == "rank":
                    # rank among selected K tokens: 0..1 (per cell)
                    if K <= 1:
                        rank01 = torch.zeros((B, K), device=x.device, dtype=torch.float32)
                    else:
                        rank01 = torch.linspace(
                            0.0, 1.0, steps=K, device=x.device, dtype=torch.float32
                        ).unsqueeze(0).expand(B, K)
                    feats.append(rank01)
                elif c == "dropout":
                    feats.append((vals == 0).to(torch.float32))
                else:
                    raise RuntimeError(f"Unhandled channel: {c!r}")
            tokens = torch.stack(feats, dim=-1)  # (B,K,C)
            key_padding_mask = None

        else:  # patch
            P = int(self.patch_size)
            T = int(self.n_tokens)
            # pad features to T*P
            pad = T * P - Fdim
            if pad > 0:
                x_pad = F.pad(x, (0, pad), mode="constant", value=0.0)  # (B, T*P)
            else:
                x_pad = x
            patches = x_pad.view(B, T, P)  # (B,T,P)

            if self.patch_proj is not None:
                tokens = self.patch_proj(patches)  # (B,T,proj_dim)
            else:
                tokens = patches  # (B,T,P)

            # Key padding mask: mark patches that are entirely padded zeros *only* if pad>0
            if pad > 0:
                # patches corresponding to padded region might be partly real; compute per-patch whether any real index exists
                # simplest: track real feature count per patch
                real_counts = torch.full((T,), P, device=x.device, dtype=torch.int64)
                # last patch may have fewer real features
                last_real = Fdim - (T - 1) * P
                if last_real < P:
                    real_counts[-1] = max(int(last_real), 0)
                # mask patches with 0 real features (rare; only if input_dim==0 which we disallow)
                key_padding_mask = (real_counts == 0).unsqueeze(0).expand(B, T)
            else:
                key_padding_mask = None

        # prepend CLS token if requested
        if self.add_cls_token:
            assert self.cls_token is not None
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            tokens = torch.cat([cls, tokens], dim=1)  # (B,1+T,D)
            if key_padding_mask is not None:
                cls_mask = torch.zeros((B, 1), device=x.device, dtype=torch.bool)
                key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        return tokens, key_padding_mask


class TransformerGaussianEncoder(GaussianEncoder):
    """
    (B,F) -> tokens (B,T,D_in) -> TransformerEncoder(d_in=D_in, d_out=2*latent_dim) -> (mu, logvar)
    """
    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        tokenizer: "_VectorToTokens",
        tcfg: "TransformerConfig",
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.vec2tok = tokenizer

        # If using learned pos-emb, TransformerConfig.max_tokens must match token count.
        if use_positional_encoding and tcfg.max_tokens is None:
            # safe default: set to tokenizer.n_tokens
            tcfg.max_tokens = int(self.vec2tok.n_tokens)

        self.encoder = TransformerEncoder(
            cfg=tcfg,
            d_in=int(self.vec2tok.d_in),
            d_out=2 * int(latent_dim),
            use_positional_encoding=bool(use_positional_encoding),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.vec2tok(x)          # (B,T,D_in)
        h = self.encoder(tokens)          # (B,2*latent_dim)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar


def build_gaussian_encoder(
    *,
    uni_cfg: UniVIConfig,
    mod_cfg: ModalityConfig,
) -> GaussianEncoder:
    """
    Factory for per-modality Gaussian encoders.

    Uses the object-based config:
      - mod_cfg.encoder_type in {"mlp","transformer"}
      - mod_cfg.transformer: TransformerConfig
      - mod_cfg.tokenizer:   TokenizerConfig
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
            raise ValueError(f"Modality {mod_cfg.name!r}: encoder_type='transformer' requires mod_cfg.transformer.")
        if mod_cfg.tokenizer is None:
            raise ValueError(f"Modality {mod_cfg.name!r}: encoder_type='transformer' requires mod_cfg.tokenizer.")

        tok_cfg = mod_cfg.tokenizer
        tcfg = mod_cfg.transformer

        # Map your TokenizerConfig -> internal _VectorToTokens modes.
        # (You can extend this mapping as you add modes.)
        mode = (tok_cfg.mode or "topk_scalar").lower().strip()

        if mode == "topk_scalar":
            tokenizer = _VectorToTokens(
                input_dim=int(mod_cfg.input_dim),
                tokenizer="linear",                  # cheap attention: F -> T
                n_tokens=int(tok_cfg.n_tokens),
                token_dim=1,
            )
        elif mode == "topk_channels":
            # multi-dim tokens: (B,T,C). easiest “consistent” version here is linear_multi.
            # If you truly want top-k-by-value selection, implement that in tokenizers.py.
            C = len(tuple(tok_cfg.channels))
            tokenizer = _VectorToTokens(
                input_dim=int(mod_cfg.input_dim),
                tokenizer="linear_multi",
                n_tokens=int(tok_cfg.n_tokens),
                token_dim=int(C),
            )
        elif mode == "patch":
            # simplest “patch-like” behavior consistent with this file: linear_multi where token_dim = patch_size
            # (true contiguous patches would need a dedicated patcher in tokenizers.py)
            D_in = int(tok_cfg.patch_proj_dim or tok_cfg.patch_size)
            tokenizer = _VectorToTokens(
                input_dim=int(mod_cfg.input_dim),
                tokenizer="linear_multi",
                n_tokens=int(tok_cfg.n_tokens),
                token_dim=D_in,
            )
        else:
            raise ValueError(f"Unknown tokenizer.mode={tok_cfg.mode!r} for modality {mod_cfg.name!r}")

        # Ensure max_tokens aligns with tokenizer output length when using learned pos emb
        if tcfg.max_tokens is None:
            tcfg.max_tokens = int(tokenizer.n_tokens)

        return TransformerGaussianEncoder(
            input_dim=int(mod_cfg.input_dim),
            latent_dim=int(uni_cfg.latent_dim),
            tokenizer=tokenizer,
            tcfg=tcfg,
            use_positional_encoding=True,
        )

    raise ValueError(f"Unknown encoder_type={kind!r} for modality {mod_cfg.name!r}")





