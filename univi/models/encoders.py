# univi/models/encoders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import torch
from torch import nn
import torch.nn.functional as F

from ..config import UniVIConfig, ModalityConfig, TokenizerConfig, TransformerConfig as CFGTransformerConfig
from .mlp import build_mlp
from .transformer import TransformerEncoder, TransformerConfig as ModelTransformerConfig


# =============================================================================
# Small config helper
# =============================================================================

@dataclass
class EncoderConfig:
    input_dim: int
    hidden_dims: List[int]
    latent_dim: int
    dropout: float = 0.1
    batchnorm: bool = True


# =============================================================================
# Base encoders
# =============================================================================

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
# Tokenization: vector -> tokens
# =============================================================================

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
        If True, prepend a learned CLS token embedding to tokens.

    Notes
    -----
    - Expects dense float input (B,F). If modality data is sparse, densify upstream.
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

        # d_in per mode
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

        self.cls_token: Optional[nn.Parameter] = None
        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_in))

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_indices: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]],
    ]:
        if x.dim() != 2:
            raise ValueError(f"_VectorToTokens expects x as (B,F); got shape {tuple(x.shape)}")
        B, Fdim = x.shape
        if Fdim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got F={Fdim}")

        key_padding_mask: Optional[torch.Tensor] = None
        meta: Dict[str, Any] = {}

        if self.mode == "topk_scalar":
            K = min(int(self.n_tokens), Fdim)
            vals, idx = torch.topk(x, k=K, dim=1, largest=True, sorted=False)  # (B,K)
            tokens = vals.unsqueeze(-1)  # (B,K,1)
            key_padding_mask = None
            if return_indices:
                meta["topk_idx"] = idx  # (B,K)

        elif self.mode == "topk_channels":
            K = min(int(self.n_tokens), Fdim)
            vals, idx = torch.topk(x, k=K, dim=1, largest=True, sorted=True)  # (B,K)
            feats = []
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
            if return_indices:
                meta["topk_idx"] = idx  # (B,K)

        else:  # patch
            P = int(self.patch_size)
            T = int(self.n_tokens)
            pad = T * P - Fdim
            if pad > 0:
                x_pad = F.pad(x, (0, pad), mode="constant", value=0.0)  # (B, T*P)
            else:
                x_pad = x
            patches = x_pad.view(B, T, P)  # (B,T,P)

            tokens = self.patch_proj(patches) if self.patch_proj is not None else patches

            if pad > 0:
                real_counts = torch.full((T,), P, device=x.device, dtype=torch.int64)
                last_real = Fdim - (T - 1) * P
                if last_real < P:
                    real_counts[-1] = max(int(last_real), 0)
                key_padding_mask = (real_counts == 0).unsqueeze(0).expand(B, T)
            else:
                key_padding_mask = None

            if return_indices:
                meta["patch_size"] = P
                meta["n_patches"] = T
                meta["pad"] = pad

        if self.add_cls_token:
            assert self.cls_token is not None
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            tokens = torch.cat([cls, tokens], dim=1)
            if key_padding_mask is not None:
                cls_mask = torch.zeros((B, 1), device=x.device, dtype=torch.bool)
                key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        if return_indices:
            return tokens, key_padding_mask, meta
        return tokens, key_padding_mask


# =============================================================================
# Per-modality transformer Gaussian encoder
# =============================================================================

def _cfg_to_model_tcfg(cfg: CFGTransformerConfig) -> ModelTransformerConfig:
    return ModelTransformerConfig(
        d_model=int(cfg.d_model),
        num_heads=int(cfg.num_heads),
        num_layers=int(cfg.num_layers),
        dim_feedforward=int(cfg.dim_feedforward),
        dropout=float(cfg.dropout),
        attn_dropout=float(cfg.attn_dropout),
        activation=str(cfg.activation),
        pooling=str(cfg.pooling),
        max_tokens=None if cfg.max_tokens is None else int(cfg.max_tokens),
    )


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
        tcfg: ModelTransformerConfig,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.vec2tok = tokenizer

        if use_positional_encoding and tcfg.max_tokens is None:
            tcfg.max_tokens = int(self.vec2tok.n_tokens + (1 if self.vec2tok.add_cls_token else 0))

        self.encoder = TransformerEncoder(
            cfg=tcfg,
            d_in=int(self.vec2tok.d_in),
            d_out=2 * int(latent_dim),
            use_positional_encoding=bool(use_positional_encoding),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, key_padding_mask = self.vec2tok(x)
        h = self.encoder(tokens, key_padding_mask=key_padding_mask)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar


# =============================================================================
# NEW: Multimodal concatenated-token transformer Gaussian encoder (fused)
# =============================================================================

class MultiModalTransformerGaussianEncoder(nn.Module):
    """
    Fused encoder over multiple modalities by concatenating tokens.

    Produces ONE fused posterior q(z | x_all). It does not replace per-modality q(z|x_m).
    """
    def __init__(
        self,
        *,
        modalities: Sequence[str],
        input_dims: Dict[str, int],
        tokenizers: Dict[str, TokenizerConfig],
        transformer_cfg: CFGTransformerConfig,
        latent_dim: int,
        add_modality_embeddings: bool = True,
        use_positional_encoding: bool = True,
    ):
        super().__init__()

        self.modalities = list(modalities)
        self.latent_dim = int(latent_dim)

        tcfg_model = _cfg_to_model_tcfg(transformer_cfg)
        d_model = int(tcfg_model.d_model)

        self.vec2tok = nn.ModuleDict()
        self.proj = nn.ModuleDict()
        self.mod_emb = nn.ParameterDict() if add_modality_embeddings else None

        total_tokens = 0
        for m in self.modalities:
            tok_cfg_in = tokenizers[m]

            # force per-modality CLS off; use only one global CLS if pooling="cls"
            tok_cfg = TokenizerConfig(
                mode=tok_cfg_in.mode,
                n_tokens=int(tok_cfg_in.n_tokens),
                channels=tuple(tok_cfg_in.channels),
                patch_size=int(tok_cfg_in.patch_size),
                patch_proj_dim=None if tok_cfg_in.patch_proj_dim is None else int(tok_cfg_in.patch_proj_dim),
                add_cls_token=False,
            )

            tok = _VectorToTokens(input_dim=int(input_dims[m]), tok=tok_cfg)
            self.vec2tok[m] = tok
            self.proj[m] = nn.Linear(int(tok.d_in), d_model, bias=True)

            if self.mod_emb is not None:
                self.mod_emb[m] = nn.Parameter(torch.zeros(1, 1, d_model))

            total_tokens += int(tok.n_tokens)

        self.pooling = str(tcfg_model.pooling).lower().strip()
        self.use_global_cls = (self.pooling == "cls")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if self.use_global_cls else None

        if use_positional_encoding and tcfg_model.max_tokens is None:
            tcfg_model.max_tokens = int(total_tokens + (1 if self.use_global_cls else 0))

        self.encoder = TransformerEncoder(
            cfg=tcfg_model,
            d_in=d_model,
            d_out=2 * int(latent_dim),
            use_positional_encoding=bool(use_positional_encoding),
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        *,
        return_token_meta: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]],
    ]:
        tokens_list: List[torch.Tensor] = []
        masks_list: List[Optional[torch.Tensor]] = []

        meta: Dict[str, Any] = {"modalities": self.modalities, "slices": {}}
        t_cursor = 0

        for m in self.modalities:
            x = x_dict[m]

            if return_token_meta:
                tok, mask, mmeta = self.vec2tok[m](x, return_indices=True)
                meta[m] = mmeta
            else:
                tok, mask = self.vec2tok[m](x, return_indices=False)

            tok = self.proj[m](tok)  # (B,T,d_model)
            if self.mod_emb is not None:
                tok = tok + self.mod_emb[m]

            Tm = tok.shape[1]
            meta["slices"][m] = (t_cursor, t_cursor + Tm)
            t_cursor += Tm

            tokens_list.append(tok)
            masks_list.append(mask)

        tokens = torch.cat(tokens_list, dim=1)  # (B, T_total, d_model)

        key_padding_mask: Optional[torch.Tensor] = None
        if any(m is not None for m in masks_list):
            B = tokens.shape[0]
            built: List[torch.Tensor] = []
            for i in range(len(self.modalities)):
                mask = masks_list[i]
                if mask is None:
                    Tm = tokens_list[i].shape[1]
                    built.append(torch.zeros((B, Tm), device=tokens.device, dtype=torch.bool))
                else:
                    built.append(mask.to(dtype=torch.bool))
            key_padding_mask = torch.cat(built, dim=1)

        if self.use_global_cls:
            assert self.cls_token is not None
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            if key_padding_mask is not None:
                cls_mask = torch.zeros((tokens.shape[0], 1), device=tokens.device, dtype=torch.bool)
                key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        h = self.encoder(tokens, key_padding_mask=key_padding_mask)
        mu, logvar = torch.chunk(h, 2, dim=-1)

        if return_token_meta:
            return mu, logvar, meta
        return mu, logvar


# =============================================================================
# Factories
# =============================================================================

def build_gaussian_encoder(*, uni_cfg: UniVIConfig, mod_cfg: ModalityConfig) -> GaussianEncoder:
    """
    Factory for per-modality Gaussian encoders.

    Supported mod_cfg.encoder_type:
      - "mlp" (default)
      - "transformer"
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

        tokenizer = _VectorToTokens(
            input_dim=int(mod_cfg.input_dim),
            tok=mod_cfg.tokenizer,  # ✅ signature is tok=TokenizerConfig
        )

        tcfg = _cfg_to_model_tcfg(mod_cfg.transformer)
        if tcfg.max_tokens is None:
            tcfg.max_tokens = int(tokenizer.n_tokens + (1 if tokenizer.add_cls_token else 0))

        return TransformerGaussianEncoder(
            input_dim=int(mod_cfg.input_dim),
            latent_dim=int(uni_cfg.latent_dim),
            tokenizer=tokenizer,
            tcfg=tcfg,
            use_positional_encoding=True,
        )

    raise ValueError(f"Unknown encoder_type={kind!r} for modality {mod_cfg.name!r}")


def build_multimodal_transformer_encoder(
    *,
    uni_cfg: UniVIConfig,
    modalities: Sequence[ModalityConfig],
    fused_modalities: Optional[Sequence[str]] = None,
) -> MultiModalTransformerGaussianEncoder:
    """
    Build the fused multimodal transformer encoder from existing per-modality configs.

    Requirements:
      - uni_cfg.fused_transformer is set
      - each fused modality has mod_cfg.tokenizer set (even if its per-modality encoder is MLP)
    """
    if uni_cfg.fused_transformer is None:
        raise ValueError("UniVIConfig.fused_transformer must be set for fused_encoder_type='multimodal_transformer'.")

    mods = {m.name: m for m in modalities}
    use_names = list(fused_modalities) if fused_modalities is not None else list(mods.keys())

    input_dims = {n: int(mods[n].input_dim) for n in use_names}
    tokenizers: Dict[str, TokenizerConfig] = {}
    for n in use_names:
        if mods[n].tokenizer is None:
            raise ValueError(f"Fused multimodal encoder requires tokenizer for modality {n!r}")
        tokenizers[n] = mods[n].tokenizer

    return MultiModalTransformerGaussianEncoder(
        modalities=use_names,
        input_dims=input_dims,
        tokenizers=tokenizers,
        transformer_cfg=uni_cfg.fused_transformer,
        latent_dim=int(uni_cfg.latent_dim),
        add_modality_embeddings=bool(getattr(uni_cfg, "fused_add_modality_embeddings", True)),
        use_positional_encoding=True,
    )

